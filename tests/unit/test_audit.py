"""Publication audit checks bytes on disk, including corruptions validation missed."""

import contextlib
import copy
import dataclasses
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path

import duckdb
import h3
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from osm_worldcover.adapters.audit import _SCHEMA, audit_build
from osm_worldcover.adapters.audit import manifest as audit_manifest
from osm_worldcover.adapters.audit import rows as audit_rows
from osm_worldcover.adapters.audit import sql as audit_sql
from osm_worldcover.adapters.audit.report import AuditProblem, AuditReport, _Checks
from osm_worldcover.adapters.audit.sql import check_aggregates
from osm_worldcover.domain.card import render
from osm_worldcover.domain.nomenclature import CLASS_LABELS
from osm_worldcover.domain.splits import assign_cell
from osm_worldcover.release_commit import release_lock

SPLITS = ("train", "validation", "test")
SETTINGS = {
    "dataset_version": "2.0.0",
    "code_repository": "https://github.com/NoeFlandre/osm-worldcover",
    "source_dataset": "owner/source",
    "source_revision": "a" * 40,
    "worldcover_version": "v200",
    "worldcover_year": 2021,
    "dominance_threshold": 0.8,
    "min_words": 10,
    "h3_resolution": 5,
    "split_seed": 20180101,
    "split_ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
    "max_polygon_area_m2": 1e10,
}
TYPES = {"string": pa.string(), "int64": pa.int64(), "double": pa.float64()}
FIELD_TYPES = {
    "polygon_id": "string",
    "osm_type": "string",
    "region": "string",
    "name": "string",
    "wikidata": "string",
    "document_id": "string",
    "project": "string",
    "language": "string",
    "title": "string",
    "url": "string",
    "text": "string",
    "lead_text": "string",
    "worldcover_label": "string",
    "centroid_wkt": "string",
    "source_pbf": "string",
    "h3_cell": "string",
    "split": "string",
    "dataset_version": "string",
    "source_dataset": "string",
    "source_revision": "string",
    "worldcover_version": "string",
    "osm_id": "int64",
    "text_words": "int64",
    "worldcover_code": "int64",
    "worldcover_year": "int64",
    "dominant_fraction": "double",
    "observed_fraction": "double",
    "lat": "double",
    "lon": "double",
    "polygon_area_m2": "double",
}
SCHEMA = pa.schema([(key, TYPES[value]) for key, value in FIELD_TYPES.items()])


def rows():
    result = {}
    for lon in range(-175, 176):
        cell = h3.latlng_to_cell(40, lon, 5)
        split = assign_cell(cell).value
        if split in result:
            continue
        row = dict.fromkeys(FIELD_TYPES)
        text = f"{split} " + " ".join(["word"] * 12)
        row.update(
            polygon_id=f"place:{split}",
            osm_type="way",
            osm_id=lon + 180,
            region="place",
            document_id=f"doc:{split}",
            project="description",
            text=text,
            text_words=13,
            worldcover_code=10,
            worldcover_label="Tree cover",
            dominant_fraction=1.0,
            observed_fraction=1.0,
            lat=40.0,
            lon=float(lon),
            centroid_wkt=f"POINT ({lon} 40)",
            polygon_area_m2=100.0,
            source_pbf="place.osm.pbf",
            h3_cell=cell,
            split=split,
            **{
                key: SETTINGS[key]
                for key in (
                    "dataset_version",
                    "source_dataset",
                    "source_revision",
                    "worldcover_version",
                    "worldcover_year",
                )
            },
        )
        result[split] = row
        if len(result) == 3:
            break
    return result


def complete_ledger():
    context = {"settings": SETTINGS}
    fingerprint = hashlib.sha256(
        json.dumps(context, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    record = {
        "stem": "place",
        "polygons_seen": 3,
        "polygons_invalid": 0,
        "polygons_accepted": 3,
        "polygons_with_examples": 3,
        "source_links": 3,
        "source_documents": 3,
        "examples": 3,
        "rejections": {},
        "text_rejections": {},
        "tiles_missing": [],
    }
    return {
        "schema_version": 1,
        "full_source_complete": True,
        "complete": True,
        "selected_complete": True,
        "scope": "full",
        "context": context,
        "build_context_sha256": fingerprint,
        "expected_regions": ["place"],
        "selected_regions": ["place"],
        "processed_regions": ["place"],
        "missing_regions": [],
        "unprocessed_selected_regions": [],
        "region_counts": {
            "expected": 1,
            "selected": 1,
            "processed": 1,
            "missing": 0,
            "unprocessed_selected": 0,
        },
        "regions": [record],
        "totals": {key: value for key, value in record.items() if key != "stem"},
        "reconciliation": {"valid": True},
    }


@pytest.fixture
def build(tmp_path):
    examples = rows()
    for split, row in examples.items():
        pq.write_table(pa.Table.from_pylist([row], schema=SCHEMA), tmp_path / f"{split}.parquet")
    counts = {**dict.fromkeys(SPLITS, 1), "total": 3}
    lons = [row["lon"] for row in examples.values()]
    manifest = {
        "settings": SETTINGS,
        "processing": complete_ledger(),
        "deduplication": {},
        "rejections": {},
        "counts": {key: counts for key in ("examples", "polygons", "documents")},
        "class_distribution": [{"code": 10, "label": "Tree cover", "examples": 3, "share": 1.0}],
        "language_distribution": [{"language": None, "examples": 3, "share": 1.0}],
        "dominant_fraction": {"p50": 1.0, "p90": 1.0, "p99": 1.0},
        "geographic_coverage": {
            "h3_cells": 3,
            "regions": 1,
            "bbox": {
                "min_lon": min(lons),
                "min_lat": 40.0,
                "max_lon": max(lons),
                "max_lat": 40.0,
            },
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with release_lock(tmp_path):
        pass
    return tmp_path


def mutate(build, file_split="train", **changes):
    path = build / f"{file_split}.parquet"
    values = pq.read_table(path).to_pylist()
    values[0].update(changes)
    pq.write_table(pa.Table.from_pylist(values, schema=SCHEMA), path)


def problems(report):
    return {problem.code for problem in report.problems}


def test_complete_release_passes_and_report_serializes(build):
    report = audit_build(build, require_complete=True)
    assert report.ok
    assert report.rows == 3
    assert report.as_dict() == {"ok": True, "rows": 3, "problems": [], "warnings": []}
    json.dumps(report.as_dict())
    assert pickle.loads(pickle.dumps(report)) == report


def test_release_fixture_schema_is_independent_and_private_alias_remains_compatible():
    assert _SCHEMA == FIELD_TYPES


def test_audit_package_public_surface_is_stable() -> None:
    """The CLI, publish and tests import these names from the audit package path."""
    from osm_worldcover.adapters import audit as package

    assert package.__all__ == ["AuditProblem", "AuditReport", "audit_build"]
    assert package.AuditProblem.__module__ == "osm_worldcover.adapters.audit"
    assert package.AuditReport.__module__ == "osm_worldcover.adapters.audit"
    assert callable(package.audit_build)


def test_audit_package_helpers_and_pickling_are_stable() -> None:
    """Importers use the retained-diagnostics helper, and problems pickle by path."""
    from osm_worldcover.adapters import audit as package

    assert callable(package._retained_text_diagnostics)
    problem = package.AuditProblem("example_code", 2, ("row",))
    assert pickle.loads(pickle.dumps(problem)) == problem


def test_complete_release_accepts_partitioned_mixed_code_provenance(build):
    path = build / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["processing"].update(
        {
            "schema_version": 2,
            "assembly_code_revision": "b" * 40,
            "code_provenance": [
                {
                    "repository": SETTINGS["code_repository"],
                    "revision": "a" * 40,
                    "regions": ["place"],
                }
            ],
        }
    )
    path.write_text(json.dumps(manifest))
    assert audit_build(build, require_complete=True).ok


@pytest.mark.parametrize(
    "groups",
    [
        [],
        [{"repository": SETTINGS["code_repository"], "revision": "a" * 40, "regions": []}],
        [
            {
                "repository": SETTINGS["code_repository"],
                "revision": "a" * 40,
                "regions": ["place", "place"],
            }
        ],
        [{"repository": SETTINGS["code_repository"], "revision": "bad", "regions": ["place"]}],
    ],
)
def test_complete_release_rejects_invalid_code_provenance(build, groups):
    path = build / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["processing"].update(
        {"schema_version": 2, "assembly_code_revision": "b" * 40, "code_provenance": groups}
    )
    path.write_text(json.dumps(manifest))
    assert "invalid_processing_ledger" in problems(audit_build(build, require_complete=True))


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        ("polygon_id", None, "required_null:polygon_id"),
        ("document_id", "", "empty_identifier:document_id"),
        ("dominant_fraction", float("nan"), "invalid_fraction:dominant_fraction"),
        ("dominant_fraction", 0.79, "below_threshold"),
        ("observed_fraction", 1.1, "invalid_fraction:observed_fraction"),
        ("observed_fraction", 0.9, "dominant_exceeds_observed"),
        ("polygon_area_m2", -1.0, "invalid_area"),
        ("lat", 91.0, "invalid_coordinates"),
        ("h3_cell", "bad", "h3_cell_mismatch"),
        ("centroid_wkt", "POINT (0 0)", "centroid_wkt_mismatch"),
        ("source_revision", "b" * 40, "provenance_mismatch:source_revision"),
        ("worldcover_label", "wrong", "invalid_label"),
        ("text", "tiny", "unusable_text"),
        ("text_words", 123, "text_word_count_mismatch"),
    ],
)
def test_invalid_row_is_reported_without_crashing(build, key, value, expected):
    mutate(build, **{key: value})
    assert expected in problems(audit_build(build))


def test_null_text_fails_gracefully(build):
    mutate(build, text=None)
    assert "invalid_text" in problems(audit_build(build))


def test_file_split_must_match_rows(build):
    mutate(build, split="validation")
    assert "file_split_mismatch" in problems(audit_build(build))


def test_missing_and_wrong_schemas_fail(build):
    (build / "test.parquet").unlink()
    assert "missing_or_invalid_parquet" in problems(audit_build(build))
    pq.write_table(pa.table({"text": ["hello"]}), build / "test.parquet")
    assert "schema_mismatch" in problems(audit_build(build))


def test_completion_is_explicit_and_separate(build):
    path = build / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.pop("processing")
    path.write_text(json.dumps(manifest))
    assert audit_build(build).ok
    assert "source_processing_incomplete" in problems(audit_build(build, require_complete=True))


def test_manifest_counts_are_reconciled(build):
    path = build / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["counts"]["examples"]["train"] = 100
    path.write_text(json.dumps(manifest))
    assert "manifest_count_mismatch:examples" in problems(audit_build(build))


def test_identical_text_different_labels_is_visible_and_optionally_fatal(build):
    text = pq.read_table(build / "train.parquet")["text"][0].as_py()
    mutate(build, "test", text=text, worldcover_code=20, worldcover_label="Shrubland")
    report = audit_build(build)
    risks = Counter({warning.code: warning.count for warning in report.warnings})
    strict_problems = problems(audit_build(build, strict_text_leakage=True))
    assert (
        risks["identical_text_cross_split"],
        risks["identical_text_conflicting_labels"],
        "identical_text_cross_split" in problems(report),
        "identical_text_cross_split" in strict_problems,
    ) == (1, 1, False, True)


def test_identical_text_and_label_on_distinct_polygons_is_a_diagnostic(build):
    text = pq.read_table(build / "train.parquet")["text"][0].as_py()
    mutate(build, "test", text=text)
    report = audit_build(build)
    assert report.ok
    assert "duplicate_polygon_text_label_record" not in problems(report)
    assert any(warning.code == "identical_text_cross_split" for warning in report.warnings)


def test_exact_polygon_text_and_label_record_is_fatal(build):
    train = pq.read_table(build / "train.parquet").to_pylist()[0]
    mutate(build, "test", text=train["text"], polygon_id=train["polygon_id"])
    assert "duplicate_polygon_text_label_record" in problems(audit_build(build))


def test_document_polygon_and_h3_leakage_are_detected(build):
    train = pq.read_table(build / "train.parquet").to_pylist()[0]
    mutate(build, "test", **{key: train[key] for key in ("document_id", "polygon_id", "h3_cell")})
    assert {"document_id_leakage", "polygon_id_leakage", "h3_cell_leakage"} <= problems(
        audit_build(build)
    )


def test_card_is_required_only_when_requested(build):
    assert audit_build(build).ok
    assert "invalid_card_or_map" in problems(audit_build(build, require_card=True))


def _card_front_matter() -> str:
    metadata = {
        "license": "cc-by-sa-4.0",
        "configs": [
            {
                "config_name": "default",
                "data_files": [{"split": split, "path": f"{split}.parquet"} for split in SPLITS],
            }
        ],
    }
    return yaml.safe_dump(metadata, sort_keys=False)


def _write_card_and_map(build, front_matter: str, body: str, image: bytes) -> None:
    (build / "README.md").write_text(f"---\n{front_matter}---\n{body}")
    (build / "worldcover_centroids.png").write_bytes(image)


def _mixed_audit_build(build) -> None:
    train = pq.read_table(build / "train.parquet").to_pylist()[0]
    mutate(
        build,
        "validation",
        polygon_id=train["polygon_id"],
        document_id=train["document_id"],
        h3_cell=train["h3_cell"],
        text=train["text"],
    )
    mutate(build, "test", text=train["text"], worldcover_code=20, worldcover_label="Shrubland")

    path = build / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["counts"]["examples"]["total"] = 2
    path.write_text(json.dumps(manifest))

    front_matter = _card_front_matter().replace("cc-by-sa-4.0", "cc-by-4.0")
    _write_card_and_map(
        build,
        front_matter,
        "![map](worldcover_centroids.png)",
        b"\x89PNG\r\n\x1a\n",
    )


def _reported_problem(code: str, *examples: str) -> dict[str, object]:
    return {"code": code, "count": 1, "examples": examples}


def _mixed_report(strict: bool) -> dict[str, object]:
    text_hash = "03F27B855A54802390EB41A873CCD28AECDDC6AC25F296AE857D6DDDB86D75F5"
    cross_split = _reported_problem("identical_text_cross_split", text_hash)
    conflicting_labels = _reported_problem("identical_text_conflicting_labels", text_hash)
    problems = [
        _reported_problem("polygon_id_leakage", "place:train"),
        _reported_problem("document_id_leakage", "doc:train"),
        _reported_problem("h3_cell_leakage", "852340b7fffffff"),
        _reported_problem("inconsistent_polygon", "place:train"),
        _reported_problem(
            "duplicate_polygon_text_label_record",
            f"place:train:{text_hash}:10",
        ),
        _reported_problem("card_license_mismatch"),
        _reported_problem("card_missing_provenance:source_dataset"),
        _reported_problem("card_missing_provenance:source_revision"),
        _reported_problem("h3_cell_mismatch", "place:train"),
        _reported_problem(
            "manifest_count_mismatch:examples",
            "{'train': 1, 'validation': 1, 'test': 1, 'total': 3}",
        ),
        _reported_problem(
            "manifest_geographic_coverage_mismatch",
            "{'h3_cells': 2, 'regions': 1, 'bbox': {'min_lon': -175.0, "
            "'min_lat': 40.0, 'max_lon': -162.0, 'max_lat': 40.0}}",
        ),
        _reported_problem("manifest_mismatch:class_distribution"),
    ]
    warnings = [cross_split, conflicting_labels]
    if strict:
        problems.insert(5, cross_split)
        warnings.remove(cross_split)
    return {"ok": False, "rows": 3, "problems": problems, "warnings": warnings}


@pytest.mark.parametrize("strict", [False, True])
def test_mixed_release_report_preserves_order_severity_counts_and_examples(build, strict):
    _mixed_audit_build(build)

    report = audit_build(build, require_card=True, strict_text_leakage=strict)

    assert report.as_dict() == _mixed_report(strict)


def test_invalid_settings_report_preserves_the_second_early_return(build):
    path = build / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["settings"]["min_words"] = 0
    manifest["settings"]["source_revision"] = "un-pinned"
    manifest.pop("processing")
    path.write_text(json.dumps(manifest))

    front_matter = _card_front_matter().replace("cc-by-sa-4.0", "cc-by-4.0")
    _write_card_and_map(
        build,
        front_matter,
        "![map](worldcover_centroids.png)",
        b"\x89PNG\r\n\x1a\n",
    )

    report = audit_build(build, require_complete=True, require_card=True)

    assert report.as_dict() == {
        "ok": False,
        "rows": 0,
        "problems": [
            _reported_problem("card_license_mismatch"),
            _reported_problem("card_missing_provenance:source_dataset"),
            _reported_problem("card_missing_provenance:source_revision"),
            _reported_problem("invalid_settings"),
            _reported_problem("source_processing_incomplete"),
            _reported_problem("unpinned_source_revision"),
        ],
        "warnings": [],
    }


def _short_text_rows(template, count):
    rows = []
    for index in range(count):
        rows.append(
            {
                **template,
                "polygon_id": f"place:train-{index}",
                "document_id": f"doc:train-{index}",
                "osm_id": 1000 + index,
                "text": f"tiny {index}",
                "text_words": 2,
            }
        )
    return rows


def _report_finding(report, code):
    return next(problem for problem in report.problems if problem.code == code)


def test_repeated_findings_accumulate_counts_and_keep_five_examples(build):
    train_path = build / "train.parquet"
    template = pq.read_table(train_path).to_pylist()[0]
    rows_to_write = _short_text_rows(template, 7)
    pq.write_table(pa.Table.from_pylist(rows_to_write, schema=SCHEMA), train_path)

    report = audit_build(build)
    finding = _report_finding(report, "unusable_text")

    assert finding.count == 7
    assert finding.examples == tuple(f"place:train-{index}" for index in range(5))


def test_card_metadata_and_assets_are_audited(build):
    front_matter = _card_front_matter()
    valid_body = (
        f"{SETTINGS['source_dataset']} {SETTINGS['source_revision']} "
        "![map](worldcover_centroids.png)"
    )
    _write_card_and_map(build, front_matter, valid_body, b"\x89PNG\r\n\x1a\n")
    assert audit_build(build, require_card=True).ok

    _write_card_and_map(build, front_matter, "", b"not a png")
    codes = problems(audit_build(build, require_card=True))
    assert {
        "card_missing_provenance:source_revision",
        "card_missing_provenance:source_dataset",
        "card_missing_coverage_map",
        "invalid_coverage_map_png",
    } <= codes


def test_required_card_matches_mixed_code_provenance(build):
    path = build / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["processing"].update(
        {
            "schema_version": 2,
            "assembly_code_revision": "b" * 40,
            "code_provenance": [
                {
                    "repository": SETTINGS["code_repository"],
                    "revision": "a" * 40,
                    "regions": ["place"],
                }
            ],
        }
    )
    path.write_text(json.dumps(manifest))
    (build / "README.md").write_text(render(manifest))
    (build / "worldcover_centroids.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    assert audit_build(build, require_complete=True, require_card=True).ok

    card_path = build / "README.md"
    card_path.write_text(card_path.read_text().replace(f"[{'a' * 40}](", "[wrong-pin]("))
    codes = problems(audit_build(build, require_complete=True, require_card=True))
    assert "card_missing_code_provenance" in codes


def test_seven_decimal_centroid_rounding_is_accepted(build):
    original = pq.read_table(build / "train.parquet").to_pylist()[0]
    lat = original["lat"] + 0.000000041
    mutate(build, lat=lat)
    assert "centroid_wkt_mismatch" not in problems(audit_build(build))


@pytest.mark.parametrize("broken", [[], {"settings": None}, {"settings": [], "counts": {}}])
def test_malformed_manifest_shape_fails_gracefully(build, broken):
    (build / "manifest.json").write_text(json.dumps(broken))
    assert "missing_or_invalid_manifest" in problems(audit_build(build))


@pytest.mark.parametrize("change", [{"h3_resolution": 5.5}, {"split_ratios": None}])
def test_invalid_settings_fail_gracefully(build, change):
    path = build / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["settings"].update(change)
    path.write_text(json.dumps(manifest))
    assert "invalid_settings" in problems(audit_build(build))


@pytest.mark.parametrize(
    "change",
    [
        {"full_source_complete": True},
        {"expected_regions": ["place", "missing"]},
        {"build_context_sha256": "bad"},
        {"region_counts": {"expected": 9, "selected": 1, "processed": 1}},
    ],
)
def test_inconsistent_completion_claim_is_rejected(build, change):
    path = build / "manifest.json"
    manifest = json.loads(path.read_text())
    if change == {"full_source_complete": True}:
        manifest["processing"] = change
    else:
        manifest["processing"].update(change)
    path.write_text(json.dumps(manifest))
    assert "invalid_processing_ledger" in problems(audit_build(build, require_complete=True))


def test_processing_examples_reconcile_dedup_removals(build):
    path = build / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["deduplication"] = {"duplicate_polygon_text_label_records": 1}
    path.write_text(json.dumps(manifest))
    assert "invalid_processing_ledger" in problems(audit_build(build, require_complete=True))


def test_processing_duplicate_text_analysis_reconciles(build):
    path = build / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["deduplication"] = {
        "duplicate_objects_across_regions": 0,
        "duplicate_polygon_text_label_records": 0,
        "documents_split_across_splits": 0,
    }
    manifest["deduplication_analysis"] = {
        "duplicate_polygon_text_label_groups": 0,
        "duplicate_records_removed": 0,
        "duplicate_record_groups_crossing_splits": 0,
        "duplicate_records_removed_from_cross_split_groups": 0,
        "duplicate_records_removed_by_text_words": {
            **dict.fromkeys((str(words) for words in range(1, 10)), 0),
            "10+": 0,
        },
        "retained_identical_text_label_groups": 0,
        "retained_identical_text_label_rows": 0,
        "retained_identical_text_label_cross_split_groups": 0,
        "retained_identical_text_label_cross_split_rows": 0,
        "identical_text_cross_split_groups": 0,
        "identical_text_cross_split_rows": 0,
    }
    path.write_text(json.dumps(manifest))

    assert audit_build(build, require_complete=True).ok


def test_processing_duplicate_text_length_histogram_must_be_complete(build):
    path = build / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["deduplication"] = {
        "duplicate_objects_across_regions": 0,
        "duplicate_polygon_text_label_records": 0,
        "documents_split_across_splits": 0,
    }
    manifest["deduplication_analysis"] = {
        "duplicate_polygon_text_label_groups": 0,
        "duplicate_records_removed": 0,
        "duplicate_record_groups_crossing_splits": 0,
        "duplicate_records_removed_from_cross_split_groups": 0,
        "duplicate_records_removed_by_text_words": {"10+": 0},
        "retained_identical_text_label_groups": 0,
        "retained_identical_text_label_rows": 0,
        "retained_identical_text_label_cross_split_groups": 0,
        "retained_identical_text_label_cross_split_rows": 0,
        "identical_text_cross_split_groups": 0,
        "identical_text_cross_split_rows": 0,
    }
    path.write_text(json.dumps(manifest))

    assert "invalid_processing_ledger" in problems(audit_build(build, require_complete=True))


_SESSION_SETTINGS = ("memory_limit", "threads", "preserve_insertion_order", "temp_directory")


class _SettingsRecorder:
    """Forward every call to a real DuckDB connection, and note its settings as it closes."""

    def __init__(self, connection) -> None:
        self.raw = connection
        self.settings_at_close: dict[str, object] = {}

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def close(self) -> None:
        self.settings_at_close = {
            name: self.raw.execute(f"SELECT current_setting('{name}')").fetchone()[0]
            for name in _SESSION_SETTINGS
        }
        self.raw.close()


@pytest.fixture
def opened_connections(monkeypatch):
    opened: list[_SettingsRecorder] = []
    connect = duckdb.connect

    def recording_connect(*args, **kwargs):
        recorder = _SettingsRecorder(connect(*args, **kwargs))
        opened.append(recorder)
        return recorder

    monkeypatch.setattr(duckdb, "connect", recording_connect)
    return opened


def test_release_audit_runs_its_aggregates_under_the_audit_memory_budget(
    build, opened_connections, reported_setting
) -> None:
    assert audit_build(build).ok

    [recorder] = opened_connections
    settings = recorder.settings_at_close
    assert settings["memory_limit"] == reported_setting("memory_limit", "256MB")
    assert settings["threads"] == 2
    assert settings["preserve_insertion_order"] is False


def test_release_audit_spills_inside_its_own_scratch_directory(build, opened_connections) -> None:
    assert audit_build(build).ok

    [recorder] = opened_connections
    spill = Path(recorder.settings_at_close["temp_directory"])
    assert spill.name == "spill"
    assert spill.parent.name.startswith("owc-audit-")


def test_aggregate_audit_closes_its_connection_when_a_query_fails(
    tmp_path, opened_connections
) -> None:
    checks = _Checks(AuditReport())

    with pytest.raises(duckdb.IOException):
        check_aggregates(
            [tmp_path / "missing.parquet"], tmp_path / "hashes.parquet", tmp_path, {}, checks, False
        )

    [recorder] = opened_connections
    with pytest.raises(duckdb.ConnectionException):
        recorder.raw.execute("SELECT 1")


def _checks() -> _Checks:
    return _Checks(AuditReport())


def _manifest_shape(**changes) -> dict:
    shape = {
        "settings": {},
        "counts": {},
        "geographic_coverage": {},
        "dominant_fraction": {},
        "class_distribution": [],
        "language_distribution": [],
    }
    shape.update(changes)
    return shape


@pytest.mark.parametrize(
    "changes, valid",
    [
        ({}, True),
        ({"processing": {}}, True),
        ({"class_distribution": [{"code": 10, "examples": 3}]}, True),
        ({"processing": []}, False),
        ({"deduplication_analysis": []}, False),
        ({"deduplication_analysis": {}}, True),
        ({"settings": None}, False),
        ({"counts": []}, False),
        ({"class_distribution": None}, False),
        ({"class_distribution": ["10"]}, False),
        ({"language_distribution": [{"language": "en"}]}, False),
    ],
)
def test_manifest_shape_checks_every_section(changes, valid) -> None:
    assert audit_manifest._valid_manifest_shape(_manifest_shape(**changes)) is valid


def test_manifest_shape_rejects_a_missing_distribution_section() -> None:
    shape = _manifest_shape()
    del shape["language_distribution"]
    assert audit_manifest._valid_manifest_shape(shape) is False


@pytest.mark.parametrize(
    "changes, valid",
    [
        ({"dominance_threshold": 0}, True),
        ({"dominance_threshold": 1}, True),
        ({"dominance_threshold": 1.0001}, False),
        ({"dominance_threshold": -0.0001}, False),
        ({"min_words": 1}, True),
        ({"min_words": 0}, False),
        ({"h3_resolution": 0}, True),
        ({"h3_resolution": 15}, True),
        ({"h3_resolution": 16}, False),
        ({"split_seed": "1"}, False),
        ({"deduplication_policy": "another-policy"}, False),
    ],
)
def test_settings_admit_exactly_the_documented_ranges(changes, valid) -> None:
    checks = _checks()
    audit_manifest._check_settings({**SETTINGS, **changes}, checks)
    assert ("invalid_settings" not in checks.counts) is valid


def test_settings_without_provenance_are_invalid_even_with_the_default_policy() -> None:
    settings = {key: value for key, value in SETTINGS.items() if key != "worldcover_year"}
    checks = _checks()
    audit_manifest._check_settings(settings, checks)
    assert checks.counts["invalid_settings"] == 1


@pytest.mark.parametrize(
    "value, rejected",
    [
        (0.0, False),
        (1.0, False),
        (1 + 5e-7, False),
        (1 + 1e-6, False),
        (1 + 5e-6, True),
        (-1e-9, True),
        (1.5, True),
        (float("nan"), True),
        (None, True),
    ],
)
def test_fraction_bounds_accept_the_closed_unit_interval(value, rejected) -> None:
    checks = _checks()
    audit_rows._check_fractions(
        {"polygon_id": "p", "dominant_fraction": value, "observed_fraction": 0.5}, checks
    )
    assert bool(checks.counts["invalid_fraction:dominant_fraction"]) is rejected
    assert "invalid_fraction:observed_fraction" not in checks.counts


@pytest.mark.parametrize(
    "value, finite",
    [
        (0, True),
        (1.5, True),
        (float("inf"), False),
        (float("nan"), False),
        (None, False),
        ("1", False),
    ],
)
def test_finite_accepts_only_real_finite_numbers(value, finite) -> None:
    assert audit_rows._finite(value) is finite


@pytest.mark.parametrize(
    "area, cap, code",
    [
        (1.0, None, None),
        (1e12, None, None),
        (0.0, None, "invalid_area"),
        (-1.0, None, "invalid_area"),
        (float("nan"), None, "invalid_area"),
        (1e10, 1e10, None),
        (1e10 * (1 + 5e-7), 1e10, None),
        (1e10 * (1 + 1e-6), 1e10, None),
        (1e10 * (1 + 5e-6), 1e10, "area_above_cap"),
    ],
)
def test_polygon_area_must_be_positive_and_within_the_cap(area, cap, code) -> None:
    checks = _checks()
    audit_rows._check_area(
        {"polygon_id": "p", "polygon_area_m2": area}, {"max_polygon_area_m2": cap}, checks
    )
    assert set(checks.counts) == ({code} if code else set())


@pytest.mark.parametrize(
    "offset, flagged",
    [(0.0, False), (-5e-13, False), (-1e-12, False), (-5e-12, True), (-0.01, True), (0.1, False)],
)
def test_dominance_below_threshold_uses_a_tiny_tolerance(offset, flagged) -> None:
    checks = _checks()
    threshold = 0.8
    audit_rows._check_dominance_threshold(
        threshold + offset, {"dominance_threshold": threshold}, "p", checks
    )
    assert bool(checks.counts["below_threshold"]) is flagged


@pytest.mark.parametrize(
    "dominant, flagged",
    [(0.5, False), (0.5 + 5e-7, False), (0.5 + 1e-6, False), (0.5 + 5e-6, True)],
)
def test_dominant_fraction_may_exceed_observed_only_by_a_micro_tolerance(dominant, flagged) -> None:
    checks = _checks()
    audit_rows._check_dominance_observation(dominant, 0.5, "p", checks)
    assert bool(checks.counts["dominant_exceeds_observed"]) is flagged


@pytest.mark.parametrize(
    "lat, lon, flagged",
    [
        (90, 180, False),
        (-90, -180, False),
        (90.5, 0, True),
        (0, 180.5, True),
        (-90.5, 0, True),
        (0, -180.5, True),
        (float("nan"), 0, True),
    ],
)
def test_coordinates_are_checked_inclusively_at_the_limits(lat, lon, flagged) -> None:
    checks = _checks()
    row = {
        "polygon_id": "p",
        "lat": lat,
        "lon": lon,
        "h3_cell": "x",
        "split": "train",
        "centroid_wkt": f"POINT ({lon} {lat})",
    }
    audit_rows._check_position(row, dict(SETTINGS), checks)
    assert bool("invalid_coordinates" in checks.counts) is flagged


@pytest.mark.parametrize(
    "lon_text, valid",
    [
        ("10.0", True),
        ("10.00000005", True),
        ("10.00000006", False),
        ("9.99999995", True),
        ("9.99999994", False),
        ("ten", False),
        ("1.2.3", False),
    ],
)
def test_centroid_wkt_tolerates_only_the_seven_decimal_rounding(lon_text, valid) -> None:
    checks = _checks()
    row = {
        "polygon_id": "p",
        "lon": 10.0,
        "lat": 45.0,
        "centroid_wkt": f"POINT ({lon_text} 45.0)",
    }
    audit_rows._check_centroid(row, checks)
    assert ("centroid_wkt_mismatch" not in checks.counts) is valid


@pytest.mark.parametrize(
    "text, expected_code",
    [
        (" ".join(["w"] * 9), "unusable_text"),
        (" ".join(["w"] * 10), None),
        ("w  x " * 5, "unnormalized_text"),
        (12, "invalid_text"),
    ],
)
def test_text_length_and_normalization_are_checked(text, expected_code) -> None:
    checks = _checks()
    row = {"polygon_id": "p", "text": text, "text_words": len(str(text).split())}
    audit_rows._check_text(row, {"min_words": 10}, checks)
    assert set(checks.counts) == ({expected_code} if expected_code else set())


def test_text_word_count_must_match_the_stored_count() -> None:
    checks = _checks()
    row = {"polygon_id": "p", "text": " ".join(["w"] * 10), "text_words": 9}
    audit_rows._check_text(row, {"min_words": 10}, checks)
    assert set(checks.counts) == {"text_word_count_mismatch"}


def test_share_is_rounded_to_six_places_against_the_scanned_row_count() -> None:
    checks = _checks()
    checks.report.rows = 3
    audit_sql._check_shares([{"examples": 1, "share": round(1 / 3, 6)}], checks)
    assert not checks.counts
    audit_sql._check_shares([{"examples": 1, "share": 0.33333}], checks)
    assert checks.counts == Counter({"manifest_share_mismatch": 1})


def test_share_is_zero_for_an_empty_release() -> None:
    checks = _checks()
    audit_sql._check_shares([{"examples": 0, "share": 0}], checks)
    assert not checks.counts
    audit_sql._check_shares([{"examples": 0, "share": 0.5}], checks)
    assert checks.counts == Counter({"manifest_share_mismatch": 1})


def test_class_labels_must_match_the_nomenclature() -> None:
    checks = _checks()
    audit_sql._check_class_labels(
        {"class_distribution": [{"code": 10, "label": CLASS_LABELS[10]}]}, checks
    )
    assert not checks.counts
    audit_sql._check_class_labels(
        {"class_distribution": [{"code": 10, "label": "Not a class"}]}, checks
    )
    assert checks.counts == Counter({"manifest_class_label_mismatch": 1})


def test_dominant_quantiles_are_compared_after_rounding_to_six_places() -> None:
    checks = _checks()
    row = (1, 1, 0, 0, 1, 1, [0.1234564, 0.9, 0.99])
    manifest = {"dominant_fraction": {"p50": 0.123456, "p90": 0.9, "p99": 0.99}}
    audit_sql._check_dominant_quantiles(row, manifest, checks)
    assert not checks.counts
    manifest["dominant_fraction"]["p50"] = 0.12346
    audit_sql._check_dominant_quantiles(row, manifest, checks)
    assert checks.counts == Counter({"manifest_quantile_mismatch": 1})


def test_an_empty_release_has_no_dominant_quantiles() -> None:
    checks = _checks()
    audit_sql._check_dominant_quantiles(
        (0, 0, None, None, None, None, None), {"dominant_fraction": {}}, checks
    )
    assert not checks.counts
    audit_sql._check_dominant_quantiles(
        (0, 0, None, None, None, None, None), {"dominant_fraction": {"p50": 0.5}}, checks
    )
    assert checks.counts == Counter({"manifest_quantile_mismatch": 1})


def test_geographic_coverage_must_match_the_aggregates_exactly() -> None:
    checks = _checks()
    row = (4, 2, 1.0, 2.0, 3.0, 4.0, None)
    manifest = {
        "geographic_coverage": {
            "h3_cells": 4,
            "regions": 2,
            "bbox": {"min_lon": 1.0, "min_lat": 2.0, "max_lon": 3.0, "max_lat": 4.0},
        }
    }
    audit_sql._check_geographic_coverage(row, manifest, checks)
    assert not checks.counts
    manifest["geographic_coverage"]["regions"] = 3
    audit_sql._check_geographic_coverage(row, manifest, checks)
    assert checks.counts == Counter({"manifest_geographic_coverage_mismatch": 1})


REVISION = "b" * 40
CODE_GROUP = {"repository": SETTINGS["code_repository"], "revision": REVISION}
ANALYSIS_COUNTERS = (
    "duplicate_polygon_text_label_groups",
    "duplicate_records_removed",
    "duplicate_record_groups_crossing_splits",
    "duplicate_records_removed_from_cross_split_groups",
    "retained_identical_text_label_groups",
    "retained_identical_text_label_rows",
    "retained_identical_text_label_cross_split_groups",
    "retained_identical_text_label_cross_split_rows",
    "identical_text_cross_split_groups",
    "identical_text_cross_split_rows",
)
DIGITS = tuple(str(words) for words in range(1, 10))


def _completion_manifest() -> dict:
    """A complete v2 ledger manifest that the completion checks accept as written."""
    ledger = copy.deepcopy(complete_ledger())
    ledger["schema_version"] = 2
    ledger["assembly_code_revision"] = REVISION
    ledger["code_provenance"] = [{**CODE_GROUP, "regions": ["place"]}]
    return {
        "settings": copy.deepcopy(SETTINGS),
        "processing": ledger,
        "deduplication": {},
        "rejections": {},
        "counts": {"examples": {"total": 3}},
    }


def _at(*path, value):
    """Return a mutation that sets one nested key of a manifest."""

    def mutate(manifest: dict) -> None:
        target = manifest
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return mutate


def _then(*mutations):
    def mutate(manifest: dict) -> None:
        for mutation in mutations:
            mutation(manifest)

    return mutate


def _rehashed(mutation):
    """Apply a context edit and recompute the digest so only that edit is judged."""

    def mutate(manifest: dict) -> None:
        mutation(manifest)
        ledger = manifest["processing"]
        context = json.dumps(ledger["context"], sort_keys=True, separators=(",", ":"))
        ledger["build_context_sha256"] = hashlib.sha256(context.encode()).hexdigest()

    return mutate


def _analysis(drops: int = 0, **counters):
    """A duplicate-text analysis whose record total matches the deduplication drops."""
    analysis = dict.fromkeys(ANALYSIS_COUNTERS, 0)
    analysis["duplicate_records_removed_by_text_words"] = dict.fromkeys((*DIGITS, "10+"), 0)
    analysis.update(counters)
    return _then(
        _at("deduplication", value={"duplicate_polygon_text_label_records": drops}),
        _at("deduplication_analysis", value=analysis),
    )


def _code(*groups):
    return _at("processing", "code_provenance", value=list(groups))


def _ledger_problems(manifest: dict) -> list[str]:
    checks = _checks()
    audit_manifest._check_completion(manifest, True, checks)
    return checks.samples["invalid_processing_ledger"]


def test_complete_ledger_with_analysis_passes_the_completion_checks() -> None:
    assert _ledger_problems(_completion_manifest()) == []
    manifest = _completion_manifest()
    _analysis()(manifest)
    assert _ledger_problems(manifest) == []


def test_a_context_revision_equal_to_the_assembly_revision_is_accepted() -> None:
    manifest = _completion_manifest()
    _rehashed(_at("processing", "context", "code_revision", value=REVISION))(manifest)
    assert _ledger_problems(manifest) == []


LEDGER_MESSAGES = [
    # Inventories and pending regions.
    (_at("processing", "expected_regions", value="place"), "expected inventory must be a list"),
    (_at("processing", "expected_regions", value=[""]), "invalid region name"),
    (_at("processing", "expected_regions", value=[1]), "invalid region name"),
    (
        _then(
            _at("processing", "expected_regions", value=[]),
            _at("processing", "region_counts", "expected", value=0),
        ),
        "empty expected source inventory",
    ),
    (_at("processing", "expected_regions", value=["place", "place"]), "duplicate expected regions"),
    (_at("processing", "region_counts", "expected", value=2), "wrong expected count"),
    (_at("processing", "selected_regions", value=None), "selected inventory must be a list"),
    (
        _then(
            _at("processing", "expected_regions", value=[]),
            _at("processing", "selected_regions", value=[]),
            _at("processing", "processed_regions", value=[]),
            _at("processing", "region_counts", "expected", value=0),
            _at("processing", "region_counts", "selected", value=0),
            _at("processing", "region_counts", "processed", value=0),
        ),
        "empty expected source inventory",
    ),
    (
        _then(
            _at("processing", "selected_regions", value=["place", "other"]),
            _at("processing", "region_counts", "selected", value=2),
        ),
        "full source inventories differ",
    ),
    (_at("processing", "missing_regions", value=["x"]), "nonempty missing_regions"),
    (
        _at("processing", "unprocessed_selected_regions", value=["x"]),
        "nonempty unprocessed_selected_regions",
    ),
    (_at("processing", "region_counts", "missing", value=1), "nonzero missing count"),
    (
        _at("processing", "region_counts", "unprocessed_selected", value=1),
        "nonzero unprocessed_selected count",
    ),
    (_at("processing", "scope", value="subset"), "subset scope cannot be published as complete"),
    (_at("processing", "complete", value=False), "complete is not true"),
    (_at("processing", "selected_complete", value=False), "selected_complete is not true"),
    # Context digest and settings.
    (_at("processing", "build_context_sha256", value="0" * 64), "build context hash mismatch"),
    (_at("processing", "schema_version", value=3), "unsupported processing ledger schema"),
    (_at("settings", "min_words", value=5), "build context settings mismatch"),
    # Code provenance.
    (_at("processing", "assembly_code_revision", value="abc"), "invalid assembly code revision"),
    (
        _rehashed(_at("processing", "context", "code_revision", value="abc")),
        "invalid context code revision",
    ),
    (
        _rehashed(_at("processing", "context", "code_revision", value="c" * 40)),
        "assembly code revision differs from the ledger context",
    ),
    (_code(), "missing region code provenance"),
    (_code("place"), "invalid code provenance group"),
    (_code({**CODE_GROUP, "repository": "", "regions": ["place"]}), "invalid code repository"),
    (
        _code({**CODE_GROUP, "repository": "https://example.com/x", "regions": ["place"]}),
        "code repository differs from the ledger context",
    ),
    (
        _code({**CODE_GROUP, "revision": "abc", "regions": ["place"]}),
        "invalid region code revision",
    ),
    (_code({**CODE_GROUP, "regions": []}), "empty code provenance region group"),
    (_code({**CODE_GROUP, "regions": [""]}), "invalid code provenance region"),
    (
        _code({**CODE_GROUP, "regions": ["place", "place"]}),
        "duplicate code provenance region",
    ),
    (
        _code({**CODE_GROUP, "regions": ["place"]}, {**CODE_GROUP, "regions": ["place"]}),
        "duplicate code provenance region",
    ),
    (
        _code({**CODE_GROUP, "regions": ["other"]}),
        "code provenance does not match processed regions",
    ),
    # Region receipts and their counters.
    (_at("processing", "regions", 0, "stem", value="other"), "region receipt inventory mismatch"),
    (
        _at("processing", "regions", 0, "polygons_seen", value=-1),
        "invalid region counter polygons_seen",
    ),
    (
        _at("processing", "regions", 0, "polygons_seen", value=True),
        "invalid region counter polygons_seen",
    ),
    (
        _at("processing", "context", "note", value=float("nan")),
        "Out of range float values are not JSON compliant: nan",
    ),
    (_at("processing", "regions", 0, "rejections", value=[]), "invalid rejections mapping"),
    (_at("processing", "regions", 0, "rejections", value={"x": -1}), "invalid rejections counter"),
    (
        _at("processing", "regions", 0, "text_rejections", value=[]),
        "invalid text_rejections mapping",
    ),
    (
        _at("processing", "regions", 0, "text_rejections", value={"x": -1}),
        "invalid text_rejections counter",
    ),
    (
        _at("processing", "regions", 0, "polygons_seen", value=4),
        "spatial region accounting mismatch",
    ),
    (
        _at("processing", "regions", 0, "polygons_with_examples", value=2),
        "text region accounting mismatch",
    ),
    (_at("processing", "regions", 0, "examples", value=2), "insufficient region examples"),
    (
        _at("processing", "totals", "polygons_seen", value=4),
        "aggregate region count mismatch: polygons_seen",
    ),
    (
        _at("processing", "totals", "rejections", value={"x": 1}),
        "aggregate rejections mismatch",
    ),
    (
        _at("processing", "totals", "text_rejections", value={"y": 1}),
        "aggregate text_rejections mismatch",
    ),
    (_at("rejections", value={"x": 1}), "manifest rejection counters mismatch"),
    (
        _at("processing", "reconciliation", value={"valid": False}),
        "ledger reconciliation not valid",
    ),
    (_at("deduplication", value={"x": -1}), "invalid deduplication counters"),
    (_at("deduplication", value={"x": 1}), "pre/post-dedup example counts mismatch"),
    # Duplicate-text analysis.
    (_analysis(duplicate_polygon_text_label_groups=-1), "invalid duplicate-text analysis counters"),
    (
        _analysis(duplicate_records_removed_by_text_words=[]),
        "invalid duplicate-record length counters",
    ),
    (
        _then(
            _analysis(),
            _at(
                "deduplication_analysis",
                "duplicate_records_removed_by_text_words",
                value={key: 0 for key in DIGITS},
            ),
        ),
        "invalid duplicate-record length counters",
    ),
    (
        _analysis(
            duplicate_records_removed_by_text_words={**dict.fromkeys(DIGITS, 0), "10+": 0, "1": 1}
        ),
        "duplicate-record length counters do not reconcile",
    ),
    (
        _then(
            _at("counts", "examples", "total", value=2),
            _analysis(drops=1),
        ),
        "duplicate-record analysis does not match deduplication total",
    ),
    (
        _analysis(duplicate_record_groups_crossing_splits=1),
        "duplicate-record split counters are inconsistent",
    ),
    (
        _analysis(duplicate_records_removed_from_cross_split_groups=1),
        "duplicate-record split counters are inconsistent",
    ),
]


@pytest.mark.parametrize("mutation, message", LEDGER_MESSAGES)
def test_each_ledger_rejection_names_its_specific_reason(mutation, message) -> None:
    manifest = _completion_manifest()
    mutation(manifest)
    assert _ledger_problems(manifest) == [message]


@contextlib.contextmanager
def _release_connection(*rows):
    """An in-memory DuckDB release view with the columns the aggregate checks read."""
    connection = duckdb.connect()
    try:
        connection.execute(
            "CREATE TABLE release (split VARCHAR, polygon_id VARCHAR, document_id VARCHAR, "
            "worldcover_code INTEGER, language VARCHAR, h3_cell VARCHAR, region VARCHAR, "
            "lon DOUBLE, lat DOUBLE, dominant_fraction DOUBLE)"
        )
        for row in rows:
            connection.execute("INSERT INTO release VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", row)
        yield connection
    finally:
        connection.close()


def _train_row(index: int = 0) -> tuple:
    return ("train", f"p{index}", f"d{index}", 10, None, f"cell{index}", "region", 1.0, 45.0, 1.0)


def _manifest_counts(counts: dict) -> dict:
    return {"counts": {key: counts for key in ("examples", "polygons", "documents")}}


def test_splits_without_rows_are_counted_as_zero_in_the_manifest_check() -> None:
    with _release_connection(_train_row()) as connection:
        checks = _checks()
        checks.report.rows = 1
        audit_sql._check_manifest(
            connection,
            _manifest_counts({"train": 1, "validation": 0, "test": 0, "total": 1}),
            checks,
        )
        assert "manifest_count_mismatch:examples" not in checks.counts
        assert "empty_dataset" not in checks.counts

        wrong = _checks()
        wrong.report.rows = 1
        audit_sql._check_manifest(
            connection,
            _manifest_counts({"train": 2, "validation": 0, "test": 0, "total": 2}),
            wrong,
        )
        assert wrong.counts["manifest_count_mismatch:examples"] == 1


def test_an_empty_release_is_reported_as_empty() -> None:
    checks = _checks()
    with _release_connection() as connection:
        audit_sql._check_manifest(connection, {"counts": {}}, checks)
    assert "empty_dataset" in checks.counts


def test_duplicate_distribution_entries_are_a_manifest_mismatch() -> None:
    entry = {"code": 10, "label": "Tree cover", "examples": 1, "share": 1.0}
    with _release_connection(_train_row()) as connection:
        duplicated = _checks()
        duplicated.report.rows = 1
        audit_sql._check_distributions(
            connection, {"class_distribution": [entry, dict(entry)]}, duplicated
        )
        assert "manifest_mismatch:class_distribution" in duplicated.counts

        single = _checks()
        single.report.rows = 1
        audit_sql._check_distributions(connection, {"class_distribution": [entry]}, single)
        assert "manifest_mismatch:class_distribution" not in single.counts


@pytest.mark.parametrize(
    "change, split, code",
    [
        ({"split": "bogus"}, "bogus", "invalid_split"),
        ({"text": "w  " + " ".join(["word"] * 19)}, "train", "unnormalized_text"),
        ({"polygon_area_m2": 1e10 * (1 + 1e-5)}, "train", "area_above_cap"),
        ({"split": "validation"}, "validation", "h3_assignment_mismatch"),
    ],
)
def test_row_checks_name_each_reason_code(change, split, code) -> None:
    checks = _checks()
    row = {**rows()["train"], **change}
    audit_rows._check_row(row, split, SETTINGS, checks)
    assert code in checks.counts


@pytest.mark.parametrize(
    "lat_text, valid",
    [
        ("45.0", True),
        ("45.00000005", True),
        ("45.00000006", False),
        ("44.99999994", False),
        ("x.5", False),
    ],
)
def test_centroid_latitude_uses_the_same_seven_decimal_tolerance(lat_text, valid) -> None:
    checks = _checks()
    row = {"polygon_id": "p", "lon": 10.0, "lat": 45.0, "centroid_wkt": f"POINT (10.0 {lat_text})"}
    audit_rows._check_centroid(row, checks)
    assert ("centroid_wkt_mismatch" not in checks.counts) is valid


@contextlib.contextmanager
def _global_connection(release_rows, hash_rows):
    """DuckDB release and hash views for the global identity and text checks."""
    connection = duckdb.connect()
    try:
        connection.execute(
            "CREATE TABLE release (split VARCHAR, polygon_id VARCHAR, document_id VARCHAR, "
            "h3_cell VARCHAR, osm_type VARCHAR, osm_id BIGINT, worldcover_code INTEGER, "
            "lat DOUBLE, lon DOUBLE, polygon_area_m2 DOUBLE)"
        )
        connection.execute(
            "CREATE TABLE hashes (text_hash BLOB, worldcover_code INTEGER, "
            "split VARCHAR, polygon_id VARCHAR)"
        )
        for row in release_rows:
            connection.execute("INSERT INTO release VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", row)
        for row in hash_rows:
            connection.execute("INSERT INTO hashes VALUES (?, ?, ?, ?)", row)
        yield connection
    finally:
        connection.close()


def _object_row(polygon_id: str, osm_id: int) -> tuple:
    return ("train", polygon_id, f"d-{polygon_id}", "cell", "way", osm_id, 10, 45.0, 1.0, 1.0)


def test_one_osm_object_under_two_polygon_ids_is_an_identity_conflict() -> None:
    checks = _checks()
    with _global_connection([_object_row("p1", 1), _object_row("p2", 1)], []) as connection:
        audit_sql._check_global(connection, checks, strict_text_leakage=False)
    assert "object_identity_conflict" in {problem.code for problem in checks.report.problems}


def test_distinct_osm_objects_are_not_an_identity_conflict() -> None:
    checks = _checks()
    with _global_connection([_object_row("p1", 1), _object_row("p2", 2)], []) as connection:
        audit_sql._check_global(connection, checks, strict_text_leakage=False)
    assert "object_identity_conflict" not in {problem.code for problem in checks.report.problems}


def test_retained_text_analysis_must_match_the_release_it_describes() -> None:
    text = bytes(32)
    keys = (
        "retained_identical_text_label_groups",
        "retained_identical_text_label_rows",
        "retained_identical_text_label_cross_split_groups",
        "retained_identical_text_label_cross_split_rows",
        "identical_text_cross_split_groups",
        "identical_text_cross_split_rows",
    )
    hash_rows = [
        (text, 10, "train", "p1"),
        (text, 10, "validation", "p2"),
        (text, 20, "train", "p3"),
    ]
    with _global_connection([], hash_rows) as connection:
        actual = audit_sql._retained_text_diagnostics(connection)
        analysis = {key: actual[key] for key in keys}
        checks = _checks()
        audit_sql._check_text_diagnostics(connection, {"deduplication_analysis": analysis}, checks)
        assert not checks.counts
        analysis[keys[1]] += 1
        audit_sql._check_text_diagnostics(connection, {"deduplication_analysis": analysis}, checks)
        assert checks.counts == Counter({"deduplication_analysis_mismatch:retained_text": 1})


def test_language_distribution_is_matched_on_its_language_key() -> None:
    manifest = {
        "class_distribution": [{"code": 10, "label": "Tree cover", "examples": 1, "share": 1.0}],
        "language_distribution": [{"language": "en", "examples": 1, "share": 1.0}],
    }
    with _release_connection(
        ("train", "p0", "d0", 10, "en", "cell0", "region", 1.0, 45.0, 1.0)
    ) as connection:
        checks = _checks()
        checks.report.rows = 1
        audit_sql._check_distributions(connection, manifest, checks)
        assert not checks.counts

        renamed = _checks()
        renamed.report.rows = 1
        manifest["language_distribution"] = [{"language": "fr", "examples": 1, "share": 1.0}]
        audit_sql._check_distributions(connection, manifest, renamed)
        assert renamed.counts == Counter({"manifest_mismatch:language_distribution": 1})


def test_share_defaults_to_zero_examples_when_an_entry_has_none() -> None:
    checks = _checks()
    checks.report.rows = 3
    audit_sql._check_shares([{"share": 0.0}], checks)
    assert not checks.counts
    audit_sql._check_shares([{"share": 0.5}], checks)
    assert checks.counts == Counter({"manifest_share_mismatch": 1})


def test_problems_are_frozen_and_reports_and_problems_keep_no_instance_dict() -> None:
    problem = AuditProblem("code", 1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        problem.count = 2  # type: ignore[misc]
    assert not hasattr(problem, "__dict__")
    assert not hasattr(AuditReport(), "__dict__")


@pytest.mark.parametrize(
    "key, value",
    [
        ("dataset_version", "9.9.9"),
        ("code_repository", "https://example.com/other"),
        ("deduplication_policy", "other-policy"),
    ],
)
def test_release_only_settings_are_not_compared_with_the_ledger(key, value) -> None:
    manifest = _completion_manifest()
    manifest["settings"][key] = value
    assert _ledger_problems(manifest) == []


def test_other_settings_must_match_the_ledger_context() -> None:
    manifest = _completion_manifest()
    manifest["settings"]["split_seed"] = 1
    assert _ledger_problems(manifest) == ["build context settings mismatch"]


def _scan_rows(tmp_path, *split_rows):
    """Scan one train shard with scan_rows; return the checks, hash rows and hash path."""
    path = tmp_path / "train.parquet"
    pq.write_table(pa.Table.from_pylist(list(split_rows), schema=SCHEMA), path)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    checks = _checks()
    hashes = audit_rows.scan_rows([path], SETTINGS, scratch, checks)
    return checks, pq.read_table(hashes).to_pylist(), hashes


def test_scan_counts_every_row_once(tmp_path) -> None:
    base = rows()["train"]
    checks, _, _ = _scan_rows(tmp_path, base, {**base, "polygon_id": "p-second"})
    assert checks.report.rows == 2


def test_rows_without_text_are_counted_but_not_fingerprinted(tmp_path) -> None:
    base = rows()["train"]
    null_text = {**base, "polygon_id": "p-null", "text": None, "text_words": 0}
    checks, fingerprints, _ = _scan_rows(tmp_path, base, null_text)
    assert checks.report.rows == 2
    assert checks.counts["invalid_text"] == 1
    assert [fingerprint["polygon_id"] for fingerprint in fingerprints] == [base["polygon_id"]]


def test_fingerprint_is_the_sha256_of_the_whitespace_normalized_text(tmp_path) -> None:
    base = rows()["train"]
    _, fingerprints, hashes = _scan_rows(
        tmp_path, {**base, "text": "alpha\tbeta \n gamma", "text_words": 3}
    )
    assert hashes.name == "hashes.parquet"
    assert fingerprints == [
        {
            "text_hash": bytes.fromhex(
                "64989ccbf3efa9c84e2afe7cee9bc5828bf0fcb91e44f8c1e591638a2c2e90e3"
            ),
            "worldcover_code": 10,
            "split": "train",
            "polygon_id": base["polygon_id"],
        }
    ]


def test_inspect_accepts_a_shard_with_the_release_schema(build) -> None:
    checks = _checks()
    audit_rows._inspect_file(build / "train.parquet", checks)
    assert checks.counts == Counter()


def test_inspect_flags_a_shard_whose_schema_differs(tmp_path) -> None:
    path = tmp_path / "train.parquet"
    pq.write_table(pa.table({"polygon_id": ["p0"]}), path)
    checks = _checks()
    audit_rows._inspect_file(path, checks)
    assert checks.counts == Counter({"schema_mismatch": 1})
    assert checks.samples == {"schema_mismatch": ["train.parquet"]}


def test_inspect_reports_an_unreadable_shard(tmp_path) -> None:
    checks = _checks()
    audit_rows._inspect_file(tmp_path / "train.parquet", checks)
    assert checks.counts == Counter({"missing_or_invalid_parquet": 1})
    samples = checks.samples["missing_or_invalid_parquet"]
    assert [sample.split(": ", 1)[0] for sample in samples] == ["train.parquet"]


def test_load_manifest_returns_a_well_formed_manifest(build) -> None:
    checks = _checks()
    manifest = audit_manifest._load_manifest(build, checks)
    assert manifest is not None
    assert manifest["settings"] == SETTINGS
    assert checks.counts == Counter()


@pytest.mark.parametrize("payload", [[], "manifest", {"settings": {}}])
def test_load_manifest_rejects_non_objects_and_incomplete_shapes(build, payload) -> None:
    (build / "manifest.json").write_text(json.dumps(payload))
    checks = _checks()
    assert audit_manifest._load_manifest(build, checks) is None
    assert checks.counts == Counter({"missing_or_invalid_manifest": 1})
    assert not checks.samples


def test_load_manifest_reports_the_json_error_for_unparseable_text(build) -> None:
    (build / "manifest.json").write_text("{not json")
    checks = _checks()
    assert audit_manifest._load_manifest(build, checks) is None
    assert checks.counts == Counter({"missing_or_invalid_manifest": 1})
    assert checks.samples["missing_or_invalid_manifest"] == [
        "Expecting property name enclosed in double quotes: line 1 column 2 (char 1)"
    ]
