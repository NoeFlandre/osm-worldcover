"""Publication audit checks bytes on disk, including corruptions validation missed."""

import hashlib
import json
import pickle
from collections import Counter

import h3
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from osm_worldcover.adapters.audit import _SCHEMA, audit_build
from osm_worldcover.domain.card import render
from osm_worldcover.domain.splits import assign_cell

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
