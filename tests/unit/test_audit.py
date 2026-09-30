"""Publication audit checks bytes on disk, including corruptions validation missed."""

import hashlib
import json
from collections import Counter

import h3
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_worldcover.adapters.audit import _SCHEMA, audit_build
from osm_worldcover.domain.splits import assign_cell

SPLITS = ("train", "validation", "test")
SETTINGS = {
    "dataset_version": "2.0.0",
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
SCHEMA = pa.schema([(key, TYPES[value]) for key, value in _SCHEMA.items()])


def rows():
    result = {}
    for lon in range(-175, 176):
        cell = h3.latlng_to_cell(40, lon, 5)
        split = assign_cell(cell).value
        if split in result:
            continue
        row = dict.fromkeys(_SCHEMA)
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
    assert report.as_dict()["ok"]
    json.dumps(report.as_dict())


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
    assert risks["identical_text_cross_split"] == 1
    assert risks["identical_text_conflicting_labels"] == 1
    assert "identical_text_cross_split" not in problems(report)
    assert "identical_text_cross_split" in problems(audit_build(build, strict_text_leakage=True))


def test_text_label_duplicates_are_always_fatal(build):
    text = pq.read_table(build / "train.parquet")["text"][0].as_py()
    mutate(build, "test", text=text)
    assert "duplicate_text_label" in problems(audit_build(build))


def test_document_polygon_and_h3_leakage_are_detected(build):
    train = pq.read_table(build / "train.parquet").to_pylist()[0]
    mutate(build, "test", **{key: train[key] for key in ("document_id", "polygon_id", "h3_cell")})
    assert {"document_id_leakage", "polygon_id_leakage", "h3_cell_leakage"} <= problems(
        audit_build(build)
    )


def test_card_is_required_only_when_requested(build):
    assert audit_build(build).ok
    assert "invalid_card_or_map" in problems(audit_build(build, require_card=True))


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
    manifest["deduplication"] = {"duplicate_examples": 1}
    path.write_text(json.dumps(manifest))
    assert "invalid_processing_ledger" in problems(audit_build(build, require_complete=True))


def test_processing_duplicate_text_analysis_reconciles(build):
    path = build / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["deduplication"] = {
        "duplicate_objects_across_regions": 0,
        "duplicate_examples": 0,
        "documents_split_across_splits": 0,
    }
    manifest["deduplication_analysis"] = {
        "duplicate_text_label_groups": 0,
        "duplicate_rows_removed": 0,
        "duplicate_groups_crossing_splits": 0,
        "duplicate_rows_removed_from_cross_split_groups": 0,
        "duplicate_rows_removed_by_text_words": {
            **dict.fromkeys((str(words) for words in range(1, 10)), 0),
            "10+": 0,
        },
    }
    path.write_text(json.dumps(manifest))

    assert audit_build(build, require_complete=True).ok


def test_processing_duplicate_text_length_histogram_must_be_complete(build):
    path = build / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["deduplication"] = {
        "duplicate_objects_across_regions": 0,
        "duplicate_examples": 0,
        "documents_split_across_splits": 0,
    }
    manifest["deduplication_analysis"] = {
        "duplicate_text_label_groups": 0,
        "duplicate_rows_removed": 0,
        "duplicate_groups_crossing_splits": 0,
        "duplicate_rows_removed_from_cross_split_groups": 0,
        "duplicate_rows_removed_by_text_words": {"10+": 0},
    }
    path.write_text(json.dumps(manifest))

    assert "invalid_processing_ledger" in problems(audit_build(build, require_complete=True))
