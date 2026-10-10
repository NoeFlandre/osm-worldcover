"""Manifest assembly."""

import json
import re

import pytest

from osm_worldcover.domain.manifest import DatasetCounts, GeographicCoverage, build
from osm_worldcover.domain.nomenclature import UnknownLandCoverCodeError


@pytest.fixture
def counts() -> DatasetCounts:
    return DatasetCounts(
        examples={"train": 80, "validation": 10, "test": 10},
        polygons={"train": 40, "validation": 5, "test": 5},
        documents={"train": 70, "validation": 9, "test": 9},
        class_distribution={10: 60, 50: 40},
        language_distribution={"en": 70, "fr": 30},
        dominant_fraction_quantiles={"p50": 0.97, "p90": 1.0},
        coverage=GeographicCoverage(h3_cells=12, bbox=(-9.0, 36.0, 31.0, 71.0), regions=3),
    )


def test_manifest_reports_totals(counts) -> None:
    manifest = build(counts, settings={"dominance_threshold": 0.8})
    assert manifest["counts"]["examples"]["total"] == 100
    assert manifest["counts"]["examples"]["train"] == 80


def test_manifest_carries_the_settings_verbatim(counts) -> None:
    manifest = build(counts, settings={"dominance_threshold": 0.8, "split_seed": 1})
    assert manifest["settings"]["dominance_threshold"] == 0.8
    assert manifest["settings"]["split_seed"] == 1


def test_class_distribution_is_labelled_and_ordered_by_code(counts) -> None:
    classes = build(counts, settings={})["class_distribution"]
    assert [c["code"] for c in classes] == [10, 50]
    assert classes[0]["label"] == "Tree cover"
    assert classes[0]["examples"] == 60
    assert classes[0]["share"] == pytest.approx(0.6)


def test_manifest_carries_named_polygon_examples(counts) -> None:
    counts.example_polygons = [{"name": "Forest", "worldcover_label": "Tree cover"}]

    examples = build(counts, settings={})["example_polygons"]

    assert examples == [{"name": "Forest", "worldcover_label": "Tree cover"}]


def test_language_distribution_is_ordered_by_count_then_name(counts) -> None:
    # "fr" is inserted before "de" but must come after it, so the tie-break is
    # really on the name rather than on insertion order.
    counts.language_distribution = {"fr": 30, "de": 30, "en": 70}
    languages = build(counts, settings={})["language_distribution"]
    assert [entry["language"] for entry in languages] == ["en", "de", "fr"]


def test_geographic_coverage_is_reported(counts) -> None:
    coverage = build(counts, settings={})["geographic_coverage"]
    assert coverage["h3_cells"] == 12
    assert coverage["regions"] == 3
    assert coverage["bbox"] == {"min_lon": -9.0, "min_lat": 36.0, "max_lon": 31.0, "max_lat": 71.0}


def test_manifest_is_json_serialisable(counts) -> None:
    json.dumps(build(counts, settings={"dominance_threshold": 0.8}))


def test_manifest_is_deterministic(counts) -> None:
    a = json.dumps(build(counts, settings={"x": 1}))
    b = json.dumps(build(counts, settings={"x": 1}))
    assert a == b


def test_an_unknown_class_code_is_refused(counts) -> None:
    counts.class_distribution = {999: 1}
    with pytest.raises(UnknownLandCoverCodeError) as raised:
        build(counts, settings={})
    assert raised.value.args == (999,)


def test_empty_dataset_reports_zero_without_dividing_by_zero(counts) -> None:
    counts.examples = {"train": 0, "validation": 0, "test": 0}
    counts.class_distribution = {}
    manifest = build(counts, settings={})
    assert manifest["counts"]["examples"]["total"] == 0
    assert manifest["class_distribution"] == []


def test_deduplication_counts_are_reported(counts) -> None:
    """How many rows were dropped, and why, belongs in the manifest."""
    counts.deduplication = {
        "duplicate_objects_across_regions": 30,
        "duplicate_polygon_text_label_records": 8821,
        "documents_split_across_splits": 30,
    }
    reported = build(counts, settings={})["deduplication"]
    assert reported["duplicate_polygon_text_label_records"] == 8821
    assert reported["documents_split_across_splits"] == 30


def test_deduplication_defaults_to_empty(counts) -> None:
    assert build(counts, settings={})["deduplication"] == {}
    assert build(counts, settings={})["deduplication_analysis"] == {}


def test_manifest_copies_nested_deduplication_analysis(counts) -> None:
    counts.deduplication_analysis = {
        "duplicate_polygon_text_label_groups": 2,
        "duplicate_records_removed": 3,
        "duplicate_records_removed_by_text_words": {"1": 1, "10+": 2},
    }

    manifest = build(counts, settings={})

    assert manifest["deduplication_analysis"]["duplicate_records_removed_by_text_words"] == {
        "1": 1,
        "10+": 2,
    }
    counts.deduplication_analysis["duplicate_records_removed_by_text_words"]["1"] = 9
    assert manifest["deduplication_analysis"]["duplicate_records_removed_by_text_words"]["1"] == 1


def test_manifest_keys_are_the_published_contract(counts) -> None:
    """The manifest is read by consumers, so its key names are an interface.

    Pinning them here means renaming one is a deliberate, visible change rather
    than something that slips out in a release.
    """
    manifest = build(counts, settings={"dominance_threshold": 0.8})
    assert _manifest_contract(manifest) == {
        "top_level": {
            "counts",
            "class_distribution",
            "example_polygons",
            "language_distribution",
            "dominant_fraction",
            "geographic_coverage",
            "rejections",
            "deduplication",
            "deduplication_analysis",
            "settings",
        },
        "count_groups": {"examples", "polygons", "documents"},
        "count_fields": {"train", "validation", "test", "total"},
        "class_fields": {"code", "label", "examples", "share"},
        "language_fields": {"language", "examples", "share"},
        "coverage_fields": {"h3_cells", "regions", "bbox"},
        "bbox_fields": {"min_lon", "min_lat", "max_lon", "max_lat"},
    }


def _manifest_contract(manifest: dict) -> dict:
    """Summarize the independently published key sets for one contract check."""
    return {
        "top_level": set(manifest),
        "count_groups": set(manifest["counts"]),
        "count_fields": set(manifest["counts"]["examples"]),
        "class_fields": set(manifest["class_distribution"][0]),
        "language_fields": set(manifest["language_distribution"][0]),
        "coverage_fields": set(manifest["geographic_coverage"]),
        "bbox_fields": set(manifest["geographic_coverage"]["bbox"]),
    }


def test_per_split_counts_keep_a_fixed_order(counts) -> None:
    """Order is part of the contract too: a manifest diff should stay readable."""
    manifest = build(counts, settings={})
    assert list(manifest["counts"]["examples"]) == ["train", "validation", "test", "total"]


def test_totals_are_the_sum_of_the_splits(counts) -> None:
    counts.examples = {"train": 7, "validation": 2, "test": 1}
    assert build(counts, settings={})["counts"]["examples"]["total"] == 10


def test_a_missing_split_counts_as_zero(counts) -> None:
    counts.polygons = {"train": 5}
    reported = build(counts, settings={})["counts"]["polygons"]
    assert reported == {"train": 5, "validation": 0, "test": 0, "total": 5}


def test_an_unspecified_language_is_null_not_the_word_none(counts) -> None:
    """Regression: a SQL NULL cast with str() published the literal "None".

    Most OSM `description` tags carry no language at all, so this was 95% of a
    release -- a language code that does not exist, presented as if it did.
    """
    counts.language_distribution = {None: 90, "en": 10}
    languages = build(counts, settings={})["language_distribution"]
    unspecified = next(entry for entry in languages if entry["language"] is None)
    assert unspecified["examples"] == 90
    assert {entry["language"] for entry in languages if entry["language"]} == {"en"}


def test_an_unspecified_language_still_sorts_by_count(counts) -> None:
    counts.language_distribution = {None: 5, "en": 90}
    languages = build(counts, settings={})["language_distribution"]
    assert [entry["language"] for entry in languages] == ["en", None]


def test_equal_counts_list_named_languages_alphabetically_then_the_unspecified(counts) -> None:
    counts.language_distribution = {"fr": 10, None: 10, "de": 10}
    languages = build(counts, settings={})["language_distribution"]
    assert [entry["language"] for entry in languages] == ["de", "fr", None]


def test_shares_are_fractions_of_all_examples_rounded_to_six_places(counts) -> None:
    counts.examples = {"train": 2, "validation": 1, "test": 0}
    counts.class_distribution = {10: 1, 50: 2}
    counts.language_distribution = {"en": 1, "fr": 2}

    manifest = build(counts, settings={})

    assert [row["share"] for row in manifest["class_distribution"]] == [0.333333, 0.666667]
    assert [row["share"] for row in manifest["language_distribution"]] == [0.666667, 0.333333]


def test_shares_are_zero_when_there_are_no_examples(counts) -> None:
    counts.examples = {}
    counts.class_distribution = {10: 0}
    counts.language_distribution = {"en": 0}

    manifest = build(counts, settings={})

    assert manifest["class_distribution"][0]["share"] == 0.0
    assert manifest["language_distribution"][0]["share"] == 0.0


@pytest.mark.parametrize(
    ("bbox", "message"),
    [
        ((1.0, 2.0, 3.0), "zip() argument 2 is shorter than argument 1"),
        ((1.0, 2.0, 3.0, 4.0, 5.0), "zip() argument 2 is longer than argument 1"),
    ],
)
def test_a_bbox_must_have_exactly_four_edges(counts, bbox, message) -> None:
    counts.coverage = GeographicCoverage(h3_cells=1, bbox=bbox, regions=1)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match=re.escape(message)):
        build(counts, settings={})


def test_deduplication_analysis_without_a_word_breakdown_gains_an_empty_one(counts) -> None:
    counts.deduplication_analysis = {"groups": 2}

    analysis = build(counts, settings={})["deduplication_analysis"]

    assert analysis == {"groups": 2, "duplicate_records_removed_by_text_words": {}}
