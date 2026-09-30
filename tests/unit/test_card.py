"""Dataset card rendering."""

from typing import Any

import pytest

from osm_worldcover.domain.card import render

# A manifest is JSON-shaped, so its values are heterogeneous by nature;
# annotating it says so instead of letting inference build a union that
# cannot be splatted.
MANIFEST: dict[str, Any] = {
    "counts": {
        "examples": {"train": 800, "validation": 100, "test": 100, "total": 1000},
        "polygons": {"train": 400, "validation": 50, "test": 50, "total": 500},
        "documents": {"train": 700, "validation": 90, "test": 90, "total": 880},
    },
    "class_distribution": [
        {"code": 10, "label": "Tree cover", "examples": 600, "share": 0.6},
        {"code": 50, "label": "Built-up", "examples": 400, "share": 0.4},
    ],
    "example_polygons": [
        {"name": "A | B", "worldcover_label": "Tree cover"},
        {"name": "Lake Azul", "worldcover_label": "Permanent water bodies"},
    ],
    "language_distribution": [
        {"language": "en", "examples": 700, "share": 0.7},
        {"language": "fr", "examples": 300, "share": 0.3},
    ],
    "dominant_fraction": {"p50": 0.97, "p90": 1.0, "p99": 1.0},
    "geographic_coverage": {
        "h3_cells": 120,
        "regions": 8,
        "bbox": {"min_lon": -9.0, "min_lat": 36.0, "max_lon": 31.0, "max_lat": 71.0},
    },
    "rejections": {"below_threshold": 50, "too_large": 3},
    "settings": {
        "dominance_threshold": 0.8,
        "source_dataset": "NoeFlandre/osm-polygon-wikidata-and-wikipedia",
        "source_revision": "abc123",
        "worldcover_version": "v200",
        "worldcover_year": 2021,
        "split_seed": 20180101,
        "h3_resolution": 5,
        "max_polygon_area_m2": 1e10,
        "min_words": 10,
    },
}


def card() -> str:
    return render(MANIFEST)


def test_card_starts_with_yaml_front_matter() -> None:
    assert card().startswith("---\n")
    assert "license:" in card().split("---")[1]


def test_front_matter_declares_the_three_splits() -> None:
    front = card().split("---")[1]
    for split in ("train", "validation", "test"):
        assert split in front


def test_card_reports_the_totals() -> None:
    assert "1,000" in card()


def test_card_embeds_the_centroid_coverage_map() -> None:
    text = card()
    assert "worldcover_centroids.png" in text
    assert "500 distinct polygons" in text
    assert "polygon centroid" in text


def test_card_shows_named_examples_with_their_esa_classes() -> None:
    text = card()
    assert "Example labelled polygons" in text
    assert "A \\| B" in text
    assert "Permanent water bodies" in text


def test_card_lists_every_class_with_its_share() -> None:
    text = card()
    assert "Tree cover" in text
    assert "Built-up" in text
    assert "60.0%" in text


def test_card_names_the_source_and_pinned_revision() -> None:
    text = card()
    assert "NoeFlandre/osm-polygon-wikidata-and-wikipedia" in text
    assert "abc123" in text


def test_card_lists_region_and_assembly_code_revisions() -> None:
    repository = "https://github.com/NoeFlandre/osm-worldcover"
    manifest = {
        **MANIFEST,
        "processing": {
            "code_provenance": [
                {"repository": repository, "revision": "a" * 40, "regions": ["alpha", "beta"]},
                {"repository": repository, "revision": "b" * 40, "regions": ["gamma"]},
            ],
            "assembly_code_revision": "c" * 40,
        },
    }
    text = render(manifest)
    assert f"{repository}/tree/{'a' * 40}" in text
    assert f"{repository}/tree/{'b' * 40}" in text
    assert f"{repository}/tree/{'c' * 40}" in text
    assert "2 regions" in text
    assert "1 region" in text


def test_card_uses_source_specific_metadata() -> None:
    manifest = {
        **MANIFEST,
        "settings": {
            **MANIFEST["settings"],
            "source": "website",
            "output_dataset": "NoeFlandre/osm-polygon-website-tag-worldcover",
            "source_text_description": "text extracted from OSM-tagged websites",
            "source_url": "https://huggingface.co/datasets/NoeFlandre/osm-polygon-website-tag",
            "dataset_license": "other",
            "text_license": "Third-party website text; source-site terms apply",
        },
    }
    text = render(manifest)
    assert "# osm-polygon-website-tag-worldcover" in text
    assert 'load_dataset("NoeFlandre/osm-polygon-website-tag-worldcover")' in text
    assert "OSM-tagged websites" in text
    assert "source-site terms apply" in text


def test_card_states_the_dominance_threshold() -> None:
    assert "80" in card()


def test_card_warns_that_the_label_describes_the_place_not_the_feature() -> None:
    """The most important caveat must not be buried or omitted."""
    assert "containing" in card()


def test_card_documents_the_leakage_guarantees() -> None:
    text = card().lower()
    assert "leak" in text or "split" in text


def test_card_exposes_record_deduplication_and_cross_split_text_diagnostics() -> None:
    manifest = {
        **MANIFEST,
        "deduplication": {"duplicate_polygon_text_label_records": 3},
        "deduplication_analysis": {
            "duplicate_polygon_text_label_groups": 2,
            "duplicate_records_removed": 3,
            "duplicate_record_groups_crossing_splits": 0,
            "duplicate_records_removed_from_cross_split_groups": 0,
            "duplicate_records_removed_by_text_words": {
                "1": 1,
                "2": 0,
                "3": 0,
                "4": 0,
                "5": 0,
                "6": 0,
                "7": 0,
                "8": 0,
                "9": 0,
                "10+": 2,
            },
            "retained_identical_text_label_groups": 2,
            "retained_identical_text_label_rows": 5,
            "retained_identical_text_label_cross_split_groups": 1,
            "retained_identical_text_label_cross_split_rows": 3,
            "identical_text_cross_split_groups": 2,
            "identical_text_cross_split_rows": 3,
        },
    }

    text = render(manifest)

    assert "1 word: 1" in text
    assert "polygon identity, normalized text and WorldCover label" in text
    assert "2 identical-text groups span splits, involving 3 rows" in text
    assert "not removed solely because text matches across polygons" in text


def test_card_is_deterministic() -> None:
    assert render(MANIFEST) == render(MANIFEST)


def test_card_handles_an_empty_class_distribution() -> None:
    manifest = {**MANIFEST, "class_distribution": []}
    assert render(manifest)


def test_card_requires_a_manifest_with_counts() -> None:
    with pytest.raises(KeyError):
        render({})


def test_card_names_an_unspecified_language_rather_than_printing_none() -> None:
    """A null language is real (an OSM description tag has none); "None" is not."""
    manifest = {
        **MANIFEST,
        "language_distribution": [
            {"language": None, "examples": 900, "share": 0.9},
            {"language": "en", "examples": 100, "share": 0.1},
        ],
    }
    text = render(manifest)
    assert "unspecified" in text
    assert "| None |" not in text


def test_card_does_not_lowercase_a_proper_noun_in_the_source_description() -> None:
    """Regression: "OpenStreetMap" was published as "openStreetMap".

    The first letter was lowercased so the phrase would read mid-sentence,
    which is wrong for every source whose description opens with a name.
    """
    manifest = {
        **MANIFEST,
        "settings": {
            **MANIFEST["settings"],
            "source_text_description": "OpenStreetMap description tag text",
        },
    }
    text = render(manifest)
    assert "OpenStreetMap description tag text" in text
    assert "openStreetMap" not in text
