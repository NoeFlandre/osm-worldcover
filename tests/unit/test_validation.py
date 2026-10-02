"""Dataset-level invariants checked before anything is published."""

import pytest

from osm_worldcover.domain.validation import (
    REQUIRED_COLUMNS,
    Check,
    ValidationReport,
    Violation,
    validate,
)


def row(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "polygon_id": "luxembourg-latest:relation:1",
        "document_id": "Q1:wikipedia:en:1:1",
        "split": "train",
        "worldcover_code": 10,
        "worldcover_label": "Tree cover",
        "dominant_fraction": 0.95,
        "text": " ".join(["word"] * 30),
        "h3_cell": "851f1d4bfffffff",
    }
    return base | over


def checks_in(rows, **kw) -> set[Check]:
    return {v.check for v in validate(rows, **kw).violations}


def _find_violation(rows, check: Check):
    return next(violation for violation in validate(rows).violations if violation.check is check)


def test_a_clean_dataset_passes() -> None:
    report = validate([row(), row(polygon_id="p2", document_id="d2", text="x " * 40)])
    assert report.ok
    assert report.violations == []


def test_same_polygon_in_two_splits_is_leakage() -> None:
    rows = [row(document_id="d1"), row(document_id="d2", split="test")]
    assert Check.POLYGON_LEAKAGE in checks_in(rows)


def test_same_polygon_twice_in_one_split_is_not_leakage() -> None:
    rows = [row(document_id="d1"), row(document_id="d2")]
    assert Check.POLYGON_LEAKAGE not in checks_in(rows)


def test_same_document_in_two_splits_is_leakage() -> None:
    rows = [row(polygon_id="p1"), row(polygon_id="p2", split="test")]
    assert Check.DOCUMENT_LEAKAGE in checks_in(rows)


def test_unknown_worldcover_code_is_rejected() -> None:
    assert Check.INVALID_LABEL in checks_in([row(worldcover_code=0)])


def test_label_not_matching_its_code_is_rejected() -> None:
    assert Check.INVALID_LABEL in checks_in([row(worldcover_label="Cropland")])


def test_dominance_below_threshold_is_rejected() -> None:
    assert Check.BELOW_THRESHOLD in checks_in([row(dominant_fraction=0.79)], threshold=0.8)


def test_dominance_exactly_at_threshold_is_accepted() -> None:
    assert Check.BELOW_THRESHOLD not in checks_in([row(dominant_fraction=0.8)], threshold=0.8)


def test_too_short_text_is_rejected() -> None:
    assert Check.UNUSABLE_TEXT in checks_in([row(text="tiny")])


def test_identical_text_and_label_on_distinct_polygons_is_allowed() -> None:
    rows = [row(polygon_id="p1", document_id="d1"), row(polygon_id="p2", document_id="d2")]
    assert Check.DUPLICATE_EXAMPLE not in checks_in(rows)


def test_exact_duplicate_polygon_text_and_label_is_rejected() -> None:
    rows = [row(polygon_id="p1", document_id="d1"), row(polygon_id="p1", document_id="d2")]
    assert Check.DUPLICATE_EXAMPLE in checks_in(rows)


def test_same_text_under_a_different_label_is_not_a_duplicate() -> None:
    rows = [
        row(polygon_id="p1", document_id="d1"),
        row(
            polygon_id="p2",
            document_id="d2",
            worldcover_code=20,
            worldcover_label="Shrubland",
        ),
    ]
    assert Check.DUPLICATE_EXAMPLE not in checks_in(rows)


def test_unknown_split_name_is_rejected() -> None:
    assert Check.INVALID_SPLIT in checks_in([row(split="holdout")])


def test_an_empty_dataset_is_reported_rather_than_silently_passing() -> None:
    report = validate([])
    assert not report.ok
    assert Check.EMPTY_DATASET in {v.check for v in report.violations}


def test_report_counts_every_offending_row_not_just_the_first() -> None:
    rows = [row(polygon_id=f"p{i}", document_id=f"d{i}", text="tiny") for i in range(3)]
    violation = next(v for v in validate(rows).violations if v.check is Check.UNUSABLE_TEXT)
    assert violation.count == 3


def test_violations_are_ordered_deterministically() -> None:
    rows = [row(split="holdout", worldcover_code=0, text="tiny")]
    first = [v.check for v in validate(rows).violations]
    second = [v.check for v in validate(rows).violations]
    assert first == second


def test_only_a_handful_of_examples_are_reported_per_check() -> None:
    """The report samples offenders; it must not grow with the dataset."""
    rows = [row(polygon_id=f"p{i}", document_id=f"d{i}", text="tiny") for i in range(20)]
    violation = _find_violation(rows, Check.UNUSABLE_TEXT)
    assert violation.count == 20
    assert len(violation.examples) == 5


def test_reported_examples_name_the_offending_polygons() -> None:
    rows = [row(polygon_id="culprit", text="tiny")]
    violation = next(v for v in validate(rows).violations if v.check is Check.UNUSABLE_TEXT)
    assert violation.examples == ("culprit",)


def test_an_inverted_bbox_error_names_the_problem() -> None:
    from osm_worldcover.domain.tiling import tiles_for_bbox

    with pytest.raises(ValueError, match="inverted bbox"):
        tiles_for_bbox((9.0, 48.0, 6.0, 51.0))


def test_the_columns_validation_needs_are_declared() -> None:
    """Declared so a caller can read only these, instead of whole rows.

    A published row carries the full article twice over (text and lead_text)
    plus titles and URLs; materialising all of that to check six fields made
    validating a global build take longer than producing it.
    """
    assert set(REQUIRED_COLUMNS) == {
        "polygon_id",
        "document_id",
        "split",
        "worldcover_code",
        "worldcover_label",
        "dominant_fraction",
        "text",
    }


def test_validation_works_on_rows_holding_only_the_required_columns() -> None:
    lean = {key: row()[key] for key in REQUIRED_COLUMNS}
    assert validate([lean]).ok


def test_the_report_counts_every_row() -> None:
    rows = [row(polygon_id=f"p{i}", document_id=f"d{i}", text="x " * (40 + i)) for i in range(3)]

    assert validate(rows).rows == 3
    assert validate(iter(rows[:1])).rows == 1


def test_an_empty_dataset_is_reported_as_exactly_that() -> None:
    assert validate([]) == ValidationReport(0, [Violation(Check.EMPTY_DATASET, 0)])


def test_each_violation_keeps_five_examples_from_the_first_five_culprits() -> None:
    rows = [row(polygon_id=f"p{i}", document_id=f"d{i}", split="bogus") for i in (6, 5, 4, 3, 2, 1)]

    (violation,) = validate(rows).violations

    assert violation == Violation(Check.INVALID_SPLIT, 6, ("p2", "p3", "p4", "p5", "p6"))


def test_a_leaking_key_is_named_as_the_example() -> None:
    rows = [row(document_id="d1"), row(document_id="d2", split="test")]

    leaks = [v for v in validate(rows).violations if v.check is Check.POLYGON_LEAKAGE]

    assert leaks == [Violation(Check.POLYGON_LEAKAGE, 1, ("luxembourg-latest:relation:1",))]


def test_identical_text_is_a_duplicate_only_within_the_same_class() -> None:
    same_class = [row(document_id="d1"), row(document_id="d2")]
    other_class = [row(document_id="d1"), row(document_id="d2", worldcover_code=20)]
    other_class[1]["worldcover_label"] = "Shrubland"

    assert Check.DUPLICATE_EXAMPLE in checks_in(same_class)
    assert Check.DUPLICATE_EXAMPLE not in checks_in(other_class)
