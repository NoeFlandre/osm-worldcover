"""Region outcome records and the rules every outcome must satisfy."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, fields
from typing import Any

from osm_worldcover.pipeline import RegionOutcome

COUNT_FIELDS = (
    "polygons_seen",
    "polygons_invalid",
    "polygons_accepted",
    "polygons_with_examples",
    "source_links",
    "source_documents",
    "examples",
)


def outcome_record(outcome: RegionOutcome) -> dict[str, Any]:
    """Serialize all accounting fields with deterministic counter order."""
    record = asdict(outcome)
    record["rejections"] = dict(sorted(outcome.rejections.items()))
    record["text_rejections"] = dict(sorted(outcome.text_rejections.items()))
    record["tiles_missing"] = sorted(set(outcome.tiles_missing))
    return record


def outcome_from_record(record: Mapping[str, Any]) -> RegionOutcome:
    """Restore a complete current-schema outcome, never invent missing counts."""
    expected = {item.name for item in fields(RegionOutcome)}
    if set(record) != expected:
        raise ValueError("region receipt does not have the current outcome schema")
    values = dict(record)
    values["rejections"] = Counter(values["rejections"])
    values["text_rejections"] = Counter(values["text_rejections"])
    outcome = RegionOutcome(**values)
    validate_outcome(outcome)
    return outcome


def _nonnegative_counts(counts: Mapping[str, Any]) -> None:
    for name, value in counts.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer, got {value!r}")


def validate_outcome(outcome: RegionOutcome) -> None:
    """Every source polygon must end in exactly one accounted-for outcome."""
    _nonnegative_counts({name: getattr(outcome, name) for name in COUNT_FIELDS})
    _nonnegative_counts(outcome.rejections)
    _nonnegative_counts(outcome.text_rejections)
    _validate_spatial_accounting(outcome)
    _validate_text_accounting(outcome)
    _validate_example_count(outcome)
    _validate_missing_tiles(outcome)


def _validate_spatial_accounting(outcome: RegionOutcome) -> None:
    if outcome.polygons_seen != (
        outcome.polygons_invalid + outcome.polygons_accepted + sum(outcome.rejections.values())
    ):
        raise ValueError(f"{outcome.stem}: spatial polygon accounting does not reconcile")


def _validate_text_accounting(outcome: RegionOutcome) -> None:
    if outcome.polygons_accepted != (
        outcome.polygons_with_examples + sum(outcome.text_rejections.values())
    ):
        raise ValueError(f"{outcome.stem}: text polygon accounting does not reconcile")


def _validate_example_count(outcome: RegionOutcome) -> None:
    if outcome.examples < outcome.polygons_with_examples:
        raise ValueError(f"{outcome.stem}: fewer examples than polygons with examples")


def _validate_missing_tiles(outcome: RegionOutcome) -> None:
    if not isinstance(outcome.tiles_missing, list) or any(
        not isinstance(tile, str) for tile in outcome.tiles_missing
    ):
        raise ValueError(f"{outcome.stem}: invalid missing-tile accounting")
