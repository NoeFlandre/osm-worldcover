"""Whole-source processing ledger: inventories, totals and reconciliation."""

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from osm_worldcover.accounting.context import RECEIPT_VERSION, BuildContext
from osm_worldcover.accounting.outcome import COUNT_FIELDS, outcome_record, validate_outcome
from osm_worldcover.accounting.provenance import (
    code_provenance_fields,
    validated_assembly_code_revision,
    validated_code_revisions,
)
from osm_worldcover.pipeline import RegionOutcome

PROCESSING_LEDGER_VERSION = 2


def processing_ledger(
    expected_regions: Sequence[str],
    selected_regions: Sequence[str],
    outcomes: Sequence[RegionOutcome],
    context: BuildContext,
    *,
    region_code_revisions: Mapping[str, str] | None = None,
    assembly_code_revision: str | None = None,
) -> dict[str, Any]:
    """Describe exactly what was processed, including resumed region outcomes.

    ``complete`` means every expected source region is accounted for. A subset
    can be ``selected_complete`` while still having missing source regions.
    Counts describe pre-deduplication region output; the assembly manifest
    records the subsequent global deduplication and final output counts.
    """
    expected, selected, processed = _validated_inventories(
        expected_regions, selected_regions, outcomes
    )
    missing, pending = _inventory_gaps(expected, selected, processed)
    revisions = validated_code_revisions(processed, context, region_code_revisions)
    assembly_code_revision = validated_assembly_code_revision(
        assembly_code_revision, context, revisions
    )
    return _processing_record(
        expected,
        selected,
        processed,
        missing,
        pending,
        outcomes,
        context,
        revisions,
        assembly_code_revision,
    )


def _inventory(values: Sequence[str], name: str) -> list[str]:
    if any(not isinstance(stem, str) or not stem for stem in values):
        raise ValueError(f"{name} must contain nonempty region names")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} contains duplicate region names")
    return sorted(values)


def _validated_inventories(
    expected_regions: Sequence[str],
    selected_regions: Sequence[str],
    outcomes: Sequence[RegionOutcome],
) -> tuple[list[str], list[str], list[str]]:
    expected = _inventory(expected_regions, "expected regions")
    selected = _inventory(selected_regions, "selected regions")
    processed = _inventory([outcome.stem for outcome in outcomes], "processed regions")
    if not set(selected) <= set(expected):
        raise ValueError(f"unknown selected regions: {sorted(set(selected) - set(expected))}")
    if not set(processed) <= set(selected):
        raise ValueError("processed regions are outside the selected source inventory")
    for outcome in outcomes:
        validate_outcome(outcome)
    return expected, selected, processed


def _inventory_gaps(
    expected: list[str], selected: list[str], processed: list[str]
) -> tuple[list[str], list[str]]:
    missing = sorted(set(expected) - set(processed))
    pending = sorted(set(selected) - set(processed))
    return missing, pending


def _processing_record(
    expected: list[str],
    selected: list[str],
    processed: list[str],
    missing: list[str],
    pending: list[str],
    outcomes: Sequence[RegionOutcome],
    context: BuildContext,
    region_code_revisions: Mapping[str, str] | None,
    assembly_code_revision: str | None,
) -> dict[str, Any]:
    record = {
        "schema_version": _processing_schema_version(region_code_revisions),
        "build_context_sha256": context.fingerprint,
        "context": context.as_dict(),
        "scope": "full" if set(selected) == set(expected) else "subset",
        "complete": not missing,
        "full_source_complete": bool(expected) and not missing,
        "selected_complete": not pending,
        "expected_regions": expected,
        "selected_regions": selected,
        "processed_regions": processed,
        "missing_regions": missing,
        "unprocessed_selected_regions": pending,
        "region_counts": {
            "expected": len(expected),
            "selected": len(selected),
            "processed": len(processed),
            "missing": len(missing),
            "unprocessed_selected": len(pending),
        },
        "totals": _totals(outcomes),
        "regions": [outcome_record(outcome) for outcome in sorted(outcomes, key=lambda x: x.stem)],
        "reconciliation": {
            "spatial": "polygons_seen = polygons_invalid + polygons_accepted + sum(rejections)",
            "text": "polygons_accepted = polygons_with_examples + sum(text_rejections)",
            "counts_are_pre_deduplication": True,
            "valid": True,
        },
    }
    record.update(code_provenance_fields(region_code_revisions, context, assembly_code_revision))
    return record


def _processing_schema_version(revisions: Mapping[str, str] | None) -> int:
    return PROCESSING_LEDGER_VERSION if revisions is not None else RECEIPT_VERSION


def _totals(outcomes: Sequence[RegionOutcome]) -> dict[str, Any]:
    total: dict[str, Any] = _count_totals(outcomes)
    total.update(_mapping_totals(outcomes))
    total["tiles_missing"] = _missing_tiles(outcomes)
    return total


def _count_totals(outcomes: Sequence[RegionOutcome]) -> dict[str, int]:
    return {name: sum(getattr(outcome, name) for outcome in outcomes) for name in COUNT_FIELDS}


def _mapping_totals(outcomes: Sequence[RegionOutcome]) -> dict[str, dict[str, int]]:
    totals = {}
    for name in ("rejections", "text_rejections"):
        counter: Counter[str] = Counter()
        for outcome in outcomes:
            counter.update(getattr(outcome, name))
        totals[name] = dict(sorted(counter.items()))
    return totals


def _missing_tiles(outcomes: Sequence[RegionOutcome]) -> list[str]:
    return sorted({tile for outcome in outcomes for tile in outcome.tiles_missing})
