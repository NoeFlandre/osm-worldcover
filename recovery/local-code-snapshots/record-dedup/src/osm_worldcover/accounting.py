"""Verifiable region completion records and whole-source processing accounting.

A Parquet filename does not prove which source revision or filters produced
it. Completion receipts bind the entire region outcome and the shard bytes to
an immutable build context. The final ledger distinguishes a completed subset
from complete coverage of the pinned source inventory.
"""

import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from osm_worldcover.config import Config
from osm_worldcover.pipeline import OUTPUT_COLUMNS, RegionOutcome

__all__ = [
    "BuildContext",
    "atomic_json",
    "file_sha256",
    "outcome_from_record",
    "outcome_record",
    "processing_ledger",
    "validate_outcome",
]

RECEIPT_VERSION = 1
PIPELINE_SCHEMA_VERSION = 3
COUNT_FIELDS = (
    "polygons_seen",
    "polygons_invalid",
    "polygons_accepted",
    "polygons_with_examples",
    "source_links",
    "source_documents",
    "examples",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@dataclass(frozen=True, slots=True)
class BuildContext:
    """The pinned input, all data-changing settings, and output contract."""

    document: dict[str, Any]

    @classmethod
    def from_config(cls, config: Config) -> "BuildContext":
        """Exclude only scratch paths, region selection and cache controls."""
        revision = config.source_revision
        if not revision or re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None:
            raise ValueError("completion receipts require a pinned source revision")
        return cls(
            {
                "receipt_version": RECEIPT_VERSION,
                "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
                "settings": config.as_manifest_settings(),
                "extra": config.extra,
                "source_recipe": asdict(config.source_recipe),
                "output_columns": list(OUTPUT_COLUMNS),
                "outcome_fields": [item.name for item in fields(RegionOutcome)],
            }
        )

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "BuildContext":
        """Restore the exact context recorded by a verified region receipt."""
        return cls(json.loads(_canonical(dict(document))))

    @property
    def fingerprint(self) -> str:
        """Stable identity independent of mapping order and scratch paths."""
        return hashlib.sha256(_canonical(self.document)).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-normalized copy, including tuple-valued recipe data."""
        return json.loads(_canonical(self.document))


def file_sha256(path: Path) -> str:
    """Hash a shard with bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    """Install a complete, flushed JSON file as one filesystem operation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    partial = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


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


def _inventory(values: Sequence[str], name: str) -> list[str]:
    if any(not isinstance(stem, str) or not stem for stem in values):
        raise ValueError(f"{name} must contain nonempty region names")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} contains duplicate region names")
    return sorted(values)


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


def processing_ledger(
    expected_regions: Sequence[str],
    selected_regions: Sequence[str],
    outcomes: Sequence[RegionOutcome],
    context: BuildContext,
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
    return _processing_record(expected, selected, processed, missing, pending, outcomes, context)


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
) -> dict[str, Any]:
    return {
        "schema_version": RECEIPT_VERSION,
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
