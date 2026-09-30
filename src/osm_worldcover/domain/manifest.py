"""Assemble the dataset manifest.

The manifest is the dataset's self-description: what it contains, how it was
filtered, and which inputs produced it. It is built from aggregates plus a
small deterministic set of representative named polygons, and its key and
element order is fixed so two builds of the same data serialise byte-identically.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from osm_worldcover.domain.nomenclature import label_for

__all__ = ["DatasetCounts", "GeographicCoverage", "build"]

SPLIT_ORDER = ("train", "validation", "test")


@dataclass(frozen=True, slots=True)
class GeographicCoverage:
    """Where in the world the dataset reaches."""

    h3_cells: int
    bbox: tuple[float, float, float, float]
    regions: int


@dataclass(slots=True)
class DatasetCounts:
    """Aggregates describing a finished build."""

    examples: dict[str, int]
    polygons: dict[str, int]
    documents: dict[str, int]
    class_distribution: dict[int, int]
    language_distribution: dict[str, int]
    dominant_fraction_quantiles: dict[str, float]
    coverage: GeographicCoverage
    rejections: dict[str, int] = field(default_factory=dict)
    deduplication: dict[str, int] = field(default_factory=dict)
    deduplication_analysis: dict[str, Any] = field(default_factory=dict)
    example_polygons: list[dict[str, str]] = field(default_factory=list)


def build(counts: DatasetCounts, settings: Mapping[str, Any]) -> dict[str, Any]:
    """Return the manifest for a finished build."""
    total = sum(counts.examples.values())
    return {
        "counts": {
            "examples": _with_total(counts.examples),
            "polygons": _with_total(counts.polygons),
            "documents": _with_total(counts.documents),
        },
        "class_distribution": [
            {
                "code": code,
                "label": label_for(code),
                "examples": count,
                "share": _share(count, total),
            }
            for code, count in sorted(counts.class_distribution.items())
        ],
        "language_distribution": [
            {"language": language, "examples": count, "share": _share(count, total)}
            # Most frequent first; ties broken by name so the order is stable.
            # An absent language sorts last among equals rather than failing to
            # compare against a string.
            for language, count in sorted(
                counts.language_distribution.items(),
                key=lambda kv: (-kv[1], kv[0] is None, kv[0] or ""),
            )
        ],
        "example_polygons": [dict(row) for row in counts.example_polygons],
        "dominant_fraction": dict(sorted(counts.dominant_fraction_quantiles.items())),
        "geographic_coverage": {
            "h3_cells": counts.coverage.h3_cells,
            "regions": counts.coverage.regions,
            "bbox": dict(
                zip(
                    ("min_lon", "min_lat", "max_lon", "max_lat"),
                    counts.coverage.bbox,
                    strict=True,
                )
            ),
        },
        "rejections": dict(sorted(counts.rejections.items())),
        "deduplication": dict(sorted(counts.deduplication.items())),
        "deduplication_analysis": _copy_deduplication_analysis(counts.deduplication_analysis),
        "settings": dict(settings),
    }


def _copy_deduplication_analysis(analysis: Mapping[str, Any]) -> dict[str, Any]:
    """Copy nested diagnostics so caller mutation cannot change the manifest."""
    if not analysis:
        return {}
    return {
        **analysis,
        "duplicate_records_removed_by_text_words": dict(
            sorted(analysis.get("duplicate_records_removed_by_text_words", {}).items())
        ),
    }


def _with_total(by_split: Mapping[str, int]) -> dict[str, int]:
    """Return per-split counts in a fixed order, plus their total."""
    ordered = {split: int(by_split.get(split, 0)) for split in SPLIT_ORDER}
    return {**ordered, "total": sum(ordered.values())}


def _share(count: int, total: int) -> float:
    return round(count / total, 6) if total else 0.0
