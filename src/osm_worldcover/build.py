"""Run a whole build: fetch each region, label it, assemble the dataset.

Regions are processed one at a time and their shards kept in memory only as
long as the build runs. Tiles are fetched and released inside each region, so
peak disk stays near a single raster even though a global run touches
thousands of them.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from osm_worldcover.accounting import (
    BuildContext,
    atomic_json,
    processing_ledger,
)
from osm_worldcover.adapters import hub
from osm_worldcover.adapters.source import RegionTables
from osm_worldcover.adapters.worldcover import WorldCoverTiles
from osm_worldcover.build_shards import ShardStore, _summed
from osm_worldcover.config import Config
from osm_worldcover.domain.revision import is_full_revision
from osm_worldcover.finalize import StreamedBuild, finalize_shards
from osm_worldcover.pipeline import RegionOutcome, run_region
from osm_worldcover.sources import DEFAULT_SOURCE, SourceRecipe

__all__ = ["BuildReport", "ShardStore", "run_build"]


Progress = Callable[[str], None]


@dataclass(slots=True)
class BuildReport:
    """The outcome of a whole build."""

    result: StreamedBuild
    regions: list[RegionOutcome] = field(default_factory=list)
    processing: dict[str, Any] = field(default_factory=dict)

    @property
    def rejections(self) -> dict[str, int]:
        """Why polygons were refused, summed over every region."""
        return _summed(self.regions)


def run_build(
    config: Config,
    regions: Sequence[str] | None = None,
    keep_tiles: bool = False,
    progress: Progress = lambda _: None,
) -> BuildReport:
    """Build the dataset from the configured source revision."""
    revision = _resolve_revision(config)
    config = config.with_overrides(source_revision=revision)

    expected = hub.list_region_stems(config.source_dataset, revision, config.source_recipe)
    stems = _selected_regions(expected, regions, config)
    context = BuildContext.from_config(config)
    # Validate the selected subset against the complete pinned inventory even
    # when no discovery is needed to choose the work itself.
    processing_ledger(expected, stems, [], context)
    _require_regions(expected, stems)
    tiles = WorldCoverTiles(
        Path(config.cache_dir) / "worldcover",
        version=config.worldcover_version,
        year=config.worldcover_year,
        max_cached_tiles=config.cached_tiles,
    )
    raw = Path(config.cache_dir) / "source"

    shards = ShardStore(Path(config.cache_dir) / "shards", context=context)
    outcomes = _run_regions(config, revision, stems, raw, tiles, shards, keep_tiles, progress)

    ledger = processing_ledger(expected, stems, outcomes, context)
    atomic_json(Path(config.cache_dir) / "processing-ledger.json", ledger)
    selected = shards.stage(stems, Path(config.cache_dir) / "assembly" / "selected-shards")
    rejections = _summed(outcomes)
    return BuildReport(
        result=finalize_shards(
            selected,
            config,
            Path(config.cache_dir) / "assembly",
            Path(config.out_dir),
            rejections,
            processing=ledger,
        ),
        regions=outcomes,
        processing=ledger,
    )


def _resolve_revision(config: Config) -> str:
    revision = config.source_revision
    if not is_full_revision(revision, allow_uppercase=True):
        return hub.resolve_revision(config.source_dataset, revision)
    return revision


def _selected_regions(
    expected: Sequence[str], regions: Sequence[str] | None, config: Config
) -> list[str]:
    selection = regions if regions is not None else config.regions
    return list(expected if selection is None else selection)


def _require_regions(expected: Sequence[str], selected: Sequence[str]) -> None:
    if not expected or not selected:
        raise ValueError("a build requires a nonempty source inventory and region selection")


def _run_regions(
    config: Config,
    revision: str,
    stems: Sequence[str],
    raw: Path,
    tiles: WorldCoverTiles,
    shards: ShardStore,
    keep_tiles: bool,
    progress: Progress,
) -> list[RegionOutcome]:
    """Process each region that has not already finished."""
    outcomes: list[RegionOutcome] = []
    for index, stem in enumerate(stems, start=1):
        previous = shards.outcome(stem)
        done = previous is not None
        # Announced as each region starts, not up front, so a long run shows
        # where it actually is.
        progress(_label(index, len(stems), stem, done=done))
        if previous is not None:
            outcomes.append(previous)
        else:
            outcomes.append(
                _process_region(config, revision, stem, raw, tiles, shards, keep_tiles, progress)
            )
    return outcomes


def _label(index: int, total: int, stem: str, done: bool) -> str:
    """The progress line for one region."""
    suffix = " (already done)" if done else ""
    return f"[{index}/{total}] {stem}{suffix}"


def _process_region(
    config: Config,
    revision: str,
    stem: str,
    raw: Path,
    tiles: WorldCoverTiles,
    shards: ShardStore,
    keep_tiles: bool,
    progress: Progress,
) -> RegionOutcome:
    """Fetch, label and record one region."""
    hub.snapshot_region(config.source_dataset, revision, stem, raw, config.source_recipe)
    if config.source == DEFAULT_SOURCE:
        tables = RegionTables.load(raw, stem)
    else:
        tables = RegionTables.load(raw, stem, config.source_recipe)
    examples, outcome = run_region(config, tables, tiles, keep_tiles=keep_tiles)
    shards.write_outcome(stem, examples, outcome)
    _release_source(raw, stem, config.source_recipe)
    progress(
        f"    {outcome.polygons_seen} polygons -> "
        f"{outcome.polygons_accepted} labelled -> {outcome.examples} examples"
    )
    return outcome


def _release_source(raw: Path, stem: str, source: SourceRecipe) -> None:
    """Delete a region's downloaded tables once its shard is written.

    The full source snapshot is ~21 GB and none of it is needed again.
    """
    paths = (
        hub.region_files(stem) if source.name == DEFAULT_SOURCE else hub.region_files(stem, source)
    )
    for path in paths:
        (raw / path).unlink(missing_ok=True)
