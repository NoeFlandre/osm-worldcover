"""Run a whole build: fetch each region, label it, assemble the dataset.

Regions are processed one at a time and their shards kept in memory only as
long as the build runs. Tiles are fetched and released inside each region, so
peak disk stays near a single raster even though a global run touches
thousands of them.
"""

import json
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from osm_worldcover.accounting import (
    RECEIPT_VERSION,
    BuildContext,
    atomic_json,
    file_sha256,
    outcome_from_record,
    outcome_record,
    processing_ledger,
    validate_outcome,
)
from osm_worldcover.adapters import hub
from osm_worldcover.adapters.source import RegionTables
from osm_worldcover.adapters.worldcover import WorldCoverTiles
from osm_worldcover.config import Config
from osm_worldcover.domain.revision import is_full_revision
from osm_worldcover.finalize import StreamedBuild, finalize_shards
from osm_worldcover.pipeline import RegionOutcome, run_region
from osm_worldcover.sources import DEFAULT_SOURCE, SourceRecipe

__all__ = ["BuildReport", "ShardStore", "run_build"]


class ShardStore:
    """Per-region results held on disk between the region pass and assembly.

    A global run produces more text than is comfortable to keep in memory, and
    takes long enough that it will sometimes be interrupted. Writing each
    region as it completes solves both: memory stays bounded by one region, and
    a restart skips whatever already finished.

    A region that produced no examples still writes a file. That is a finished
    result, and without it every restart would retry the empty regions forever.
    """

    def __init__(self, directory: Path, context: BuildContext | None = None) -> None:
        self.context = context
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, stem: str) -> Path:
        if not stem or Path(stem).name != stem or stem in {".", ".."}:
            raise ValueError(f"unsafe region name: {stem!r}")
        return self.directory / f"{stem}.parquet"

    def has(self, stem: str) -> bool:
        """Whether ``stem`` has already been processed to completion."""
        if self.context is not None:
            return self.outcome(stem) is not None
        path = self.path_for(stem)
        return path.exists() and path.stat().st_size > 0

    def write(
        self, stem: str, frame: pd.DataFrame, rejections: Mapping[str, int] | None = None
    ) -> None:
        """Write a generic shard; build paths must use ``write_outcome`` instead."""
        if self.context is not None:
            raise ValueError("a context-bound store requires write_outcome with full accounting")
        self._receipt_for(stem).unlink(missing_ok=True)
        self._write_frame(stem, frame)
        atomic_json(self._counters_for(stem), dict(sorted((rejections or {}).items())))

    def _write_frame(self, stem: str, frame: pd.DataFrame) -> None:
        path = self.path_for(stem)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=self.directory)
        os.close(descriptor)
        partial = Path(temporary)
        try:
            frame.to_parquet(partial, index=False)
            with partial.open("rb") as stream:
                os.fsync(stream.fileno())
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)

    def _receipt_for(self, stem: str) -> Path:
        return self.path_for(stem).with_suffix(".complete.json")

    def write_outcome(self, stem: str, frame: pd.DataFrame, outcome: RegionOutcome) -> None:
        """Commit a shard and its complete context-bound, hashed receipt.

        The receipt is installed last. A crash before that leaves no reusable
        completion marker, or leaves an old receipt whose hash cannot match.
        """
        if self.context is None:
            raise ValueError("write_outcome requires a pinned build context")
        validate_outcome(outcome)
        self._validate_frame(stem, frame, outcome)
        self._write_frame(stem, frame)
        path = self.path_for(stem)
        atomic_json(
            self._receipt_for(stem),
            {
                "schema_version": RECEIPT_VERSION,
                "build_context_sha256": self.context.fingerprint,
                "context": self.context.as_dict(),
                "shard": {
                    "filename": path.name,
                    "sha256": file_sha256(path),
                    "bytes": path.stat().st_size,
                    "rows": len(frame),
                    "columns": list(frame.columns),
                },
                "outcome": outcome_record(outcome),
            },
        )
        atomic_json(self._counters_for(stem), dict(sorted(outcome.rejections.items())))

    @staticmethod
    def _validate_frame(stem: str, frame: pd.DataFrame, outcome: RegionOutcome) -> None:
        if outcome.stem != stem or outcome.examples != len(frame):
            raise ValueError(f"{stem}: shard identity or row count differs from outcome")
        polygons = int(frame["polygon_id"].nunique()) if len(frame) else 0
        if polygons != outcome.polygons_with_examples:
            raise ValueError(f"{stem}: shard polygon count differs from outcome")

    def outcome(self, stem: str) -> RegionOutcome | None:
        """Return only a fully verified completion; legacy/stale files are pending."""
        if self.context is None:
            raise ValueError("reading completion receipts requires a pinned build context")
        try:
            return self._verified_outcome(stem)
        except (OSError, ValueError, TypeError, KeyError):
            return None

    def _verified_outcome(self, stem: str) -> RegionOutcome | None:
        receipt = json.loads(self._receipt_for(stem).read_text())
        if not isinstance(receipt, dict):
            return None
        return self._verify_receipt(stem, receipt)

    def _verify_receipt(self, stem: str, receipt: dict[str, Any]) -> RegionOutcome | None:
        if not self._matching_context(receipt):
            return None
        path = self.path_for(stem)
        shard = receipt["shard"]
        if not self._receipt_matches_shard(path, shard):
            return None
        return self._receipt_outcome(stem, path, shard, receipt)

    @staticmethod
    def _receipt_outcome(
        stem: str, path: Path, shard: Mapping[str, Any], receipt: Mapping[str, Any]
    ) -> RegionOutcome | None:
        metadata = pq.read_metadata(path)
        if shard["rows"] != metadata.num_rows or shard["columns"] != metadata.schema.names:
            return None
        outcome = outcome_from_record(receipt["outcome"])
        if outcome.stem != stem or outcome.examples != metadata.num_rows:
            return None
        return outcome

    @staticmethod
    def _receipt_matches_shard(path: Path, shard: Mapping[str, Any]) -> bool:
        return (
            shard["filename"] == path.name
            and shard["bytes"] == path.stat().st_size
            and shard["sha256"] == file_sha256(path)
        )

    def _matching_context(self, receipt: Mapping[str, Any]) -> bool:
        assert self.context is not None
        return (
            receipt.get("schema_version") == RECEIPT_VERSION
            and receipt.get("build_context_sha256") == self.context.fingerprint
            and receipt.get("context") == self.context.as_dict()
        )

    def verified_outcomes(self, stems: Sequence[str]) -> list[RegionOutcome]:
        """Load a complete requested set, refusing unverifiable assembly inputs."""
        outcomes = [self.outcome(stem) for stem in stems]
        missing = _missing_outcome_stems(stems, outcomes)
        if missing:
            raise ValueError(f"missing or unverifiable region completion receipts: {missing}")
        return [outcome for outcome in outcomes if outcome is not None]

    def stage(self, stems: Sequence[str], directory: Path) -> Path:
        """Expose exactly the verified selected shards to the streaming assembler."""
        self.verified_outcomes(stems)
        directory = Path(directory)
        self._validate_staging_directory(directory)
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
        for stem in stems:
            self._stage_shard(stem, directory)
        return directory

    def _validate_staging_directory(self, directory: Path) -> None:
        resolved = directory.resolve()
        source = self.directory.resolve()
        if resolved == source or resolved in source.parents:
            raise ValueError("staging directory must not replace the shard store")

    def _stage_shard(self, stem: str, directory: Path) -> None:
        source = self.path_for(stem)
        target = directory / source.name
        try:
            target.hardlink_to(source)
        except OSError:
            shutil.copy2(source, target)

    def _counters_for(self, stem: str) -> Path:
        return self.directory / f"{stem}.rejections.json"

    def rejections(self) -> dict[str, int]:
        """Sum legacy counters, or verified outcomes for a context-bound store."""
        if self.context is not None:
            stems = [path.stem for path in self.directory.glob("*.parquet")]
            outcomes = self.verified_outcomes(stems)
            return _summed(outcomes)
        total: Counter[str] = Counter()
        for path in sorted(self.directory.glob("*.rejections.json")):
            total.update(json.loads(path.read_text()))
        return dict(sorted(total.items()))

    def read(self) -> list[pd.DataFrame]:
        """Read every non-empty shard, in a deterministic order."""
        paths = sorted(self.directory.glob("*.parquet"))
        if self.context is not None:
            self.verified_outcomes([path.stem for path in paths])
        return _nonempty_frames(paths)


def _missing_outcome_stems(
    stems: Sequence[str], outcomes: Sequence[RegionOutcome | None]
) -> list[str]:
    return [stem for stem, outcome in zip(stems, outcomes, strict=True) if outcome is None]


def _nonempty_frames(paths: Sequence[Path]) -> list[pd.DataFrame]:
    frames = [pd.read_parquet(path) for path in paths]
    return [frame for frame in frames if len(frame) > 0]


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


def _summed(outcomes: Sequence[RegionOutcome]) -> dict[str, int]:
    """Why polygons were refused, summed over every region."""
    total: Counter[str] = Counter()
    for outcome in outcomes:
        total.update(outcome.rejections)
    return dict(total)


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
