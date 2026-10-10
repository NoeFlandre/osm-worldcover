"""Assemble region shards into the published dataset.

Three different problems are resolved here, and conflating them would get at
least one of them wrong:

* Geofabrik's regional extracts overlap, so one OSM object appears in several
  regions under different ``polygon_id`` values. Those are the *same* object,
  and one region is chosen for all of its documents.
* Distinct polygons can carry byte-identical text and labels; these remain
  separate examples, with cross-split collisions reported as diagnostics.
* One article can describe several distant places. Those fall in different
  cells and therefore different splits, which would put the document in train
  *and* test.

Nothing is ever held whole. A global build is several times the memory of the
machine that produces it, so row-local work happens a shard at a time, the
global work is left to DuckDB over files, and the result is streamed to Parquet
in batches rather than collected first.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import pandas as pd
import pyarrow.parquet as pq

from osm_worldcover.adapters._duckdb import open_session
from osm_worldcover.adapters.sql import sql_literal
from osm_worldcover.adapters.writer import write_batches, write_manifest
from osm_worldcover.config import Config
from osm_worldcover.domain import manifest as manifest_module
from osm_worldcover.domain.manifest import DatasetCounts
from osm_worldcover.domain.text import dedup_key, word_count
from osm_worldcover.domain.validation import (
    REQUIRED_COLUMNS,
    ValidationReport,
    validate,
)
from osm_worldcover.release_commit import (
    commit_release,
    core_inventory,
    create_stage,
    discard_stage,
    release_write_lock,
    stage_inventory,
    transaction_pending,
)

from .aggregate import _aggregate, _count
from .analysis import _deduplication_analysis, _retained_text_diagnostics
from .enrich import _assign_splits, _attach_provenance, _with_stable_text_types
from .staged import _validate_staged_manifest

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

__all__ = ["StreamedBuild", "finalize_shards"]


# DuckDB's working-memory budget for the global deduplication pass.
_DEDUP_MEMORY_LIMIT: Final = "2GB"


@dataclass(slots=True)
class StreamedBuild:
    """A finished build, written to disk without ever being held in memory."""

    rows: int
    paths: list[Path]
    manifest: dict[str, Any]
    report: ValidationReport
    duplicates_across_regions: int = 0
    duplicate_records: int = 0
    documents_split_across_splits: int = 0


def finalize_shards(
    shard_dir: Path,
    config: Config,
    work_dir: Path,
    out_dir: Path,
    rejections: dict[str, int] | None = None,
    processing: dict[str, Any] | None = None,
) -> StreamedBuild:
    """Assemble region shards into the dataset written under ``out_dir``."""
    shard_dir = Path(shard_dir).absolute()
    work_dir, out_dir = Path(work_dir).absolute(), Path(out_dir).resolve()
    target = out_dir / f"v{config.dataset_version}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with release_write_lock(target):
        return _finalize_locked(Path(shard_dir), config, work_dir, target, rejections, processing)


def _finalize_locked(
    shard_dir: Path,
    config: Config,
    work_dir: Path,
    target: Path,
    rejections: dict[str, int] | None,
    processing: dict[str, Any] | None,
) -> StreamedBuild:
    enriched = work_dir / "enriched"
    if enriched.exists():
        shutil.rmtree(enriched)
    enriched.mkdir(parents=True)

    if _enrich_shards(shard_dir, enriched, config) == 0:
        return StreamedBuild(0, [], {}, validate([]))

    connection, dropped, deduplication_analysis = _deduplicate(enriched)
    try:
        return _build_candidate(
            connection,
            config,
            target,
            dropped,
            deduplication_analysis,
            rejections,
            processing,
        )
    finally:
        connection.close()


def _build_candidate(
    connection: DuckDBPyConnection,
    config: Config,
    target: Path,
    dropped: dict[str, int],
    deduplication_analysis: dict[str, Any],
    rejections: dict[str, int] | None,
    processing: dict[str, Any] | None,
) -> StreamedBuild:
    stage = create_stage(target)
    try:
        rows, manifest, report = _prepare_candidate(
            connection,
            config,
            stage,
            dropped,
            deduplication_analysis,
            rejections,
            processing,
        )
        if not report.ok:
            discard_stage(target, stage)
            return _streamed_result(rows, [], manifest, report, dropped)
        _retain_or_promote_candidate(target, stage)
        return _streamed_result(rows, _target_paths(target), manifest, report, dropped)
    except BaseException:
        _discard_candidate_after_error(target, stage)
        raise


def _prepare_candidate(
    connection: DuckDBPyConnection,
    config: Config,
    stage: Path,
    dropped: dict[str, int],
    deduplication_analysis: dict[str, Any],
    rejections: dict[str, int] | None,
    processing: dict[str, Any] | None,
) -> tuple[int, dict[str, Any], ValidationReport]:
    paths, rows = _write_splits(connection, stage)
    counts = _aggregate(connection, rejections or {}, dropped, deduplication_analysis)
    manifest = _build_candidate_manifest(counts, config, processing)
    manifest_path = write_manifest(manifest, stage / "manifest.json")
    report = _validate_written(paths, config)
    _validate_staged_manifest(manifest_path, paths, manifest, rows)
    return rows, manifest, report


def _build_candidate_manifest(
    counts: DatasetCounts, config: Config, processing: dict[str, Any] | None
) -> dict[str, Any]:
    manifest = manifest_module.build(counts, config.as_manifest_settings())
    if processing is not None:
        manifest["processing"] = processing
    return manifest


def _retain_or_promote_candidate(target: Path, stage: Path) -> None:
    inventory = stage_inventory(stage)
    if core_inventory(target) == inventory:
        discard_stage(target, stage)
    else:
        commit_release(target, stage, inventory)


def _discard_candidate_after_error(target: Path, stage: Path) -> None:
    if not transaction_pending(target):
        with suppress(Exception):
            discard_stage(target, stage)


def _streamed_result(
    rows: int,
    paths: list[Path],
    manifest: dict[str, Any],
    report: ValidationReport,
    dropped: dict[str, int],
) -> StreamedBuild:
    return StreamedBuild(
        rows=rows,
        paths=paths,
        manifest=manifest,
        report=report,
        duplicates_across_regions=dropped["duplicate_objects_across_regions"],
        duplicate_records=dropped["duplicate_polygon_text_label_records"],
        documents_split_across_splits=dropped["documents_split_across_splits"],
    )


def _target_paths(target: Path) -> list[Path]:
    return [
        *(target / f"{split}.parquet" for split in manifest_module.SPLIT_ORDER),
        target / "manifest.json",
    ]


def _enrich_shards(shard_dir: Path, enriched: Path, config: Config) -> int:
    """Add the row-local columns to each shard in turn. Returns rows seen."""
    total = 0
    for path in sorted(shard_dir.glob("*.parquet")):
        frame = pd.read_parquet(path)
        if len(frame) == 0 or "polygon_id" not in frame.columns:
            continue
        frame = _assign_splits(frame, config)
        frame = _attach_provenance(frame, config)
        frame = _with_stable_text_types(frame)
        frame["text_words"] = frame["text"].map(word_count).astype("int64")
        frame["_dedup_key"] = [
            dedup_key(text, str(code))
            for text, code in zip(frame["text"], frame["worldcover_code"], strict=True)
        ]
        frame.to_parquet(enriched / path.name, index=False)
        total += len(frame)
    return total


def _deduplicate(enriched: Path) -> tuple[DuckDBPyConnection, dict[str, int], dict[str, Any]]:
    """Collapse duplicates and split conflicts across every shard, using DuckDB.

    The open connection is returned so the surviving rows can be streamed out
    rather than collected. It is closed here if anything fails before that.
    Every ordering is fully specified, so no survivor depends on the order
    files happened to be read in.
    """
    source = sql_literal(str(enriched / "*.parquet"))
    connection = open_session(_DEDUP_MEMORY_LIMIT, enriched.parent / "duckdb-spill")
    try:
        before = _count(connection, f"SELECT count(*) FROM read_parquet({source})")

        # One region per OSM object, so an object cannot wear two polygon_ids.
        connection.execute(
            f"""
            CREATE TEMP TABLE objects AS
            WITH home_region AS (
                SELECT osm_type, osm_id, region AS _home_region FROM (
                    SELECT osm_type, osm_id, region, row_number() OVER (
                        PARTITION BY osm_type, osm_id ORDER BY region
                    ) AS _rank
                    FROM (SELECT DISTINCT osm_type, osm_id, region FROM read_parquet({source}))
                ) WHERE _rank = 1
            ),
            canonical AS (
                SELECT r.* FROM read_parquet({source}) r
                JOIN home_region h USING (osm_type, osm_id)
                WHERE r.region = h._home_region
            )
            SELECT * EXCLUDE (_rank) FROM (
                SELECT *, row_number() OVER (
                    PARTITION BY osm_type, osm_id, document_id ORDER BY polygon_id
                ) AS _rank
                FROM canonical
            ) WHERE _rank = 1
            """
        )
        after_objects = _count(connection, "SELECT count(*) FROM objects")
        analysis = _deduplication_analysis(connection)

        # Remove only repeated records for the same stable polygon, text and label.
        connection.execute(
            """
            CREATE TEMP TABLE examples AS
            SELECT * EXCLUDE (_rank) FROM (
                SELECT *, row_number() OVER (
                    PARTITION BY polygon_id, _dedup_key ORDER BY document_id
                ) AS _rank
                FROM objects
            ) WHERE _rank = 1
            """
        )
        after_examples = _count(connection, "SELECT count(*) FROM examples")

        # One split per document: the one holding most of its rows.
        connection.execute(
            """
            CREATE TEMP TABLE kept AS
            WITH document_home AS (
                SELECT document_id, split AS _home FROM (
                    SELECT document_id, split, count(*) AS n, row_number() OVER (
                        PARTITION BY document_id ORDER BY count(*) DESC, split
                    ) AS _rank
                    FROM examples GROUP BY document_id, split
                ) WHERE _rank = 1
            )
            SELECT e.* FROM examples e
            JOIN document_home h USING (document_id)
            WHERE e.split = h._home
            """
        )
        after = _count(connection, "SELECT count(*) FROM kept")
        analysis.update(_retained_text_diagnostics(connection))
        connection.execute("DROP TABLE objects")
        connection.execute("DROP TABLE examples")
    except BaseException:
        connection.close()
        raise

    return (
        connection,
        {
            "duplicate_objects_across_regions": before - after_objects,
            "duplicate_polygon_text_label_records": after_objects - after_examples,
            "documents_split_across_splits": after_examples - after,
        },
        analysis,
    )


def _write_splits(connection: DuckDBPyConnection, target: Path) -> tuple[list[Path], int]:
    """Stream each split from DuckDB into its own Parquet file."""
    paths: list[Path] = []
    rows = 0
    connection.execute("SET arrow_large_buffer_size = true")
    for split in manifest_module.SPLIT_ORDER:
        path = target / f"{split}.parquet"
        reader = connection.execute(
            "SELECT * EXCLUDE (_dedup_key) FROM kept "
            "WHERE split = ? ORDER BY polygon_id, document_id",
            [split],
        ).to_arrow_reader()
        rows += write_batches(reader, path)
        paths.append(path)
    return paths, rows


def _validate_written(paths: Sequence[Path], config: Config) -> ValidationReport:
    """Validate the dataset that was actually written, by streaming it back."""

    def rows() -> Iterable[dict[str, Any]]:
        for path in paths:
            batches = pq.ParquetFile(path).iter_batches(
                batch_size=8192, columns=list(REQUIRED_COLUMNS)
            )
            for batch in batches:
                yield from batch.to_pylist()

    return validate(rows(), threshold=config.threshold, min_words=config.effective_min_words)
