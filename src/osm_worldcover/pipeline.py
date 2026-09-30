"""Turn one region's source tables into labelled text examples.

The unit of work is a region, and within it a WorldCover tile: polygons are
grouped by the tiles they touch so each ~94 MB raster is fetched once, used for
every polygon over it, and then discarded. Nothing here holds more than one
region and its tiles in memory.

Labelling and example assembly are kept apart. Labelling answers "what covers
this polygon"; assembly answers "which articles describe it". Only polygons
that survive the first question reach the second, so the expensive text join
runs on a fraction of the rows.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import geopandas as gpd
import pandas as pd
import shapely

from osm_worldcover.adapters.source import RegionTables
from osm_worldcover.adapters.worldcover import (
    TileNotPublishedError,
    TileSource,
    class_coverage,
)
from osm_worldcover.config import Config
from osm_worldcover.domain import nomenclature
from osm_worldcover.domain.dominance import (
    DominanceOutcome,
    OverlappingCoverageError,
    RejectionReason,
    decide,
)
from osm_worldcover.domain.geometry import is_usable_polygon
from osm_worldcover.domain.text import is_usable, normalise, word_count
from osm_worldcover.domain.tiling import Tile, tiles_for_bbox

__all__ = ["RegionOutcome", "label_polygons", "prepare_polygons", "run_region", "to_examples"]


@dataclass(slots=True)
class RegionOutcome:
    """What became of one region."""

    stem: str
    polygons_seen: int = 0
    polygons_invalid: int = 0
    polygons_accepted: int = 0
    polygons_with_examples: int = 0
    source_links: int = 0
    source_documents: int = 0
    text_rejections: Counter[str] = field(default_factory=Counter)
    examples: int = 0
    rejections: Counter[str] = field(default_factory=Counter)
    tiles_missing: list[str] = field(default_factory=list)


def prepare_polygons(polygons: pd.DataFrame) -> tuple[gpd.GeoDataFrame, int]:
    """Parse the stored GeoJSON geometries and drop the unusable ones.

    Returns the usable polygons and how many were discarded, so the count can
    be reported rather than silently absorbed.
    """
    if len(polygons) == 0:
        empty = gpd.GeoDataFrame(polygons.assign(geometry=[]), geometry="geometry", crs="EPSG:4326")
        return empty, 0

    geometries = shapely.from_geojson(polygons["geometry"].to_numpy(), on_invalid="ignore")
    frame = gpd.GeoDataFrame(
        polygons.drop(columns=["geometry"]), geometry=geometries, crs="EPSG:4326"
    )
    usable = frame.geometry.map(is_usable_polygon).to_numpy()
    return frame[usable].reset_index(drop=True), int((~usable).sum())


def tiles_for_frame(frame: gpd.GeoDataFrame) -> list[tuple[Tile, ...]]:
    """Return, per row, the tiles its geometry touches."""
    return [
        tuple(tiles_for_bbox(bounds)) for bounds in frame.geometry.bounds.itertuples(index=False)
    ]


def label_polygons(
    frame: gpd.GeoDataFrame,
    tiles: TileSource,
    threshold: float,
    outcome: RegionOutcome,
    keep_tiles: bool = False,
    max_area_m2: float | None = None,
) -> pd.DataFrame:
    """Label every polygon in ``frame`` that one class dominates.

    Polygons are grouped by the tiles they touch, so each raster is fetched
    once, used for every polygon over it, and released before the next group.
    Oversized polygons are dropped first, before anything is downloaded.
    """
    frame = _within_size_cap(frame, max_area_m2, outcome)
    blocks = _label_every_group(frame, tiles, threshold, outcome, keep_tiles)
    if not blocks:
        return pd.DataFrame()
    labelled = pd.concat(blocks, ignore_index=True)
    outcome.polygons_accepted = len(labelled)
    return labelled


def _label_every_group(
    frame: gpd.GeoDataFrame,
    tiles: TileSource,
    threshold: float,
    outcome: RegionOutcome,
    keep_tiles: bool,
) -> list[pd.DataFrame]:
    """Label each group of polygons sharing a tile set."""
    if len(frame) == 0:
        return []
    # groupby widens its key to Hashable and its group to DataFrame; both are
    # narrower than that here by construction.
    grouped = cast("gpd.GeoDataFrame", frame.assign(_tiles=tiles_for_frame(frame)))
    blocks = [
        _process_group(
            cast("gpd.GeoDataFrame", group),
            cast("tuple[Tile, ...]", tile_set),
            tiles,
            threshold,
            outcome,
            keep_tiles,
        )
        for tile_set, group in grouped.groupby("_tiles", sort=True)
    ]
    return [b for b in blocks if b is not None]


def _within_size_cap(
    frame: gpd.GeoDataFrame, max_area_m2: float | None, outcome: RegionOutcome
) -> gpd.GeoDataFrame:
    """Drop polygons above the cap, counting them.

    Screened on the source's own ``area_m2`` before any tile is fetched, so a
    continent-scale polygon costs nothing rather than ~10 GB of download.
    """
    if max_area_m2 is None:
        return frame
    keep = frame["area_m2"].to_numpy() <= max_area_m2
    rejected = int((~keep).sum())
    if rejected:
        outcome.rejections[RejectionReason.TOO_LARGE.value] += rejected
    return cast("gpd.GeoDataFrame", frame[keep].reset_index(drop=True))


def _process_group(
    group: gpd.GeoDataFrame,
    tile_set: Sequence[Tile],
    tiles: TileSource,
    threshold: float,
    outcome: RegionOutcome,
    keep_tiles: bool,
) -> pd.DataFrame | None:
    """Label one group of polygons sharing a tile set, then release the tiles."""
    try:
        paths = _fetch(tiles, tile_set, outcome)
        if not paths:
            outcome.rejections[RejectionReason.NO_VALID_CLASS.value] += len(group)
            return None
        return _label_group(group, paths, threshold, outcome)
    finally:
        if not keep_tiles:
            for tile in tile_set:
                tiles.discard(tile)


def _fetch(tiles: TileSource, tile_set: Sequence[Tile], outcome: RegionOutcome) -> list[Path]:
    """Download every published tile in ``tile_set``, noting the ones that are not.

    Paths are de-duplicated: reading one raster twice would count its coverage
    twice and make an otherwise valid polygon look doubly covered.
    """
    paths: dict[Path, None] = {}
    for tile in tile_set:
        try:
            paths[tiles.ensure(tile)] = None
        except TileNotPublishedError:
            outcome.tiles_missing.append(tile.name)
    return list(paths)


def _label_group(
    group: gpd.GeoDataFrame,
    paths: Sequence[Path],
    threshold: float,
    outcome: RegionOutcome,
) -> pd.DataFrame | None:
    """Return the rows of ``group`` that a single class dominates, or ``None``."""
    coverages = class_coverage(paths, group)
    verdicts = [_verdict(c, threshold) for c in coverages]
    kept = [i for i, v in enumerate(verdicts) if v.accepted]
    _tally(verdicts, set(kept), outcome)
    if not kept:
        return None
    return _block(group, kept, verdicts, coverages)


def _block(
    group: gpd.GeoDataFrame,
    kept: Sequence[int],
    verdicts: Sequence[DominanceOutcome],
    coverages: Sequence[dict[int, float]],
) -> pd.DataFrame:
    """Attach the label columns to the rows that were accepted."""
    block = group.iloc[list(kept)].copy()
    block["worldcover_code"] = [verdicts[i].code for i in kept]
    block["dominant_fraction"] = [verdicts[i].fraction for i in kept]
    block["observed_fraction"] = [sum(coverages[i].values()) for i in kept]
    return block.drop(columns=["_tiles"])


def _tally(verdicts: Sequence[DominanceOutcome], kept: set[int], outcome: RegionOutcome) -> None:
    """Record why each rejected polygon was rejected."""
    for index, verdict in enumerate(verdicts):
        if index not in kept and verdict.reason is not None:
            outcome.rejections[verdict.reason.value] += 1


def _verdict(coverage: dict[int, float], threshold: float) -> DominanceOutcome:
    """Apply the dominance rule to one polygon's coverage shares.

    Shares are already relative to the polygon, so its area is 1 by construction.
    """
    try:
        return decide(coverage, polygon_area=1.0, threshold=threshold)
    except OverlappingCoverageError:
        # Double-counted coverage means the tiles overlapped, which they must
        # not; refuse the polygon rather than trust the arithmetic.
        return DominanceOutcome(False, None, 0.0, RejectionReason.NO_VALID_CLASS)


def to_examples(
    labelled: pd.DataFrame,
    tables: RegionTables,
    min_words: int,
    outcome: RegionOutcome | None = None,
) -> pd.DataFrame:
    """Join labels to text and account for every polygon lost at each stage.

    A polygon is rejected at its last surviving stage, so the mutually
    exclusive counters sum to labelled polygons without any usable example.
    Multiple documents can still produce multiple examples for one polygon.
    """
    if len(labelled) == 0:
        return pd.DataFrame()
    links = tables.links[["polygon_id", "document_id"]]
    joined = labelled.merge(links, on="polygon_id", how="inner")
    _text_loss(labelled, joined, "no_source_document", outcome)
    linked = joined.merge(tables.documents, on="document_id", how="inner", suffixes=("", "_doc"))
    _text_loss(joined, linked, "missing_document", outcome)
    fetched = linked[linked["fetch_status"] == "ok"].copy()
    _text_loss(linked, fetched, "document_fetch_failed", outcome)
    fetched["text"] = fetched["full_text"].fillna("").map(normalise)
    nonempty = fetched[fetched["text"].map(bool).astype(bool)]
    _text_loss(fetched, nonempty, "empty_text", outcome)
    kept = nonempty[nonempty["text"].map(lambda t: is_usable(t, min_words)).astype(bool)]
    _text_loss(nonempty, kept, "text_too_short", outcome)
    return kept.reset_index(drop=True)


def _text_loss(
    before: pd.DataFrame,
    after: pd.DataFrame,
    reason: str,
    outcome: RegionOutcome | None,
) -> None:
    """Count polygons that lose their final candidate at this text stage."""
    if outcome is not None:
        count = before["polygon_id"].nunique() - after["polygon_id"].nunique()
        if count:
            outcome.text_rejections[reason] += int(count)


def run_region(
    config: Config,
    tables: RegionTables,
    tiles: TileSource,
    keep_tiles: bool = False,
) -> tuple[pd.DataFrame, RegionOutcome]:
    """Produce every example for one region."""
    outcome = RegionOutcome(
        stem=tables.stem, source_links=len(tables.links), source_documents=len(tables.documents)
    )
    frame, invalid = prepare_polygons(tables.polygons)
    outcome.polygons_seen = len(tables.polygons)
    outcome.polygons_invalid = invalid

    labelled = label_polygons(
        frame,
        tiles,
        config.threshold,
        outcome,
        keep_tiles,
        max_area_m2=config.max_polygon_area_m2,
    )
    examples = to_examples(labelled, tables, config.effective_min_words, outcome)
    examples = _shape(examples)
    outcome.examples = len(examples)
    outcome.polygons_with_examples = int(examples["polygon_id"].nunique())
    return examples, outcome


OUTPUT_COLUMNS: Sequence[str] = (
    "polygon_id",
    "osm_type",
    "osm_id",
    "region",
    "name",
    "wikidata",
    "document_id",
    "project",
    "language",
    "title",
    "url",
    "text",
    "lead_text",
    "text_words",
    "worldcover_code",
    "worldcover_label",
    "dominant_fraction",
    "observed_fraction",
    "lat",
    "lon",
    "centroid_wkt",
    "polygon_area_m2",
    "source_pbf",
)


#: Published columns that are text. A region whose rows all lack one of these
#: would otherwise write it as a NULL-typed Parquet column, and a reader taking
#: its schema from the first file would then refuse the rest.
TEXT_COLUMNS: Sequence[str] = (
    "polygon_id",
    "osm_type",
    "region",
    "name",
    "wikidata",
    "document_id",
    "project",
    "language",
    "title",
    "url",
    "text",
    "lead_text",
    "worldcover_label",
    "centroid_wkt",
    "source_pbf",
)


def _shape(examples: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns and project onto the published schema."""
    if len(examples) == 0:
        return pd.DataFrame(columns=list(OUTPUT_COLUMNS))

    examples = examples.copy()
    examples["worldcover_label"] = examples["worldcover_code"].map(nomenclature.label_for)
    examples["text_words"] = examples["text"].map(word_count)
    examples["polygon_area_m2"] = examples["area_m2"]
    examples["centroid_wkt"] = shapely.to_wkt(
        shapely.points(examples["lon"].to_numpy(), examples["lat"].to_numpy()),
        rounding_precision=7,
    )
    if "language_doc" in examples.columns:
        # The document's own language is authoritative; the link table's copy
        # is only a hint.
        examples["language"] = examples["language_doc"].fillna(examples["language"])
    return _project(examples)


def _project(examples: pd.DataFrame) -> pd.DataFrame:
    """Narrow to the published schema, filling anything the source omitted."""
    for column in OUTPUT_COLUMNS:
        if column not in examples.columns:
            examples[column] = None
    return examples[list(OUTPUT_COLUMNS)]
