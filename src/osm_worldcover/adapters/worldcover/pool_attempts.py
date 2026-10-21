"""Source-wide WorldCover evidence without text or dominance filtering."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd
import shapely
from rasterio.errors import RasterioError
from shapely.errors import ShapelyError
from shapely.geometry.base import BaseGeometry

from osm_worldcover.adapters.worldcover.extract import class_coverage
from osm_worldcover.adapters.worldcover.tiles import TileNotPublishedError, TileSource
from osm_worldcover.domain.geometry import is_usable_polygon
from osm_worldcover.domain.tiling import Tile, tiles_for_bbox


@dataclass(frozen=True, slots=True)
class SourceWideCoverageAttempt:
    """One raw polygon's raster attempt, including failures and uncovered tiles."""

    source_polygon_id: str
    source_row_ordinal: int
    class_fractions: tuple[tuple[int, float], ...]
    attempted_tiles: tuple[str, ...]
    missing_tiles: tuple[str, ...]
    extraction_error: str | None = None


def iter_sourcewide_coverage(
    polygons: pd.DataFrame,
    tiles: TileSource,
) -> Iterator[SourceWideCoverageAttempt]:
    """Measure every raw polygon row, before document or label eligibility joins.

    ``polygons`` must be the unfiltered source polygon table with ``polygon_id``
    and GeoJSON ``geometry`` columns. The iterator yields exactly one result per
    row, including malformed or unusable geometries. Missing ESA tiles remain
    uncovered support in the extractor's full-polygon denominator; operational
    tile or raster errors are recorded as exclusions by downstream adapters.
    """

    if "polygon_id" not in polygons or "geometry" not in polygons:
        raise ValueError("raw WorldCover input requires polygon_id and geometry columns")
    seen_ids: set[str] = set()
    for ordinal, row in enumerate(polygons[["polygon_id", "geometry"]].itertuples(index=False)):
        raw_id = row.polygon_id
        source_polygon_id = "" if raw_id is None or pd.isna(raw_id) else str(raw_id).strip()
        if not source_polygon_id:
            yield SourceWideCoverageAttempt(
                source_polygon_id=f"row:{ordinal}",
                source_row_ordinal=ordinal,
                class_fractions=(),
                attempted_tiles=(),
                missing_tiles=(),
                extraction_error="source_polygon_id_missing",
            )
            continue
        if source_polygon_id in seen_ids:
            raise ValueError(f"raw WorldCover input repeats polygon_id {source_polygon_id!r}")
        seen_ids.add(source_polygon_id)
        geometry = _decode_polygon(row.geometry)
        if geometry is None or not is_usable_polygon(geometry):
            yield SourceWideCoverageAttempt(
                source_polygon_id=source_polygon_id,
                source_row_ordinal=ordinal,
                class_fractions=(),
                attempted_tiles=(),
                missing_tiles=(),
                extraction_error="geometry_unusable",
            )
            continue
        yield _measure_polygon(source_polygon_id, ordinal, geometry, tiles)


def _measure_polygon(
    source_polygon_id: str,
    ordinal: int,
    geometry: BaseGeometry,
    tile_source: TileSource,
) -> SourceWideCoverageAttempt:
    """Fetch each touched tile once for this row and retain every outcome."""

    tile_set = tuple(tiles_for_bbox(geometry.bounds))
    paths: list[Path] = []
    attempted: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    try:
        paths, attempted, missing, errors = _fetch_tiles(tile_set, tile_source)
        if errors:
            return SourceWideCoverageAttempt(
                source_polygon_id,
                ordinal,
                (),
                attempted,
                missing,
                f"tile_read_failed:{','.join(sorted(set(errors)))}",
            )
        fractions = _coverage_fractions(geometry, paths)
        return SourceWideCoverageAttempt(
            source_polygon_id,
            ordinal,
            fractions,
            attempted,
            missing,
        )
    except (OSError, RasterioError, RuntimeError, ValueError, ShapelyError) as error:
        return SourceWideCoverageAttempt(
            source_polygon_id,
            ordinal,
            (),
            attempted,
            missing,
            f"raster_measurement_failed:{type(error).__name__}",
        )
    finally:
        for tile in tile_set:
            tile_source.discard(tile)


def _fetch_tiles(
    tile_set: tuple[Tile, ...],
    tile_source: TileSource,
) -> tuple[list[Path], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Attempt every touched tile and report unavailable versus failed reads."""

    paths: list[Path] = []
    attempted: list[str] = []
    missing: list[str] = []
    errors: list[str] = []
    for tile in tile_set:
        attempted.append(tile.name)
        try:
            paths.append(tile_source.ensure(tile))
        except TileNotPublishedError:
            missing.append(tile.name)
        except (OSError, RasterioError, RuntimeError, ValueError) as error:
            errors.append(type(error).__name__)
    return paths, tuple(attempted), tuple(missing), tuple(sorted(set(errors)))


def _coverage_fractions(
    geometry: BaseGeometry,
    paths: list[Path],
) -> tuple[tuple[int, float], ...]:
    """Measure class fractions on the unchanged full source geometry."""

    if not paths:
        return ()
    frame = gpd.GeoDataFrame(geometry=[geometry], crs="EPSG:4326")
    measured = class_coverage(paths, frame)[0]
    return tuple(sorted((int(code), float(value)) for code, value in measured.items()))


def _decode_polygon(value: object) -> BaseGeometry | None:
    if isinstance(value, BaseGeometry):
        return value
    if not isinstance(value, str) or not value:
        return None
    try:
        return shapely.from_geojson(value, on_invalid="ignore")
    except (ShapelyError, ValueError):
        return None
