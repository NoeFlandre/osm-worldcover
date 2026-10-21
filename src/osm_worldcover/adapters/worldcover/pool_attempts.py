"""Source-wide WorldCover evidence without text or dominance filtering."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import shapely
from rasterio.errors import RasterioError
from shapely.errors import ShapelyError
from shapely.geometry.base import BaseGeometry

from osm_worldcover.adapters.worldcover.geodesic_coverage import (
    AREA_MEASUREMENT,
    geodesic_class_coverage_with_tile_evidence,
)
from osm_worldcover.adapters.worldcover.tiles import TileNotPublishedError, TileSource
from osm_worldcover.domain.geometry import is_usable_polygon
from osm_worldcover.domain.tiling import Tile, tiles_for_bbox


@dataclass(frozen=True, slots=True)
class SourceWideCoverageAttempt:
    """One raw polygon's raster attempt, including failures and uncovered tiles."""

    source_polygon_id: str
    source_row_ordinal: int
    polygon_area_m2: float | None
    class_areas_m2: tuple[tuple[int, float], ...]
    area_measurement: str
    attempted_tiles: tuple[str, ...]
    missing_tiles: tuple[str, ...]
    extraction_error: str | None = None
    class_evidence_ids: tuple[tuple[int, tuple[str, ...]], ...] = ()


def iter_sourcewide_coverage(
    polygons: pd.DataFrame | Iterable[pd.DataFrame],
    tiles: TileSource,
) -> Iterator[SourceWideCoverageAttempt]:
    """Measure every raw polygon row, before document or label eligibility joins.

    ``polygons`` must be the unfiltered source table, either as one DataFrame
    or as an iterator of bounded DataFrame row groups. Each frame has
    ``polygon_id`` and GeoJSON ``geometry`` columns. The iterator yields exactly
    one result per raw row, including malformed or unusable geometries. Missing
    ESA tiles remain uncovered support in the full-polygon denominator;
    operational tile or raster errors are recorded by downstream adapters.
    """

    frames = (polygons,) if isinstance(polygons, pd.DataFrame) else polygons
    ordinal = 0
    for frame in frames:
        if "polygon_id" not in frame or "geometry" not in frame:
            raise ValueError("raw WorldCover input requires polygon_id and geometry columns")
        for row in frame[["polygon_id", "geometry"]].itertuples(index=False):
            raw_id = row.polygon_id
            source_polygon_id = "" if raw_id is None or pd.isna(raw_id) else str(raw_id).strip()
            if not source_polygon_id:
                yield SourceWideCoverageAttempt(
                    source_polygon_id=f"row:{ordinal}",
                    source_row_ordinal=ordinal,
                    polygon_area_m2=None,
                    class_areas_m2=(),
                    area_measurement=AREA_MEASUREMENT,
                    attempted_tiles=(),
                    missing_tiles=(),
                    extraction_error="source_polygon_id_missing",
                )
                ordinal += 1
                continue
            geometry = _decode_polygon(row.geometry)
            if geometry is None or not is_usable_polygon(geometry):
                yield SourceWideCoverageAttempt(
                    source_polygon_id=source_polygon_id,
                    source_row_ordinal=ordinal,
                    polygon_area_m2=None,
                    class_areas_m2=(),
                    area_measurement=AREA_MEASUREMENT,
                    attempted_tiles=(),
                    missing_tiles=(),
                    extraction_error="geometry_unusable",
                )
                ordinal += 1
                continue
            yield _measure_polygon(source_polygon_id, ordinal, geometry, tiles)
            ordinal += 1


def _measure_polygon(
    source_polygon_id: str,
    ordinal: int,
    geometry: BaseGeometry,
    tile_source: TileSource,
) -> SourceWideCoverageAttempt:
    """Fetch each touched tile once for this row and retain every outcome."""

    tile_set = tuple(tiles_for_bbox(geometry.bounds))
    attempted: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    try:
        tile_paths, attempted, missing, errors = _fetch_tiles(tile_set, tile_source)
        if errors:
            return SourceWideCoverageAttempt(
                source_polygon_id,
                ordinal,
                None,
                (),
                AREA_MEASUREMENT,
                attempted,
                missing,
                f"tile_read_failed:{','.join(sorted(set(errors)))}",
            )
        polygon_area_m2, class_areas_m2, class_evidence_ids = _coverage_areas(geometry, tile_paths)
        return SourceWideCoverageAttempt(
            source_polygon_id,
            ordinal,
            polygon_area_m2,
            class_areas_m2,
            AREA_MEASUREMENT,
            attempted,
            missing,
            class_evidence_ids=class_evidence_ids,
        )
    except (OSError, RasterioError, RuntimeError, ValueError, ShapelyError) as error:
        return SourceWideCoverageAttempt(
            source_polygon_id,
            ordinal,
            None,
            (),
            AREA_MEASUREMENT,
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
) -> tuple[list[tuple[str, Path]], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Attempt every touched tile and report unavailable versus failed reads."""

    paths: list[tuple[str, Path]] = []
    attempted: list[str] = []
    missing: list[str] = []
    errors: list[str] = []
    for tile in tile_set:
        attempted.append(tile.name)
        try:
            paths.append((tile.name, tile_source.ensure(tile)))
        except TileNotPublishedError:
            missing.append(tile.name)
        except (OSError, RasterioError, RuntimeError, ValueError) as error:
            errors.append(type(error).__name__)
    return paths, tuple(attempted), tuple(missing), tuple(sorted(set(errors)))


def _coverage_areas(
    geometry: BaseGeometry,
    tile_paths: list[tuple[str, Path]],
) -> tuple[float, tuple[tuple[int, float], ...], tuple[tuple[int, tuple[str, ...]], ...]]:
    """Measure square-metre support on the unchanged full source geometry."""

    return geodesic_class_coverage_with_tile_evidence(tile_paths, geometry)


def _decode_polygon(value: object) -> BaseGeometry | None:
    if isinstance(value, BaseGeometry):
        return value
    if not isinstance(value, str) or not value:
        return None
    try:
        return shapely.from_geojson(value, on_invalid="ignore")
    except (ShapelyError, ValueError):
        return None
