"""WGS84 geodesic area evidence for source-wide WorldCover attempts.

The older WorldCover release path measures pixel support and polygon area in
EPSG:4326 square degrees. This module is a separate pool adapter: it weights
fully covered raster cells by their WGS84 ellipsoidal area and measures every
boundary-cell intersection geodesically. NoData cells and unavailable tiles
contribute no numerator while the complete source polygon remains the
denominator.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyproj
import rasterio
import shapely
from exactextract import exact_extract
from rasterio.features import rasterize
from shapely.geometry import box

from osm_worldcover.adapters.worldcover.geometry import (
    _bounded_pixel_chunks,
    _map_geometry,
    _pixel_geometry,
)

AREA_MEASUREMENT = "pyproj.Geod(ellps=WGS84); boundary-cell intersections geodesic; v1"
AREA_UNIT = "m^2"
_GEOD = pyproj.Geod(ellps="WGS84")
_OPS = ("cell_id", "coverage", "values")
_MAX_CELLS_IN_MEMORY = 2_000_000
_BOUNDARY_HALO_PIXELS = 0.01
_BOUNDARY_ROW_CHUNK = 256
_BOUNDARY_CELL_CHUNK = 16_384
_READ_ROW_BAND = 64
_GEOGRAPHIC_CRS = pyproj.CRS.from_epsg(4326)


def geodesic_class_coverage(
    raster_paths: Iterable[Path], geometry: shapely.Geometry
) -> tuple[float, tuple[tuple[int, float], ...]]:
    """Return full-polygon area and per-class WGS84 support areas in square metres.

    Raster tiles must be north-up EPSG:4326 grids. Interior pixel support is
    weighted by the ellipsoidal area of each pixel row. Boundary cells are
    measured from the geodesic area of the source geometry intersected with
    each cell, so multipart geometries, holes, and partial cells retain their
    physical area. The denominator is measured once from the unchanged source
    geometry, including any uncovered and NoData portion.
    """
    if geometry.is_empty or not geometry.is_valid:
        raise ValueError("WorldCover geodesic coverage requires a valid non-empty polygon")
    polygon_area = _geodesic_area(geometry)
    if not math.isfinite(polygon_area) or polygon_area <= 0.0:
        raise ValueError("WorldCover source polygon has no positive WGS84 geodesic area")

    totals: defaultdict[int, float] = defaultdict(float)
    for path in raster_paths:
        with rasterio.open(path) as dataset:
            _validate_dataset(path, dataset)
            pixel_geometry = _pixel_geometry(geometry, dataset.transform)
            for pixel_chunk in _bounded_pixel_chunks(pixel_geometry, dataset.width, dataset.height):
                geometry_chunk = _map_geometry(pixel_chunk, dataset.transform)
                frame = gpd.GeoDataFrame(geometry=[geometry_chunk], crs=dataset.crs)
                result = exact_extract(
                    str(path),
                    frame,
                    list(_OPS),
                    max_cells_in_memory=_MAX_CELLS_IN_MEMORY,
                    output="pandas",
                )
                cell_ids = np.asarray(result["cell_id"].iloc[0], dtype=np.int64)
                coverage = np.asarray(result["coverage"].iloc[0], dtype=float)
                values = np.asarray(result["values"].iloc[0], dtype=float)
                boundary_ids = _boundary_cell_ids(pixel_chunk, dataset.width, dataset.height)
                boundary_mask = np.isin(cell_ids, boundary_ids, assume_unique=False)
                _add_interior_cells(
                    totals,
                    cell_ids[~boundary_mask],
                    coverage[~boundary_mask],
                    values[~boundary_mask],
                    dataset.transform,
                    dataset.width,
                )
                _add_boundary_cells(
                    totals,
                    boundary_ids,
                    geometry_chunk,
                    dataset,
                )

    class_areas = tuple(sorted((code, area) for code, area in totals.items() if area > 0.0))
    support_total = sum(area for _code, area in class_areas)
    if support_total > polygon_area * (1.0 + 1e-8):
        raise ValueError("WorldCover class support exceeds the full geodesic polygon area")
    return polygon_area, class_areas


def _validate_dataset(path: Path, dataset: rasterio.DatasetReader) -> None:
    if dataset.crs is None or not pyproj.CRS.from_user_input(dataset.crs).equals(_GEOGRAPHIC_CRS):
        raise ValueError(f"WorldCover pool raster must use EPSG:4326: {path}")
    transform = dataset.transform
    if transform.b != 0.0 or transform.d != 0.0 or transform.a <= 0.0 or transform.e >= 0.0:
        raise ValueError(f"WorldCover pool raster must be a north-up geographic grid: {path}")


def _geodesic_area(geometry: shapely.Geometry) -> float:
    return sum(
        abs(
            float(
                _GEOD.geometry_area_perimeter(shapely.orient_polygons(polygon, exterior_cw=False))[
                    0
                ]
            )
        )
        for polygon in _polygonal_parts(geometry)
    )


def _polygonal_parts(geometry: shapely.Geometry) -> Iterator[shapely.Geometry]:
    if geometry.geom_type == "Polygon":
        yield geometry
    elif geometry.geom_type in {"MultiPolygon", "GeometryCollection"}:
        for part in geometry.geoms:
            yield from _polygonal_parts(part)


def _add_interior_cells(
    totals: defaultdict[int, float],
    cell_ids: np.ndarray,
    coverage: np.ndarray,
    values: np.ndarray,
    transform: rasterio.Affine,
    width: int,
) -> None:
    valid = np.isfinite(coverage) & (coverage > 0.0) & np.isfinite(values) & (values != 0.0)
    if not np.any(valid):
        return
    rows = cell_ids[valid] // width
    classes = values[valid].astype(np.int64)
    fractions = coverage[valid]
    for row in np.unique(rows):
        selected = rows == row
        area = _cell_area(int(row), transform)
        for code in np.unique(classes[selected]):
            share = float(fractions[selected & (classes == code)].sum()) * area
            if share > 0.0:
                totals[int(code)] += share


def _cell_area(row: int, transform: rasterio.Affine) -> float:
    left = transform.c
    right = transform.c + transform.a
    top = transform.f + row * transform.e
    bottom = top + transform.e
    return _geodesic_area(box(left, bottom, right, top))


def _boundary_cell_ids(pixel_geometry: shapely.Geometry, width: int, height: int) -> np.ndarray:
    if pixel_geometry.is_empty:
        return np.array([], dtype=np.int64)
    min_x, min_y, max_x, max_y = pixel_geometry.bounds
    left = max(0, math.floor(min_x) - 1)
    right = min(width, math.ceil(max_x) + 1)
    top = max(0, math.floor(min_y) - 1)
    bottom = min(height, math.ceil(max_y) + 1)
    if left >= right or top >= bottom:
        return np.array([], dtype=np.int64)
    boundary = shapely.buffer(pixel_geometry.boundary, _BOUNDARY_HALO_PIXELS)
    ids: list[np.ndarray] = []
    for row in range(top, bottom, _BOUNDARY_ROW_CHUNK):
        strip_top = max(0, row - 1)
        strip_bottom = min(height, row + _BOUNDARY_ROW_CHUNK + 1)
        part = shapely.intersection(boundary, box(left, strip_top, right, strip_bottom))
        if part.is_empty:
            continue
        min_part_x, min_part_y, max_part_x, max_part_y = part.bounds
        part_left = max(left, math.floor(min_part_x) - 1)
        part_right = min(right, math.ceil(max_part_x) + 1)
        part_top = max(0, math.floor(min_part_y) - 1)
        part_bottom = min(height, math.ceil(max_part_y) + 1)
        mask = rasterize(
            [(part, 1)],
            out_shape=(part_bottom - part_top, part_right - part_left),
            transform=rasterio.Affine.translation(part_left, part_top),
            all_touched=True,
            dtype="uint8",
        )
        rows, columns = np.nonzero(mask)
        ids.append((rows + part_top) * width + columns + part_left)
    if not ids:
        return np.array([], dtype=np.int64)
    return np.unique(np.concatenate(ids).astype(np.int64, copy=False))


def _add_boundary_cells(
    totals: defaultdict[int, float],
    cell_ids: np.ndarray,
    geometry: shapely.Geometry,
    dataset: rasterio.DatasetReader,
) -> None:
    for start in range(0, len(cell_ids), _BOUNDARY_CELL_CHUNK):
        ids = cell_ids[start : start + _BOUNDARY_CELL_CHUNK]
        rows, columns, values, valid = _read_cell_values(ids, dataset)
        if not np.any(valid):
            continue
        left = dataset.transform.c + columns * dataset.transform.a
        right = left + dataset.transform.a
        top = dataset.transform.f + rows * dataset.transform.e
        bottom = top + dataset.transform.e
        cells = shapely.box(left, bottom, right, top)
        intersections = shapely.intersection(geometry, cells)
        for intersection, value in zip(intersections[valid], values[valid], strict=True):
            area = _geodesic_area(intersection) if not intersection.is_empty else 0.0
            if area > 0.0:
                totals[int(value)] += area


def _read_cell_values(
    cell_ids: np.ndarray, dataset: rasterio.DatasetReader
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = cell_ids // dataset.width
    columns = cell_ids % dataset.width
    values = np.zeros(len(cell_ids), dtype=np.int64)
    valid = np.zeros(len(cell_ids), dtype=bool)
    band_ids = rows // _READ_ROW_BAND
    for band in np.unique(band_ids):
        positions = np.flatnonzero(band_ids == band)
        band_rows = rows[positions]
        band_columns = columns[positions]
        top = int(band * _READ_ROW_BAND)
        left = int(band_columns.min())
        right = int(band_columns.max()) + 1
        bottom = min(top + _READ_ROW_BAND, dataset.height)
        data = dataset.read(1, window=((top, bottom), (left, right)), masked=True)
        selected = data[band_rows - top, band_columns - left]
        selected_valid = ~np.ma.getmaskarray(selected)
        selected_values = np.asarray(np.ma.getdata(selected), dtype=float)
        selected_valid &= np.isfinite(selected_values) & (selected_values != 0.0)
        values[positions[selected_valid]] = selected_values[selected_valid].astype(np.int64)
        valid[positions] = selected_valid
    return rows, columns, values, valid
