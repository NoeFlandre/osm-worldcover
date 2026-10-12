import math
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Final

import geopandas as gpd
import numpy as np
import pyproj
import rasterio
import shapely
from exactextract import exact_extract
from rasterio.features import rasterize
from shapely.geometry import box

from .geometry import (
    _bounded_pixel_chunks,
    _feature_batches,
    _map_geometry,
    _needs_spatial_chunks,
    _pixel_geometry,
)

#: exactextract operations needed to correct pixel-boundary coverage.
_OPS: Final[Sequence[str]] = ("cell_id", "coverage", "values")

#: A small pixel-space halo makes all-touched rasterization retain cells that
#: meet a polygon boundary at a single grid corner. GEOS measures the original
#: polygon against those cells; the halo only selects candidates for checking.
_BOUNDARY_HALO_PIXELS: Final[float] = 0.01
_BOUNDARY_ROW_CHUNK: Final[int] = 256
# Boundary cells omitted by exactextract are read in windows this many rows tall.
_MISSING_READ_ROWS: Final[int] = 64
_MAX_CELLS_IN_MEMORY: Final[int] = 2_000_000
_BOUNDARY_CELLS_PER_CHUNK: Final[int] = 16_384
_ACCUMULATION_CELL_CHUNK: Final[int] = 1_000_000


def class_coverage(raster_paths: Iterable[Path], frame: gpd.GeoDataFrame) -> list[dict[int, float]]:
    """Return, per row of ``frame``, each class's share of that row's area.

    Shares are relative to the polygon's own area, so they sum to at most one
    and fall short wherever the polygon was not observed. Contributions from
    multiple non-overlapping rasters are summed.
    """
    if len(frame) == 0:
        return []

    totals: list[defaultdict[int, float]] = [defaultdict(float) for _ in range(len(frame))]
    # Areas are measured in the raster's own planar units, for both the cells
    # and the polygons. Only their ratio is used, so it is exact even though
    # the units are degrees and a degree is not a constant ground distance.
    areas = shapely.area(frame.geometry.to_numpy())

    for path in raster_paths:
        _add_raster(totals, path, frame, areas)

    return [dict(sorted(total.items())) for total in totals]


def _add_raster(
    totals: list[defaultdict[int, float]],
    path: Path,
    frame: gpd.GeoDataFrame,
    areas: np.ndarray,
) -> None:
    """Add one raster's contribution to every polygon's running totals."""
    with rasterio.open(path) as dataset:
        _check_same_crs(frame.crs, dataset.crs, path)
        transform = dataset.transform
        cell_area = abs(transform.a * transform.e - transform.b * transform.d)
        for start, stop in _feature_batches(frame, transform, dataset.width, dataset.height):
            ordinary, spatial = _partition_feature_batch(
                frame, start, stop, transform, dataset.width, dataset.height
            )
            _add_ordinary_features(totals, path, frame, ordinary, dataset, cell_area, areas)
            for index, pixel_geometry in spatial:
                _add_spatial_feature(
                    totals,
                    path,
                    frame.crs,
                    index,
                    pixel_geometry,
                    transform,
                    dataset,
                    cell_area,
                    areas[index],
                )


def _check_same_crs(frame_crs, raster_crs, path: Path) -> None:
    """Reject polygons and raster in different CRSs; their cells would not line up."""
    if frame_crs is None or raster_crs is None:
        return
    if pyproj.CRS.from_user_input(frame_crs) != pyproj.CRS.from_user_input(raster_crs):
        raise ValueError(f"frame CRS {frame_crs} does not match CRS of raster {path}: {raster_crs}")


def _partition_feature_batch(
    frame: gpd.GeoDataFrame,
    start: int,
    stop: int,
    transform: rasterio.Affine,
    width: int,
    height: int,
) -> tuple[list[int], list[tuple[int, shapely.Geometry]]]:
    """Separate ordinary features from geometries requiring spatial chunks."""
    ordinary = []
    spatial = []
    for index in range(start, stop):
        pixel_geometry = _pixel_geometry(frame.geometry.iloc[index], transform)
        if _needs_spatial_chunks(pixel_geometry, width, height):
            spatial.append((index, pixel_geometry))
        else:
            ordinary.append(index)
    return ordinary, spatial


def _add_ordinary_features(
    totals: list[defaultdict[int, float]],
    path: Path,
    frame: gpd.GeoDataFrame,
    indices: list[int],
    dataset: rasterio.DatasetReader,
    cell_area: float,
    areas: np.ndarray,
) -> None:
    """Extract one bounded feature batch and accumulate its class coverage."""
    if indices:
        _extract_and_accumulate(
            totals,
            path,
            frame.iloc[indices],
            indices,
            list(frame.geometry.iloc[indices]),
            dataset,
            cell_area,
            [float(areas[index]) for index in indices],
        )


def _add_spatial_feature(
    totals: list[defaultdict[int, float]],
    path: Path,
    crs,
    index: int,
    pixel_geometry: shapely.Geometry,
    transform: rasterio.Affine,
    dataset: rasterio.DatasetReader,
    cell_area: float,
    polygon_area: float,
) -> None:
    """Extract bounded spatial pieces while retaining the original area denominator."""
    for pixel_chunk in _bounded_pixel_chunks(pixel_geometry, dataset.width, dataset.height):
        geometry_chunk = _map_geometry(pixel_chunk, transform)
        chunk_frame = gpd.GeoDataFrame(geometry=[geometry_chunk], crs=crs)
        _extract_and_accumulate(
            totals,
            path,
            chunk_frame,
            [index],
            [geometry_chunk],
            dataset,
            cell_area,
            [polygon_area],
        )


def _extract_and_accumulate(
    totals: list[defaultdict[int, float]],
    path: Path,
    frame: gpd.GeoDataFrame,
    indices: list[int],
    geometries: list[shapely.Geometry],
    dataset: rasterio.DatasetReader,
    cell_area: float,
    polygon_areas: Sequence[float],
) -> None:
    """Keep exactextract output batches small and accumulate each feature in order."""
    result = exact_extract(
        str(path),
        frame,
        list(_OPS),
        max_cells_in_memory=_MAX_CELLS_IN_MEMORY,
        output="pandas",
    )
    for index, geometry, polygon_area, cell_ids, coverage, values in zip(
        indices,
        geometries,
        polygon_areas,
        result["cell_id"],
        result["coverage"],
        result["values"],
        strict=True,
    ):
        _accumulate_corrected(
            totals[index],
            cell_ids,
            coverage,
            values,
            geometry,
            dataset,
            cell_area,
            polygon_area,
        )


def _accumulate_corrected(
    into: defaultdict[int, float],
    cell_ids: Iterable[int],
    coverage: Iterable[float],
    values: Iterable[float],
    geometry: shapely.Geometry,
    dataset: rasterio.DatasetReader,
    cell_area: float,
    polygon_area: float,
) -> None:
    """Replace exactextract's boundary-cell values with independent GEOS areas."""
    if polygon_area <= 0.0 or not math.isfinite(polygon_area):
        return
    ids = np.asarray(cell_ids, dtype=np.int64)
    fractions = np.asarray(coverage, dtype=float)
    # exactextract omits NoData cells by default; this call never sets default_value.
    classes = np.asarray(values)
    pixel_geometry = _pixel_geometry(geometry, dataset.transform)
    scale = cell_area / polygon_area
    excluded = _accumulate_boundary_cells(into, ids, classes, pixel_geometry, dataset, scale)
    _accumulate_stable_cells(into, classes, fractions, excluded, scale)


def _accumulate_boundary_cells(
    into: defaultdict[int, float],
    ids: np.ndarray,
    classes: np.ndarray,
    pixel_geometry: shapely.Geometry,
    dataset: rasterio.DatasetReader,
    scale: float,
) -> np.ndarray:
    """Replace candidate boundary pixels and return ids excluded from stable sums."""
    excluded = np.zeros(len(ids), dtype=bool)
    for candidate_ids, rows, columns, corrected in _boundary_cell_chunks(
        pixel_geometry, dataset.width, dataset.height
    ):
        present, positions = _find_candidates(ids, candidate_ids)
        excluded[positions[present]] = True
        candidate_classes = _matched_candidate_classes(candidate_ids, classes, present, positions)
        missing = ~present & (corrected > 0.0)
        _read_missing_classes(
            dataset,
            rows[missing],
            columns[missing],
            candidate_classes,
            np.flatnonzero(missing),
        )
        _add_class_coverage(into, candidate_classes, corrected, scale)
    return excluded


def _matched_candidate_classes(
    candidate_ids: np.ndarray,
    classes: np.ndarray,
    present: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    """Map exactextract values onto candidate cells, leaving unmatched cells empty."""
    candidate_classes = np.full(len(candidate_ids), np.nan)
    matched_candidates = np.flatnonzero(present)
    matched_positions = positions[matched_candidates]
    candidate_classes[matched_candidates] = classes[matched_positions]
    return candidate_classes


def _boundary_cell_chunks(
    geometry: shapely.Geometry, width: int, height: int
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield boundary-cell coverage in bounded candidate arrays."""
    if geometry.is_empty:
        return
    min_x, min_y, max_x, max_y = geometry.bounds
    col_start = max(0, math.floor(min_x) - 1)
    col_stop = min(width, math.ceil(max_x) + 1)
    row_start = max(0, math.floor(min_y) - 1)
    row_stop = min(height, math.ceil(max_y) + 1)
    if col_start >= col_stop or row_start >= row_stop:
        return
    boundary = shapely.buffer(geometry.boundary, _BOUNDARY_HALO_PIXELS)
    candidate_ids = _rasterized_boundary_ids(
        boundary, col_start, col_stop, row_start, row_stop, width, height
    )
    for start in range(0, len(candidate_ids), _BOUNDARY_CELLS_PER_CHUNK):
        cell_ids = candidate_ids[start : start + _BOUNDARY_CELLS_PER_CHUNK]
        rows, columns = cell_ids // width, cell_ids % width
        cells = shapely.box(columns, rows, columns + 1, rows + 1)
        corrected = np.asarray(shapely.area(shapely.intersection(geometry, cells)), dtype=float)
        yield cell_ids, rows, columns, corrected


def _rasterized_boundary_ids(
    boundary: shapely.Geometry,
    col_start: int,
    col_stop: int,
    row_start: int,
    row_stop: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Rasterize boundary strips with bounded memory, then return unique IDs."""
    ids: list[np.ndarray] = []
    for row in range(row_start, row_stop, _BOUNDARY_ROW_CHUNK):
        strip_top = max(0, row - 1)
        strip_bottom = min(height, row + _BOUNDARY_ROW_CHUNK + 1)
        part = shapely.intersection(boundary, box(col_start, strip_top, col_stop, strip_bottom))
        if part.is_empty:
            continue
        min_x, min_y, max_x, max_y = part.bounds
        left = max(col_start, math.floor(min_x) - 1)
        right = min(col_stop, math.ceil(max_x) + 1)
        top = max(0, math.floor(min_y) - 1)
        bottom = min(height, math.ceil(max_y) + 1)
        mask = rasterize(
            [(part, 1)],
            out_shape=(bottom - top, right - left),
            transform=rasterio.Affine.translation(left, top),
            all_touched=True,
            dtype="uint8",
        )
        rows, columns = np.nonzero(mask)
        ids.append((rows + top) * width + columns + left)
    if not ids:
        return np.array([], dtype=np.int64)
    return np.unique(np.concatenate(ids).astype(np.int64, copy=False))


def _find_candidates(cell_ids: np.ndarray, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find result positions without reordering the potentially huge cell array."""
    if not len(cell_ids) or not len(candidates):
        return np.zeros(len(candidates), dtype=bool), np.zeros(len(candidates), dtype=np.int64)
    found = np.zeros(len(candidates), dtype=bool)
    positions = np.zeros(len(candidates), dtype=np.int64)
    if _is_sorted(cell_ids):
        positions = np.searchsorted(cell_ids, candidates)
        in_range = positions < len(cell_ids)
        found[in_range] = cell_ids[positions[in_range]] == candidates[in_range]
        return found, positions
    for start in range(0, len(cell_ids), _ACCUMULATION_CELL_CHUNK):
        stop = min(start + _ACCUMULATION_CELL_CHUNK, len(cell_ids))
        chunk = cell_ids[start:stop]
        offsets = np.searchsorted(candidates, chunk)
        in_range = offsets < len(candidates)
        matched = np.zeros(len(chunk), dtype=bool)
        matched[in_range] = candidates[offsets[in_range]] == chunk[in_range]
        found[offsets[matched]] = True
        positions[offsets[matched]] = start + np.flatnonzero(matched)
    return found, positions


def _is_sorted(values: np.ndarray) -> bool:
    """Check monotonic order in bounded slices, avoiding a full-size mask."""
    previous: int | None = None
    for start in range(0, len(values), _ACCUMULATION_CELL_CHUNK):
        chunk = values[start : start + _ACCUMULATION_CELL_CHUNK]
        if not _chunk_follows(previous, chunk):
            return False
        previous = int(chunk[-1])
    return True


def _chunk_follows(previous: int | None, chunk: np.ndarray) -> bool:
    """Check sorted order within a chunk and across its leading boundary."""
    starts_after_previous = previous is None or previous <= chunk[0]
    return starts_after_previous and not np.any(chunk[1:] < chunk[:-1])


def _accumulate_stable_cells(
    into: defaultdict[int, float],
    classes: np.ndarray,
    fractions: np.ndarray,
    excluded: np.ndarray,
    scale: float,
) -> None:
    """Accumulate non-boundary cells in bounded slices of exactextract output."""
    for start in range(0, len(classes), _ACCUMULATION_CELL_CHUNK):
        stop = min(start + _ACCUMULATION_CELL_CHUNK, len(classes))
        valid = ~excluded[start:stop]
        valid &= np.isfinite(classes[start:stop]) & np.isfinite(fractions[start:stop])
        valid &= fractions[start:stop] > 0.0
        _add_class_coverage(into, classes[start:stop][valid], fractions[start:stop][valid], scale)


def _read_missing_classes(
    dataset: rasterio.DatasetReader,
    rows: np.ndarray,
    columns: np.ndarray,
    classes: np.ndarray,
    positions: np.ndarray,
) -> None:
    """Read valid pixel classes for boundary cells omitted by exactextract.

    Cells are read one window per band of ``_MISSING_READ_ROWS`` rows, not one
    GDAL read per cell.
    """
    if not len(rows) == len(columns) == len(positions):
        raise ValueError("rows, columns and positions must have the same length")
    if len(rows) == 0:
        return
    rows, columns, positions = (np.asarray(a, dtype=np.intp) for a in (rows, columns, positions))
    order = np.argsort(rows, kind="stable")
    rows, columns, positions = rows[order], columns[order], positions[order]
    bands = rows // _MISSING_READ_ROWS
    starts = np.flatnonzero(np.r_[True, bands[1:] != bands[:-1]])
    for start, stop in zip(starts, np.r_[starts[1:], len(rows)], strict=True):
        _read_band_classes(
            dataset,
            rows[start:stop],
            columns[start:stop],
            classes,
            positions[start:stop],
        )


def _read_band_classes(
    dataset: rasterio.DatasetReader,
    rows: np.ndarray,
    columns: np.ndarray,
    classes: np.ndarray,
    positions: np.ndarray,
) -> None:
    """Read one window covering a band's cells and fill the unmasked ones."""
    top, left = int(rows.min()), int(columns.min())
    window = dataset.read(
        1,
        window=((top, int(rows.max()) + 1), (left, int(columns.max()) + 1)),
        masked=True,
    )
    values = window[rows - top, columns - left]
    valid = ~np.ma.getmaskarray(values)
    classes[positions[valid]] = values.data[valid].astype(float)


def _add_class_coverage(
    into: defaultdict[int, float],
    values: np.ndarray,
    coverage: np.ndarray,
    scale: float,
) -> None:
    """Add valid per-cell areas to class totals."""
    valid = np.isfinite(values) & np.isfinite(coverage) & (coverage > 0.0)
    for value in np.unique(values[valid]):
        share = float(coverage[valid & (values == value)].sum()) * scale
        if share > 0.0:
            into[int(value)] += share
