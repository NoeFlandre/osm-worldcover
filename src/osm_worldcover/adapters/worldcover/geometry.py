import math
from collections.abc import Iterator
from typing import Final

import geopandas as gpd
import numpy as np
import rasterio
import shapely
from shapely.affinity import affine_transform
from shapely.geometry import box

# exactextract returns one cell-level result array per feature. Bound each
# batch by both feature count and polygon area so one busy tile cannot make a
# whole region's result arrays resident at once. A 2,000 km² batch covers at
# most about 20 million 10 m cells; larger polygons are processed alone.
_MAX_FEATURES_PER_BATCH: Final[int] = 128
_MAX_BATCH_AREA_M2: Final[float] = 2_000_000_000.0
_MAX_BATCH_PIXEL_BBOX_CELLS: Final[int] = 2_000_000
_MAX_GEOMETRY_COORDINATES: Final[int] = 25_000
_MAX_GEOMETRY_SPLIT_DEPTH: Final[int] = 32


def _feature_batches(
    frame: gpd.GeoDataFrame,
    transform: rasterio.Affine | None = None,
    width: int | None = None,
    height: int | None = None,
) -> Iterator[tuple[int, int]]:
    """Yield feature ranges bounded by count, source area, and raster envelope."""
    areas = _feature_areas(frame)
    pixel_cells = _feature_pixel_cells(frame, transform, width, height)
    yield from _yield_feature_batches(areas, pixel_cells)


def _feature_pixel_cells(
    frame: gpd.GeoDataFrame,
    transform: rasterio.Affine | None,
    width: int | None,
    height: int | None,
) -> np.ndarray:
    """Return each feature's raster-envelope cell bound, or zero without a raster."""
    if transform is None or width is None or height is None:
        return np.zeros(len(frame), dtype=np.int64)
    return np.fromiter(
        (
            _pixel_bbox_cells(_pixel_geometry(geometry, transform), width, height)
            for geometry in frame.geometry
        ),
        dtype=np.int64,
        count=len(frame),
    )


def _yield_feature_batches(areas: np.ndarray, pixel_cells: np.ndarray) -> Iterator[tuple[int, int]]:
    """Yield consecutive ranges within both per-batch budgets."""
    start = 0
    batch_area = 0.0
    batch_pixel_cells = 0
    for index, (value, cells) in enumerate(zip(areas, pixel_cells, strict=True)):
        area = float(value)
        feature_count = index - start
        if feature_count and _batch_exceeds_limits(
            feature_count, batch_area, area, batch_pixel_cells, int(cells)
        ):
            yield start, index
            start = index
            batch_area = 0.0
            batch_pixel_cells = 0
        batch_area += area
        batch_pixel_cells += int(cells)
    if start < len(areas):
        yield start, len(areas)


def _pixel_bbox_cells(geometry: shapely.Geometry, width: int, height: int) -> int:
    """Bound all candidate cells in a geometry's raster-space envelope."""
    if geometry.is_empty:
        return 0
    min_x, min_y, max_x, max_y = geometry.bounds
    left = max(0, math.floor(min_x) - 1)
    right = min(width, math.ceil(max_x) + 1)
    top = max(0, math.floor(min_y) - 1)
    bottom = min(height, math.ceil(max_y) + 1)
    return max(0, right - left) * max(0, bottom - top)


def _needs_spatial_chunks(geometry: shapely.Geometry, width: int, height: int) -> bool:
    """Whether one feature can exceed the per-result cell or coordinate budget."""
    return (
        _pixel_bbox_cells(geometry, width, height) > _MAX_BATCH_PIXEL_BBOX_CELLS
        or shapely.get_num_coordinates(geometry) > _MAX_GEOMETRY_COORDINATES
    )


def _bounded_pixel_chunks(
    geometry: shapely.Geometry, width: int, height: int
) -> Iterator[shapely.Geometry]:
    """Yield raster-clipped pieces with bounded raster envelopes and coordinates."""
    clipped = shapely.intersection(geometry, box(0, 0, width, height))
    if not clipped.is_empty:
        yield from _split_pixel_geometry(clipped, width, height, depth=0)


_AREAL_TYPES = ("Polygon", "MultiPolygon")


def _areal_part(geometry: shapely.Geometry) -> shapely.Geometry:
    """Keep only the polygonal parts of a clipped geometry.

    Clipping can return a GeometryCollection that also holds zero-area lines or
    points, which exactextract rejects as mixed-type. Those parts add no area, so
    dropping them changes no coverage.
    """
    if geometry.geom_type in _AREAL_TYPES:
        return geometry
    areal = [part for part in shapely.get_parts(geometry) if part.geom_type in _AREAL_TYPES]
    return shapely.union_all(areal) if areal else shapely.Polygon()


def _split_pixel_geometry(
    geometry: shapely.Geometry, width: int, height: int, depth: int
) -> Iterator[shapely.Geometry]:
    """Recursively split a large feature on pixel-aligned lines."""
    geometry = _areal_part(geometry)
    if geometry.is_empty:
        return
    if _geometry_is_bounded(geometry, width, height):
        yield geometry
        return
    if depth >= _MAX_GEOMETRY_SPLIT_DEPTH:
        raise ValueError(
            "could not bound a WorldCover geometry after "
            f"{_MAX_GEOMETRY_SPLIT_DEPTH} spatial splits"
        )
    first, second = _split_pixel_geometry_once(geometry)
    if _split_is_stalled(geometry, first, second):
        raise ValueError("WorldCover geometry split did not reduce its spatial extent")
    yield from _nonempty_pixel_children(first, second, width, height, depth + 1)


def _geometry_is_bounded(geometry: shapely.Geometry, width: int, height: int) -> bool:
    """Return whether one geometry fits both spatial memory budgets."""
    pixel_cells = _pixel_bbox_cells(geometry, width, height)
    coordinates = shapely.get_num_coordinates(geometry)
    return pixel_cells <= _MAX_BATCH_PIXEL_BBOX_CELLS and coordinates <= _MAX_GEOMETRY_COORDINATES


def _split_pixel_geometry_once(
    geometry: shapely.Geometry,
) -> tuple[shapely.Geometry, shapely.Geometry]:
    """Bisect a geometry along its longest pixel-space axis."""
    min_x, min_y, max_x, max_y = geometry.bounds
    x_midpoint = _pixel_aligned_midpoint(min_x, max_x)
    y_midpoint = _pixel_aligned_midpoint(min_y, max_y)
    split_x = _choose_split_axis(max_x - min_x, max_y - min_y, x_midpoint, y_midpoint)
    if split_x:
        middle = x_midpoint if x_midpoint is not None else (min_x + max_x) / 2.0
        first_box = box(min_x - 1.0, min_y - 1.0, middle, max_y + 1.0)
        second_box = box(middle, min_y - 1.0, max_x + 1.0, max_y + 1.0)
    else:
        middle = y_midpoint if y_midpoint is not None else (min_y + max_y) / 2.0
        first_box = box(min_x - 1.0, min_y - 1.0, max_x + 1.0, middle)
        second_box = box(min_x - 1.0, middle, max_x + 1.0, max_y + 1.0)
    first = shapely.intersection(geometry, first_box)
    second = shapely.intersection(geometry, second_box)
    return first, second


def _pixel_aligned_midpoint(minimum: float, maximum: float) -> float | None:
    """Return an interior integer pixel boundary near the midpoint, if one exists."""
    midpoint = (minimum + maximum) / 2.0
    for candidate in (math.floor(midpoint), math.ceil(midpoint)):
        if minimum < candidate < maximum:
            return float(candidate)
    return None


def _choose_split_axis(
    x_span: float,
    y_span: float,
    x_midpoint: float | None,
    y_midpoint: float | None,
) -> bool:
    """Prefer the longest axis with an integer pixel split; fall back if needed."""
    if x_midpoint is None and y_midpoint is None:
        return x_span >= y_span
    if x_midpoint is None:
        return False
    if y_midpoint is None:
        return True
    return x_span >= y_span


def _split_is_stalled(
    geometry: shapely.Geometry,
    first: shapely.Geometry,
    second: shapely.Geometry,
) -> bool:
    """Detect a split that fails to create any smaller geometry."""
    return (first.is_empty and second.is_empty) or (
        first.equals(geometry) and second.equals(geometry)
    )


def _nonempty_pixel_children(
    first: shapely.Geometry,
    second: shapely.Geometry,
    width: int,
    height: int,
    depth: int,
) -> Iterator[shapely.Geometry]:
    """Yield bounded chunks from the nonempty halves of one spatial split."""
    for piece in (first, second):
        if not piece.is_empty:
            yield from _split_pixel_geometry(piece, width, height, depth)


def _map_geometry(geometry: shapely.Geometry, transform: rasterio.Affine) -> shapely.Geometry:
    """Return a pixel-space geometry to the raster's map coordinates."""
    return affine_transform(
        geometry,
        [transform.a, transform.b, transform.d, transform.e, transform.c, transform.f],
    )


def _feature_areas(frame: gpd.GeoDataFrame) -> np.ndarray:
    """Return source areas, or zeros when only the feature-count limit applies."""
    if "area_m2" in frame:
        return frame["area_m2"].to_numpy(dtype=float, copy=False)
    return np.zeros(len(frame), dtype=float)


def _batch_exceeds_limits(
    feature_count: int,
    batch_area: float,
    next_area: float,
    batch_pixel_cells: int = 0,
    next_pixel_cells: int = 0,
) -> bool:
    """Whether adding another feature would exceed either extraction bound."""
    return (
        feature_count >= _MAX_FEATURES_PER_BATCH
        or batch_area + next_area > _MAX_BATCH_AREA_M2
        or batch_pixel_cells + next_pixel_cells > _MAX_BATCH_PIXEL_BBOX_CELLS
    )


def _pixel_geometry(geometry: shapely.Geometry, transform: rasterio.Affine) -> shapely.Geometry:
    """Express a raster-CRS geometry in pixel coordinates for stable overlay."""
    inverse = ~transform
    return affine_transform(
        geometry,
        [inverse.a, inverse.b, inverse.d, inverse.e, inverse.c, inverse.f],
    )
