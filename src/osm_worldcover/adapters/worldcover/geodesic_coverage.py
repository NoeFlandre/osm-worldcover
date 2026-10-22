"""WGS84 equal-area evidence for source-wide WorldCover attempts.

The older WorldCover release path measures pixel support and polygon area in
EPSG:4326 square degrees. This module is a separate pool adapter: it measures
the full source polygon, raster cells, and clipped supports in the same WGS84
ellipsoidal cylindrical equal-area projection. Geographic edges are densified
before projection. NoData cells and unavailable tiles contribute no numerator
while the complete source polygon remains the denominator.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
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
from osm_worldcover.domain.nomenclature import CLASS_LABELS

AREA_MEASUREMENT = (
    "WGS84 ellipsoidal CEA (lat_ts=30, lon_0=0, EPSG:4326 input); "
    "straight lon/lat source edges; canonical full-polygon boundary reused for clipping; "
    "CEA curve-to-chord boundary area envelope controls the ratio decision; invalid coarse "
    "projections refine or fail closed; GEOS overlay and floating-point accumulation error "
    "are excluded; unresolved enclosures fail closed; v4"
)
AREA_UNIT = "m^2"
_OPS = ("cell_id", "coverage", "values")
_MAX_CELLS_IN_MEMORY = 2_000_000
_BOUNDARY_HALO_PIXELS = 0.01
_BOUNDARY_ROW_CHUNK = 256
_BOUNDARY_CELL_CHUNK = 16_384
_READ_ROW_BAND = 64
_GEOGRAPHIC_CRS = pyproj.CRS.from_epsg(4326)
_AREA_CRS = pyproj.CRS.from_proj4(
    "+proj=cea +lat_ts=30 +lon_0=0 +datum=WGS84 +units=m +no_defs +type=crs"
)
_AREA_TRANSFORMER = pyproj.Transformer.from_crs(_GEOGRAPHIC_CRS, _AREA_CRS, always_xy=True)
_AREA_DENSIFY_DEGREES = 0.01


class WorldCoverAreaMeasurementError(ValueError):
    """A geometry or threshold could not be measured within declared limits."""


_AREA_REFINEMENT_STEPS = (0.001, 0.0001)
_MAX_DENSIFIED_COORDINATES = 1_000_000
_MAX_BOUNDARY_CELL_IDS = 2_000_000
_WGS84_SEMI_MAJOR_METRES = 6_378_137.0
_WGS84_ECCENTRICITY_SQUARED = 0.0066943799901413165


@dataclass(frozen=True, slots=True)
class _AreaMeasurement:
    densify_step_degrees: float
    polygon_area_m2: float
    class_areas_m2: tuple[tuple[int, float], ...]
    class_evidence: tuple[tuple[int, tuple[str, ...]], ...]
    ratios: dict[int, float]
    projected_chord_boundary_error_bound_m2: float
    class_support_error_bound_m2: float


def equal_area_class_coverage(
    raster_paths: Iterable[Path], geometry: shapely.Geometry
) -> tuple[float, tuple[tuple[int, float], ...]]:
    """Return full-polygon and per-class WGS84 equal-area support in square metres.

    Raster tiles must be north-up EPSG:4326 grids. Interior pixel support is
    weighted by the projected area of each pixel row. Boundary cells are
    clipped and measured in the same equal-area plane as the full source
    geometry, so multipart geometries, holes, and partial cells retain their
    physical area. The denominator is measured once from the unchanged source
    geometry, including any uncovered and NoData portion.
    """
    polygon_area, class_areas, _class_evidence = _equal_area_class_coverage(
        ((None, path) for path in raster_paths), geometry
    )
    return polygon_area, class_areas


def equal_area_class_coverage_with_tile_evidence(
    tile_paths: Iterable[tuple[str, Path]], geometry: shapely.Geometry
) -> tuple[float, tuple[tuple[int, float], ...], tuple[tuple[int, tuple[str, ...]], ...]]:
    """Measure class areas and identify the tiles that contribute to each class."""
    return _equal_area_class_coverage(tile_paths, geometry)


def equal_area_class_coverage_with_tile_evidence_detailed(
    tile_paths: Iterable[tuple[str, Path]], geometry: shapely.Geometry
) -> tuple[
    float,
    tuple[tuple[int, float], ...],
    tuple[tuple[int, tuple[str, ...]], ...],
    str | None,
]:
    """Measure support and return diagnostics for the CEA boundary error bound."""
    return _equal_area_class_coverage_detailed(tile_paths, geometry)


def geodesic_class_coverage(
    raster_paths: Iterable[Path], geometry: shapely.Geometry
) -> tuple[float, tuple[tuple[int, float], ...]]:
    """Compatibility alias; this adapter now measures in an equal-area projection."""
    return equal_area_class_coverage(raster_paths, geometry)


def geodesic_class_coverage_with_tile_evidence(
    tile_paths: Iterable[tuple[str, Path]], geometry: shapely.Geometry
) -> tuple[float, tuple[tuple[int, float], ...], tuple[tuple[int, tuple[str, ...]], ...]]:
    """Compatibility alias for the earlier adapter name."""
    return equal_area_class_coverage_with_tile_evidence(tile_paths, geometry)


def _equal_area_class_coverage(
    raster_sources: Iterable[tuple[str | None, Path]], geometry: shapely.Geometry
) -> tuple[float, tuple[tuple[int, float], ...], tuple[tuple[int, tuple[str, ...]], ...]]:
    area, supports, evidence, _diagnostics = _equal_area_class_coverage_detailed(
        raster_sources, geometry
    )
    return area, supports, evidence


def _equal_area_class_coverage_detailed(
    raster_sources: Iterable[tuple[str | None, Path]], geometry: shapely.Geometry
) -> tuple[
    float,
    tuple[tuple[int, float], ...],
    tuple[tuple[int, tuple[str, ...]], ...],
    str | None,
]:
    _validate_source_geometry(geometry)
    sources = tuple(raster_sources)
    measurements = _refinement_measurements(sources, geometry)
    selected = measurements[-1]
    diagnostics = _format_refinement_diagnostics(measurements)
    _validate_support_denominator(selected.polygon_area_m2, selected.class_areas_m2)
    return (
        selected.polygon_area_m2,
        selected.class_areas_m2,
        selected.class_evidence,
        diagnostics,
    )


def _refinement_measurements(
    sources: tuple[tuple[str | None, Path], ...],
    geometry: shapely.Geometry,
) -> list[_AreaMeasurement]:
    measurements = _collect_area_measurements(sources, geometry)
    _validate_area_measurements(measurements)
    return measurements


def _collect_area_measurements(
    sources: tuple[tuple[str | None, Path], ...],
    geometry: shapely.Geometry,
) -> list[_AreaMeasurement]:
    measurements = []
    for step in (_AREA_DENSIFY_DEGREES, *_AREA_REFINEMENT_STEPS):
        if measurements and not _needs_area_refinement(tuple(measurements)):
            break
        measurement = _measure_valid_projected_step(sources, geometry, step)
        if measurement is not None:
            measurements.append(measurement)
    return measurements


def _validate_area_measurements(measurements: list[_AreaMeasurement]) -> None:
    if not measurements:
        raise WorldCoverAreaMeasurementError("worldcover_projected_boundary_invalid_at_all_steps")
    if _needs_area_refinement(tuple(measurements)):
        _raise_unresolved_threshold_error(tuple(measurements))


def _measure_valid_projected_step(
    sources: tuple[tuple[str | None, Path], ...], geometry: shapely.Geometry, step: float
) -> _AreaMeasurement | None:
    try:
        return _measure_at_step(sources, geometry, step)
    except WorldCoverAreaMeasurementError as error:
        if str(error) != "worldcover_projected_boundary_invalid":
            raise
        return None


def _raise_unresolved_threshold_error(measurements: tuple[_AreaMeasurement, ...]) -> None:
    measurement = measurements[-1]
    code = min(_unresolved_threshold_codes(measurements))
    lower, upper = _class_ratio_enclosure(measurement, code)
    raise WorldCoverAreaMeasurementError(
        "coverage_accuracy_bound_unresolved_at_0.8:"
        f"class={code};step={measurement.densify_step_degrees:g};"
        f"ratio_enclosure={lower:.12g}:{upper:.12g};"
        "projected_chord_boundary_error_bound_m2="
        f"{measurement.projected_chord_boundary_error_bound_m2:.12g};"
        "class_support_error_bound_m2="
        f"{measurement.class_support_error_bound_m2:.12g}"
    )


def _unresolved_threshold_codes(measurements: tuple[_AreaMeasurement, ...]) -> tuple[int, ...]:
    codes = set(CLASS_LABELS)
    codes.update(code for measurement in measurements for code in measurement.ratios)
    latest = measurements[-1]
    return tuple(
        code
        for code in sorted(codes)
        if _ratio_enclosure_crosses_threshold(*_class_ratio_enclosure(latest, code))
    )


def _needs_area_refinement(measurements: tuple[_AreaMeasurement, ...]) -> bool:
    return bool(_unresolved_threshold_codes(measurements))


def _ratio_enclosure_crosses_threshold(lower: float, upper: float) -> bool:
    return lower < 0.8 <= upper


def _class_ratio_enclosure(measurement: _AreaMeasurement, code: int) -> tuple[float, float]:
    area = measurement.polygon_area_m2
    area_error = measurement.projected_chord_boundary_error_bound_m2
    support_error = measurement.class_support_error_bound_m2
    if area <= area_error:
        return -math.inf, math.inf
    support = dict(measurement.class_areas_m2).get(code, 0.0)
    return (
        (support - support_error) / (area + area_error),
        (support + support_error) / (area - area_error),
    )


def _format_refinement_diagnostics(
    measurements: list[_AreaMeasurement],
) -> str:
    class_codes = sorted(set().union(*(measurement.ratios.keys() for measurement in measurements)))
    step_values = ",".join(f"{measurement.densify_step_degrees:g}" for measurement in measurements)
    classes = tuple(_format_class_ratio_bounds(code, measurements) for code in class_codes)
    return (
        "cea_chord_area_envelope;steps="
        f"{step_values};projected_chord_boundary_error_bound_m2="
        f"{measurements[-1].projected_chord_boundary_error_bound_m2:.12g};"
        f"class_support_error_m2={measurements[-1].class_support_error_bound_m2:.12g};"
        + ";".join(classes)
    )


def _format_class_ratio_bounds(
    code: int,
    measurements: list[_AreaMeasurement],
) -> str:
    ratios = tuple(measurement.ratios.get(code, 0.0) for measurement in measurements)
    minimum, maximum = min(ratios), max(ratios)
    lower, upper = _class_ratio_enclosure(measurements[-1], code)
    return (
        f"class={code},observed_range={minimum:.12g}:{maximum:.12g},"
        f"ratio_enclosure={lower:.12g}:{upper:.12g}"
    )


def _measure_at_step(
    raster_sources: tuple[tuple[str | None, Path], ...],
    geometry: shapely.Geometry,
    densify_degrees: float,
) -> _AreaMeasurement:
    area_geometry = _equal_area_geometry(geometry, densify_degrees=densify_degrees)
    if not area_geometry.is_valid:
        raise WorldCoverAreaMeasurementError("worldcover_projected_boundary_invalid")
    polygon_area = float(shapely.area(area_geometry))
    _validate_polygon_area(polygon_area)
    class_areas, class_evidence = _aggregate_tile_support(
        raster_sources,
        geometry,
        area_geometry=area_geometry,
    )
    _validate_support_denominator(polygon_area, class_areas)
    boundary_error = _projected_chord_area_error_bound(geometry, densify_degrees)
    return _AreaMeasurement(
        densify_step_degrees=densify_degrees,
        polygon_area_m2=polygon_area,
        class_areas_m2=class_areas,
        class_evidence=class_evidence,
        ratios=_support_ratios(polygon_area, class_areas),
        projected_chord_boundary_error_bound_m2=boundary_error,
        class_support_error_bound_m2=(
            math.nextafter(boundary_error * len(raster_sources), math.inf)
            if boundary_error
            else 0.0
        ),
    )


def _projected_chord_area_error_bound(geometry: shapely.Geometry, densify_degrees: float) -> float:
    """Bound projected area between straight lon/lat edges and their CEA chords.

    In ellipsoidal CEA, x is linear in longitude and y is a function q(latitude).
    For each densified subsegment, the interpolation error of y is bounded by
    max(abs(q'')) * dlat**2 / 8; integrating the sharper t*(1-t)/2 envelope
    gives the area term below. The global WGS84 bound is 2/(1-e**2). This bounds
    only curve-to-chord area; it is not a certified bound on GEOS overlay or
    floating-point accumulation error.
    """
    q_second_derivative_bound = math.nextafter(2.0 / (1.0 - _WGS84_ECCENTRICITY_SQUARED), math.inf)
    area_factor = math.nextafter(
        _WGS84_SEMI_MAJOR_METRES**2 * q_second_derivative_bound / 24.0,
        math.inf,
    )
    error_bound = 0.0
    for polygon in shapely.get_parts(geometry):
        for ring in (polygon.exterior, *polygon.interiors):
            coordinates = shapely.get_coordinates(ring)
            for start, end in pairwise(coordinates):
                segment_bound = _projected_segment_chord_area_error_bound(
                    start, end, densify_degrees, area_factor
                )
                if segment_bound:
                    error_bound = math.nextafter(error_bound + segment_bound, math.inf)
    return error_bound


def _projected_segment_chord_area_error_bound(
    start: np.ndarray,
    end: np.ndarray,
    densify_degrees: float,
    area_factor: float,
) -> float:
    delta_lon_degrees = float(end[0] - start[0])
    delta_lat_degrees = float(end[1] - start[1])
    if delta_lon_degrees == 0.0 or delta_lat_degrees == 0.0:
        return 0.0
    segment_count = max(
        1,
        math.ceil(math.hypot(delta_lon_degrees, delta_lat_degrees) / densify_degrees),
    )
    delta_lon = math.radians(abs(delta_lon_degrees))
    delta_lat = math.radians(delta_lat_degrees)
    segment_bound = (
        area_factor * delta_lon * delta_lat * delta_lat / (segment_count * segment_count)
    )
    return math.nextafter(segment_bound * (1.0 + 32.0 * math.ulp(1.0)), math.inf)


def _support_ratios(
    polygon_area: float, class_areas: tuple[tuple[int, float], ...]
) -> dict[int, float]:
    return {code: area / polygon_area for code, area in class_areas}


def _validate_source_geometry(geometry: shapely.Geometry) -> None:
    if (
        geometry.is_empty
        or not geometry.is_valid
        or geometry.geom_type not in {"Polygon", "MultiPolygon"}
    ):
        raise WorldCoverAreaMeasurementError("worldcover_polygon_invalid_or_unsupported_type")
    for polygon in shapely.get_parts(geometry):
        _validate_polygon_global_domain(polygon)


def _validate_polygon_global_domain(polygon: shapely.Polygon) -> None:
    _validate_ring_longitudes(polygon)
    _validate_polygon_bounds(polygon.bounds)


def _validate_polygon_bounds(bounds: tuple[float, float, float, float]) -> None:
    min_x, min_y, max_x, max_y = bounds
    _validate_longitude_bounds(min_x, max_x)
    _validate_latitude_bounds(min_y, max_y)
    _validate_antimeridian_span(min_x, max_x)


def _validate_longitude_bounds(min_x: float, max_x: float) -> None:
    if min_x < -180.0 or max_x > 180.0:
        raise WorldCoverAreaMeasurementError("worldcover_pole_or_crs_edge_unsupported")


def _validate_latitude_bounds(min_y: float, max_y: float) -> None:
    if min_y <= -89.999999 or max_y >= 89.999999:
        raise WorldCoverAreaMeasurementError("worldcover_pole_or_crs_edge_unsupported")


def _validate_antimeridian_span(min_x: float, max_x: float) -> None:
    if max_x - min_x >= 180.0:
        raise WorldCoverAreaMeasurementError("worldcover_antimeridian_span_unsupported")


def _validate_ring_longitudes(polygon: shapely.Polygon) -> None:
    for ring in (polygon.exterior, *polygon.interiors):
        longitudes = shapely.get_coordinates(ring)[:, 0]
        if len(longitudes) > 1 and bool(np.any(np.abs(np.diff(longitudes)) > 180.0)):
            raise WorldCoverAreaMeasurementError("worldcover_longitude_wrapping_unsupported")


def _validate_polygon_area(polygon_area: float) -> None:
    if not math.isfinite(polygon_area) or polygon_area <= 0.0:
        raise ValueError("WorldCover source polygon has no positive WGS84 equal-area")


def _validate_support_denominator(
    polygon_area: float,
    class_areas: tuple[tuple[int, float], ...],
) -> None:
    support_total = sum(area for _code, area in class_areas)
    if support_total > polygon_area * (1.0 + 1e-8):
        raise ValueError("WorldCover class support exceeds the full equal-area polygon area")


def _aggregate_tile_support(
    raster_sources: Iterable[tuple[str | None, Path]],
    geometry: shapely.Geometry,
    *,
    area_geometry: shapely.Geometry,
) -> tuple[tuple[tuple[int, float], ...], tuple[tuple[int, tuple[str, ...]], ...]]:
    """Sum each tile's class areas and retain contributing source IDs."""
    totals: defaultdict[int, float] = defaultdict(float)
    evidence_by_class: defaultdict[int, set[str]] = defaultdict(set)
    for source_id, path in raster_sources:
        _merge_tile_support(
            totals,
            evidence_by_class,
            source_id,
            _measure_tile(
                path,
                geometry,
                area_geometry=area_geometry,
            ),
        )
    class_areas = tuple(sorted((code, area) for code, area in totals.items() if area > 0.0))
    class_evidence = tuple(
        (code, tuple(sorted(source_ids))) for code, source_ids in sorted(evidence_by_class.items())
    )
    return class_areas, class_evidence


def _merge_tile_support(
    totals: defaultdict[int, float],
    evidence_by_class: defaultdict[int, set[str]],
    source_id: str | None,
    tile_totals: defaultdict[int, float],
) -> None:
    """Merge positive area from one tile and attribute its source ID."""
    for code, area in tile_totals.items():
        if area > 0.0:
            totals[code] += area
            if source_id is not None:
                evidence_by_class[code].add(source_id)


def _measure_tile(
    path: Path,
    geometry: shapely.Geometry,
    *,
    area_geometry: shapely.Geometry,
) -> defaultdict[int, float]:
    """Measure class areas in one raster tile without retaining cell records."""
    tile_totals: defaultdict[int, float] = defaultdict(float)
    with rasterio.open(path) as dataset:
        _validate_dataset(path, dataset)
        pixel_geometry = _pixel_geometry(geometry, dataset.transform)
        seen_boundary_cells: set[int] = set()
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
            partial_ids = cell_ids[coverage != 1.0]
            if len(partial_ids):
                boundary_ids = np.unique(np.concatenate((boundary_ids, partial_ids)))
            boundary_mask = np.isin(cell_ids, boundary_ids, assume_unique=False)
            _add_interior_cells(
                tile_totals,
                cell_ids[~boundary_mask],
                values[~boundary_mask],
                dataset.transform,
                dataset.width,
            )
            boundary_ids = _unseen_boundary_cells(boundary_ids, seen_boundary_cells)
            _add_boundary_cells(
                tile_totals,
                boundary_ids,
                area_geometry,
                dataset,
            )
    return tile_totals


def _validate_dataset(path: Path, dataset: rasterio.DatasetReader) -> None:
    _validate_dataset_crs(path, dataset)
    _validate_dataset_transform(path, dataset.transform)


def _validate_dataset_crs(path: Path, dataset: rasterio.DatasetReader) -> None:
    if dataset.crs is None or not pyproj.CRS.from_user_input(dataset.crs).equals(_GEOGRAPHIC_CRS):
        raise ValueError(f"WorldCover pool raster must use EPSG:4326: {path}")


def _validate_dataset_transform(path: Path, transform: rasterio.Affine) -> None:
    if transform.b != 0.0 or transform.d != 0.0 or transform.a <= 0.0 or transform.e >= 0.0:
        raise ValueError(f"WorldCover pool raster must be a north-up geographic grid: {path}")


def _equal_area_geometry(
    geometry: shapely.Geometry, *, densify_degrees: float = _AREA_DENSIFY_DEGREES
) -> shapely.Geometry:
    """Densify lon/lat edges, then transform them into the pinned equal-area CRS."""
    existing_coordinates = int(np.asarray(shapely.get_num_coordinates(geometry)).sum())
    total_length = float(np.asarray(shapely.length(geometry)).sum())
    estimated_coordinates = existing_coordinates + math.ceil(total_length / densify_degrees)
    if estimated_coordinates > _MAX_DENSIFIED_COORDINATES:
        raise WorldCoverAreaMeasurementError(
            "worldcover_area_boundary_densification_coordinate_limit_exceeded"
        )
    densified = shapely.segmentize(geometry, max_segment_length=densify_degrees)
    return shapely.transform(densified, _AREA_TRANSFORMER.transform, interleaved=False)


def _equal_area_m2(
    geometry: shapely.Geometry, *, densify_degrees: float = _AREA_DENSIFY_DEGREES
) -> float:
    return float(shapely.area(_equal_area_geometry(geometry, densify_degrees=densify_degrees)))


def _add_interior_cells(
    totals: defaultdict[int, float],
    cell_ids: np.ndarray,
    values: np.ndarray,
    transform: rasterio.Affine,
    width: int,
) -> None:
    valid = np.isfinite(values) & (values != 0.0)
    if not np.any(valid):
        return
    rows = cell_ids[valid] // width
    classes = values[valid].astype(np.int64)
    for row in np.unique(rows):
        selected = rows == row
        area = _cell_area(int(row), transform)
        for code in np.unique(classes[selected]):
            cell_count = int(np.count_nonzero(selected & (classes == code)))
            if cell_count > 0:
                totals[int(code)] += cell_count * area


def _cell_area(
    row: int,
    transform: rasterio.Affine,
) -> float:
    left = transform.c
    right = transform.c + transform.a
    top = transform.f + row * transform.e
    bottom = top + transform.e
    return float(shapely.area(_equal_area_cell_geometry(box(left, bottom, right, top))))


def _equal_area_cell_geometry(cells: shapely.Geometry) -> shapely.Geometry:
    """Transform axis-aligned geographic cells exactly under cylindrical equal-area."""
    return shapely.transform(cells, _AREA_TRANSFORMER.transform, interleaved=False)


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
    return _rasterize_boundary(pixel_geometry, left, top, right, bottom, width, height)


def _rasterize_boundary(
    pixel_geometry: shapely.Geometry,
    left: int,
    top: int,
    right: int,
    bottom: int,
    width: int,
    height: int,
) -> np.ndarray:
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
    return _unique_boundary_ids(ids)


def _unique_boundary_ids(ids: list[np.ndarray]) -> np.ndarray:
    if not ids:
        return np.array([], dtype=np.int64)
    return np.unique(np.concatenate(ids).astype(np.int64, copy=False))


def _add_boundary_cells(
    totals: defaultdict[int, float],
    cell_ids: np.ndarray,
    area_geometry: shapely.Geometry,
    dataset: rasterio.DatasetReader,
) -> None:
    for start in range(0, len(cell_ids), _BOUNDARY_CELL_CHUNK):
        _add_boundary_cell_chunk(
            totals,
            cell_ids[start : start + _BOUNDARY_CELL_CHUNK],
            area_geometry,
            dataset,
        )


def _add_boundary_cell_chunk(
    totals: defaultdict[int, float],
    cell_ids: np.ndarray,
    area_geometry: shapely.Geometry,
    dataset: rasterio.DatasetReader,
) -> None:
    rows, columns, values, valid = _read_cell_values(cell_ids, dataset)
    if not np.any(valid):
        return
    left = dataset.transform.c + columns * dataset.transform.a
    right = left + dataset.transform.a
    top = dataset.transform.f + rows * dataset.transform.e
    bottom = top + dataset.transform.e
    cells = shapely.box(left, bottom, right, top)
    area_cells = _equal_area_cell_geometry(cells)
    intersections = shapely.intersection(area_geometry, area_cells)
    for intersection, value in zip(intersections[valid], values[valid], strict=True):
        if intersection.is_empty:
            continue
        area = float(shapely.area(intersection))
        if area > 0.0:
            totals[int(value)] += area


def _unseen_boundary_cells(cell_ids: np.ndarray, seen: set[int]) -> np.ndarray:
    """Deduplicate chunks before measuring full-polygon cell intersections."""
    if not len(cell_ids):
        return cell_ids
    fresh = []
    for raw_id in cell_ids:
        cell_id = int(raw_id)
        if cell_id in seen:
            continue
        if len(seen) >= _MAX_BOUNDARY_CELL_IDS:
            raise ValueError("WorldCover tile boundary-cell deduplication limit exceeded")
        seen.add(cell_id)
        fresh.append(cell_id)
    return np.asarray(fresh, dtype=np.int64)


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
