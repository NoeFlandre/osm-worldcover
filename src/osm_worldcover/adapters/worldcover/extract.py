from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Final

import geopandas as gpd
import numpy as np
import pyproj
import rasterio
import shapely
from exactextract import exact_extract

from .boundary import _accumulate_corrected
from .geometry import (
    _bounded_pixel_chunks,
    _feature_batches,
    _map_geometry,
    _needs_spatial_chunks,
    _pixel_geometry,
)

#: exactextract operations needed to correct pixel-boundary coverage.
_OPS: Final[Sequence[str]] = ("cell_id", "coverage", "values")

_MAX_CELLS_IN_MEMORY: Final[int] = 2_000_000


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
