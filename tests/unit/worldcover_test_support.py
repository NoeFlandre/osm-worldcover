"""Shared helpers for the WorldCover adapter unit tests."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import shapely
from rasterio.windows import Window

import osm_worldcover.adapters.worldcover as worldcover

BOUNDARY_FIXTURES = Path(__file__).parents[1] / "fixtures" / "worldcover-boundary"


def one(geom) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"polygon_id": ["p"]}, geometry=[geom], crs="EPSG:4326")


class ExactExtractRecorder:
    """Capture the real adapter call and outputs for raster contract checks."""

    def __init__(self) -> None:
        self.original = worldcover.extract.exact_extract
        self.calls = []
        self.values = []

    def __call__(self, raster, features, operations, **kwargs):
        result = self.original(raster, features, operations, **kwargs)
        self.calls.append((raster, features.copy(), operations, kwargs))
        self.values.extend(result["values"])
        return result


def _assert_nodata_values_are_unmasked(recorder: ExactExtractRecorder) -> None:
    assert all(not np.any(np.ma.getmaskarray(values)) for values in recorder.values)


def _assert_nodata_calls_omit_default(recorder: ExactExtractRecorder) -> None:
    assert all(call[2] == ["cell_id", "coverage", "values"] for call in recorder.calls)
    assert all("default_value" not in call[3] for call in recorder.calls)


def _assert_nodata_coverage(coverage: dict[int, float]) -> None:
    assert coverage == {10: pytest.approx(0.5)}
    assert sum(coverage.values()) == pytest.approx(0.5)


def _recorded_pixel_bounds(recorder: ExactExtractRecorder) -> list[tuple[int, int]]:
    bounds = []
    for raster, features, _, _ in recorder.calls:
        with rasterio.open(raster) as dataset:
            for geometry in features.geometry:
                pixel_geometry = worldcover.geometry._pixel_geometry(geometry, dataset.transform)
                bounds.append(
                    (
                        worldcover.geometry._pixel_bbox_cells(
                            pixel_geometry, dataset.width, dataset.height
                        ),
                        shapely.get_num_coordinates(pixel_geometry),
                    )
                )
    return bounds


def _assert_pixel_bounds(bounds: list[tuple[int, int]], minimum: int) -> None:
    assert len(bounds) > minimum
    assert all(cells <= 9 and coordinates <= 12 for cells, coordinates in bounds)


def _assert_accepted_chunk_label(actual, expected) -> None:
    assert (actual.code, actual.accepted) == (expected.code, True)
    assert actual.code == 50
    assert actual.fraction == pytest.approx(expected.fraction, abs=1e-12)


def _assert_rejected_chunk_label(actual, expected) -> None:
    assert (actual.code, actual.accepted) == (expected.code, False)
    assert actual.code == 50
    assert actual.fraction == pytest.approx(expected.fraction, abs=1e-12)


def _sparse_full_grid_fixture(name: str, metadata: dict, tmp_path: Path) -> Path:
    """Restore a small tile window at its full WorldCover grid origin."""
    window_path = BOUNDARY_FIXTURES / f"{name}-window.tif"
    with rasterio.open(window_path) as window:
        values = window.read(1)
    output = tmp_path / f"{name}-full-grid.tif"
    with rasterio.open(
        output,
        "w",
        driver="GTiff",
        width=metadata["width"],
        height=metadata["height"],
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=rasterio.Affine(*metadata["transform"]),
        nodata=0,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress="DEFLATE",
        SPARSE_OK="TRUE",
    ) as dataset:
        dataset.write(
            values,
            1,
            window=Window(metadata["col0"], metadata["row0"], values.shape[1], values.shape[0]),
        )
    return output


def _bounds(geometries) -> list[tuple[float, ...]]:
    return [tuple(g.bounds) for g in geometries]


def _halo_cells(geometry, width, height) -> set[int]:
    """Independent oracle: cells whose square meets the buffered boundary."""
    halo = shapely.buffer(geometry.boundary, worldcover.boundary._BOUNDARY_HALO_PIXELS)
    return {
        row * width + col
        for row in range(height)
        for col in range(width)
        if halo.intersects(shapely.box(col, row, col + 1, row + 1))
    }


class ReadCounter:
    """Wrap a raster dataset and count pixel reads, delegating all else."""

    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.reads = 0

    def read(self, *args, **kwargs):
        self.reads += 1
        return self.dataset.read(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.dataset, name)


def _boundary_ids(geometry, width, height) -> np.ndarray:
    chunks = list(worldcover.boundary._boundary_cell_chunks(geometry, width, height))
    return np.concatenate([c[0] for c in chunks]) if chunks else np.array([], dtype=np.int64)
