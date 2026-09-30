"""Shared fixtures.

The raster fixtures are deliberately tiny and hand-laid so expected coverage
fractions can be reasoned about exactly rather than approximated.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from osm_worldcover.domain.tiling import Tile


class FixedTiles:
    """Return one deterministic raster and record tile-cache calls."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.ensured: list[Tile] = []
        self.discarded: list[Tile] = []

    def ensure(self, tile: Tile) -> Path:
        self.ensured.append(tile)
        return self.path

    def discard(self, tile: Tile) -> None:
        self.discarded.append(tile)


def write_raster(path: Path, values: np.ndarray, origin=(0.0, 4.0), pixel=1.0) -> Path:
    """Write ``values`` as a 1-band byte GeoTIFF in WGS84 with ``pixel``-degree cells."""
    transform = from_origin(origin[0], origin[1], pixel, pixel)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=transform,
        nodata=0,
    ) as dst:
        dst.write(values, 1)
    return path


@pytest.fixture
def half_and_half(tmp_path: Path) -> Path:
    """A 4x4 raster: left half class 10, right half class 50."""
    values = np.zeros((4, 4), dtype="uint8")
    values[:, :2] = 10
    values[:, 2:] = 50
    return write_raster(tmp_path / "half.tif", values)


@pytest.fixture
def with_nodata(tmp_path: Path) -> Path:
    """A 4x4 raster: top half class 10, bottom half no-data (0)."""
    values = np.zeros((4, 4), dtype="uint8")
    values[:2, :] = 10
    return write_raster(tmp_path / "nodata.tif", values)


@pytest.fixture
def square() -> gpd.GeoDataFrame:
    """One polygon covering the whole 4x4 raster extent."""
    return gpd.GeoDataFrame(
        {"polygon_id": ["p1"]},
        geometry=[Polygon([(0, 0), (0, 4), (4, 4), (4, 0)])],
        crs="EPSG:4326",
    )
