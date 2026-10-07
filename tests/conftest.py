"""Shared fixtures.

The raster fixtures are deliberately tiny and hand-laid so expected coverage
fractions can be reasoned about exactly rather than approximated.
"""

import io
import urllib.error
from collections import Counter
from email.message import Message
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from osm_worldcover.accounting import BuildContext
from osm_worldcover.adapters.worldcover import TileNotPublishedError
from osm_worldcover.config import Config
from osm_worldcover.domain.tiling import Tile
from osm_worldcover.pipeline import RegionOutcome


class FakeResponse(io.BytesIO):
    """What ``urlopen`` returns: a readable body and its response headers."""

    def __init__(self, body: bytes, headers: Message) -> None:
        super().__init__(body)
        self.headers = headers


class BrokenResponse(FakeResponse):
    """Delivers its body, then fails the way a dropped or stalled connection does."""

    def __init__(self, body: bytes, error: BaseException, headers: Message) -> None:
        super().__init__(body, headers)
        self.error = error

    def read(self, size: int | None = -1) -> bytes:
        chunk = super().read(size)
        if chunk:
            return chunk
        raise self.error


class FakeUrlopen:
    """Stand in for ``urllib.request.urlopen``, recording each ``(url, timeout)``.

    Serves ``body``, declaring ``content_length`` (default: the body's length).
    ``status`` fails the request with an HTTP error, ``error`` raises before any
    body arrives, and ``error_after_body`` delivers the body and then raises.
    """

    def __init__(
        self,
        body: bytes = b"tif",
        *,
        content_length: int | None = None,
        status: int | None = None,
        error: BaseException | None = None,
        error_after_body: BaseException | None = None,
    ) -> None:
        self.body = body
        self.content_length = len(body) if content_length is None else content_length
        self.status = status
        self.error = error
        self.error_after_body = error_after_body
        self.calls: list[tuple[str, float | None]] = []

    def __call__(self, url: str, timeout: float | None = None) -> FakeResponse:
        self.calls.append((url, timeout))
        if self.status is not None:
            raise urllib.error.HTTPError(url, self.status, "boom", Message(), None)
        if self.error is not None:
            raise self.error
        headers = Message()
        headers["Content-Length"] = str(self.content_length)
        if self.error_after_body is not None:
            return BrokenResponse(self.body, self.error_after_body, headers)
        return FakeResponse(self.body, headers)


@pytest.fixture
def config():
    """A configuration pinned to an immutable source revision."""
    return Config(source_revision="a" * 40)


@pytest.fixture
def context(config):
    """The build context completion receipts are bound to."""
    return BuildContext.from_config(config)


@pytest.fixture
def outcome():
    """A fully reconciled region outcome named ``alpha``."""
    return RegionOutcome(
        "alpha",
        polygons_seen=12,
        polygons_invalid=1,
        polygons_accepted=8,
        polygons_with_examples=2,
        examples=3,
        source_links=10,
        source_documents=9,
        rejections=Counter({"below_threshold": 3}),
        text_rejections=Counter({"text_too_short": 4, "empty_text": 2}),
        tiles_missing=["N00E000"],
    )


@pytest.fixture
def examples():
    """Three examples over two polygons, matching :func:`outcome`."""
    return pd.DataFrame({"polygon_id": ["p1", "p1", "p2"], "text": ["a", "b", "c"]})


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


class MissingTiles(FixedTiles):
    """A tile source whose product publishes none of the requested tiles."""

    def ensure(self, tile: Tile) -> Path:
        self.ensured.append(tile)
        raise TileNotPublishedError(tile.name)


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
