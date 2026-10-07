"""Tiles are reused between polygon groups instead of re-downloaded."""

import pytest
from tests.conftest import FakeUrlopen

from osm_worldcover.adapters import worldcover as wc
from osm_worldcover.adapters.worldcover import WorldCoverTiles
from osm_worldcover.domain.tiling import Tile


@pytest.fixture
def downloads(monkeypatch) -> list[tuple[str, float | None]]:
    fake = FakeUrlopen(b"tif")
    monkeypatch.setattr(wc.urllib.request, "urlopen", fake)
    return fake.calls


def test_a_released_tile_is_reused_while_it_fits_in_the_cache(tmp_path, downloads) -> None:
    """Adjacent polygon groups share tiles; re-fetching 94 MB each time is waste."""
    tiles = WorldCoverTiles(tmp_path, max_cached_tiles=4)
    tile = Tile(48, 6)
    tiles.ensure(tile)
    tiles.discard(tile)
    tiles.ensure(tile)
    assert len(downloads) == 1


def test_the_cache_evicts_the_least_recently_used_tile(tmp_path, downloads) -> None:
    tiles = WorldCoverTiles(tmp_path, max_cached_tiles=2)
    a, b, c = Tile(0, 0), Tile(0, 3), Tile(0, 6)
    for tile in (a, b, c):
        tiles.ensure(tile)
        tiles.discard(tile)
    assert not tiles.path_for(a).exists()
    assert tiles.path_for(c).exists()


def test_tiles_still_in_use_are_not_evicted(tmp_path, downloads) -> None:
    tiles = WorldCoverTiles(tmp_path, max_cached_tiles=1)
    a, b = Tile(0, 0), Tile(0, 3)
    tiles.ensure(a)
    tiles.ensure(b)  # a is still held, so it must survive
    assert tiles.path_for(a).exists()
    assert tiles.path_for(b).exists()


def test_a_zero_capacity_cache_deletes_immediately(tmp_path, downloads) -> None:
    tiles = WorldCoverTiles(tmp_path, max_cached_tiles=0)
    tile = Tile(48, 6)
    tiles.ensure(tile)
    tiles.discard(tile)
    assert not tiles.path_for(tile).exists()


def test_cached_bytes_are_bounded_by_the_capacity(tmp_path, downloads) -> None:
    tiles = WorldCoverTiles(tmp_path, max_cached_tiles=3)
    for i in range(10):
        tile = Tile(0, i * 3)
        tiles.ensure(tile)
        tiles.discard(tile)
    assert len(list(tmp_path.glob("*.tif"))) <= 3
