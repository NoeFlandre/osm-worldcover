"""WorldCover tile cache tests."""

import urllib.error

import pytest
from tests.conftest import FakeUrlopen

import osm_worldcover.adapters.worldcover as worldcover
from osm_worldcover.adapters.worldcover import TileNotPublishedError, WorldCoverTiles
from osm_worldcover.domain.tiling import Tile


class TestTileCache:
    def test_defaults_describe_the_published_product(self, tmp_path) -> None:
        tiles = WorldCoverTiles(str(tmp_path), base_url="https://example.test/rootX//")
        name = "ESA_WorldCover_10m_2021_v200_N48E006_Map.tif"
        assert (
            tiles.cache_dir,
            tiles.version,
            tiles.year,
            tiles.max_cached_tiles,
            tiles.url_for(Tile(48, 6)),
            tiles.path_for(Tile(48, 6)),
        ) == (
            tmp_path,
            "v200",
            2021,
            8,
            f"https://example.test/rootX/v200/2021/map/{name}",
            tmp_path / name,
        )

    def test_download_lands_atomically_in_the_cache(self, tmp_path, monkeypatch) -> None:
        fake = FakeUrlopen(b"tile")
        monkeypatch.setattr(worldcover.tiles.urllib.request, "urlopen", fake)
        tiles = WorldCoverTiles(tmp_path / "nested" / "cache")
        tile = Tile(48, 6)

        path = tiles.ensure(tile)

        assert (path, path.read_bytes(), tiles._in_use) == (
            tiles.path_for(tile),
            b"tile",
            {tile.name},
        )
        assert fake.calls == [(tiles.url_for(tile), worldcover.tiles.DOWNLOAD_TIMEOUT_SECONDS)]
        assert [p.name for p in path.parent.iterdir()] == [path.name]

    @pytest.mark.parametrize("cached", [b"cached", b"1"])
    def test_a_cached_tile_is_not_downloaded_again(self, tmp_path, monkeypatch, cached) -> None:
        fake = FakeUrlopen()
        monkeypatch.setattr(worldcover.tiles.urllib.request, "urlopen", fake)
        tiles = WorldCoverTiles(tmp_path)
        tiles.path_for(Tile(48, 6)).write_bytes(cached)
        assert (tiles.ensure(Tile(48, 6)).read_bytes(), fake.calls) == (cached, [])

    def test_an_empty_cached_tile_is_downloaded_again(self, tmp_path, monkeypatch) -> None:
        fake = FakeUrlopen(b"tile")
        monkeypatch.setattr(worldcover.tiles.urllib.request, "urlopen", fake)
        tiles = WorldCoverTiles(tmp_path)
        tiles.path_for(Tile(48, 6)).write_bytes(b"")
        assert tiles.ensure(Tile(48, 6)).read_bytes() == b"tile"
        assert len(fake.calls) == 1

    def test_an_absent_tile_is_reported_as_unpublished_and_leaves_no_partial_file(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(worldcover.tiles.urllib.request, "urlopen", FakeUrlopen(status=404))
        tiles = WorldCoverTiles(tmp_path)
        with pytest.raises(TileNotPublishedError, match=r"^N48E006 is not published$") as caught:
            tiles.ensure(Tile(48, 6))
        assert isinstance(caught.value.__cause__, urllib.error.HTTPError)
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize("status", [403, 500])
    def test_other_http_errors_propagate_and_clean_up(self, tmp_path, monkeypatch, status) -> None:
        monkeypatch.setattr(worldcover.tiles.urllib.request, "urlopen", FakeUrlopen(status=status))
        with pytest.raises(urllib.error.HTTPError) as caught:
            WorldCoverTiles(tmp_path).ensure(Tile(48, 6))
        assert caught.value.code == status
        assert not isinstance(caught.value, TileNotPublishedError)
        assert list(tmp_path.iterdir()) == []

    def test_an_ensured_tile_survives_the_release_of_others(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(worldcover.tiles.urllib.request, "urlopen", FakeUrlopen())
        tiles = WorldCoverTiles(tmp_path, max_cached_tiles=1)
        first, second = Tile(1, 1), Tile(2, 2)
        first_path = tiles.ensure(first)
        tiles.discard(first)
        assert first_path.exists()
        tiles.ensure(first)
        tiles.discard(second)
        assert second.name in tiles._released
        assert first.name not in tiles._released
        assert first_path.exists()

    def test_least_recently_released_tiles_are_evicted_first(self, tmp_path) -> None:
        tiles = WorldCoverTiles(tmp_path, max_cached_tiles=2)
        a, b, c = Tile(1, 1), Tile(2, 2), Tile(3, 3)
        for tile in (a, b, c):
            tiles.path_for(tile).write_bytes(b"x")
            tiles._in_use.add(tile.name)
        for released in (a, b, a, c):  # re-releasing a refreshes it: b is now the oldest
            tiles.discard(released)
        on_disk = [tiles.path_for(tile).exists() for tile in (a, b, c)]
        assert (on_disk, list(tiles._released), tiles._in_use) == (
            [True, False, True],
            [a.name, c.name],
            set(),
        )

    def test_evicting_a_tile_already_gone_from_disk_is_harmless(self, tmp_path) -> None:
        tiles = WorldCoverTiles(tmp_path, max_cached_tiles=0)
        tiles.discard(Tile(1, 1))
        assert tiles._released == {}

    def test_the_cache_holds_exactly_its_capacity(self, tmp_path) -> None:
        tiles = WorldCoverTiles(tmp_path, max_cached_tiles=1)
        a, b = Tile(1, 1), Tile(2, 2)
        for tile in (a, b):
            tiles.path_for(tile).write_bytes(b"x")
        tiles.discard(a)
        assert tiles.path_for(a).exists()
        tiles.discard(b)
        assert not tiles.path_for(a).exists()
        assert tiles.path_for(b).exists()
