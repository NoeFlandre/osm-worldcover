"""Tile caching and download behaviour.

Downloads are faked at ``urlopen`` except for one test, which stalls a real
loopback socket to prove the timeout is applied on the wire.
"""

import socket
import urllib.error
from email.message import Message

import pytest
from tests.conftest import FakeUrlopen

from osm_worldcover.adapters import worldcover as wc
from osm_worldcover.adapters.worldcover import (
    TileNotPublishedError,
    WorldCoverTiles,
)
from osm_worldcover.domain.tiling import Tile

TILE = Tile(48, 6)


def test_an_absent_tile_is_downloaded(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(wc.urllib.request, "urlopen", FakeUrlopen(b"tif"))
    path = WorldCoverTiles(tmp_path).ensure(TILE)
    assert path.exists()
    assert path.read_bytes() == b"tif"


def test_the_download_is_made_with_a_timeout(tmp_path, monkeypatch) -> None:
    fake = FakeUrlopen()
    monkeypatch.setattr(wc.urllib.request, "urlopen", fake)
    tiles = WorldCoverTiles(tmp_path)
    tiles.ensure(TILE)
    assert fake.calls == [(tiles.url_for(TILE), wc.DOWNLOAD_TIMEOUT_SECONDS)]


def test_a_stalled_download_times_out_and_leaves_no_partial_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(wc, "DOWNLOAD_TIMEOUT_SECONDS", 0.2)
    # The kernel completes the connection from the backlog, but nothing ever
    # answers it, so the response read stalls until the timeout fires.
    with socket.create_server(("127.0.0.1", 0)) as server:
        port = server.getsockname()[1]
        tiles = WorldCoverTiles(tmp_path, base_url=f"http://127.0.0.1:{port}")
        with pytest.raises((TimeoutError, urllib.error.URLError)):
            tiles.ensure(TILE)
    assert list(tmp_path.iterdir()) == []


def test_a_cached_tile_is_not_downloaded_again(tmp_path, monkeypatch) -> None:
    tiles = WorldCoverTiles(tmp_path)
    monkeypatch.setattr(wc.urllib.request, "urlopen", FakeUrlopen())
    tiles.ensure(TILE)

    def explode(*_args, **_kwargs):
        raise AssertionError("should not download a cached tile")

    monkeypatch.setattr(wc.urllib.request, "urlopen", explode)
    assert tiles.ensure(TILE).exists()


def test_an_empty_cached_file_is_treated_as_absent(tmp_path, monkeypatch) -> None:
    """A zero-byte file is the fingerprint of an interrupted download."""
    tiles = WorldCoverTiles(tmp_path)
    tiles.path_for(TILE).parent.mkdir(parents=True, exist_ok=True)
    tiles.path_for(TILE).write_bytes(b"")
    monkeypatch.setattr(wc.urllib.request, "urlopen", FakeUrlopen(b"real"))
    assert tiles.ensure(TILE).read_bytes() == b"real"


def test_an_unpublished_tile_raises(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(wc.urllib.request, "urlopen", FakeUrlopen(status=404))
    with pytest.raises(TileNotPublishedError):
        WorldCoverTiles(tmp_path).ensure(TILE)


def test_other_http_errors_are_not_swallowed(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(wc.urllib.request, "urlopen", FakeUrlopen(status=500))
    with pytest.raises(urllib.error.HTTPError):
        WorldCoverTiles(tmp_path).ensure(TILE)


def test_a_failed_download_leaves_no_partial_file(tmp_path, monkeypatch) -> None:
    not_found = urllib.error.HTTPError("url", 404, "Not Found", Message(), None)
    monkeypatch.setattr(
        wc.urllib.request,
        "urlopen",
        FakeUrlopen(b"half", error_after_body=not_found),
    )
    tiles = WorldCoverTiles(tmp_path)
    with pytest.raises(TileNotPublishedError):
        tiles.ensure(TILE)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.URLError("connection reset"),
        ConnectionResetError("connection reset"),
        TimeoutError("stalled"),
        KeyboardInterrupt(),
    ],
    ids=["url-error", "connection-reset", "timeout", "keyboard-interrupt"],
)
def test_a_non_http_failure_mid_download_leaves_no_partial_file(
    tmp_path, monkeypatch, error
) -> None:
    monkeypatch.setattr(
        wc.urllib.request,
        "urlopen",
        FakeUrlopen(b"half", error_after_body=error),
    )
    with pytest.raises(type(error)):
        WorldCoverTiles(tmp_path).ensure(TILE)
    assert list(tmp_path.iterdir()) == []


def test_a_connection_failure_leaves_no_partial_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        wc.urllib.request,
        "urlopen",
        FakeUrlopen(error=urllib.error.URLError("refused")),
    )
    with pytest.raises(urllib.error.URLError):
        WorldCoverTiles(tmp_path).ensure(TILE)
    assert list(tmp_path.iterdir()) == []


def test_a_body_shorter_than_its_declared_length_is_rejected(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        wc.urllib.request,
        "urlopen",
        FakeUrlopen(b"tif", content_length=10),
    )
    with pytest.raises(urllib.error.ContentTooShortError):
        WorldCoverTiles(tmp_path).ensure(TILE)
    assert list(tmp_path.iterdir()) == []


def test_discard_releases_a_tile_for_eviction(tmp_path, monkeypatch) -> None:
    """Discard releases rather than deletes; see test_tile_cache for eviction."""
    monkeypatch.setattr(wc.urllib.request, "urlopen", FakeUrlopen())
    tiles = WorldCoverTiles(tmp_path, max_cached_tiles=0)
    tiles.ensure(TILE)
    tiles.discard(TILE)
    assert not tiles.path_for(TILE).exists()


def test_discarding_an_absent_tile_is_harmless(tmp_path) -> None:
    WorldCoverTiles(tmp_path).discard(TILE)
