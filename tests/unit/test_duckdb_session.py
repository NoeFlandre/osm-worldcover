"""A DuckDB session has the configured budget and spill space, and is always closed."""

import duckdb
import pytest

from osm_worldcover.adapters._duckdb import open_session, session


def _setting(connection, name):
    return connection.execute(f"SELECT current_setting('{name}')").fetchone()[0]


@pytest.fixture
def opened(monkeypatch):
    """Every connection the code under test opens, in the order it opened them."""
    connections = []
    connect = duckdb.connect

    def recording_connect(*args, **kwargs):
        connection = connect(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(duckdb, "connect", recording_connect)
    return connections


def test_open_session_applies_the_budget_threads_ordering_and_spill_directory(
    tmp_path, reported_setting
) -> None:
    connection = open_session("256MB", tmp_path / "spill")
    try:
        assert _setting(connection, "memory_limit") == reported_setting("memory_limit", "256MB")
        assert _setting(connection, "threads") == 2
        assert _setting(connection, "preserve_insertion_order") is False
        assert _setting(connection, "temp_directory") == str(tmp_path / "spill")
    finally:
        connection.close()


def test_open_session_closes_the_connection_when_setup_fails(tmp_path, opened) -> None:
    with pytest.raises(duckdb.Error):
        open_session("not a size", tmp_path)

    [connection] = opened
    with pytest.raises(duckdb.ConnectionException):
        connection.execute("SELECT 1")


def test_session_yields_an_open_connection_and_closes_it_afterwards(tmp_path) -> None:
    with session("256MB", tmp_path) as connection:
        assert connection.execute("SELECT 1").fetchone() == (1,)

    with pytest.raises(duckdb.ConnectionException):
        connection.execute("SELECT 1")


def test_session_closes_the_connection_when_the_block_fails(tmp_path, opened) -> None:
    with pytest.raises(RuntimeError, match="boom"), session("256MB", tmp_path):
        raise RuntimeError("boom")

    [connection] = opened
    with pytest.raises(duckdb.ConnectionException):
        connection.execute("SELECT 1")
