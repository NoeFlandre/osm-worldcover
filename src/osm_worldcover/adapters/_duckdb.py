"""Open DuckDB sessions configured the same way for every adapter that works over files.

Each caller chooses its own memory budget and spill directory, because those
depend on the job. The thread count and the insertion-order setting do not, so
they are fixed here.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Final

import duckdb

from osm_worldcover.adapters.sql import sql_literal

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

__all__ = ["open_session", "session"]

_THREADS: Final = 2


def open_session(memory_limit: str, spill_dir: Path) -> DuckDBPyConnection:
    """Open a connection with ``memory_limit`` and ``spill_dir`` applied.

    The caller owns the returned connection and must close it. If the session
    cannot be configured, the connection is closed here before the error is raised.
    """
    connection = duckdb.connect()
    try:
        connection.execute(f"SET memory_limit = {sql_literal(memory_limit)}")
        connection.execute(f"SET threads = {_THREADS}")
        connection.execute("SET preserve_insertion_order = false")
        connection.execute("SET temp_directory = ?", [str(spill_dir)])
    except BaseException:
        connection.close()
        raise
    return connection


@contextmanager
def session(memory_limit: str, spill_dir: Path) -> Iterator[DuckDBPyConnection]:
    """Yield a configured connection, and close it however the block ends."""
    connection = open_session(memory_limit, spill_dir)
    try:
        yield connection
    finally:
        connection.close()
