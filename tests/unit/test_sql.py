"""SQL literals built from paths must read back as the same text, whatever the path holds."""

import duckdb
import pytest

from osm_worldcover.adapters.sql import sql_literal


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("plain", "'plain'"),
        ("", "''"),
        ("o'brien", "'o''brien'"),
        ("''", "''''''"),
        ("/tmp/with space/*.parquet", "'/tmp/with space/*.parquet'"),
    ],
)
def test_sql_literal_wraps_the_value_and_doubles_quotes(value: str, expected: str) -> None:
    assert sql_literal(value) == expected


@pytest.mark.parametrize(
    "value",
    ["plain", "", "o'brien", "''", "/data/it's here", "x'); DROP TABLE t; --"],
)
def test_sql_literal_reads_back_as_the_same_text(value: str) -> None:
    (read_back,) = duckdb.connect().execute(f"SELECT {sql_literal(value)}").fetchone()
    assert read_back == value
