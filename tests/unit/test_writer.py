"""Writing a build to disk."""

import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_worldcover.adapters.writer import (
    PARQUET_ROW_GROUP_SIZE,
    read_manifest,
    write_batches,
    write_manifest,
)
from osm_worldcover.release_commit import release_lock


def reader(n: int = 3, max_chunksize: int = 2) -> pa.RecordBatchReader:
    table = pa.table(
        {
            "polygon_id": [f"p{i}" for i in range(n)],
            "text": [f"t{i}" for i in range(n)],
            "worldcover_code": [10] * n,
        }
    )
    return table.to_reader(max_chunksize=max_chunksize)


def test_every_row_is_written(tmp_path) -> None:
    path = tmp_path / "train.parquet"
    assert write_batches(reader(5), path) == 5
    assert len(pd.read_parquet(path)) == 5


def test_rows_survive_the_round_trip(tmp_path) -> None:
    path = tmp_path / "train.parquet"
    write_batches(reader(3), path)
    assert pd.read_parquet(path)["polygon_id"].tolist() == ["p0", "p1", "p2"]


def test_a_multi_batch_stream_is_written_as_one_file(tmp_path) -> None:
    """The point of streaming: a split larger than memory costs one batch."""
    path = tmp_path / "train.parquet"
    write_batches(reader(7), path)
    assert pq.ParquetFile(path).metadata.num_rows == 7


def test_a_large_input_batch_is_split_into_viewer_safe_row_groups(tmp_path) -> None:
    path = tmp_path / "train.parquet"
    rows = PARQUET_ROW_GROUP_SIZE + 1

    assert write_batches(reader(rows, max_chunksize=rows), path) == rows

    metadata = pq.ParquetFile(path).metadata
    assert metadata.num_row_groups == 2
    assert all(
        metadata.row_group(index).num_rows <= PARQUET_ROW_GROUP_SIZE
        for index in range(metadata.num_row_groups)
    )


def test_an_empty_stream_still_produces_a_file_with_the_schema(tmp_path) -> None:
    """A consumer expecting three splits should find three."""
    path = tmp_path / "test.parquet"
    assert write_batches(reader(0), path) == 0
    assert path.exists()
    assert "polygon_id" in pq.ParquetFile(path).schema_arrow.names


def test_writing_twice_is_byte_identical(tmp_path) -> None:
    """A rebuild of the same data must not produce a different file."""
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    write_batches(reader(4), a)
    write_batches(reader(4), b)
    assert a.read_bytes() == b.read_bytes()


def test_manifest_is_written_as_readable_json(tmp_path) -> None:
    path = write_manifest({"counts": {"examples": {"total": 3}}}, tmp_path / "manifest.json")
    text = path.read_text()
    assert text.endswith("\n")
    assert json.loads(text)["counts"]["examples"]["total"] == 3


def test_manifest_round_trips_through_read_manifest(tmp_path) -> None:
    with release_lock(tmp_path):
        write_manifest({"counts": {"examples": {"total": 7}}}, tmp_path / "manifest.json")
    assert read_manifest(tmp_path)["counts"]["examples"]["total"] == 7


def test_read_manifest_rejects_a_missing_build(tmp_path) -> None:
    target = tmp_path / "nope"
    with release_lock(target):
        pass
    with pytest.raises(FileNotFoundError):
        read_manifest(target)
