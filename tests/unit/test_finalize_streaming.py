"""Assembly must not depend on the whole dataset fitting in memory.

At ~4.9 KB per row a global build is roughly 10 GB of DataFrame, so the final
pass reads shards one at a time and does the global work in DuckDB over files.
"""

import duckdb
import pandas as pd
import pytest
from tests.unit.finalize_test_support import (
    TEXT,
    run,
    shard,
    with_columns,
    written,
)
from tests.unit.finalize_test_support import (
    fast_release_fsyncs as fast_release_fsyncs,
)
from tests.unit.finalize_test_support import (
    five_places as five_places,
)
from tests.unit.finalize_test_support import (
    shards as shards,
)
from tests.unit.finalize_test_support import (
    two_repeated_polygons as two_repeated_polygons,
)

from osm_worldcover.config import Config
from osm_worldcover.finalize import (
    _deduplicate,
    _enrich_shards,
    finalize_shards,
)


def test_a_single_shard_becomes_a_dataset(shards, tmp_path) -> None:
    shard(shards / "a.parquet", n=3)
    result = finalize_shards(shards, Config(), tmp_path / "work", tmp_path / "work" / "out")
    assert result.rows == 3
    assert {p.stem for p in result.paths if p.suffix == ".parquet"} == {
        "train",
        "validation",
        "test",
    }


def test_every_row_gets_a_cell_and_a_split(shards, tmp_path) -> None:
    shard(shards / "a.parquet", n=3)
    result = finalize_shards(shards, Config(), tmp_path / "work", tmp_path / "work" / "out")
    all_rows = written(result)
    assert all_rows["split"].isin(["train", "validation", "test"]).all()
    assert all_rows["h3_cell"].str.len().gt(0).all()


def test_the_same_object_from_two_regions_is_kept_once(shards, tmp_path) -> None:
    shard(shards / "a.parquet", n=1, region="luxembourg")
    shard(shards / "b.parquet", n=1, region="belgium")
    result = finalize_shards(shards, Config(), tmp_path / "work", tmp_path / "work" / "out")
    assert result.rows == 1
    assert result.duplicates_across_regions == 1


def test_cross_region_dedup_is_deterministic(shards, tmp_path) -> None:
    shard(shards / "a.parquet", n=1, region="luxembourg")
    shard(shards / "b.parquet", n=1, region="belgium")
    first = finalize_shards(shards, Config(), tmp_path / "w1", tmp_path / "w1" / "out")
    second = finalize_shards(shards, Config(), tmp_path / "w2", tmp_path / "w2" / "out")
    kept = written(first)["region"].tolist()
    assert kept == written(second)["region"].tolist()
    assert kept == ["belgium"]


def test_identical_text_and_label_on_distinct_polygons_is_retained(shards, tmp_path) -> None:
    shared = "same text here " * 5
    shard(shards / "a.parquet", n=1, start=0, text=shared)
    shard(
        shards / "b.parquet",
        n=1,
        start=9,
        region="belgium",
        text=shared,
        lat=-33.9,
        lon=151.2,
    )
    result = finalize_shards(shards, Config(), tmp_path / "work", tmp_path / "work" / "out")
    assert result.rows == 2
    assert result.duplicate_records == 0
    assert result.manifest["deduplication_analysis"]["retained_identical_text_label_groups"] == 1


def test_repeated_records_of_the_same_polygon_text_and_label_are_removed(shards, tmp_path) -> None:
    path = shards / "a.parquet"
    shard(path, n=2, start=0, text=TEXT)
    frame = pd.read_parquet(path)
    frame.loc[1, "polygon_id"] = frame.loc[0, "polygon_id"]
    frame.loc[1, "osm_id"] = frame.loc[0, "osm_id"]
    frame.loc[1, "document_id"] = "same-polygon:second-document"
    frame.to_parquet(path, index=False)

    result = finalize_shards(shards, Config(), tmp_path / "work", tmp_path / "work" / "out")

    analysis = result.manifest["deduplication_analysis"]
    assert (
        result.rows,
        result.duplicate_records,
        result.manifest["deduplication"]["duplicate_polygon_text_label_records"],
        analysis["duplicate_polygon_text_label_groups"],
        analysis["duplicate_records_removed"],
    ) == (1, 1, 1, 1, 1)


def test_record_removals_and_retained_cross_split_text_are_reported(shards, tmp_path) -> None:
    shard(
        shards / "a.parquet",
        n=2,
        start=0,
        region="alpha",
        text="Park",
        lat=49.6,
        lon=6.1,
    )
    shard(
        shards / "b.parquet",
        start=2,
        region="beta",
        text="Park",
        lat=-33.9,
        lon=151.2,
    )
    shard(
        shards / "c.parquet",
        n=2,
        start=20,
        text="one two three four five six seven eight nine ten eleven twelve",
    )
    repeated = pd.read_parquet(shards / "c.parquet")
    repeated.loc[1, "polygon_id"] = repeated.loc[0, "polygon_id"]
    repeated.loc[1, "osm_id"] = repeated.loc[0, "osm_id"]
    repeated.loc[1, "document_id"] = "same-polygon:second-document"
    repeated.to_parquet(shards / "c.parquet", index=False)

    result = finalize_shards(shards, Config(), tmp_path / "work", tmp_path / "work" / "out")

    analysis = result.manifest["deduplication_analysis"]
    assert result.duplicate_records == 1
    assert analysis == {
        "duplicate_polygon_text_label_groups": 1,
        "duplicate_records_removed": 1,
        "duplicate_record_groups_crossing_splits": 0,
        "duplicate_records_removed_from_cross_split_groups": 0,
        "duplicate_records_removed_by_text_words": {
            "1": 0,
            "2": 0,
            "3": 0,
            "4": 0,
            "5": 0,
            "6": 0,
            "7": 0,
            "8": 0,
            "9": 0,
            "10+": 1,
        },
        "retained_identical_text_label_groups": 1,
        "retained_identical_text_label_rows": 3,
        "retained_identical_text_label_cross_split_groups": 1,
        "retained_identical_text_label_cross_split_rows": 3,
        "identical_text_cross_split_groups": 1,
        "identical_text_cross_split_rows": 3,
    }


def test_identical_text_under_different_labels_is_kept(shards, tmp_path) -> None:
    shard(shards / "a.parquet", n=1, start=0, text="same text here " * 5, code=10)
    shard(shards / "b.parquet", n=1, start=9, text="same text here " * 5, code=50)
    assert finalize_shards(shards, Config(), tmp_path / "work", tmp_path / "work" / "out").rows == 2


def test_deduplication_uses_the_work_directory_for_duckdb_spill(shards, tmp_path) -> None:
    shard(shards / "a.parquet")
    work = tmp_path / "work"
    enriched = work / "enriched"
    enriched.mkdir(parents=True)
    assert _enrich_shards(shards, enriched, Config()) == 1

    connection, _, _ = _deduplicate(enriched)
    try:
        spill_dir = connection.execute("SELECT current_setting('temp_directory')").fetchone()[0]
    finally:
        connection.close()

    assert spill_dir == str(work / "duckdb-spill")


def test_deduplication_runs_under_its_memory_budget_and_two_threads(
    shards, tmp_path, reported_setting
) -> None:
    shard(shards / "a.parquet")
    enriched = tmp_path / "work" / "enriched"
    enriched.mkdir(parents=True)
    assert _enrich_shards(shards, enriched, Config()) == 1

    connection, _, _ = _deduplicate(enriched)
    try:
        settings = {
            name: connection.execute(f"SELECT current_setting('{name}')").fetchone()[0]
            for name in ("memory_limit", "threads", "preserve_insertion_order")
        }
    finally:
        connection.close()

    assert settings == {
        "memory_limit": reported_setting("memory_limit", "2GB"),
        "threads": 2,
        "preserve_insertion_order": False,
    }


def test_output_is_sorted_deterministically(shards, tmp_path) -> None:
    shard(shards / "a.parquet", n=6)
    a = finalize_shards(shards, Config(), tmp_path / "w1", tmp_path / "w1" / "out")
    b = finalize_shards(shards, Config(), tmp_path / "w2", tmp_path / "w2" / "out")
    assert written(a)["polygon_id"].tolist() == written(b)["polygon_id"].tolist()


def test_reused_work_dir_does_not_retain_old_enriched_rows(shards, tmp_path) -> None:
    work = tmp_path / "work"
    shard(shards / "old.parquet")
    finalize_shards(shards, Config(), work, work / "out")

    (shards / "old.parquet").unlink()
    shard(shards / "new.parquet", n=2, start=10)
    result = finalize_shards(shards, Config(), work, work / "out")

    assert result.rows == 2


def test_a_polygon_text_pair_under_two_labels_keeps_both_records(shards, tmp_path) -> None:
    shard(shards / "a.parquet", n=2, text="same text here " * 5)
    with_columns(shards / "a.parquet", polygon_id="p", osm_id=7, worldcover_code=[10, 50])
    result = run(shards, tmp_path)
    assert (result.rows, result.duplicate_records) == (2, 0)


def test_duplicate_records_in_different_splits_are_reported_as_crossing(shards, tmp_path) -> None:
    shard(shards / "a.parquet", n=3, text="one two three")
    with_columns(
        shards / "a.parquet",
        polygon_id="p",
        osm_id=7,
        lat=[49.6, 49.6, -33.9],
        lon=[6.1, 6.1, 151.2],
    )
    analysis = run(shards, tmp_path).manifest["deduplication_analysis"]
    assert {
        key: value
        for key, value in analysis.items()
        if key.startswith("duplicate_record") and isinstance(value, int)
    } == {
        "duplicate_records_removed": 2,
        "duplicate_record_groups_crossing_splits": 1,
        "duplicate_records_removed_from_cross_split_groups": 2,
    }


def test_deduplication_leaves_only_the_kept_table_open(shards, tmp_path) -> None:
    shard(shards / "a.parquet")
    enriched = tmp_path / "work" / "enriched"
    enriched.mkdir(parents=True)
    _enrich_shards(shards, enriched, Config())

    connection, _, _ = _deduplicate(enriched)
    try:
        tables = connection.execute("SHOW TABLES").fetchall()
    finally:
        connection.close()

    assert tables == [("kept",)]


def test_deduplication_closes_its_connection_when_a_statement_fails(tmp_path, monkeypatch) -> None:
    opened = []
    connect = duckdb.connect

    def recording_connect(*args, **kwargs):
        connection = connect(*args, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr(duckdb, "connect", recording_connect)
    empty = tmp_path / "work" / "enriched"
    empty.mkdir(parents=True)  # no shards, so the first read fails

    with pytest.raises(duckdb.IOException):
        _deduplicate(empty)

    [connection] = opened
    with pytest.raises(duckdb.ConnectionException):
        connection.execute("SELECT 1")


def test_a_work_directory_with_a_quote_in_its_name_is_deduplicated(shards, tmp_path) -> None:
    shard(shards / "a.parquet", n=1, region="luxembourg")
    shard(shards / "b.parquet", n=1, region="belgium")
    work = tmp_path / "o'brien" / "work"

    result = finalize_shards(shards, Config(), work, work / "out")

    assert result.rows == 1
    assert result.duplicates_across_regions == 1


def test_the_finished_build_lists_its_splits_then_its_manifest(shards, tmp_path) -> None:
    shard(shards / "a.parquet")
    result = run(shards, tmp_path)
    target = tmp_path / "work" / "out" / f"v{Config().dataset_version}"
    assert result.paths == [
        target / "train.parquet",
        target / "validation.parquet",
        target / "test.parquet",
        target / "manifest.json",
    ]
    assert list((tmp_path / "work" / "enriched").glob("*.parquet"))
