"""Assembly must not depend on the whole dataset fitting in memory.

At ~4.9 KB per row a global build is roughly 10 GB of DataFrame, so the final
pass reads shards one at a time and does the global work in DuckDB over files.
"""

import json

import h3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_worldcover.adapters.writer import write_batches
from osm_worldcover.config import Config
from osm_worldcover.domain import manifest as manifest_module
from osm_worldcover.domain.validation import REQUIRED_COLUMNS
from osm_worldcover.finalize import (
    _assign_splits,
    _count,
    _deduplicate,
    _enrich_shards,
    _write_splits,
    finalize_shards,
)


def written(result) -> pd.DataFrame:
    """Every published row, read back from the files that were written."""
    splits = [p for p in result.paths if p.suffix == ".parquet"]
    return pd.concat([pd.read_parquet(p) for p in splits], ignore_index=True)


@pytest.fixture(autouse=True)
def fast_release_fsyncs(monkeypatch):
    import osm_worldcover.release_commit as release

    monkeypatch.setattr(release, "fsync_file", lambda _path: None)
    monkeypatch.setattr(release, "fsync_directory", lambda _path: None)


TEXT = " ".join(["word"] * 30)


def shard(
    path,
    n=1,
    start=0,
    region="luxembourg",
    code=10,
    text=None,
    text_words=None,
    lat=49.6,
    lon=6.1,
):
    pd.DataFrame(_shard_rows(n, start, region, code, text, text_words, lat, lon)).to_parquet(
        path, index=False
    )


def _shard_texts(text, ids: range) -> list[str]:
    return [text or f"{TEXT} {index}" for index in ids]


def _shard_text_words(text_words, texts: list[str]) -> list[int]:
    return [text_words or len(value.split()) for value in texts]


def _shard_rows(n, start, region, code, text, text_words, lat, lon) -> dict:
    ids = range(start, start + n)
    texts = _shard_texts(text, ids)
    return {
        "polygon_id": [f"{region}-latest:way:{index}" for index in ids],
        "osm_type": "way",
        "osm_id": list(ids),
        "region": region,
        "name": "N",
        "wikidata": "Q1",
        "document_id": [f"d{index}" for index in ids],
        "project": "wikipedia",
        "language": "en",
        "title": "T",
        "url": "u",
        "text": texts,
        "lead_text": "lead",
        "text_words": _shard_text_words(text_words, texts),
        "worldcover_code": code,
        "worldcover_label": {10: "Tree cover"}.get(code, "Built-up"),
        "dominant_fraction": 0.95,
        "observed_fraction": 1.0,
        "lat": lat,
        "lon": lon,
        "centroid_wkt": "POINT (6.1 49.6)",
        "polygon_area_m2": 1000.0,
        "source_pbf": f"{region}-latest.osm.pbf",
    }


@pytest.fixture
def shards(tmp_path):
    d = tmp_path / "shards"
    d.mkdir()
    return d


def _clear_shards(shards):
    for path in shards.glob("*.parquet"):
        path.unlink()


def _assert_empty_rerun_preserves_release(result, target, before):
    import hashlib

    assert result.rows == 0
    assert result.paths == []
    assert _file_snapshot(target) == before
    assert (
        hashlib.sha256((target / "manifest.json").read_bytes()).hexdigest()
        == before["manifest.json"][0]
    )


def _assert_empty_result(result):
    assert result.rows == 0
    assert result.paths == []


def _assert_release_recovered_and_clean(target, before, out):
    import osm_worldcover.release_commit as release

    assert release.inventory_release(target) == before
    assert not list(out.glob(f".{target.name}.stage-*"))
    assert not list(out.glob(f".{target.name}.backup-*"))
    assert not (out / f".{target.name}.transaction.json").exists()


def _assert_failed_validation_result(result):
    assert result.rows == 8
    assert result.paths == []
    assert not result.report.ok


def _assert_old_release_unchanged_and_unstaged(target, before, out):
    assert _file_snapshot(target) == before
    assert not list(out.glob(f".{target.name}.stage-*"))


def _recover_twice_and_assert_snapshot(target, before):
    import osm_worldcover.release_commit as release

    release.recover_release(target)
    release.recover_release(target)
    assert _file_snapshot(target) == before


def _assert_unchanged_rerun_result(result):
    assert result.rows == 3
    assert {path.name for path in result.paths} == {
        "train.parquet",
        "validation.parquet",
        "test.parquet",
        "manifest.json",
    }


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


def test_the_result_validates(shards, tmp_path) -> None:
    shard(shards / "a.parquet", n=5)
    assert finalize_shards(shards, Config(), tmp_path / "work", tmp_path / "work" / "out").report.ok


def test_manifest_counts_match_the_rows(shards, tmp_path) -> None:
    shard(shards / "a.parquet", n=7)
    result = finalize_shards(shards, Config(), tmp_path / "work", tmp_path / "work" / "out")
    assert result.manifest["counts"]["examples"]["total"] == result.rows


def test_manifest_includes_a_named_polygon_example(shards, tmp_path) -> None:
    shard(shards / "a.parquet", n=3)

    result = finalize_shards(shards, Config(), tmp_path / "work", tmp_path / "work" / "out")

    assert result.manifest["example_polygons"] == [{"name": "N", "worldcover_label": "Tree cover"}]


def test_provenance_is_attached(shards, tmp_path) -> None:
    shard(shards / "a.parquet", n=1)
    result = finalize_shards(
        shards, Config(source_revision="abc"), tmp_path / "work", tmp_path / "work" / "out"
    )
    row = written(result).iloc[0]
    assert row["source_revision"] == "abc"
    assert row["worldcover_version"] == "v200"


def test_output_is_sorted_deterministically(shards, tmp_path) -> None:
    shard(shards / "a.parquet", n=6)
    a = finalize_shards(shards, Config(), tmp_path / "w1", tmp_path / "w1" / "out")
    b = finalize_shards(shards, Config(), tmp_path / "w2", tmp_path / "w2" / "out")
    assert written(a)["polygon_id"].tolist() == written(b)["polygon_id"].tolist()


def test_an_empty_shard_directory_is_reported(shards, tmp_path) -> None:
    result = finalize_shards(shards, Config(), tmp_path / "work", tmp_path / "work" / "out")
    assert result.rows == 0
    assert not result.report.ok


def test_empty_shards_are_ignored(shards, tmp_path) -> None:
    pd.DataFrame().to_parquet(shards / "empty.parquet", index=False)
    shard(shards / "a.parquet", n=2)
    assert finalize_shards(shards, Config(), tmp_path / "work", tmp_path / "work" / "out").rows == 2


def test_reused_work_dir_does_not_retain_old_enriched_rows(shards, tmp_path) -> None:
    work = tmp_path / "work"
    shard(shards / "old.parquet")
    finalize_shards(shards, Config(), work, work / "out")

    (shards / "old.parquet").unlink()
    shard(shards / "new.parquet", n=2, start=10)
    result = finalize_shards(shards, Config(), work, work / "out")

    assert result.rows == 2


def _published_files(out) -> list[str]:
    release = out / f"v{Config().dataset_version}"
    return sorted(p.name for p in release.iterdir()) if release.exists() else []


def test_a_rerun_that_publishes_nothing_preserves_the_previous_release(shards, tmp_path) -> None:
    out = tmp_path / "out"
    shard(shards / "a.parquet", n=3)
    finalize_shards(shards, Config(), tmp_path / "w1", out)
    release = out / f"v{Config().dataset_version}"
    (release / "audit.json").write_text("local audit report")
    before = {path.name: path.read_bytes() for path in release.iterdir()}

    empty = tmp_path / "empty"
    empty.mkdir()
    result = finalize_shards(empty, Config(), tmp_path / "w2", out)

    assert result.rows == 0
    assert result.paths == []
    assert _published_files(out) == sorted(before)
    assert {path.name: path.read_bytes() for path in release.iterdir()} == before


def test_a_failed_rerun_leaves_the_previous_release_intact(shards, tmp_path, monkeypatch) -> None:
    out = tmp_path / "out"
    shard(shards / "a.parquet", n=3)
    finalize_shards(shards, Config(), tmp_path / "w1", out)
    before = {
        name: (out / f"v{Config().dataset_version}" / name).read_bytes()
        for name in _published_files(out)
    }

    later = tmp_path / "later"
    later.mkdir()
    shard(later / "b.parquet", n=5, start=100)
    calls = []

    def fail_second_split(reader, path):
        calls.append(path)
        if len(calls) == 2:
            raise OSError("disk full")
        return write_batches(reader, path)

    monkeypatch.setattr("osm_worldcover.finalize.write_batches", fail_second_split)
    with pytest.raises(OSError, match="disk full"):
        finalize_shards(later, Config(), tmp_path / "w2", out)

    assert _published_files(out) == sorted(before)
    assert {
        name: (out / f"v{Config().dataset_version}" / name).read_bytes() for name in before
    } == before
    assert sorted(p.name for p in out.iterdir()) == [f"v{Config().dataset_version}"]


def test_a_failed_first_run_publishes_nothing(shards, tmp_path, monkeypatch) -> None:
    out = tmp_path / "out"
    shard(shards / "a.parquet", n=3)

    def fail(reader, path):
        raise OSError("disk full")

    monkeypatch.setattr("osm_worldcover.finalize.write_batches", fail)
    with pytest.raises(OSError, match="disk full"):
        finalize_shards(shards, Config(), tmp_path / "w1", out)

    assert _published_files(out) == []
    assert not any(out.iterdir())


def test_a_successful_rerun_replaces_every_file_of_the_previous_release(shards, tmp_path) -> None:
    out = tmp_path / "out"
    shard(shards / "a.parquet", n=3)
    finalize_shards(shards, Config(), tmp_path / "w1", out)

    (shards / "a.parquet").unlink()
    shard(shards / "b.parquet", n=5, start=100)
    result = finalize_shards(shards, Config(), tmp_path / "w2", out)

    release = out / f"v{Config().dataset_version}"
    manifest = json.loads((release / "manifest.json").read_text())
    published = sum(
        len(pd.read_parquet(release / f"{s}.parquet")) for s in manifest_module.SPLIT_ORDER
    )
    assert result.rows == 5
    assert manifest["counts"]["examples"]["total"] == published == 5
    assert sorted(p.name for p in out.iterdir()) == [f"v{Config().dataset_version}"]


def test_split_writes_enable_large_arrow_string_buffers(tmp_path) -> None:
    import duckdb

    connection = duckdb.connect()
    try:
        connection.execute(
            "CREATE TEMP TABLE kept AS "
            "SELECT * FROM (VALUES ('train', 'p1', 'd1', 'key')) "
            "AS rows(split, polygon_id, document_id, _dedup_key)"
        )

        paths, rows = _write_splits(connection, tmp_path)

        assert rows == 1
        setting = connection.execute("SELECT current_setting('arrow_large_buffer_size')").fetchone()
        assert setting is not None
        assert str(setting[0]).lower() == "true"
        assert {path.stem for path in paths} == {"train", "validation", "test"}
    finally:
        connection.close()


def test_shards_are_never_all_held_in_memory(shards, tmp_path, monkeypatch) -> None:
    """Guard the property that matters: one shard is read at a time."""
    import osm_worldcover.finalize as module

    live = 0
    peak = 0
    original = pd.read_parquet

    def counting_read(*args, **kwargs):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        frame = original(*args, **kwargs)
        live -= 1
        return frame

    monkeypatch.setattr(module.pd, "read_parquet", counting_read)
    for i in range(5):
        shard(shards / f"s{i}.parquet", n=2, start=i * 10)
    finalize_shards(shards, Config(), tmp_path / "work", tmp_path / "work" / "out")
    assert peak == 1


def linked(path, polygon_ids, document_id, lats, lons, region="alpha", codes=None):
    """Rows sharing one document across several polygons.

    Rows share one document across polygons to exercise its one-home-split rule.
    Identical text and labels on distinct polygons remain separate records.
    """
    n = len(polygon_ids)
    codes = codes or [10] * n
    labels = {10: "Tree cover", 20: "Shrubland", 30: "Grassland", 50: "Built-up"}
    pd.DataFrame(
        {
            "polygon_id": [f"{region}-latest:way:{i}" for i in polygon_ids],
            "osm_type": ["way"] * n,
            "osm_id": list(polygon_ids),
            "region": [region] * n,
            "name": ["N"] * n,
            "wikidata": ["Q1"] * n,
            "document_id": [document_id] * n,
            "project": ["wikipedia"] * n,
            "language": ["en"] * n,
            "title": ["T"] * n,
            "url": ["u"] * n,
            "text": [f"{TEXT} {document_id}"] * n,
            "lead_text": ["lead"] * n,
            "text_words": [31] * n,
            "worldcover_code": list(codes),
            "worldcover_label": [labels[c] for c in codes],
            "dominant_fraction": [0.95] * n,
            "observed_fraction": [1.0] * n,
            "lat": list(lats),
            "lon": list(lons),
            "centroid_wkt": ["POINT (0 0)"] * n,
            "polygon_area_m2": [1000.0] * n,
            "source_pbf": [f"{region}-latest.osm.pbf"] * n,
        }
    ).to_parquet(path, index=False)


class TestDocumentLeakage:
    """One article can describe several distant polygons.

    Those polygons fall in different H3 cells and therefore different splits,
    which would put the same document in train and test. Found by running the
    assembly over real shards: 23 documents leaked across splits.
    """

    def test_a_document_spanning_distant_places_never_straddles_splits(
        self, shards, tmp_path
    ) -> None:
        linked(
            shards / "a.parquet",
            polygon_ids=[1, 2, 3, 4],
            document_id="shared",
            lats=[49.6, 35.7, -33.9, 60.2],  # Luxembourg, Tokyo, Sydney, Helsinki
            lons=[6.1, 139.7, 151.2, 24.9],
            codes=[10, 20, 30, 50],
        )
        result = finalize_shards(shards, Config(), tmp_path / "work", tmp_path / "work" / "out")
        rows = written(result)
        assert rows["split"].nunique() == 1
        assert result.report.ok

    def test_the_dropped_rows_are_counted(self, shards, tmp_path) -> None:
        linked(
            shards / "a.parquet",
            polygon_ids=[1, 2, 3, 4],
            document_id="shared",
            lats=[49.6, 35.7, -33.9, 60.2],
            lons=[6.1, 139.7, 151.2, 24.9],
            codes=[10, 20, 30, 50],
        )
        result = finalize_shards(shards, Config(), tmp_path / "work", tmp_path / "work" / "out")
        assert result.documents_split_across_splits > 0

    def test_a_document_confined_to_one_cell_keeps_every_row(self, shards, tmp_path) -> None:
        linked(
            shards / "a.parquet",
            polygon_ids=[1, 2, 3],
            document_id="local",
            lats=[49.600, 49.601, 49.602],
            lons=[6.100, 6.101, 6.102],
            codes=[10, 20, 30],
        )
        result = finalize_shards(shards, Config(), tmp_path / "work", tmp_path / "work" / "out")
        assert result.rows == 3
        assert result.documents_split_across_splits == 0

    def test_the_surviving_split_is_deterministic(self, shards, tmp_path) -> None:
        for name in ("a", "b"):
            linked(
                shards / f"{name}.parquet",
                polygon_ids=[1, 2, 3, 4],
                document_id="shared",
                lats=[49.6, 35.7, -33.9, 60.2],
                lons=[6.1, 139.7, 151.2, 24.9],
                codes=[10, 20, 30, 50],
            )
        first = finalize_shards(shards, Config(), tmp_path / "w1", tmp_path / "w1" / "out")
        second = finalize_shards(shards, Config(), tmp_path / "w2", tmp_path / "w2" / "out")
        assert written(first)["split"].tolist() == written(second)["split"].tolist()


class TestOneRegionPerObject:
    """The same OSM object must not appear under two polygon_ids.

    Geofabrik extracts overlap, and a region prefix is part of polygon_id, so
    picking the region per (object, document) let one object wear two ids.
    """

    def test_an_object_in_two_regions_uses_one_region_for_every_document(
        self, shards, tmp_path
    ) -> None:
        for region in ("luxembourg", "belgium"):
            shard(shards / f"{region}.parquet", n=2, start=0, region=region)
        result = finalize_shards(shards, Config(), tmp_path / "work", tmp_path / "work" / "out")
        rows = written(result)
        assert rows["region"].nunique() == 1


class TestShardSchemaDrift:
    """Shards must present one schema, whatever values a region happened to hold.

    Regression: a description-tag region whose rows all lack a language wrote
    `language` as a NULL-typed Parquet column. DuckDB takes the schema from the
    first file it reads, so combining that shard with one holding real strings
    failed with "failed to cast column language from VARCHAR to NULL" — and
    which file came first decided whether a global build succeeded at all.
    """

    def _shard_without_language(self, path, n=2, start=100):
        frame = pd.read_parquet(path) if path.exists() else None
        assert frame is None
        shard(path, n=n, start=start)
        written = pd.read_parquet(path)
        written["language"] = None
        written.to_parquet(path, index=False)

    def test_a_shard_with_no_language_still_combines(self, shards, tmp_path) -> None:
        shard(shards / "with_language.parquet", n=2, start=0)
        self._shard_without_language(shards / "no_language.parquet")
        result = finalize_shards(shards, Config(), tmp_path / "work", tmp_path / "work" / "out")
        assert result.rows == 4

    def test_the_published_language_column_stays_a_string(self, shards, tmp_path) -> None:
        shard(shards / "with_language.parquet", n=2, start=0)
        self._shard_without_language(shards / "no_language.parquet")
        result = finalize_shards(shards, Config(), tmp_path / "work", tmp_path / "work" / "out")
        split = next(p for p in result.paths if p.name == "train.parquet")
        field = pq.ParquetFile(split).schema_arrow.field("language")
        assert pa.types.is_string(field.type) or pa.types.is_large_string(field.type)

    def test_combining_does_not_depend_on_which_shard_is_read_first(self, shards, tmp_path) -> None:
        """Reversing the names reverses the read order; the result must not move."""
        shard(shards / "a_with.parquet", n=2, start=0)
        self._shard_without_language(shards / "z_without.parquet")
        first = finalize_shards(shards, Config(), tmp_path / "w1", tmp_path / "w1" / "out")

        other = tmp_path / "other"
        other.mkdir()
        shard(other / "z_with.parquet", n=2, start=0)
        self._shard_without_language(other / "a_without.parquet")
        second = finalize_shards(other, Config(), tmp_path / "w2", tmp_path / "w2" / "out")

        assert first.rows == second.rows


def run(shards, tmp_path, config=None):
    """Finalize ``shards`` under ``tmp_path``, returning the streamed build."""
    return finalize_shards(shards, config or Config(), tmp_path / "work", tmp_path / "work" / "out")


def with_columns(path, **columns) -> None:
    """Overwrite columns of a shard written by :func:`shard`."""
    frame = pd.read_parquet(path)
    for name, value in columns.items():
        frame[name] = value
    frame.to_parquet(path, index=False)


FIVE_PLACES = [  # Luxembourg, Tokyo, Sydney, Helsinki, Nairobi: five distinct H3 cells
    (49.6, 6.1),
    (35.7, 139.7),
    (-33.9, 151.2),
    (60.2, 24.9),
    (-1.3, 36.8),
]


@pytest.fixture
def five_places(shards):
    """One row per place, each in its own region, with distinct fractions, labels, languages."""
    fractions = [0.8, 0.85, 0.9000004, 0.95, 0.9876543217]
    languages = ["en", "fr", None, "en", "en"]
    codes = [10, 10, 50, 10, 50]
    for index, (lat, lon) in enumerate(FIVE_PLACES):
        path = shards / f"r{index}.parquet"
        shard(path, start=index, region=f"r{index}", code=codes[index], lat=lat, lon=lon)
        with_columns(path, dominant_fraction=fractions[index], language=languages[index])
    return shards


def test_manifest_reports_dominance_quantiles_to_six_places(five_places, tmp_path) -> None:
    manifest = run(five_places, tmp_path).manifest
    assert manifest["dominant_fraction"] == {"p50": 0.9, "p90": 0.972593, "p99": 0.986148}


def test_manifest_reports_geographic_coverage(five_places, tmp_path) -> None:
    coverage = run(five_places, tmp_path).manifest["geographic_coverage"]
    assert coverage == {
        "h3_cells": 5,
        "regions": 5,
        "bbox": {"min_lon": 6.1, "min_lat": -33.9, "max_lon": 151.2, "max_lat": 60.2},
    }


def test_manifest_keeps_an_absent_language_absent(five_places, tmp_path) -> None:
    manifest = run(five_places, tmp_path).manifest
    assert manifest["language_distribution"] == [
        {"language": "en", "examples": 3, "share": 0.6},
        {"language": "fr", "examples": 1, "share": 0.2},
        {"language": None, "examples": 1, "share": 0.2},
    ]
    assert [row["examples"] for row in manifest["class_distribution"]] == [3, 2]


def repeated_polygon(path, n, start, text) -> None:
    """A shard whose ``n`` documents all describe one polygon with the same text."""
    shard(path, n=n, start=start, text=text)
    with_columns(path, polygon_id=f"p{start}", osm_id=start)


@pytest.fixture
def two_repeated_polygons(shards):
    """Two groups lose records: two removed at three words, one removed at twelve."""
    repeated_polygon(shards / "short.parquet", 3, 0, "one two three")
    twelve = "one two three four five six seven eight nine ten eleven twelve"
    repeated_polygon(shards / "long.parquet", 2, 10, twelve)
    return shards


def test_deduplication_analysis_counts_groups_and_removed_records(
    two_repeated_polygons, tmp_path
) -> None:
    result = run(two_repeated_polygons, tmp_path)
    analysis = result.manifest["deduplication_analysis"]
    assert result.duplicate_records == 3
    assert analysis["duplicate_polygon_text_label_groups"] == 2
    assert analysis["duplicate_records_removed"] == 3


def test_deduplication_analysis_buckets_removed_records_by_text_length(
    two_repeated_polygons, tmp_path
) -> None:
    buckets = run(two_repeated_polygons, tmp_path).manifest["deduplication_analysis"][
        "duplicate_records_removed_by_text_words"
    ]
    assert buckets == {**{str(words): 0 for words in range(1, 10)}, "3": 2, "10+": 1}


def test_deduplication_analysis_reports_zero_for_an_empty_long_bucket(shards, tmp_path) -> None:
    shard(shards / "a.parquet", n=2, text="one two three")
    with_columns(shards / "a.parquet", polygon_id="p", osm_id=7)

    buckets = run(shards, tmp_path).manifest["deduplication_analysis"]
    assert buckets["duplicate_records_removed_by_text_words"]["10+"] == 0
    assert buckets["duplicate_records_removed_by_text_words"]["3"] == 1


def test_split_assignment_defaults_to_the_configured_ratios() -> None:
    config = Config(train_ratio=0.0, validation_ratio=1.0, test_ratio=0.0, h3_resolution=3)
    frame = pd.DataFrame({"lat": [49.6, 35.7], "lon": [6.1, 139.7]})
    assigned = _assign_splits(frame, config)
    assert assigned["split"].tolist() == ["validation", "validation"]
    assert [h3.get_resolution(cell) for cell in assigned["h3_cell"]] == [3, 3]


def test_split_assignment_follows_the_configured_seed() -> None:
    frame = pd.DataFrame({"lat": [49.6], "lon": [6.1]})
    assert _assign_splits(frame, Config())["split"].tolist() == ["train"]
    assert _assign_splits(frame, Config(split_seed=9))["split"].tolist() == ["test"]


def test_enrichment_skips_empty_shards_and_shards_without_polygons(shards, tmp_path) -> None:
    shard(shards / "a_good.parquet", n=2)
    shard(shards / "b_empty.parquet", n=1)
    pd.read_parquet(shards / "b_empty.parquet").iloc[0:0].to_parquet(
        shards / "b_empty.parquet", index=False
    )
    shard(shards / "c_no_polygon.parquet", n=1)
    pd.read_parquet(shards / "c_no_polygon.parquet").drop(columns="polygon_id").to_parquet(
        shards / "c_no_polygon.parquet", index=False
    )
    enriched = tmp_path / "enriched"
    enriched.mkdir()

    assert _enrich_shards(shards, enriched, Config()) == 2
    assert [path.name for path in enriched.glob("*.parquet")] == ["a_good.parquet"]


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


def test_every_row_carries_its_provenance(shards, tmp_path) -> None:
    shard(shards / "a.parquet")
    config = Config(
        source_revision="abc",
        source_dataset="owner/dataset",
        dataset_version="9.9.9",
        worldcover_version="v100",
        worldcover_year=2020,
    )
    row = written(run(shards, tmp_path, config)).iloc[0]
    assert row[
        ["dataset_version", "source_dataset", "source_revision", "worldcover_version"]
    ].tolist() == ["9.9.9", "owner/dataset", "abc", "v100"]
    assert row["worldcover_year"] == 2020


def test_enrichment_recounts_words_and_totals_every_shard(shards, tmp_path) -> None:
    shard(shards / "a.parquet", n=2, text_words=999)
    shard(shards / "b.parquet", n=3, start=10)
    enriched = tmp_path / "enriched"
    enriched.mkdir()

    assert _enrich_shards(shards, enriched, Config()) == 5

    frame = pd.read_parquet(enriched / "a.parquet")
    assert frame["text_words"].tolist() == [31, 31]
    assert frame["text_words"].dtype == "int64"
    assert frame["_dedup_key"].str.len().gt(0).all()


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


def test_an_empty_build_has_no_files_and_no_manifest(shards, tmp_path) -> None:
    result = run(shards, tmp_path)
    assert (result.rows, result.paths, result.manifest) == (0, [], {})


def test_empty_rerun_preserves_the_last_release_without_claiming_new_paths(
    shards, tmp_path
) -> None:
    shard(shards / "first.parquet", n=3)
    out = tmp_path / "out"
    target = out / f"v{Config().dataset_version}"
    first = finalize_shards(shards, Config(), tmp_path / "work", out)
    (target / "audit.json").write_text("local audit report")
    before = _file_snapshot(target)

    _clear_shards(shards)
    second = finalize_shards(shards, Config(), tmp_path / "work", out)

    _assert_empty_rerun_preserves_release(second, target, before)
    assert first.paths


def test_finalize_shards_accepts_a_symlinked_output_directory(shards, tmp_path):
    shard(shards / "first.parquet", n=3)
    real_out = tmp_path / "real-out"
    real_out.mkdir()
    output_alias = tmp_path / "out-link"
    output_alias.symlink_to(real_out, target_is_directory=True)

    result = finalize_shards(
        shards,
        Config(),
        tmp_path / "work",
        output_alias,
    )

    assert result.rows == 3
    assert {path.name for path in result.paths} == {
        "train.parquet",
        "validation.parquet",
        "test.parquet",
        "manifest.json",
    }
    assert all(path.is_file() for path in result.paths)
    assert list(real_out.glob(f"v{Config().dataset_version}"))


def test_empty_rerun_recovers_an_interrupted_promotion_before_counting_rows(
    shards, tmp_path, monkeypatch
) -> None:
    import osm_worldcover.release_commit as release

    shard(shards / "first.parquet", n=3)
    out = tmp_path / "out"
    target = out / f"v{Config().dataset_version}"
    finalize_shards(shards, Config(), tmp_path / "work", out)
    before = release.inventory_release(target)
    original = release._rename_directory

    def stop_after_backup(source, destination):
        original(source, destination)
        if source == target:
            raise SystemExit("simulated process interruption")

    monkeypatch.setattr(release, "_rename_directory", stop_after_backup)
    with release.release_lock(target):
        stage = release.create_stage(target)
        for name in release.CORE_FILES:
            (stage / name).write_bytes(f"uncommitted:{name}".encode())
        new = release.stage_inventory(stage)
        with pytest.raises(SystemExit):
            release.commit_release(target, stage, new)
    monkeypatch.setattr(release, "_rename_directory", original)

    _clear_shards(shards)
    result = finalize_shards(shards, Config(), tmp_path / "work", out)

    _assert_empty_result(result)
    _assert_release_recovered_and_clean(target, before, out)


def test_failed_staged_validation_does_not_promote_or_replace_the_old_release(
    shards, tmp_path, monkeypatch
) -> None:
    import osm_worldcover.finalize as module

    shard(shards / "first.parquet", n=3)
    out = tmp_path / "out"
    target = out / f"v{Config().dataset_version}"
    finalize_shards(shards, Config(), tmp_path / "work", out)
    before = _file_snapshot(target)
    shard(shards / "second.parquet", n=5, start=100)
    invalid = module.ValidationReport(8, [module.validate([]).violations[0]])
    monkeypatch.setattr(module, "_validate_written", lambda *_args: invalid)

    result = finalize_shards(shards, Config(), tmp_path / "work", out)

    _assert_failed_validation_result(result)
    _assert_old_release_unchanged_and_unstaged(target, before, out)
    _recover_twice_and_assert_snapshot(target, before)


@pytest.mark.parametrize("failed_split", ["train", "validation", "test"])
def test_failed_split_write_never_changes_the_previous_release(
    shards, tmp_path, monkeypatch, failed_split
) -> None:
    import osm_worldcover.finalize as module

    shard(shards / "first.parquet", n=3)
    out = tmp_path / "out"
    target = out / f"v{Config().dataset_version}"
    finalize_shards(shards, Config(), tmp_path / "work", out)
    before = _file_snapshot(target)
    shard(shards / "second.parquet", n=5, start=100)
    original = module.write_batches

    def interrupted(reader, path):
        if path.name == f"{failed_split}.parquet":
            path.write_bytes(b"partial split")
            raise OSError("injected split write interruption")
        return original(reader, path)

    monkeypatch.setattr(module, "write_batches", interrupted)

    with pytest.raises(OSError, match="injected split write interruption"):
        finalize_shards(shards, Config(), tmp_path / "work", out)

    import osm_worldcover.release_commit as release

    release.recover_release(target)
    release.recover_release(target)
    assert _file_snapshot(target) == before
    assert not list(out.glob(f".{target.name}.stage-*"))


def test_failed_manifest_write_never_changes_the_previous_release(
    shards, tmp_path, monkeypatch
) -> None:
    import osm_worldcover.finalize as module

    shard(shards / "first.parquet", n=3)
    out = tmp_path / "out"
    target = out / f"v{Config().dataset_version}"
    finalize_shards(shards, Config(), tmp_path / "work", out)
    before = _file_snapshot(target)
    shard(shards / "second.parquet", n=5, start=100)

    def interrupted(manifest, path):
        path.write_text("partial manifest")
        raise OSError("injected manifest write interruption")

    monkeypatch.setattr(module, "write_manifest", interrupted)

    with pytest.raises(OSError, match="injected manifest write interruption"):
        finalize_shards(shards, Config(), tmp_path / "work", out)

    import osm_worldcover.release_commit as release

    release.recover_release(target)
    release.recover_release(target)
    assert _file_snapshot(target) == before
    assert not list(out.glob(f".{target.name}.stage-*"))


def test_unchanged_rerun_keeps_release_files_and_publication_sidecars_untouched(
    shards, tmp_path
) -> None:
    shard(shards / "first.parquet", n=3)
    out = tmp_path / "out"
    target = out / f"v{Config().dataset_version}"
    finalize_shards(shards, Config(), tmp_path / "work", out)
    (target / "README.md").write_text("publication sidecar")
    (target / "audit.json").write_text("local audit report")
    before = _file_snapshot(target)

    result = finalize_shards(shards, Config(), tmp_path / "work", out)

    _assert_unchanged_rerun_result(result)
    _assert_old_release_unchanged_and_unstaged(target, before, out)
    _recover_twice_and_assert_snapshot(target, before)


def _file_snapshot(directory):
    import hashlib

    return {
        path.name: (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def test_the_written_dataset_is_validated_against_the_configured_threshold(
    shards, tmp_path
) -> None:
    shard(shards / "a.parquet")
    assert run(shards, tmp_path, Config(threshold=0.9)).report.ok
    assert not run(shards, tmp_path, Config(threshold=0.99)).report.ok


def test_the_written_dataset_is_validated_against_the_effective_minimum_words(
    shards, tmp_path
) -> None:
    shard(shards / "a.parquet")
    assert run(shards, tmp_path, Config(min_words=31)).report.ok
    assert not run(shards, tmp_path, Config(min_words=32)).report.ok


def test_validation_streams_only_the_required_columns(shards, tmp_path, monkeypatch) -> None:
    requested = []

    class Spy(pq.ParquetFile):
        def iter_batches(self, *args, **kwargs):
            requested.append((kwargs.get("batch_size"), kwargs.get("columns")))
            return super().iter_batches(*args, **kwargs)

    monkeypatch.setattr(pq, "ParquetFile", Spy)
    shard(shards / "a.parquet")
    run(shards, tmp_path)
    assert requested == [(8192, list(REQUIRED_COLUMNS))] * 3


def test_a_count_with_no_row_is_refused() -> None:
    import duckdb

    with pytest.raises(RuntimeError, match="count query returned no row: SELECT 1 WHERE false"):
        _count(duckdb.connect(), "SELECT 1 WHERE false")
