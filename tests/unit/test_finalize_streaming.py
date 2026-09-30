"""Assembly must not depend on the whole dataset fitting in memory.

At ~4.9 KB per row a global build is roughly 10 GB of DataFrame, so the final
pass reads shards one at a time and does the global work in DuckDB over files.
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_worldcover.config import Config
from osm_worldcover.finalize import _deduplicate, _enrich_shards, _write_splits, finalize_shards


def written(result) -> pd.DataFrame:
    """Every published row, read back from the files that were written."""
    splits = [p for p in result.paths if p.suffix == ".parquet"]
    return pd.concat([pd.read_parquet(p) for p in splits], ignore_index=True)


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
    texts = [text or f"{TEXT} {i}" for i in range(start, start + n)]
    pd.DataFrame(
        {
            "polygon_id": [f"{region}-latest:way:{i}" for i in range(start, start + n)],
            "osm_type": ["way"] * n,
            "osm_id": list(range(start, start + n)),
            "region": [region] * n,
            "name": ["N"] * n,
            "wikidata": ["Q1"] * n,
            "document_id": [f"d{i}" for i in range(start, start + n)],
            "project": ["wikipedia"] * n,
            "language": ["en"] * n,
            "title": ["T"] * n,
            "url": ["u"] * n,
            "text": texts,
            "lead_text": ["lead"] * n,
            "text_words": [text_words or len(value.split()) for value in texts],
            "worldcover_code": [code] * n,
            "worldcover_label": ["Tree cover" if code == 10 else "Built-up"] * n,
            "dominant_fraction": [0.95] * n,
            "observed_fraction": [1.0] * n,
            "lat": [lat] * n,
            "lon": [lon] * n,
            "centroid_wkt": ["POINT (6.1 49.6)"] * n,
            "polygon_area_m2": [1000.0] * n,
            "source_pbf": [f"{region}-latest.osm.pbf"] * n,
        }
    ).to_parquet(path, index=False)


@pytest.fixture
def shards(tmp_path):
    d = tmp_path / "shards"
    d.mkdir()
    return d


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

    assert result.rows == 1
    assert result.duplicate_records == 1
    assert result.manifest["deduplication"]["duplicate_polygon_text_label_records"] == 1
    analysis = result.manifest["deduplication_analysis"]
    assert analysis["duplicate_polygon_text_label_groups"] == 1
    assert analysis["duplicate_records_removed"] == 1


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
        assert setting[0]
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
