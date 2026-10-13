"""Assembly must not depend on the whole dataset fitting in memory.

At ~4.9 KB per row a global build is roughly 10 GB of DataFrame, so the final
pass reads shards one at a time and does the global work in DuckDB over files.
"""

import h3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tests.unit.finalize_test_support import (
    fast_release_fsyncs as fast_release_fsyncs,
)
from tests.unit.finalize_test_support import (
    five_places as five_places,
)
from tests.unit.finalize_test_support import (
    linked,
    run,
    shard,
    with_columns,
    written,
)
from tests.unit.finalize_test_support import (
    shards as shards,
)
from tests.unit.finalize_test_support import (
    two_repeated_polygons as two_repeated_polygons,
)

from osm_worldcover.config import Config
from osm_worldcover.domain.validation import REQUIRED_COLUMNS
from osm_worldcover.finalize import (
    _assign_splits,
    _count,
    _enrich_shards,
    finalize_shards,
)


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


def test_an_empty_shard_directory_is_reported(shards, tmp_path) -> None:
    result = finalize_shards(shards, Config(), tmp_path / "work", tmp_path / "work" / "out")
    assert result.rows == 0
    assert not result.report.ok


def test_empty_shards_are_ignored(shards, tmp_path) -> None:
    pd.DataFrame().to_parquet(shards / "empty.parquet", index=False)
    shard(shards / "a.parquet", n=2)
    assert finalize_shards(shards, Config(), tmp_path / "work", tmp_path / "work" / "out").rows == 2


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
