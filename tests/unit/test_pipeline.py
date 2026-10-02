"""Region pipeline: geometry preparation, labelling and example assembly."""

import shutil

import pandas as pd
import pytest
from tests.conftest import FixedTiles, MissingTiles

from osm_worldcover.adapters.source import RegionTables
from osm_worldcover.config import Config
from osm_worldcover.domain.dominance import DominanceOutcome, RejectionReason
from osm_worldcover.domain.tiling import Tile
from osm_worldcover.pipeline import (
    RegionOutcome,
    _verdict,
    label_polygons,
    prepare_polygons,
    run_region,
    to_examples,
)

SQUARE_GEOJSON = '{"type":"Polygon","coordinates":[[[0,0],[0,4],[4,4],[4,0],[0,0]]]}'
LEFT_GEOJSON = '{"type":"Polygon","coordinates":[[[0,0],[0,4],[2,4],[2,0],[0,0]]]}'
BOWTIE_GEOJSON = '{"type":"Polygon","coordinates":[[[0,0],[1,1],[1,0],[0,1],[0,0]]]}'


def polygons_frame(**over) -> pd.DataFrame:
    base = {
        "polygon_id": ["p1"],
        "region": ["r"],
        "osm_type": ["way"],
        "osm_id": [1],
        "wikidata": ["Q1"],
        "name": ["N"],
        "lat": [2.0],
        "lon": [2.0],
        "geometry": [SQUARE_GEOJSON],
        "area_m2": [1000.0],
        "source_pbf": ["r.osm.pbf"],
    }
    return pd.DataFrame(base | over)


class TestPreparePolygons:
    def test_valid_geometry_is_parsed(self) -> None:
        frame, invalid = prepare_polygons(polygons_frame())
        assert invalid == 0
        assert frame.geometry.iloc[0].area == pytest.approx(16.0)

    def test_invalid_geometry_is_dropped_and_counted(self) -> None:
        frame, invalid = prepare_polygons(polygons_frame(geometry=[BOWTIE_GEOJSON]))
        assert len(frame) == 0
        assert invalid == 1

    def test_an_empty_table_yields_an_empty_frame(self) -> None:
        frame, invalid = prepare_polygons(polygons_frame().iloc[0:0])
        assert len(frame) == 0
        assert invalid == 0

    @pytest.mark.parametrize(
        "geometry",
        [None, "", "not GeoJSON", "null", '{"type":"Polygon"}', BOWTIE_GEOJSON],
    )
    def test_null_or_malformed_geometry_does_not_discard_valid_neighbors(self, geometry) -> None:
        polygons = pd.concat(
            [polygons_frame(polygon_id=["bad"], geometry=[geometry]), polygons_frame()],
            ignore_index=True,
        )
        frame, invalid = prepare_polygons(polygons)
        assert invalid == 1
        assert frame["polygon_id"].tolist() == ["p1"]


class TestLabelPolygons:
    def test_a_dominated_polygon_is_labelled(self, half_and_half) -> None:
        frame, _ = prepare_polygons(polygons_frame(geometry=[LEFT_GEOJSON]))
        outcome = RegionOutcome("r")
        labelled = label_polygons(frame, FixedTiles(half_and_half), 0.8, outcome)
        assert labelled["worldcover_code"].tolist() == [10]
        assert labelled["dominant_fraction"].iloc[0] == pytest.approx(1.0)
        assert outcome.polygons_accepted == 1

    def test_an_evenly_split_polygon_is_rejected_below_threshold(self, half_and_half) -> None:
        frame, _ = prepare_polygons(polygons_frame())
        outcome = RegionOutcome("r")
        labelled = label_polygons(frame, FixedTiles(half_and_half), 0.8, outcome)
        assert len(labelled) == 0
        assert outcome.rejections["below_threshold"] == 1

    def test_a_half_unobserved_polygon_is_rejected(self, with_nodata) -> None:
        frame, _ = prepare_polygons(polygons_frame())
        outcome = RegionOutcome("r")
        labelled = label_polygons(frame, FixedTiles(with_nodata), 0.8, outcome)
        assert len(labelled) == 0
        assert outcome.rejections["below_threshold"] == 1

    def test_tiles_are_released_after_use(self, half_and_half) -> None:
        frame, _ = prepare_polygons(polygons_frame(geometry=[LEFT_GEOJSON]))
        tiles = FixedTiles(half_and_half)
        label_polygons(frame, tiles, 0.8, RegionOutcome("r"))
        assert tiles.discarded == tiles.ensured

    def test_tiles_are_kept_when_asked(self, half_and_half) -> None:
        frame, _ = prepare_polygons(polygons_frame(geometry=[LEFT_GEOJSON]))
        tiles = FixedTiles(half_and_half)
        label_polygons(frame, tiles, 0.8, RegionOutcome("r"), keep_tiles=True)
        assert tiles.discarded == []


def tables_for(labelled_id: str = "p1", **over) -> RegionTables:
    documents = pd.DataFrame(
        {
            "document_id": ["d1"],
            "language": ["en"],
            "title": ["T"],
            "url": ["u"],
            "lead_text": ["lead"],
            "full_text": [" ".join(["word"] * 40)],
            "article_length_words": [40],
            "fetch_status": ["ok"],
            "license": ["CC BY-SA 4.0"],
            "project": ["wikipedia"],
        }
    )
    for key, value in over.items():
        documents[key] = value
    return RegionTables(
        stem="r",
        polygons=polygons_frame(),
        links=pd.DataFrame(
            {
                "polygon_id": [labelled_id],
                "document_id": ["d1"],
                "project": ["wikipedia"],
                "language": ["en"],
                "link_sources": ["[]"],
            }
        ),
        documents=documents,
    )


def _region_result_summary(examples: pd.DataFrame, outcome: RegionOutcome) -> dict:
    """Expose output shape and counters as one region-level contract value."""
    row = examples.iloc[0]
    return {
        "seen": outcome.polygons_seen,
        "examples": outcome.examples,
        "label": row["worldcover_code"],
        "label_name": row["worldcover_label"],
        "dominance": row["dominant_fraction"],
        "text_words": row["text_words"],
        "centroid": row["centroid_wkt"],
        "area": row["polygon_area_m2"],
    }


def _invalid_geometry_summary(examples: pd.DataFrame, outcome: RegionOutcome, tiles) -> dict:
    return {
        "examples_empty": examples.empty,
        "seen": outcome.polygons_seen,
        "invalid": outcome.polygons_invalid,
        "accepted": outcome.polygons_accepted,
        "with_examples": outcome.polygons_with_examples,
        "text_rejections": outcome.text_rejections,
        "rejections": outcome.rejections,
        "source_links": outcome.source_links,
        "source_documents": outcome.source_documents,
        "tiles_ensured": tiles.ensured,
    }


def _text_rejection_summary(examples: pd.DataFrame, outcome: RegionOutcome) -> dict:
    return {
        "polygon_ids": examples["polygon_id"].tolist(),
        "seen": outcome.polygons_seen,
        "accepted": outcome.polygons_accepted,
        "invalid": outcome.polygons_invalid,
        "rejections": outcome.rejections,
        "source_links": outcome.source_links,
        "source_documents": outcome.source_documents,
        "examples": outcome.examples,
        "with_examples": outcome.polygons_with_examples,
        "text_rejections": outcome.text_rejections,
    }


def _usable_document_summary(examples: pd.DataFrame, outcome: RegionOutcome) -> dict:
    return {
        "document_ids": set(examples["document_id"]),
        "examples": outcome.examples,
        "accepted": outcome.polygons_accepted,
        "with_examples": outcome.polygons_with_examples,
        "source_links": outcome.source_links,
        "source_documents": outcome.source_documents,
        "text_rejections": outcome.text_rejections,
    }


class TestToExamples:
    def _labelled(self) -> pd.DataFrame:
        frame = polygons_frame()
        frame["worldcover_code"] = [10]
        frame["dominant_fraction"] = [1.0]
        frame["observed_fraction"] = [1.0]
        return frame

    def test_one_row_per_polygon_document_pair(self) -> None:
        rows = to_examples(self._labelled(), tables_for(), min_words=10)
        assert len(rows) == 1
        assert rows["document_id"].iloc[0] == "d1"

    def test_failed_fetches_are_dropped(self) -> None:
        rows = to_examples(self._labelled(), tables_for(fetch_status="error"), min_words=10)
        assert len(rows) == 0

    def test_short_text_is_dropped(self) -> None:
        rows = to_examples(self._labelled(), tables_for(full_text="too short"), min_words=10)
        assert len(rows) == 0

    def test_a_polygon_with_no_links_yields_nothing(self) -> None:
        rows = to_examples(self._labelled(), tables_for(labelled_id="other"), min_words=10)
        assert len(rows) == 0

    def test_no_labelled_polygons_yields_nothing(self) -> None:
        assert len(to_examples(pd.DataFrame(), tables_for(), min_words=10)) == 0


class TestRunRegion:
    def test_a_region_produces_shaped_examples(self, half_and_half) -> None:
        tables = tables_for()
        tables = RegionTables(
            stem="r",
            polygons=polygons_frame(geometry=[LEFT_GEOJSON]),
            links=tables.links,
            documents=tables.documents,
        )
        config = Config()
        examples, outcome = run_region(config, tables, FixedTiles(half_and_half))
        assert _region_result_summary(examples, outcome) == {
            "seen": 1,
            "examples": 1,
            "label": 10,
            "label_name": "Tree cover",
            "dominance": pytest.approx(1.0),
            "text_words": 40,
            "centroid": "POINT (2 2)",
            "area": 1000.0,
        }

    def test_a_region_whose_polygons_are_all_rejected_yields_no_examples(
        self, half_and_half
    ) -> None:
        examples, outcome = run_region(Config(), tables_for(), FixedTiles(half_and_half))
        assert len(examples) == 0
        assert outcome.examples == 0

    @pytest.mark.parametrize("geometry", [None, "malformed GeoJSON"])
    def test_invalid_geometry_is_accounted_for_without_fetching_tiles(
        self, half_and_half, geometry
    ) -> None:
        tables = tables_for()
        tables = RegionTables(
            stem="r",
            polygons=polygons_frame(geometry=[geometry]),
            links=tables.links,
            documents=tables.documents,
        )
        tiles = FixedTiles(half_and_half)

        examples, outcome = run_region(Config(), tables, tiles)

        assert _invalid_geometry_summary(examples, outcome, tiles) == {
            "examples_empty": True,
            "seen": 1,
            "invalid": 1,
            "accepted": 0,
            "with_examples": 0,
            "text_rejections": {},
            "rejections": {},
            "source_links": 1,
            "source_documents": 1,
            "tiles_ensured": [],
        }


class TestTextRejectionAccounting:
    """Every labelled polygon is retained or counted once at its last text stage."""

    def test_each_text_stage_accounts_for_its_rejected_polygon(self, half_and_half) -> None:
        polygon_ids = ["no-link", "missing", "failed", "empty", "short", "kept"]
        polygons = pd.concat(
            [polygons_frame(polygon_id=[pid], geometry=[LEFT_GEOJSON]) for pid in polygon_ids],
            ignore_index=True,
        )
        source = tables_for()
        documents = pd.concat(
            [
                source.documents.assign(document_id="failed", fetch_status="error"),
                source.documents.assign(document_id="empty", full_text=" \t\n "),
                source.documents.assign(document_id="short", full_text="Small wooded garden"),
                source.documents.assign(document_id="kept"),
            ],
            ignore_index=True,
        )
        tables = RegionTables(
            stem="r",
            polygons=polygons,
            links=pd.DataFrame({"polygon_id": polygon_ids[1:], "document_id": polygon_ids[1:]}),
            documents=documents,
        )

        examples, outcome = run_region(Config(), tables, FixedTiles(half_and_half))

        assert _text_rejection_summary(examples, outcome) == {
            "polygon_ids": ["kept"],
            "seen": 6,
            "accepted": 6,
            "invalid": 0,
            "rejections": {},
            "source_links": 5,
            "source_documents": 4,
            "examples": 1,
            "with_examples": 1,
            "text_rejections": {
                "no_source_document": 1,
                "missing_document": 1,
                "document_fetch_failed": 1,
                "empty_text": 1,
                "text_too_short": 1,
            },
        }

    def test_one_usable_document_prevents_counting_a_polygon_as_rejected(
        self, half_and_half
    ) -> None:
        source = tables_for()
        document_ids = ["missing", "failed", "empty", "short", "kept-a", "kept-b"]
        documents = pd.concat(
            [
                source.documents.assign(document_id="failed", fetch_status="error"),
                source.documents.assign(document_id="empty", full_text=None),
                source.documents.assign(document_id="short", full_text="Garden"),
                source.documents.assign(document_id="kept-a"),
                source.documents.assign(document_id="kept-b"),
            ],
            ignore_index=True,
        )
        tables = RegionTables(
            stem="r",
            polygons=polygons_frame(geometry=[LEFT_GEOJSON]),
            links=pd.DataFrame({"polygon_id": ["p1"] * 6, "document_id": document_ids}),
            documents=documents,
        )

        examples, outcome = run_region(Config(), tables, FixedTiles(half_and_half))

        assert _usable_document_summary(examples, outcome) == {
            "document_ids": {"kept-a", "kept-b"},
            "examples": 2,
            "accepted": 1,
            "with_examples": 1,
            "source_links": 6,
            "source_documents": 5,
            "text_rejections": {},
        }

    def test_last_surviving_document_determines_polygon_rejection_stage(
        self, half_and_half
    ) -> None:
        source = tables_for()
        documents = pd.concat(
            [
                source.documents.assign(document_id="failed", fetch_status="error"),
                source.documents.assign(document_id="blank", full_text=""),
                source.documents.assign(document_id="short-a", full_text="Garden"),
                source.documents.assign(document_id="short-b", full_text="Wooded garden"),
            ],
            ignore_index=True,
        )
        tables = RegionTables(
            stem="r",
            polygons=polygons_frame(geometry=[LEFT_GEOJSON]),
            links=pd.DataFrame({"polygon_id": ["p1"] * 4, "document_id": documents["document_id"]}),
            documents=documents,
        )

        examples, outcome = run_region(Config(), tables, FixedTiles(half_and_half))

        assert examples.empty
        assert outcome.polygons_accepted == 1
        assert outcome.polygons_with_examples == outcome.examples == 0
        assert outcome.text_rejections == {"text_too_short": 1}


class TestTileDeduplication:
    """Regression: one raster serving several tiles must not be read twice.

    Reading it twice doubles every coverage share, which trips the overlapping
    coverage guard and silently rejects a perfectly good polygon.
    """

    def test_a_polygon_spanning_tiles_is_not_double_counted(self, half_and_half) -> None:
        # The fixture raster spans two tile rows, so this polygon's tile set
        # resolves to the same file twice.
        frame, _ = prepare_polygons(polygons_frame(geometry=[LEFT_GEOJSON]))
        outcome = RegionOutcome("r")
        labelled = label_polygons(frame, FixedTiles(half_and_half), 0.8, outcome)
        assert len(labelled) == 1
        assert labelled["dominant_fraction"].iloc[0] == pytest.approx(1.0)
        assert outcome.rejections == {}


class TestPolygonSizeCap:
    """Continent-scale polygons are refused before any raster is fetched.

    The largest polygon in the source is 10.2 million km2 -- about 10^11 pixels
    spread over ~100 tiles, roughly 10 GB of download for a single row.
    """

    def test_a_polygon_above_the_cap_is_refused(self, half_and_half) -> None:
        frame, _ = prepare_polygons(polygons_frame(geometry=[LEFT_GEOJSON], area_m2=[2e10]))
        outcome = RegionOutcome("r")
        tiles = FixedTiles(half_and_half)
        labelled = label_polygons(frame, tiles, 0.8, outcome, max_area_m2=1e10)
        assert len(labelled) == 0
        assert outcome.rejections["too_large"] == 1

    def test_an_oversized_polygon_costs_no_tile_download(self, half_and_half) -> None:
        frame, _ = prepare_polygons(polygons_frame(geometry=[LEFT_GEOJSON], area_m2=[2e10]))
        tiles = FixedTiles(half_and_half)
        label_polygons(frame, tiles, 0.8, RegionOutcome("r"), max_area_m2=1e10)
        assert tiles.ensured == []

    def test_a_polygon_exactly_at_the_cap_is_kept(self, half_and_half) -> None:
        frame, _ = prepare_polygons(polygons_frame(geometry=[LEFT_GEOJSON], area_m2=[1e10]))
        labelled = label_polygons(
            frame, FixedTiles(half_and_half), 0.8, RegionOutcome("r"), max_area_m2=1e10
        )
        assert len(labelled) == 1

    def test_no_cap_keeps_everything(self, half_and_half) -> None:
        frame, _ = prepare_polygons(polygons_frame(geometry=[LEFT_GEOJSON], area_m2=[1e30]))
        labelled = label_polygons(
            frame, FixedTiles(half_and_half), 0.8, RegionOutcome("r"), max_area_m2=None
        )
        assert len(labelled) == 1

    def test_the_cap_is_applied_by_a_whole_region_run(self, half_and_half) -> None:
        tables = tables_for()
        tables = RegionTables(
            stem="r",
            polygons=polygons_frame(geometry=[LEFT_GEOJSON], area_m2=[2e10]),
            links=tables.links,
            documents=tables.documents,
        )
        examples, outcome = run_region(
            Config(max_polygon_area_m2=1e10), tables, FixedTiles(half_and_half)
        )
        assert len(examples) == 0
        assert outcome.rejections["too_large"] == 1

    def test_surviving_polygons_are_renumbered_and_rejections_accumulate(
        self, half_and_half
    ) -> None:
        polygons = many_polygons(["huge", "a", "b"], [LEFT_GEOJSON] * 3, area_m2=[2e10, 1.0, 1.0])
        frame, _ = prepare_polygons(polygons)
        outcome = RegionOutcome("r")
        label_polygons(frame, FixedTiles(half_and_half), 0.8, outcome, max_area_m2=1e10)
        labelled = label_polygons(frame, FixedTiles(half_and_half), 0.8, outcome, max_area_m2=1e10)
        assert labelled["polygon_id"].tolist() == ["a", "b"]
        assert labelled.index.tolist() == [0, 1]
        assert "index" not in labelled.columns
        assert outcome.rejections["too_large"] == 2


def many_polygons(polygon_ids, geometries, **over) -> pd.DataFrame:
    """One polygon per id, so a region can hold polygons over different tile sets."""
    rows = [
        polygons_frame(polygon_id=[pid], osm_id=[index], geometry=[geometry]).assign(
            **{name: values[index] for name, values in over.items()}
        )
        for index, (pid, geometry) in enumerate(zip(polygon_ids, geometries, strict=True))
    ]
    return pd.concat(rows, ignore_index=True)


LOWER_LEFT_GEOJSON = '{"type":"Polygon","coordinates":[[[0,0],[0,2],[2,2],[2,0],[0,0]]]}'
STRADDLE_GEOJSON = '{"type":"Polygon","coordinates":[[[1,0],[1,2],[3,2],[3,0],[1,0]]]}'


class TestGroupedLabelling:
    """Polygons sharing a tile set are labelled together, groups in tile order."""

    def test_groups_are_labelled_in_tile_order_and_renumbered(self, half_and_half) -> None:
        # "big" touches two tile rows, "small" only one, whatever the input order.
        frame, _ = prepare_polygons(
            many_polygons(["big", "small"], [LEFT_GEOJSON, LOWER_LEFT_GEOJSON])
        )
        tiles = FixedTiles(half_and_half)

        labelled = label_polygons(frame, tiles, 0.8, RegionOutcome("r"))

        assert labelled["polygon_id"].tolist() == ["small", "big"]
        assert labelled.index.tolist() == [0, 1]
        assert [tile.name for tile in tiles.ensured] == ["N00E000", "N00E000", "N03E000"]

    def test_every_group_is_labelled_with_the_threshold_and_keep_policy(
        self, half_and_half
    ) -> None:
        frame, _ = prepare_polygons(
            many_polygons(["big", "small"], [SQUARE_GEOJSON, STRADDLE_GEOJSON])
        )
        tiles = FixedTiles(half_and_half)
        outcome = RegionOutcome("r")

        labelled = label_polygons(frame, tiles, 0.8, outcome, keep_tiles=True)

        assert len(labelled) == 0
        assert outcome.rejections == {"below_threshold": 2}
        assert tiles.discarded == []

    def test_unpublished_tiles_are_named_and_their_polygons_counted_per_group(
        self, half_and_half
    ) -> None:
        frame, _ = prepare_polygons(
            many_polygons(["big", "small", "small2"], [LEFT_GEOJSON] + [LOWER_LEFT_GEOJSON] * 2)
        )
        outcome = RegionOutcome("r")
        tiles = MissingTiles(half_and_half)

        labelled = label_polygons(frame, tiles, 0.8, outcome)

        assert len(labelled) == 0
        assert outcome.rejections == {"no_valid_class": 3}
        assert sorted(outcome.tiles_missing) == ["N00E000", "N00E000", "N03E000"]
        assert tiles.discarded == tiles.ensured

    def test_overlapping_coverage_refuses_the_polygon(self, half_and_half, tmp_path) -> None:
        twin = tmp_path / "twin.tif"
        shutil.copy(half_and_half, twin)

        class TwoRasters(FixedTiles):
            """Distinct files per tile row: a polygon over both is covered twice."""

            def ensure(self, tile):
                super().ensure(tile)
                return twin if tile.lat else half_and_half

        frame, _ = prepare_polygons(polygons_frame(geometry=[LEFT_GEOJSON]))
        outcome = RegionOutcome("r")

        labelled = label_polygons(frame, TwoRasters(half_and_half), 0.8, outcome)

        assert len(labelled) == 0
        assert outcome.rejections == {"no_valid_class": 1}

    def test_the_fetched_tiles_are_exactly_those_the_polygon_touches(self, half_and_half) -> None:
        frame, _ = prepare_polygons(polygons_frame(geometry=[LEFT_GEOJSON]))
        tiles = FixedTiles(half_and_half)
        label_polygons(frame, tiles, 0.8, RegionOutcome("r"))
        assert tiles.ensured == [Tile(0, 0), Tile(3, 0)]


class TestVerdict:
    def test_the_threshold_decides_between_label_and_rejection(self) -> None:
        assert _verdict({10: 0.9}, 0.95) == DominanceOutcome(
            False, 10, 0.9, RejectionReason.BELOW_THRESHOLD
        )
        assert _verdict({10: 0.9}, 0.5) == DominanceOutcome(True, 10, 0.9)

    def test_double_counted_coverage_is_refused_without_a_label(self) -> None:
        assert _verdict({10: 0.8, 50: 0.8}, 0.5) == DominanceOutcome(
            False, None, 0.0, RejectionReason.NO_VALID_CLASS
        )


class TestAcceptedPolygonColumns:
    def test_a_labelled_polygon_records_its_dominance_and_observed_coverage(
        self, with_nodata
    ) -> None:
        frame, _ = prepare_polygons(polygons_frame())
        labelled = label_polygons(frame, FixedTiles(with_nodata), 0.4, RegionOutcome("r"))
        row = labelled.iloc[0]
        assert (row["worldcover_code"], row["dominant_fraction"]) == (10, pytest.approx(0.5))
        assert row["observed_fraction"] == pytest.approx(0.5)

    def test_rejections_of_one_reason_add_up_across_polygons(self, half_and_half) -> None:
        frame, _ = prepare_polygons(many_polygons(["a", "b"], [SQUARE_GEOJSON] * 2))
        outcome = RegionOutcome("r")
        label_polygons(frame, FixedTiles(half_and_half), 0.8, outcome)
        assert outcome.rejections == {"below_threshold": 2}


class TestPreparedFrame:
    def test_an_empty_table_keeps_its_columns_and_the_wgs84_geometry_column(self) -> None:
        empty = polygons_frame().iloc[0:0]
        frame, _ = prepare_polygons(empty)
        assert frame.geometry.name == "geometry"
        assert frame.crs.to_epsg() == 4326
        assert list(frame.columns) == [*empty.columns]

    def test_usable_polygons_are_renumbered_without_an_index_column(self) -> None:
        polygons = pd.concat(
            [polygons_frame(polygon_id=["bad"], geometry=[BOWTIE_GEOJSON]), polygons_frame()],
            ignore_index=True,
        )
        frame, _ = prepare_polygons(polygons)
        assert frame.index.tolist() == [0]
        assert "index" not in frame.columns
        assert frame.crs.to_epsg() == 4326


class TestExamplesIndexAndAccumulation:
    def test_kept_examples_are_renumbered_without_an_index_column(self) -> None:
        source = tables_for().documents
        documents = pd.concat(
            [source.assign(document_id="d0", fetch_status="error"), source], ignore_index=True
        )
        links = pd.DataFrame({"polygon_id": ["p1", "p1"], "document_id": ["d0", "d1"]})
        tables = RegionTables("r", polygons_frame(), links, documents)
        rows = to_examples(polygons_frame().assign(worldcover_code=10), tables, 10)
        assert rows.index.tolist() == [0]
        assert "index" not in rows.columns

    def test_text_rejections_accumulate_across_calls(self) -> None:
        labelled = polygons_frame().assign(worldcover_code=10)
        outcome = RegionOutcome("r")
        for _ in range(2):
            to_examples(labelled, tables_for(full_text="short"), 10, outcome)
        assert outcome.text_rejections == {"text_too_short": 2}

    def test_document_columns_that_clash_with_labelled_ones_get_a_doc_suffix(self) -> None:
        labelled = polygons_frame().assign(worldcover_code=10, language="xx")
        rows = to_examples(labelled, tables_for(), 10)
        assert rows[["language", "language_doc"]].iloc[0].tolist() == ["xx", "en"]


class TestShapedExamples:
    def test_centroids_keep_seven_decimals(self, half_and_half) -> None:
        tables = tables_for()
        tables = RegionTables(
            "r",
            polygons_frame(geometry=[LEFT_GEOJSON], lat=[3.123456789], lon=[2.987654321]),
            tables.links,
            tables.documents,
        )
        examples, _ = run_region(Config(), tables, FixedTiles(half_and_half))
        assert examples["centroid_wkt"].tolist() == ["POINT (2.9876543 3.1234568)"]

    def test_the_document_language_wins_and_the_polygon_language_is_the_fallback(
        self, half_and_half
    ) -> None:
        source = tables_for()
        documents = pd.concat(
            [
                source.documents.assign(document_id="has", language="fr"),
                source.documents.assign(document_id="lacks", language=None),
            ],
            ignore_index=True,
        )
        tables = RegionTables(
            "r",
            many_polygons(["p1", "p2"], [LEFT_GEOJSON] * 2, language=["en", "en"]),
            pd.DataFrame({"polygon_id": ["p1", "p2"], "document_id": ["has", "lacks"]}),
            documents,
        )
        examples, _ = run_region(Config(), tables, FixedTiles(half_and_half))
        assert dict(zip(examples["polygon_id"], examples["language"], strict=True)) == {
            "p1": "fr",
            "p2": "en",
        }

    def test_columns_the_source_omits_are_published_as_missing(self, half_and_half) -> None:
        tables = tables_for()
        tables = RegionTables(
            "r",
            polygons_frame(geometry=[LEFT_GEOJSON]).drop(columns=["wikidata"]),
            tables.links,
            tables.documents,
        )
        examples, _ = run_region(Config(), tables, FixedTiles(half_and_half))
        assert examples["wikidata"].tolist() == [None]

    def test_run_region_names_the_region_and_honours_the_keep_policy(self, half_and_half) -> None:
        tables = tables_for()
        tables = RegionTables(
            "r", polygons_frame(geometry=[LEFT_GEOJSON]), tables.links, tables.documents
        )
        kept, released = FixedTiles(half_and_half), FixedTiles(half_and_half)
        _, outcome = run_region(Config(), tables, released)
        run_region(Config(), tables, kept, keep_tiles=True)
        assert outcome.stem == "r"
        assert released.discarded == released.ensured
        assert kept.discarded == []
