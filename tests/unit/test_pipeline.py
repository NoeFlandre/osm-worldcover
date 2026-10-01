"""Region pipeline: geometry preparation, labelling and example assembly."""

import pandas as pd
import pytest
from tests.conftest import FixedTiles

from osm_worldcover.adapters.source import RegionTables
from osm_worldcover.config import Config
from osm_worldcover.pipeline import (
    RegionOutcome,
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
