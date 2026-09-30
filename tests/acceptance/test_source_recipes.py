"""End-to-end acceptance checks for the non-Wikidata source recipes."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from tests.conftest import FixedTiles, write_raster

from osm_worldcover.adapters.source import RegionTables
from osm_worldcover.config import Config
from osm_worldcover.finalize import finalize_shards
from osm_worldcover.pipeline import run_region


def test_description_source_flows_through_the_shared_pipeline(tmp_path: Path) -> None:
    source = tmp_path / "description" / "data"
    source.mkdir(parents=True)
    text = "Small public wooded garden"
    polygon = Polygon([(0, 0), (0, 4), (2, 4), (2, 0)])
    pd.DataFrame(
        {
            "source_pbf": ["alpha-latest.osm.pbf"],
            "osm_type": ["way"],
            "osm_id": [7],
            "osm_url": ["https://www.openstreetmap.org/way/7"],
            "name": ["Description place"],
            "description": [text],
            "localized_descriptions": [[{"key": "fr", "value": "Jardin"}]],
            "area_m2": [1000.0],
            "geometry": [polygon.wkb],
        }
    ).to_parquet(source / "alpha-latest.parquet", index=False)

    _assert_published_examples(
        tmp_path, Config(source="description"), source.parent, {text, "Jardin"}
    )


def test_website_source_flows_through_the_shared_pipeline(tmp_path: Path) -> None:
    source = tmp_path / "website" / "polygons"
    source.mkdir(parents=True)
    text = " ".join(["website"] * 40)
    polygon = Polygon([(0, 0), (0, 4), (2, 4), (2, 0)])
    pd.DataFrame(
        {
            "polygon_id": ["alpha-latest:way/8"],
            "region": ["alpha"],
            "source_pbf": ["alpha-latest.osm.pbf"],
            "osm_type": ["way"],
            "osm_id": [8],
            "name": ["Website place"],
            "geometry": [json.dumps(polygon.__geo_interface__)],
            "lat": [2.0],
            "lon": [1.0],
            "area_m2": [1000.0],
            "website": ["https://example.org"],
            "website_text": [text],
            "website_text_status": ["success"],
            "website_language": ["eng_Latn"],
            "contact_website": [None],
            "contact_website_text": [None],
            "contact_website_text_status": [None],
            "contact_website_language": [None],
        }
    ).to_parquet(source / "alpha-latest.parquet", index=False)

    _assert_published_examples(tmp_path, Config(source="website"), source.parent, {text})


def _assert_published_examples(
    tmp_path: Path, config: Config, source: Path, expected_texts: set[str]
) -> None:
    """Run one source adapter through labelling, assembly, and manifest output."""
    raster = write_raster(tmp_path / "worldcover.tif", np.full((4, 4), 10, dtype="uint8"))
    tables = RegionTables.load(source, "alpha-latest", config.source_recipe)
    examples, outcome = run_region(config, tables, FixedTiles(raster))
    assert outcome.polygons_accepted == 1
    assert set(examples["text"]) == expected_texts
    assert len(examples) == len(expected_texts)

    shards = tmp_path / "shards"
    shards.mkdir()
    examples.to_parquet(shards / "alpha-latest.parquet", index=False)
    release = finalize_shards(shards, config, tmp_path / "assembly", tmp_path / "out", {})
    assert release.rows == len(expected_texts)
    assert release.report.ok
    assert release.manifest["settings"]["min_words"] == config.effective_min_words
    rows = pd.concat([pd.read_parquet(path) for path in release.paths if path.suffix == ".parquet"])
    assert set(rows["text"]) == expected_texts
    assert set(rows["worldcover_label"]) == {"Tree cover"}
