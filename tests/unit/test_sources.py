"""Source recipes normalize heterogeneous public inputs into one contract."""

import json

import pandas as pd
import pytest
from shapely.geometry import Polygon
from shapely.wkb import dumps

from osm_worldcover.adapters.source import RegionTables
from osm_worldcover.sources import recipe_for


def test_recipes_name_the_three_public_outputs() -> None:
    assert recipe_for("wikidata").output_dataset == "NoeFlandre/osm-wikidata-worldcover"
    assert (
        recipe_for("description").output_dataset
        == "NoeFlandre/osm-polygon-description-tag-worldcover"
    )
    assert recipe_for("website").output_dataset == "NoeFlandre/osm-polygon-website-tag-worldcover"


def test_description_rows_become_one_document_per_description_value(tmp_path) -> None:
    path = tmp_path / "data" / "luxembourg-latest.parquet"
    path.parent.mkdir()
    pd.DataFrame(
        {
            "source_pbf": ["luxembourg-latest.osm.pbf"],
            "osm_type": ["way"],
            "osm_id": [7],
            "osm_url": ["https://www.openstreetmap.org/way/7"],
            "name": ["Garden"],
            "description": ["A public garden with trees."],
            "localized_descriptions": [[{"key": "fr", "value": "Un jardin public."}]],
            "geometry_type": ["MultiPolygon"],
            "area_m2": [1000.0],
            "geometry": [dumps(Polygon([(6, 49), (6, 50), (7, 50), (7, 49), (6, 49)]))],
        }
    ).to_parquet(path, index=False)

    tables = RegionTables.load(tmp_path, "luxembourg-latest", source="description")

    assert tables.polygons.loc[0, "polygon_id"] == "luxembourg-latest:way/7"
    assert tables.polygons.loc[0, "region"] == "luxembourg"
    assert tables.polygons.loc[0, "lat"] == 49.5
    assert json.loads(tables.polygons.loc[0, "geometry"])["type"] == "Polygon"
    assert set(tables.documents["full_text"]) == {
        "A public garden with trees.",
        "Un jardin public.",
    }
    assert set(tables.documents["language"].dropna()) == {"fr"}
    assert tables.documents["language"].isna().sum() == 1
    assert set(tables.links["document_id"]) == {
        "luxembourg-latest:way/7:description",
        "luxembourg-latest:way/7:description:fr",
    }


def test_website_rows_become_successful_website_documents(tmp_path) -> None:
    path = tmp_path / "polygons" / "luxembourg-latest.parquet"
    path.parent.mkdir()
    pd.DataFrame(
        {
            "polygon_id": ["luxembourg-latest:way/8"],
            "region": ["luxembourg"],
            "source_pbf": ["luxembourg-latest.osm.pbf"],
            "osm_type": ["way"],
            "osm_id": [8],
            "name": ["Museum"],
            "geometry": [
                json.dumps(Polygon([(6, 49), (6, 50), (7, 50), (7, 49), (6, 49)]).__geo_interface__)
            ],
            "lat": [49.5],
            "lon": [6.5],
            "area_m2": [1000.0],
            "website": ["https://museum.example"],
            "website_text": ["Museum opening hours and collections."],
            "website_text_status": ["success"],
            "website_language": ["eng_Latn"],
            "contact_website": ["https://contact.example"],
            "contact_website_text": ["Contact the museum."],
            "contact_website_text_status": ["success"],
            "contact_website_language": ["eng_Latn"],
        }
    ).to_parquet(path, index=False)

    tables = RegionTables.load(tmp_path, "luxembourg-latest", source="website")

    assert len(tables.documents) == 2
    assert set(tables.documents["project"]) == {"website", "contact_website"}
    assert set(tables.documents["language"]) == {"eng_Latn"}
    assert set(tables.documents["full_text"]) == {
        "Museum opening hours and collections.",
        "Contact the museum.",
    }


def test_description_adapter_tolerates_missing_optional_columns(tmp_path) -> None:
    path = tmp_path / "data" / "luxembourg-latest.parquet"
    path.parent.mkdir()
    pd.DataFrame(
        {
            "source_pbf": ["luxembourg-latest.osm.pbf"],
            "osm_type": ["way"],
            "osm_id": [9],
            "osm_url": [None],
            "name": ["Unnamed place"],
            "description": ["A description with enough words."],
            "area_m2": [1000.0],
            "geometry": [dumps(Polygon([(6, 49), (6, 50), (7, 50), (7, 49), (6, 49)]))],
        }
    ).to_parquet(path, index=False)

    tables = RegionTables.load(tmp_path, "luxembourg-latest", source="description")

    assert len(tables.documents) == 1
    assert tables.documents.loc[0, "language"] is None


def test_missing_description_shard_is_an_empty_region(tmp_path) -> None:
    tables = RegionTables.load(tmp_path, "missing-latest", source="description")

    assert tables.polygons.empty
    assert tables.links.empty
    assert tables.documents.empty


@pytest.mark.parametrize("missing_text", [None, "", "  \n\t\u3000 "])
def test_blank_base_and_localized_descriptions_are_not_documents(tmp_path, missing_text) -> None:
    path = tmp_path / "data" / "alpha-latest.parquet"
    path.parent.mkdir()
    pd.DataFrame(
        {
            "source_pbf": ["alpha-latest.osm.pbf"],
            "osm_type": ["way"],
            "osm_id": [1],
            "description": [missing_text],
            "localized_descriptions": [[{"key": "fr", "value": missing_text}]],
            "area_m2": [1000.0],
            "geometry": [dumps(Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]))],
        }
    ).to_parquet(path, index=False)

    tables = RegionTables.load(tmp_path, "alpha-latest", source="description")

    assert len(tables.polygons) == 1
    assert tables.documents.empty
    assert tables.links.empty
