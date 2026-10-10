"""Source recipes normalize heterogeneous public inputs into one contract."""

import json
from typing import get_args

import pandas as pd
import pytest
from shapely.geometry import Polygon
from shapely.wkb import dumps

from osm_worldcover.adapters import source as source_module
from osm_worldcover.adapters.source import RegionTables
from osm_worldcover.sources import Layout, recipe_for, source_names


def _assert_description_polygon(tables: RegionTables) -> None:
    assert (
        tables.polygons.loc[0, "polygon_id"],
        tables.polygons.loc[0, "region"],
        tables.polygons.loc[0, "lat"],
        json.loads(tables.polygons.loc[0, "geometry"])["type"],
    ) == ("luxembourg-latest:way/7", "luxembourg", 49.5, "Polygon")


def _assert_description_documents(tables: RegionTables) -> None:
    assert (
        set(tables.documents["full_text"]),
        set(tables.documents["language"].dropna()),
        tables.documents["language"].isna().sum(),
    ) == (
        {"A public garden with trees.", "Un jardin public."},
        {"fr"},
        1,
    )


def _assert_description_links(tables: RegionTables) -> None:
    assert set(tables.links["document_id"]) == {
        "luxembourg-latest:way/7:description",
        "luxembourg-latest:way/7:description:fr",
    }


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

    _assert_description_polygon(tables)
    _assert_description_documents(tables)
    _assert_description_links(tables)


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


def test_source_url_is_derived_from_the_dataset() -> None:
    for name in ("wikidata", "description", "website"):
        recipe = recipe_for(name)
        assert recipe.source_url == f"https://huggingface.co/datasets/{recipe.source_dataset}"


def test_region_paths_are_the_files_each_loader_reads(tmp_path, monkeypatch) -> None:
    from osm_worldcover.adapters import source as source_module

    for name in ("wikidata", "description", "website"):
        recipe = recipe_for(name)
        read: list[str] = []

        def spy(path, columns, read=read):
            read.append(path.relative_to(tmp_path).as_posix())
            return pd.DataFrame({column: [] for column in columns})

        monkeypatch.setattr(source_module, "_read", spy)
        import osm_worldcover.adapters.source_profiles as profiles

        monkeypatch.setattr(profiles, "_read", spy)
        RegionTables.load(tmp_path, "alpha", recipe)

        assert sorted(read) == sorted(recipe.region_paths("alpha")), name


def test_unknown_layout_raises_a_clear_error(tmp_path) -> None:
    import dataclasses

    recipe = dataclasses.replace(recipe_for("website"), name="bogus")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="no table loader for source layout 'bogus'"):
        RegionTables.load(tmp_path, "alpha", recipe)


def test_layouts_and_table_loaders_cover_each_other_exactly() -> None:
    layouts = set(get_args(Layout))

    assert set(source_module._LOADERS) == layouts
    assert {recipe_for(name).name for name in source_names()} == layouts


def test_unknown_source_lists_the_sorted_choices() -> None:
    with pytest.raises(ValueError) as error:
        recipe_for("nope")

    assert str(error.value) == (
        "unknown source 'nope'; choose one of: description, website, wikidata"
    )
