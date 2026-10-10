"""Exact-value tests for the description and website source normalizers."""

import dataclasses
import math
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import shapely

from osm_worldcover.adapters import source_profiles as sp

POINT_WKB = shapely.to_wkb(shapely.Point(10.0, 20.0))
SQUARE = shapely.Polygon([(0, 0), (4, 0), (4, 2), (0, 2)])


def _records(frame: pd.DataFrame) -> list[dict]:
    clean = frame.astype(object).where(frame.notna(), None)
    return clean.to_dict("records")


def _columns(region: sp.NormalizedRegion) -> list[list[str]]:
    return [list(t.columns) for t in (region.polygons, region.links, region.documents)]


_CANONICAL = [sp.POLYGON_COLUMNS, sp.LINK_COLUMNS, [*sp.DOCUMENT_COLUMNS, "project"]]


def _write(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


_STRUCT = pa.list_(pa.struct([("key", pa.string()), ("value", pa.string())]))
_DESCRIPTION_TYPES = {
    "source_pbf": pa.string(),
    "osm_type": pa.string(),
    "osm_id": pa.int64(),
    "osm_url": pa.string(),
    "name": pa.string(),
    "description": pa.string(),
    "localized_descriptions": _STRUCT,
    "area_m2": pa.float64(),
    "geometry": pa.binary(),
}


def _description_table(rows: list[dict]) -> pa.Table:
    renamed = [{**row, "localized_descriptions": row["localized"]} for row in rows]
    return pa.table(
        {
            name: pa.array([row[name] for row in renamed], kind)
            for name, kind in _DESCRIPTION_TYPES.items()
        }
    )


def _description_row(**overrides):
    row = {
        "source_pbf": "x.pbf",
        "osm_type": "way",
        "osm_id": 7,
        "osm_url": "https://osm/way/7",
        "name": "Park",
        "description": "  A park  ",
        "localized": [
            {"key": "fr", "value": " Un parc "},
            {"key": "description:de", "value": "Ein Park"},
            {"key": "es", "value": "   "},
            {"key": "it", "value": None},
        ],
        "area_m2": 8.0,
        "geometry": shapely.to_wkb(SQUARE),
    }
    row.update(overrides)
    return row


def test_description_region_is_exact(tmp_path) -> None:
    _write(
        tmp_path / "data" / "europe-latest.parquet",
        _description_table([_description_row()]),
    )

    region = sp.load_description_region(tmp_path, "europe-latest")

    assert _columns(region) == _CANONICAL
    assert _records(region.polygons)[0] == {
        "polygon_id": "europe-latest:way/7",
        "region": "europe",
        "osm_type": "way",
        "osm_id": 7,
        "wikidata": None,
        "name": "Park",
        "lat": 1.0,
        "lon": 2.0,
        "geometry": shapely.to_geojson(SQUARE),
        "area_m2": 8.0,
        "source_pbf": "x.pbf",
    }
    assert _records(region.links) == [
        {
            "polygon_id": "europe-latest:way/7",
            "document_id": f"europe-latest:way/7:{key}",
            "project": "description",
            "language": language,
            "link_sources": "[]",
        }
        for key, language in [
            ("description", None),
            ("description:fr", "fr"),
            ("description:de", "de"),
        ]
    ]
    docs = _records(region.documents)
    assert docs[1] == {
        "document_id": "europe-latest:way/7:description:fr",
        "language": "fr",
        "title": "Park",
        "url": "https://osm/way/7",
        "lead_text": None,
        "full_text": "Un parc",
        "article_length_words": None,
        "fetch_status": "ok",
        "license": "ODbL",
        "project": "description",
    }


def test_description_without_text_yields_polygon_but_no_documents(tmp_path) -> None:
    row = _description_row(description=None, localized=None)
    _write(tmp_path / "data" / "s.parquet", _description_table([row]))

    region = sp.load_description_region(tmp_path, "s")

    assert _columns(region) == _CANONICAL
    assert region.polygons["region"].tolist() == ["s"]
    assert (region.links.empty, region.documents.empty) == (True, True)


def test_description_with_blank_base_text_is_skipped(tmp_path) -> None:
    row = _description_row(description="   ", localized=[{"key": "fr", "value": "Salut"}])
    _write(tmp_path / "data" / "s.parquet", _description_table([row]))

    region = sp.load_description_region(tmp_path, "s")

    assert region.links["document_id"].tolist() == ["s:way/7:description:fr"]


@pytest.mark.parametrize("loader", [sp.load_description_region, sp.load_website_region])
def test_missing_shard_gives_empty_canonical_tables(tmp_path, loader) -> None:
    region = loader(tmp_path, "absent")

    assert _columns(region) == _CANONICAL
    assert [t.empty for t in (region.polygons, region.links, region.documents)] == [True] * 3


def test_empty_shard_gives_empty_canonical_tables(tmp_path) -> None:
    _write(tmp_path / "data" / "e.parquet", _description_table([]))

    region = sp.load_description_region(tmp_path, "e")

    assert _columns(region) == _CANONICAL
    assert region.polygons.empty


def _website_table(rows: list[dict], drop: tuple[str, ...] = ()) -> pa.Table:
    columns = {name: [r.get(name) for r in rows] for name in sp._WEBSITE_COLUMNS}
    for name in drop:
        del columns[name]
    return pa.table(columns)


def _website_row(**overrides):
    row = {
        "polygon_id": "s:node/1",
        "region": "reg",
        "source_pbf": "x.pbf",
        "osm_type": "node",
        "osm_id": 1,
        "name": "Cafe",
        "geometry": "{}",
        "lat": 1.5,
        "lon": 2.5,
        "area_m2": 3.0,
        "website": "https://cafe.example",
        "website_text": "  Hello  ",
        "website_text_status": "success",
        "website_language": "en",
        "contact_website": "https://contact.example",
        "contact_website_text": "Contact us",
        "contact_website_text_status": "success",
        "contact_website_language": None,
    }
    row.update(overrides)
    return row


def test_website_region_is_exact(tmp_path) -> None:
    _write(tmp_path / "polygons" / "s.parquet", _website_table([_website_row()]))

    region = sp.load_website_region(tmp_path, "s")

    assert _records(region.polygons)[0] == {
        "polygon_id": "s:node/1",
        "region": "reg",
        "osm_type": "node",
        "osm_id": 1,
        "wikidata": None,
        "name": "Cafe",
        "lat": 1.5,
        "lon": 2.5,
        "geometry": "{}",
        "area_m2": 3.0,
        "source_pbf": "x.pbf",
    }
    assert _records(region.links) == [
        {
            "polygon_id": "s:node/1",
            "document_id": "s:node/1:website",
            "project": "website",
            "language": "en",
            "link_sources": "[]",
        },
        {
            "polygon_id": "s:node/1",
            "document_id": "s:node/1:contact_website",
            "project": "contact_website",
            "language": None,
            "link_sources": "[]",
        },
    ]


def test_website_documents_are_exact(tmp_path) -> None:
    _write(tmp_path / "polygons" / "s.parquet", _website_table([_website_row()]))

    first, second = _records(sp.load_website_region(tmp_path, "s").documents)

    assert first == {
        "document_id": "s:node/1:website",
        "language": "en",
        "title": "Cafe",
        "url": "https://cafe.example",
        "lead_text": None,
        "full_text": "Hello",
        "article_length_words": None,
        "fetch_status": "ok",
        "license": None,
        "project": "website",
    }
    assert (second["url"], second["full_text"], second["project"]) == (
        "https://contact.example",
        "Contact us",
        "contact_website",
    )


def test_website_documents_require_success_and_usable_text(tmp_path) -> None:
    rows = [
        _website_row(website_text_status="failed"),
        _website_row(
            polygon_id="s:node/2",
            website_text=None,
            contact_website_text="Contact us",
        ),
        _website_row(polygon_id="s:node/3", contact_website_text_status="failed"),
    ]
    _write(tmp_path / "polygons" / "s.parquet", _website_table(rows))

    region = sp.load_website_region(tmp_path, "s")

    assert region.documents["document_id"].tolist() == [
        "s:node/1:contact_website",
        "s:node/2:contact_website",
        "s:node/3:website",
    ]
    assert region.links["document_id"].tolist() == region.documents["document_id"].tolist()


def test_website_region_falls_back_to_stem_and_fills_missing_columns(tmp_path) -> None:
    rows = [_website_row(region=None)]
    _write(
        tmp_path / "polygons" / "north-latest.parquet",
        _website_table(rows, drop=("contact_website_text", "website_language")),
    )

    region = sp.load_website_region(tmp_path, "north-latest")

    assert region.polygons.iloc[0]["region"] == "north"
    assert region.documents["project"].tolist() == ["website"]
    assert region.documents.iloc[0]["language"] is None


@pytest.mark.parametrize(
    ("function", "value", "expected"),
    [
        (sp._nullable_string, math.nan, None),
        (sp._nullable_string, None, None),
        (sp._nullable_string, 5, "5"),
        (sp._nullable_string, pd.NA, "<NA>"),
        (sp._usable_text, math.nan, None),
        (sp._usable_text, None, None),
        (sp._usable_text, "  x ", "x"),
        (sp._usable_text, "   ", None),
        (sp._usable_text, 0, "0"),
        (sp._usable_text, 1.5, "1.5"),
    ],
)
def test_text_normalizers(function, value, expected) -> None:
    assert function(value) == expected


@pytest.mark.parametrize(
    ("function", "value", "expected"),
    [
        (sp._tag_key, "fr", "description:fr"),
        (sp._tag_key, "description:fr", "description:fr"),
        (sp._language_suffix, "description", None),
        (sp._language_suffix, "description:fr", "fr"),
        (sp._language_suffix, "description:", None),
        (sp._language_suffix, "other", "other"),
    ],
)
def test_tag_key_and_language_suffix(function, value, expected) -> None:
    assert function(value) == expected


@pytest.mark.parametrize("value", [None, "abc", b"abc", 5])
def test_pairs_ignores_non_collections(value) -> None:
    assert list(sp._pairs(value)) == []


def test_pairs_accepts_only_complete_mappings() -> None:
    value = [
        {"key": "a", "value": " 1 "},
        {"key": "b"},
        {"value": "2"},
        {"key": "c", "value": None},
        {"key": "d", "value": ""},
        "oops",
        {"key": 5, "value": 6},
    ]
    assert list(sp._pairs(value)) == [("a", "1"), ("5", "6")]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ({"key": "k", "value": "v"}, ("k", "v")),
        ({"key": "k"}, None),
        ({"value": "v"}, None),
        (["key", "value"], None),
    ],
)
def test_pair(value, expected) -> None:
    assert sp._pair(value) == expected


def test_polygon_id() -> None:
    assert sp._polygon_id("s", {"osm_type": "way", "osm_id": 3.0}) == "s:way/3"


@pytest.mark.parametrize(
    ("stem", "expected"),
    [("a-latest", "a"), ("a-latest-latest", "a-latest"), ("latest-a", "latest-a")],
)
def test_region_from_stem(stem, expected) -> None:
    assert sp._region_from_stem(stem) == expected


def test_read_keeps_requested_order_and_fills_absent_columns(tmp_path) -> None:
    path = tmp_path / "t.parquet"
    _write(path, pa.table({"b": [1, 2], "a": [3, 4], "extra": [5, 6]}))

    frame = sp._read(path, ["a", "b", "c"])

    assert list(frame.columns) == ["a", "b", "c"]
    assert [frame["a"].tolist(), frame["c"].tolist()] == [[3, 4], [None, None]]
    assert str(frame["c"].dtype) == "object"


def test_read_of_absent_file_is_an_empty_object_frame(tmp_path) -> None:
    missing = sp._read(tmp_path / "none.parquet", ["x", "y"])

    assert list(missing.columns) == ["x", "y"]
    assert missing.empty
    assert set(map(str, missing.dtypes)) == {"object"}


def test_fill_missing_adds_object_columns_in_requested_order() -> None:
    frame = pd.DataFrame({"a": [1, 2]})
    out = sp._fill_missing(frame, ["z", "a", "y"])
    assert list(out.columns) == ["z", "a", "y"]
    assert out["z"].tolist() == [None, None]
    assert out["y"].dtype == object


def test_frame_has_stable_columns_even_when_empty() -> None:
    assert list(sp._frame([], ["a", "b"]).columns) == ["a", "b"]
    out = sp._frame([{"b": 1, "a": 2}], ["a", "b"])
    assert _records(out) == [{"a": 2, "b": 1}]


def test_description_region_handles_multiple_rows(tmp_path) -> None:
    rows = [
        _description_row(osm_id=1, description="one", localized=[]),
        _description_row(osm_id=2, description="two", localized=[], name="Other"),
    ]
    _write(tmp_path / "data" / "s.parquet", _description_table(rows))

    region = sp.load_description_region(tmp_path, "s")

    assert region.polygons["polygon_id"].tolist() == ["s:way/1", "s:way/2"]
    assert region.polygons["name"].tolist() == ["Park", "Other"]
    assert region.documents["title"].tolist() == ["Park", "Other"]


def _assert_null_columns(documents: pd.DataFrame, columns: list[str]) -> None:
    """Columns must be explicitly None, not NaN from a missing key."""
    for column in columns:
        assert all(value is None for value in documents[column]), column


def test_documents_carry_explicit_null_columns(tmp_path) -> None:
    _write(tmp_path / "data" / "d.parquet", _description_table([_description_row()]))
    _write(tmp_path / "polygons" / "w.parquet", _website_table([_website_row()]))

    described = sp.load_description_region(tmp_path, "d").documents
    website = sp.load_website_region(tmp_path, "w").documents

    _assert_null_columns(described, ["lead_text", "article_length_words"])
    _assert_null_columns(website, ["lead_text", "article_length_words", "license"])
    assert set(described["license"]) == {"ODbL"}


def test_normalized_regions_are_immutable_value_objects() -> None:
    region = sp.NormalizedRegion(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    with pytest.raises(dataclasses.FrozenInstanceError):
        region.polygons = pd.DataFrame()  # type: ignore[misc]


def test_contact_website_language_is_read_from_its_own_column(tmp_path) -> None:
    columns = {name: [value] for name, value in _website_row(contact_website_language="fr").items()}
    _write(tmp_path / "polygons" / "s.parquet", pa.table(columns))

    documents = _records(sp.load_website_region(tmp_path, "s").documents)

    contact = [document for document in documents if document["project"] == "contact_website"]
    assert [document["language"] for document in contact] == ["fr"]


def test_public_surface_is_pinned() -> None:
    assert sp.__all__ == [
        "DOCUMENT_COLUMNS",
        "LINK_COLUMNS",
        "POLYGON_COLUMNS",
        "NormalizedRegion",
        "load_description_region",
        "load_website_region",
    ]
