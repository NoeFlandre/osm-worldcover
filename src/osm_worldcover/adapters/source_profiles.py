"""Normalize the non-Wikidata source repositories.

The functions here are deliberately at the I/O boundary. They decode source
geometry and turn source-specific text fields into the canonical polygon,
link, and document tables consumed by the shared pipeline.
"""

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd
import pyarrow.parquet as pq
import shapely

from osm_worldcover.sources import DESCRIPTION_DIR, POLYGONS_DIR

__all__ = [
    "DOCUMENT_COLUMNS",
    "LINK_COLUMNS",
    "POLYGON_COLUMNS",
    "NormalizedRegion",
    "load_description_region",
    "load_website_region",
]

#: Canonical table columns shared by every source layout. Normalized document
#: rows add the source-specific project value after these base columns.
POLYGON_COLUMNS: Final[list[str]] = [
    "polygon_id",
    "region",
    "osm_type",
    "osm_id",
    "wikidata",
    "name",
    "lat",
    "lon",
    "geometry",
    "area_m2",
    "source_pbf",
]
LINK_COLUMNS: Final[list[str]] = [
    "polygon_id",
    "document_id",
    "project",
    "language",
    "link_sources",
]
DOCUMENT_COLUMNS: Final[list[str]] = [
    "document_id",
    "language",
    "title",
    "url",
    "lead_text",
    "full_text",
    "article_length_words",
    "fetch_status",
    "license",
]
_NORMALIZED_DOCUMENT_COLUMNS: Final[tuple[str, ...]] = (*DOCUMENT_COLUMNS, "project")

_DESCRIPTION_COLUMNS = (
    "source_pbf",
    "osm_type",
    "osm_id",
    "osm_url",
    "name",
    "description",
    "localized_descriptions",
    "area_m2",
    "geometry",
)
_WEBSITE_COLUMNS = (
    "polygon_id",
    "region",
    "source_pbf",
    "osm_type",
    "osm_id",
    "name",
    "geometry",
    "lat",
    "lon",
    "area_m2",
    "website",
    "website_text",
    "website_text_status",
    "website_language",
    "contact_website",
    "contact_website_text",
    "contact_website_text_status",
    "contact_website_language",
)


@dataclass(frozen=True, slots=True)
class NormalizedRegion:
    """Canonical tables for one source region."""

    polygons: pd.DataFrame
    links: pd.DataFrame
    documents: pd.DataFrame


def load_description_region(root: Path, stem: str) -> NormalizedRegion:
    """Read a description-tag shard and expose each description as a document."""
    raw = _read(root / DESCRIPTION_DIR / f"{stem}.parquet", _DESCRIPTION_COLUMNS)
    if raw.empty:
        return _empty_region()

    polygons = _description_polygons(raw, stem)
    links, documents = _description_documents(raw, stem)
    return NormalizedRegion(polygons, links, documents)


def load_website_region(root: Path, stem: str) -> NormalizedRegion:
    """Read a website-tag shard and expose successful website texts as documents."""
    raw = _read(root / POLYGONS_DIR / f"{stem}.parquet", _WEBSITE_COLUMNS)
    if raw.empty:
        return _empty_region()

    polygons = _website_polygons(raw, stem)
    links, documents = _website_documents(raw)
    return NormalizedRegion(polygons, links, documents)


def _description_polygons(raw: pd.DataFrame, stem: str) -> pd.DataFrame:
    """Decode GeoParquet WKB and derive the canonical spatial columns."""
    geometries = shapely.from_wkb(raw["geometry"].tolist())
    centroids = shapely.centroid(geometries)
    polygon_ids = [
        f"{stem}:{osm_type}/{int(osm_id)}"
        for osm_type, osm_id in zip(raw["osm_type"], raw["osm_id"], strict=True)
    ]
    source_pbf = raw["source_pbf"].tolist()
    return pd.DataFrame(
        {
            "polygon_id": polygon_ids,
            "region": [_region_from_stem(stem)] * len(raw),
            "osm_type": raw["osm_type"].tolist(),
            "osm_id": raw["osm_id"].tolist(),
            "wikidata": [None] * len(raw),
            "name": raw["name"].tolist(),
            "lat": shapely.get_y(centroids),
            "lon": shapely.get_x(centroids),
            "geometry": [str(value) for value in shapely.to_geojson(geometries)],
            "area_m2": raw["area_m2"].tolist(),
            "source_pbf": source_pbf,
        }
    )[POLYGON_COLUMNS]


def _description_documents(raw: pd.DataFrame, stem: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn base and localized descriptions into canonical document rows."""
    links: list[dict[str, Any]] = []
    documents: list[dict[str, Any]] = []
    for record in raw.to_dict("records"):
        polygon_id = _polygon_id(stem, record)
        values = [("description", record.get("description"))]
        values += [
            (_tag_key(key), value) for key, value in _pairs(record.get("localized_descriptions"))
        ]
        for tag_key, value in values:
            text = _usable_text(value)
            if text is None:
                continue
            language = _language_suffix(tag_key)
            document_id = f"{polygon_id}:{tag_key}"
            links.append(
                {
                    "polygon_id": polygon_id,
                    "document_id": document_id,
                    "project": "description",
                    "language": language,
                    "link_sources": "[]",
                }
            )
            documents.append(
                {
                    "document_id": document_id,
                    "language": language,
                    "title": record.get("name"),
                    "url": record.get("osm_url"),
                    "lead_text": None,
                    "full_text": text,
                    "article_length_words": None,
                    "fetch_status": "ok",
                    "license": "ODbL",
                    "project": "description",
                }
            )
    return _frame(links, LINK_COLUMNS), _frame(documents, _NORMALIZED_DOCUMENT_COLUMNS)


def _website_polygons(raw: pd.DataFrame, stem: str) -> pd.DataFrame:
    """Project website rows onto the canonical polygon contract."""
    region = raw["region"].fillna(_region_from_stem(stem)).tolist()
    polygons = pd.DataFrame(
        {
            "polygon_id": raw["polygon_id"].tolist(),
            "region": region,
            "osm_type": raw["osm_type"].tolist(),
            "osm_id": raw["osm_id"].tolist(),
            "wikidata": [None] * len(raw),
            "name": raw["name"].tolist(),
            "lat": raw["lat"].tolist(),
            "lon": raw["lon"].tolist(),
            "geometry": raw["geometry"].tolist(),
            "area_m2": raw["area_m2"].tolist(),
            "source_pbf": raw["source_pbf"].tolist(),
        }
    )
    return polygons[POLYGON_COLUMNS]


def _website_documents(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expose successful website and contact-website text independently."""
    links: list[dict[str, Any]] = []
    documents: list[dict[str, Any]] = []
    for record in raw.to_dict("records"):
        for project, prefix in (("website", "website"), ("contact_website", "contact_website")):
            if record.get(f"{prefix}_text_status") != "success":
                continue
            text = _usable_text(record.get(f"{prefix}_text"))
            if text is None:
                continue
            polygon_id = str(record["polygon_id"])
            document_id = f"{polygon_id}:{project}"
            language = _nullable_string(record.get(f"{prefix}_language"))
            links.append(
                {
                    "polygon_id": polygon_id,
                    "document_id": document_id,
                    "project": project,
                    "language": language,
                    "link_sources": "[]",
                }
            )
            documents.append(
                {
                    "document_id": document_id,
                    "language": language,
                    "title": record.get("name"),
                    "url": record.get(project),
                    "lead_text": None,
                    "full_text": text,
                    "article_length_words": record.get(f"{prefix}_word_count"),
                    "fetch_status": "ok",
                    "license": None,
                    "project": project,
                }
            )
    return _frame(links, LINK_COLUMNS), _frame(documents, _NORMALIZED_DOCUMENT_COLUMNS)


def _polygon_id(stem: str, record: Mapping[str, Any]) -> str:
    """Build the stable polygon identity used by the other source adapters."""
    return f"{stem}:{record['osm_type']}/{int(record['osm_id'])}"


def _pairs(value: object) -> Iterator[tuple[str, str]]:
    """Yield key/value pairs from an Arrow list-of-struct value."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        return
    yield from _valid_pairs(value)


def _valid_pairs(value: Iterable[object]) -> Iterator[tuple[str, str]]:
    """Filter an Arrow collection down to complete key/value pairs."""
    for pair in value:
        parsed = _pair(pair)
        if parsed is not None:
            yield parsed


def _pair(value: object) -> tuple[str, str] | None:
    """Parse one localized-description struct, if it has both fields."""
    if not isinstance(value, Mapping):
        return None
    if "key" not in value or "value" not in value:
        return None
    text = _usable_text(value["value"])
    return None if text is None else (str(value["key"]), text)


def _tag_key(key: str) -> str:
    """Preserve the source's localized suffix as an OSM tag key."""
    return key if key.startswith("description:") else f"description:{key}"


def _language_suffix(tag_key: str) -> str | None:
    """Return an opaque localized suffix, or null for the base description."""
    if tag_key == "description":
        return None
    return tag_key.removeprefix("description:") or None


def _usable_text(value: object) -> str | None:
    """Return trimmed non-empty source text."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def _nullable_string(value: object) -> str | None:
    """Convert nullable Arrow string values to ordinary Python strings."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return str(value)


def _region_from_stem(stem: str) -> str:
    """Match the source repositories' human-readable region convention."""
    return stem.removesuffix("-latest")


def _read(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    """Read only needed columns and tolerate old/partial local fixtures."""
    if not path.exists():
        return _empty_columns(columns)
    available = set(pq.ParquetFile(path).schema_arrow.names)
    frame = pd.read_parquet(path, columns=[column for column in columns if column in available])
    return _fill_missing(frame, columns)


def _empty_columns(columns: Sequence[str]) -> pd.DataFrame:
    """Return an empty frame with a requested schema."""
    return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})


def _fill_missing(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Add absent optional columns and apply the canonical order."""
    for missing in set(columns) - set(frame.columns):
        frame[missing] = pd.Series([None] * len(frame), dtype="object")
    return frame[list(columns)]


def _frame(rows: list[dict[str, Any]], columns: Sequence[str]) -> pd.DataFrame:
    """Return rows with a stable column order, including the empty case."""
    return pd.DataFrame(rows, columns=list(columns))


def _empty_region() -> NormalizedRegion:
    """Return empty canonical tables for a missing or empty source shard."""
    return NormalizedRegion(
        pd.DataFrame(columns=POLYGON_COLUMNS),
        pd.DataFrame(columns=LINK_COLUMNS),
        pd.DataFrame(columns=_NORMALIZED_DOCUMENT_COLUMNS),
    )
