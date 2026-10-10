"""Read the source dataset's tables.

The source publishes one Parquet file per Geofabrik region for each table, so
a region is the natural unit of work: its polygons, its polygon-document links
and its documents are read together and never need a global join.

Files are addressed on a local snapshot directory. Downloading that snapshot is
a separate concern (see :mod:`osm_worldcover.adapters.hub`), which
keeps this module usable against a fixture directory in tests.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Self

import pandas as pd
import pyarrow.parquet as pq

from osm_worldcover.adapters.source_profiles import (
    DOCUMENT_COLUMNS,
    LINK_COLUMNS,
    POLYGON_COLUMNS,
    NormalizedRegion,
    load_description_region,
    load_website_region,
)
from osm_worldcover.sources import (
    DEFAULT_SOURCE,
    DOCUMENT_PROJECTS,
    DOCUMENTS_DIR,
    LINKS_DIR,
    POLYGONS_DIR,
    Layout,
    SourceRecipe,
    recipe_for,
)

__all__ = [
    "DOCUMENT_COLUMNS",
    "PROJECTS",
    "RegionTables",
    "load_documents",
    "load_links",
    "load_polygons",
    "region_stems",
]

#: The two text corpora the source links polygons to.
PROJECTS: Final[tuple[str, ...]] = DOCUMENT_PROJECTS


def region_stems(root: Path, source: str | SourceRecipe = DEFAULT_SOURCE) -> list[str]:
    """Return every region name in ``root``, sorted for reproducible iteration."""
    recipe = _recipe(source)
    directory = root / recipe.region_prefix.removesuffix("/")
    return sorted(p.stem for p in directory.glob("*.parquet"))


def load_polygons(root: Path, stem: str) -> pd.DataFrame:
    """Load one region's polygon table."""
    return _read(root / POLYGONS_DIR / f"{stem}.parquet", POLYGON_COLUMNS)


def load_links(root: Path, stem: str) -> pd.DataFrame:
    """Load one region's polygon-document links."""
    return _read(root / LINKS_DIR / f"{stem}.parquet", LINK_COLUMNS)


def load_documents(root: Path, stem: str, project: str) -> pd.DataFrame:
    """Load one region's documents for ``project``.

    Returns an empty, correctly shaped frame when the region has no sidecar for
    that project, which is the normal case for Wikivoyage.
    """
    return _read(root / project / DOCUMENTS_DIR / f"{stem}.parquet", DOCUMENT_COLUMNS)


@dataclass(frozen=True, slots=True)
class RegionTables:
    """One region's three tables, read together."""

    stem: str
    polygons: pd.DataFrame
    links: pd.DataFrame
    documents: pd.DataFrame

    @classmethod
    def load(cls, root: Path, stem: str, source: str | SourceRecipe = DEFAULT_SOURCE) -> Self:
        """Read every table for ``stem``, with both text projects concatenated."""
        recipe = _recipe(source)
        try:
            loader = _LOADERS[recipe.name]
        except KeyError:
            raise ValueError(f"no table loader for source layout {recipe.name!r}") from None
        polygons, links, documents = loader(root, stem)
        return cls(stem, polygons, links, documents)


_Tables = tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]


def _load_wikidata(root: Path, stem: str) -> _Tables:
    """Read the Wikidata layout, concatenating both text projects."""
    frames = []
    for project in PROJECTS:
        frame = load_documents(root, stem, project)
        frame["project"] = project
        frames.append(frame)
    return (
        load_polygons(root, stem),
        load_links(root, stem),
        pd.concat(frames, ignore_index=True),
    )


def _normalized(loader: Callable[[Path, str], NormalizedRegion]) -> Callable[[Path, str], _Tables]:
    """Adapt a source-profile loader to the common table-triple shape."""

    def load(root: Path, stem: str) -> _Tables:
        region = loader(root, stem)
        return region.polygons, region.links, region.documents

    return load


_LOADERS: Final[dict[Layout, Callable[[Path, str], _Tables]]] = {
    "wikidata": _load_wikidata,
    "description": _normalized(load_description_region),
    "website": _normalized(load_website_region),
}


def _recipe(source: str | SourceRecipe) -> SourceRecipe:
    """Accept a recipe object at internal seams and a name at public seams."""
    return source if isinstance(source, SourceRecipe) else recipe_for(source)


def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read ``columns`` from ``path``, tolerating a missing file or column."""
    if not path.exists():
        return _empty(columns)
    available = set(pq.ParquetFile(path).schema_arrow.names)
    frame = pd.read_parquet(path, columns=[c for c in columns if c in available])
    return _with_all_columns(frame, columns)


def _empty(columns: list[str]) -> pd.DataFrame:
    """An empty frame carrying the expected columns, so callers need no special case."""
    return pd.DataFrame({name: pd.Series(dtype="object") for name in columns})


def _with_all_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Fill in any column the source omitted, then project onto ``columns``."""
    for missing in set(columns) - set(frame.columns):
        frame[missing] = pd.Series([None] * len(frame), dtype="object")
    return frame[columns]
