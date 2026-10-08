"""The supported input and output dataset recipes.

Recipes provide metadata, path rules, and text-eligibility defaults.
Source-specific row normalization lives in :mod:`osm_worldcover.adapters.source`,
while the WorldCover pipeline consumes the same canonical tables for every recipe.
"""

from dataclasses import dataclass
from typing import Final, Literal

from osm_worldcover.domain.identity import HUB_NAMESPACE, WIKIDATA_OUTPUT_DATASET
from osm_worldcover.domain.text import DEFAULT_MIN_WORDS

__all__ = [
    "DEFAULT_SOURCE",
    "DESCRIPTION_DIR",
    "DOCUMENTS_DIR",
    "DOCUMENT_PROJECTS",
    "LINKS_DIR",
    "POLYGONS_DIR",
    "Layout",
    "SourceRecipe",
    "recipe_for",
]

#: The recipe name doubles as the layout that selects the table reader.
Layout = Literal["wikidata", "description", "website"]

#: Directory names inside the source repositories. Recipes and readers both use these.
POLYGONS_DIR: Final[str] = "polygons"
LINKS_DIR: Final[str] = "polygon_document_links"
DESCRIPTION_DIR: Final[str] = "data"
DOCUMENTS_DIR: Final[str] = "documents"
#: The two text corpora the Wikidata source links polygons to.
DOCUMENT_PROJECTS: Final[tuple[str, ...]] = ("wikipedia", "wikivoyage")

_HUB_DATASETS_URL: Final[str] = "https://huggingface.co/datasets/"
_PARQUET: Final[str] = "{stem}.parquet"

DEFAULT_SOURCE: Final[str] = "wikidata"


@dataclass(frozen=True, slots=True)
class SourceRecipe:
    """Immutable contract for one source repository."""

    name: Layout
    source_dataset: str
    output_dataset: str
    display_name: str
    text_description: str
    dataset_license: str
    text_license: str
    region_prefix: str
    region_paths_template: tuple[str, ...]
    min_words: int = DEFAULT_MIN_WORDS

    @property
    def source_url(self) -> str:
        """Hub URL of the source dataset."""
        return f"{_HUB_DATASETS_URL}{self.source_dataset}"

    def region_paths(self, stem: str) -> tuple[str, ...]:
        """Return the source-repository files needed for ``stem``."""
        return tuple(path.format(stem=stem) for path in self.region_paths_template)


_RECIPES: Final[dict[str, SourceRecipe]] = {
    "wikidata": SourceRecipe(
        name="wikidata",
        source_dataset=f"{HUB_NAMESPACE}/osm-polygon-wikidata-and-wikipedia",
        output_dataset=WIKIDATA_OUTPUT_DATASET,
        display_name="OSM Wikidata WorldCover",
        text_description="Wikipedia and Wikivoyage article text",
        dataset_license="cc-by-sa-4.0",
        text_license="CC BY-SA 4.0",
        region_prefix=f"{POLYGONS_DIR}/",
        region_paths_template=(
            f"{POLYGONS_DIR}/{_PARQUET}",
            f"{LINKS_DIR}/{_PARQUET}",
            *(f"{project}/{DOCUMENTS_DIR}/{_PARQUET}" for project in DOCUMENT_PROJECTS),
        ),
    ),
    "description": SourceRecipe(
        name="description",
        source_dataset=f"{HUB_NAMESPACE}/osm-polygon-description-tag",
        output_dataset=f"{HUB_NAMESPACE}/osm-polygon-description-tag-worldcover",
        display_name="OSM Description Tag WorldCover",
        text_description="OpenStreetMap description and localized-description tag text",
        dataset_license="odbl",
        text_license="Open Database License (ODbL)",
        region_prefix=f"{DESCRIPTION_DIR}/",
        region_paths_template=(f"{DESCRIPTION_DIR}/{_PARQUET}",),
        min_words=1,
    ),
    "website": SourceRecipe(
        name="website",
        source_dataset=f"{HUB_NAMESPACE}/osm-polygon-website-tag",
        output_dataset=f"{HUB_NAMESPACE}/osm-polygon-website-tag-worldcover",
        display_name="OSM Website Tag WorldCover",
        text_description="Text extracted from websites linked by OSM website tags",
        dataset_license="other",
        text_license="Third-party website text; source-site terms apply",
        region_prefix=f"{POLYGONS_DIR}/",
        region_paths_template=(f"{POLYGONS_DIR}/{_PARQUET}",),
    ),
}


def recipe_for(name: str) -> SourceRecipe:
    """Return a named recipe, with a useful error for an unknown source."""
    try:
        return _RECIPES[name]
    except KeyError as error:
        choices = ", ".join(sorted(_RECIPES))
        raise ValueError(f"unknown source {name!r}; choose one of: {choices}") from error


def source_names() -> tuple[str, ...]:
    """Return the registered source names, sorted, for CLI help."""
    return tuple(sorted(_RECIPES))
