"""The supported input and output dataset recipes.

Recipes provide metadata, path rules, and text-eligibility defaults.
Source-specific row normalization lives in :mod:`osm_worldcover.adapters.source`,
while the WorldCover pipeline consumes the same canonical tables for every recipe.
"""

from dataclasses import dataclass
from typing import Final

from osm_worldcover.domain.text import DEFAULT_MIN_WORDS

__all__ = ["DEFAULT_SOURCE", "SourceRecipe", "recipe_for"]

DEFAULT_SOURCE: Final[str] = "wikidata"


@dataclass(frozen=True, slots=True)
class SourceRecipe:
    """Immutable contract for one source repository."""

    name: str
    source_dataset: str
    output_dataset: str
    source_url: str
    display_name: str
    layout: str
    text_description: str
    dataset_license: str
    text_license: str
    region_prefix: str
    region_paths_template: tuple[str, ...]
    min_words: int = DEFAULT_MIN_WORDS

    def region_paths(self, stem: str) -> tuple[str, ...]:
        """Return the source-repository files needed for ``stem``."""
        return tuple(path.format(stem=stem) for path in self.region_paths_template)


_RECIPES: Final[dict[str, SourceRecipe]] = {
    "wikidata": SourceRecipe(
        name="wikidata",
        source_dataset="NoeFlandre/osm-polygon-wikidata-and-wikipedia",
        output_dataset="NoeFlandre/osm-wikidata-worldcover",
        source_url="https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-and-wikipedia",
        display_name="OSM Wikidata WorldCover",
        layout="wikidata",
        text_description="Wikipedia and Wikivoyage article text",
        dataset_license="cc-by-sa-4.0",
        text_license="CC BY-SA 4.0",
        region_prefix="polygons/",
        region_paths_template=(
            "polygons/{stem}.parquet",
            "polygon_document_links/{stem}.parquet",
            "wikipedia/documents/{stem}.parquet",
            "wikivoyage/documents/{stem}.parquet",
        ),
    ),
    "description": SourceRecipe(
        name="description",
        source_dataset="NoeFlandre/osm-polygon-description-tag",
        output_dataset="NoeFlandre/osm-polygon-description-tag-worldcover",
        source_url="https://huggingface.co/datasets/NoeFlandre/osm-polygon-description-tag",
        display_name="OSM Description Tag WorldCover",
        layout="description",
        text_description="OpenStreetMap description and localized-description tag text",
        dataset_license="odbl",
        text_license="Open Database License (ODbL)",
        region_prefix="data/",
        region_paths_template=("data/{stem}.parquet",),
        min_words=1,
    ),
    "website": SourceRecipe(
        name="website",
        source_dataset="NoeFlandre/osm-polygon-website-tag",
        output_dataset="NoeFlandre/osm-polygon-website-tag-worldcover",
        source_url="https://huggingface.co/datasets/NoeFlandre/osm-polygon-website-tag",
        display_name="OSM Website Tag WorldCover",
        layout="website",
        text_description="Text extracted from websites linked by OSM website tags",
        dataset_license="other",
        text_license="Third-party website text; source-site terms apply",
        region_prefix="polygons/",
        region_paths_template=("polygons/{stem}.parquet",),
    ),
}


def recipe_for(name: str) -> SourceRecipe:
    """Return a named recipe, with a useful error for an unknown source."""
    try:
        return _RECIPES[name]
    except KeyError as error:
        choices = ", ".join(sorted(_RECIPES))
        raise ValueError(f"unknown source {name!r}; choose one of: {choices}") from error
