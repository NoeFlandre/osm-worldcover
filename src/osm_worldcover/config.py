"""Run configuration.

Every knob that changes the published data lives here and is copied into the
manifest, so a dataset can always be traced back to the settings that made it.
"""

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Self

from osm_worldcover.adapters.worldcover import DEFAULT_CACHED_TILES
from osm_worldcover.domain.dominance import DEFAULT_THRESHOLD
from osm_worldcover.domain.splits import DEFAULT_RATIOS, DEFAULT_RESOLUTION, DEFAULT_SEED
from osm_worldcover.sources import DEFAULT_SOURCE, SourceRecipe, recipe_for

__all__ = [
    "DEFAULT_CACHED_TILES",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_DATASET_VERSION",
    "DEFAULT_MAX_POLYGON_AREA_KM2",
    "DEFAULT_OUT_DIR",
    "DEFAULT_SOURCE",
    "DEFAULT_SOURCE_DATASET",
    "DEFAULT_THRESHOLD",
    "Config",
]

DEFAULT_SOURCE_DATASET = recipe_for(DEFAULT_SOURCE).source_dataset

DEDUPLICATION_POLICY = "polygon_id+normalized_text+worldcover_code"

#: Polygons larger than this are refused before any raster is read.
#: Zonal-statistics cost is linear in area: 10,000 km2 is ~10^8 pixels and
#: about 1.4 s, while the largest polygon in the source -- 10.2 million km2 --
#: would need ~100 tiles and 10^11 pixels for a single row. Capping here bounds
#: the worst case to roughly a second and costs 1,099 of 1,259,424 polygons
#: (0.087%), which are provinces, countries and continents whose articles
#: describe history and governance rather than the ground beneath them.
DEFAULT_MAX_POLYGON_AREA_KM2 = 10_000.0
M2_PER_KM2 = 1e6
DEFAULT_MAX_POLYGON_AREA_M2 = DEFAULT_MAX_POLYGON_AREA_KM2 * M2_PER_KM2

DEFAULT_OUT_DIR = Path("data/out")
DEFAULT_CACHE_DIR = Path("data/cache")
DEFAULT_DATASET_VERSION = "1.1.0"

#: Equal-area projection used whenever a real-world area is needed.
EQUAL_AREA_CRS = "EPSG:6933"
CODE_REPOSITORY = "https://github.com/NoeFlandre/osm-worldcover"


@dataclass(frozen=True, slots=True)
class Config:
    """Settings for one dataset build."""

    out_dir: Path = DEFAULT_OUT_DIR
    cache_dir: Path = DEFAULT_CACHE_DIR

    worldcover_version: str = "v200"
    worldcover_year: int = 2021
    cached_tiles: int = DEFAULT_CACHED_TILES

    source: str = DEFAULT_SOURCE
    source_dataset: str = DEFAULT_SOURCE_DATASET
    source_revision: str | None = None
    regions: tuple[str, ...] | None = None

    threshold: float = DEFAULT_THRESHOLD
    max_polygon_area_m2: float | None = DEFAULT_MAX_POLYGON_AREA_M2
    # None selects the source recipe's policy; an explicit integer overrides it.
    min_words: int | None = None
    h3_resolution: int = DEFAULT_RESOLUTION
    split_seed: int = DEFAULT_SEED
    train_ratio: float = DEFAULT_RATIOS.train
    validation_ratio: float = DEFAULT_RATIOS.validation
    test_ratio: float = DEFAULT_RATIOS.test

    dataset_version: str = DEFAULT_DATASET_VERSION
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Resolve the named recipe while retaining custom repository overrides."""
        recipe_for(self.source)
        if self.source != DEFAULT_SOURCE and self.source_dataset == DEFAULT_SOURCE_DATASET:
            object.__setattr__(self, "source_dataset", recipe_for(self.source).source_dataset)
        if self.min_words is not None:
            _validate_min_words(self.min_words)

    @classmethod
    def from_cli(cls, *, max_area_km2: float, **options: Any) -> Self:
        """Build from command-line options, converting the area from km2 to m2."""
        return cls(max_polygon_area_m2=max_area_km2 * M2_PER_KM2, **options)

    @property
    def source_recipe(self) -> SourceRecipe:
        """Return the immutable metadata for this build's source."""
        return recipe_for(self.source)

    @property
    def effective_min_words(self) -> int:
        """Resolve a text-length override or the source-specific default."""
        return self.source_recipe.min_words if self.min_words is None else self.min_words

    def with_overrides(self, **over: Any) -> Self:
        """Return a copy with ``over`` applied, ignoring ``None`` values."""
        return replace(self, **{k: v for k, v in over.items() if v is not None})

    def as_manifest_settings(self) -> dict[str, Any]:
        """The subset of settings recorded in the manifest."""
        return {
            "dataset_version": self.dataset_version,
            "deduplication_policy": DEDUPLICATION_POLICY,
            "source": self.source,
            "worldcover_version": self.worldcover_version,
            "worldcover_year": self.worldcover_year,
            "source_dataset": self.source_dataset,
            "source_revision": self.source_revision,
            "source_url": self.source_recipe.source_url,
            "code_repository": CODE_REPOSITORY,
            "source_display_name": self.source_recipe.display_name,
            "source_text_description": self.source_recipe.text_description,
            "output_dataset": self.source_recipe.output_dataset,
            "dataset_license": self.source_recipe.dataset_license,
            "text_license": self.source_recipe.text_license,
            "dominance_threshold": self.threshold,
            "max_polygon_area_m2": self.max_polygon_area_m2,
            "min_words": self.effective_min_words,
            "h3_resolution": self.h3_resolution,
            "split_seed": self.split_seed,
            "split_ratios": {
                "train": self.train_ratio,
                "validation": self.validation_ratio,
                "test": self.test_ratio,
            },
            "equal_area_crs": EQUAL_AREA_CRS,
        }


def _validate_min_words(value: int) -> None:
    """Require a positive integer so an override can never admit empty text."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("min_words must be a positive integer or null for the source default")
