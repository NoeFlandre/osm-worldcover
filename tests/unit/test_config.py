"""Run configuration loading and overrides."""

import dataclasses
from pathlib import Path

import pytest

from osm_worldcover.config import Config


def test_defaults_are_usable_without_arguments() -> None:
    config = Config()
    assert config.threshold == 0.8
    assert config.worldcover_version == "v200"


def test_named_source_selects_its_input_dataset() -> None:
    config = Config(source="website")
    assert config.source_dataset == "NoeFlandre/osm-polygon-website-tag"
    assert config.source_recipe.output_dataset == "NoeFlandre/osm-polygon-website-tag-worldcover"


@pytest.mark.parametrize("source, minimum", [("description", 1), ("wikidata", 10), ("website", 10)])
def test_source_specific_text_defaults_are_recorded(source: str, minimum: int) -> None:
    config = Config(source=source)
    assert config.effective_min_words == minimum
    assert config.as_manifest_settings()["min_words"] == minimum


@pytest.mark.parametrize("source", ["description", "wikidata", "website"])
@pytest.mark.parametrize("minimum", [1, 4, 10, 20])
def test_explicit_text_minimum_overrides_recipe(source: str, minimum: int) -> None:
    config = Config(source=source, min_words=minimum)
    assert config.effective_min_words == minimum
    assert config.as_manifest_settings()["min_words"] == minimum


def test_overriding_source_resolves_its_text_default() -> None:
    config = Config().with_overrides(source="description")
    assert config.effective_min_words == 1
    assert config.with_overrides(source="website").effective_min_words == 10


def test_overriding_source_preserves_an_explicit_text_minimum() -> None:
    config = Config(min_words=10).with_overrides(source="description")
    assert config.effective_min_words == 10
    assert config.with_overrides(min_words=None).effective_min_words == 10
    assert config.with_overrides(min_words=4).effective_min_words == 4


@pytest.mark.parametrize("minimum", [0, -1, 1.5, "10", True, False])
def test_text_minimum_must_be_a_positive_integer(minimum: object) -> None:
    with pytest.raises(ValueError, match="min_words must be a positive integer"):
        Config(min_words=minimum)


def test_overrides_ignore_none_so_unset_cli_flags_do_not_clobber() -> None:
    config = Config(threshold=0.9).with_overrides(threshold=None, source_revision="abc")
    assert config.threshold == 0.9
    assert config.source_revision == "abc"


def test_manifest_settings_expose_every_knob_that_changes_the_data() -> None:
    settings = Config(source_revision="r1").as_manifest_settings()
    assert settings["source_revision"] == "r1"
    assert settings["dominance_threshold"] == 0.8
    assert settings["split_ratios"] == {"train": 0.8, "validation": 0.1, "test": 0.1}
    assert settings["worldcover_version"] == "v200"


def test_invalid_min_words_message_names_the_constraint() -> None:
    with pytest.raises(
        ValueError, match=r"^min_words must be a positive integer or null for the source default$"
    ):
        Config(min_words=0)


def test_defaults_pin_the_worldcover_year_and_the_ten_thousand_km2_cap() -> None:
    config = Config()
    assert config.worldcover_year == 2021
    assert config.max_polygon_area_m2 == 1e10


def test_config_is_an_immutable_value_object() -> None:
    config = Config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.threshold = 0.5  # type: ignore[misc]
    assert not hasattr(config, "__dict__")


def test_explicit_source_dataset_survives_a_named_source() -> None:
    config = Config(source="website", source_dataset="example/custom-dataset")
    assert config.source_dataset == "example/custom-dataset"


@pytest.mark.parametrize("length, accepted", [(64, True), (65, False)])
def test_dataset_version_is_limited_to_sixty_four_characters(length: int, accepted: bool) -> None:
    version = "a" * length
    if accepted:
        assert Config(dataset_version=version).dataset_version == version
    else:
        with pytest.raises(ValueError, match="invalid dataset version"):
            Config(dataset_version=version)


MANIFEST_SETTING_KEYS = {
    "dataset_version",
    "deduplication_policy",
    "source",
    "worldcover_version",
    "worldcover_year",
    "source_dataset",
    "source_revision",
    "source_url",
    "code_repository",
    "source_display_name",
    "source_text_description",
    "output_dataset",
    "dataset_license",
    "text_license",
    "dominance_threshold",
    "max_polygon_area_m2",
    "min_words",
    "h3_resolution",
    "split_seed",
    "split_ratios",
    "equal_area_crs",
}


def test_manifest_settings_name_every_recorded_knob_exactly() -> None:
    settings = Config().as_manifest_settings()
    assert set(settings) == MANIFEST_SETTING_KEYS
    assert settings["deduplication_policy"] == "polygon_id+normalized_text+worldcover_code"
    assert settings["equal_area_crs"] == "EPSG:6933"


def test_default_paths_and_versions_are_pinned() -> None:
    config = Config()
    assert config.out_dir == Path("data/out")
    assert config.cache_dir == Path("data/cache")
    assert config.dataset_version == "1.1.0"
    assert config.worldcover_version == "v200"


def test_public_surface_is_pinned() -> None:
    from osm_worldcover import config

    assert config.__all__ == [
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
