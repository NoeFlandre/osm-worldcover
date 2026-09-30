"""Run configuration loading and overrides."""

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
        Config.from_mapping({"min_words": minimum})


@pytest.mark.parametrize("setting, expected", [("null", 1), ("4", 4), ("10", 10)])
def test_yaml_preserves_source_text_policy(tmp_path: Path, setting: str, expected: int) -> None:
    path = tmp_path / "description.yaml"
    path.write_text(f"source: description\nmin_words: {setting}\n")
    assert Config.from_yaml(path).effective_min_words == expected


def test_from_mapping_coerces_paths() -> None:
    config = Config.from_mapping({"out_dir": "a/b", "cache_dir": "c"})
    assert config.out_dir == Path("a/b")
    assert config.cache_dir == Path("c")


def test_from_mapping_coerces_regions_to_a_tuple() -> None:
    assert Config.from_mapping({"regions": ["a", "b"]}).regions == ("a", "b")


def test_from_mapping_keeps_regions_none() -> None:
    assert Config.from_mapping({"regions": None}).regions is None


def test_from_mapping_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="unknown config keys"):
        Config.from_mapping({"nope": 1})


def test_from_yaml_reads_a_file(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("threshold: 0.9\nregions: [luxembourg-latest]\n")
    config = Config.from_yaml(path)
    assert config.threshold == 0.9
    assert config.regions == ("luxembourg-latest",)


def test_from_yaml_accepts_an_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text("")
    assert Config.from_yaml(path).threshold == 0.8


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
