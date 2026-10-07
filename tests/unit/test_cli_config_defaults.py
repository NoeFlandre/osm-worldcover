"""The CLI and Config declare each default once; these tests lock them together."""

from pathlib import Path

import pytest
import typer.main

from osm_worldcover.cli import _build_config, app
from osm_worldcover.config import Config


def _defaults(command: str) -> dict[str, object]:
    group = typer.main.get_command(app)
    return {param.name: param.default for param in group.commands[command].params}


@pytest.mark.parametrize(
    ("option", "field"),
    [
        ("source", "source"),
        ("out", "out_dir"),
        ("cache", "cache_dir"),
        ("threshold", "threshold"),
        ("cached_tiles", "cached_tiles"),
        ("dataset_version", "dataset_version"),
    ],
)
def test_build_option_defaults_equal_config_defaults(option, field) -> None:
    assert _defaults("build")[option] == getattr(Config(), field)


def test_build_area_default_is_config_area_in_km2() -> None:
    assert _defaults("build")["max_area_km2"] * 1e6 == Config().max_polygon_area_m2


def test_verify_threshold_defaults_to_the_manifest_then_config() -> None:
    """None means "read it from the build's manifest"; see test_cli for the fallback."""
    assert _defaults("verify")["threshold"] is None


def test_effective_defaults_are_unchanged() -> None:
    config = _build_config(
        "wikidata", Path("data/out"), Path("data/cache"), 0.8, 10_000.0, 8, None, "1.1.0"
    )

    assert config == Config()
    assert (config.max_polygon_area_m2, config.cached_tiles, config.dataset_version) == (
        1e10,
        8,
        "1.1.0",
    )


def test_from_cli_converts_square_kilometres_to_square_metres() -> None:
    config = Config.from_cli(max_area_km2=2.5, threshold=0.9)

    assert (config.max_polygon_area_m2, config.threshold) == (2_500_000.0, 0.9)


def test_assemble_defaults_derive_from_config_and_are_unchanged() -> None:
    defaults = _defaults("assemble")

    assert defaults["out"] == Config().out_dir == Path("data/out")
    assert defaults["work"] == Config().cache_dir / "assembly" == Path("data/cache/assembly")
