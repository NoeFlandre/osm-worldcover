"""Build verification CLI tests."""

import json

import pytest
from tests.cli_test_helpers import frame
from tests.unit.cli_test_support import MANIFEST, _seed_release_lock, write_build
from typer.testing import CliRunner

from osm_worldcover import cli

runner = CliRunner()


def test_verify_accepts_a_sound_build(tmp_path) -> None:
    build = tmp_path / "v1.0.0"
    build.mkdir(parents=True)
    for split in ("train", "validation", "test"):
        frame()[frame()["split"] == split].to_parquet(build / f"{split}.parquet", index=False)
    _seed_release_lock(build)
    outcome = runner.invoke(cli.app, ["verify", str(build)])
    assert outcome.exit_code == 0
    assert "every guarantee holds" in outcome.output


@pytest.mark.parametrize("minimum, expected_exit", [(1, 0), (4, 0), (5, 1), (10, 1)])
def test_verify_uses_the_manifest_text_threshold(tmp_path, minimum, expected_exit) -> None:
    build = tmp_path / "v1.0.0"
    build.mkdir()
    rows = frame(1)
    rows["text"] = ["Small public wooded garden"]
    rows.to_parquet(build / "train.parquet", index=False)
    (build / "manifest.json").write_text(
        json.dumps({"settings": {"source": "description", "min_words": minimum}})
    )
    _seed_release_lock(build)

    outcome = runner.invoke(cli.app, ["verify", str(build)])

    assert outcome.exit_code == expected_exit, outcome.output
    assert ("every guarantee holds" in outcome.output) == (expected_exit == 0)
    assert ("unusable_text" in outcome.output) == (expected_exit == 1)


def test_verify_uses_ten_words_for_legacy_manifest_without_settings(tmp_path) -> None:
    build = tmp_path / "v1.0.0"
    build.mkdir()
    rows = frame(1)
    rows["text"] = ["Small public wooded garden"]
    rows.to_parquet(build / "train.parquet", index=False)
    (build / "manifest.json").write_text(json.dumps(MANIFEST))
    _seed_release_lock(build)

    outcome = runner.invoke(cli.app, ["verify", str(build)])

    assert outcome.exit_code == 1
    assert "unusable_text" in outcome.output


def test_verify_rejects_a_build_that_breaks_a_guarantee(tmp_path) -> None:
    build = tmp_path / "v1.0.0"
    build.mkdir(parents=True)
    bad = frame(2)
    bad.loc[:, "dominant_fraction"] = 0.4
    bad.to_parquet(build / "train.parquet", index=False)
    _seed_release_lock(build)
    outcome = runner.invoke(cli.app, ["verify", str(build)])
    assert outcome.exit_code == 1
    assert outcome.output.splitlines() == ["rows: 2", "FAILED below_threshold: 2 ('p0', 'p1')"]


@pytest.mark.parametrize(
    "code",
    [10.5, 10.0, None, float("nan"), float("inf"), float("-inf"), "10", "bad", True],
)
def test_verify_rejects_non_integer_worldcover_codes(tmp_path, code) -> None:
    build = tmp_path / "v1.0.0"
    build.mkdir()
    rows = frame(1)
    rows["worldcover_code"] = [code]
    rows.to_parquet(build / "train.parquet", index=False)
    _seed_release_lock(build)

    outcome = runner.invoke(cli.app, ["verify", str(build)])

    assert outcome.exit_code == 1, outcome.output
    assert "FAILED invalid_label: 1 ('p0',)" in outcome.output
    assert "every guarantee holds" not in outcome.output


def test_verify_enforces_the_manifest_dominance_threshold(tmp_path) -> None:
    rows = frame(2)
    rows["dominant_fraction"] = 0.85
    build = write_build(tmp_path, rows, {"dominance_threshold": 0.9})

    outcome = runner.invoke(cli.app, ["verify", str(build)])

    assert outcome.exit_code == 1, outcome.output
    assert "below_threshold" in outcome.output


def test_verify_accepts_rows_meeting_the_manifest_dominance_threshold(tmp_path) -> None:
    rows = frame(2)
    rows["dominant_fraction"] = 0.85
    build = write_build(tmp_path, rows, {"dominance_threshold": 0.8})

    outcome = runner.invoke(cli.app, ["verify", str(build)])

    assert outcome.exit_code == 0, outcome.output


def test_an_explicit_threshold_overrides_the_manifest(tmp_path) -> None:
    rows = frame(2)
    rows["dominant_fraction"] = 0.85
    build = write_build(tmp_path, rows, {"dominance_threshold": 0.9})

    outcome = runner.invoke(cli.app, ["verify", str(build), "--threshold", "0.8"])

    assert outcome.exit_code == 0, outcome.output


def test_verify_falls_back_to_the_default_threshold_for_legacy_manifests(tmp_path) -> None:
    rows = frame(2)
    rows["dominant_fraction"] = 0.79
    build = write_build(tmp_path, rows, {"min_words": 10})

    outcome = runner.invoke(cli.app, ["verify", str(build)])

    assert outcome.exit_code == 1, outcome.output
    assert "below_threshold" in outcome.output


@pytest.mark.parametrize(
    "bad_threshold",
    [-1, 0, 1.5, float("nan"), float("inf"), float("-inf"), None, "0.8", True, [0.8], {"a": 1}],
)
def test_verify_rejects_an_invalid_manifest_dominance_threshold(tmp_path, bad_threshold) -> None:
    rows = frame(2)
    rows["dominant_fraction"] = 0.1
    build = write_build(tmp_path, rows, {"dominance_threshold": bad_threshold})

    outcome = runner.invoke(cli.app, ["verify", str(build)])

    assert outcome.exit_code == 1, outcome.output
    assert "dominance threshold" in outcome.output
    assert "every guarantee holds" not in outcome.output


@pytest.mark.parametrize("bad_threshold", ["nan", "inf", "0", "-0.5", "1.2"])
def test_verify_rejects_an_invalid_explicit_dominance_threshold(tmp_path, bad_threshold) -> None:
    rows = frame(2)
    rows["dominant_fraction"] = 0.1
    build = write_build(tmp_path, rows, {"dominance_threshold": 0.8})

    outcome = runner.invoke(cli.app, ["verify", str(build), "--threshold", bad_threshold])

    assert outcome.exit_code == 1, outcome.output
    assert "dominance threshold" in outcome.output
    assert "every guarantee holds" not in outcome.output


@pytest.mark.parametrize("bad_fraction", [float("nan"), 1.5, -0.1])
def test_verify_rejects_a_fraction_outside_zero_to_one(tmp_path, bad_fraction) -> None:
    rows = frame(2)
    rows["dominant_fraction"] = bad_fraction
    build = write_build(tmp_path, rows)

    outcome = runner.invoke(cli.app, ["verify", str(build)])

    assert outcome.exit_code == 1, outcome.output
    assert "invalid_fraction" in outcome.output
    assert "OK" not in outcome.output


def test_verify_refuses_a_directory_with_no_splits(tmp_path) -> None:
    _seed_release_lock(tmp_path)
    outcome = runner.invoke(cli.app, ["verify", str(tmp_path)])
    assert outcome.exit_code == 1
    assert "no splits" in outcome.output
