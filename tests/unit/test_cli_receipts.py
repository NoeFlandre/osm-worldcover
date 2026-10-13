"""Receipt verification and recovery behavior for CLI assembly."""

import pandas as pd
import pytest
from tests.unit.cli_assemble_test_support import (
    _damage_receipt,
    _derived_settings_summary,
    _mixed_provenance_summary,
    _mixed_receipt_workers,
    _worker_receipt_summary,
    assemble_args,
    assembled_manifest,
    verified_shard,
)
from typer.testing import CliRunner

from osm_worldcover import cli
from osm_worldcover.accounting import BuildContext
from osm_worldcover.build import ShardStore
from osm_worldcover.config import Config
from osm_worldcover.pipeline import RegionOutcome

runner = CliRunner()


def test_assemble_derives_all_settings_from_verified_receipts(tmp_path, monkeypatch):
    directory = tmp_path / "shards"
    config = Config(
        source="description",
        source_dataset="owner/custom-description",
        source_revision="a" * 40,
        min_words=10,
        threshold=0.9,
        max_polygon_area_m2=None,
        worldcover_version="v100",
        worldcover_year=2020,
        h3_resolution=4,
        split_seed=17,
        train_ratio=0.7,
        validation_ratio=0.2,
        test_ratio=0.1,
        dataset_version="2.1.0",
        extra={"policy": "custom"},
    )
    verified_shard(directory, config=config)
    seen = []

    def inventory(repo, revision, source):
        seen.append((repo, revision, source.name))
        return ["alpha"]

    monkeypatch.setattr(cli.hub, "list_region_stems", inventory)
    outcome = runner.invoke(cli.app, assemble_args(tmp_path, directory))
    manifest = assembled_manifest(tmp_path, "2.1.0")
    assert outcome.exit_code == 0, outcome.output
    assert _derived_settings_summary(manifest, config, seen) == {
        "settings": config.as_manifest_settings(),
        "min_words": 10,
        "complete": True,
        "context": BuildContext.from_config(config).as_dict(),
        "inventory_calls": [("owner/custom-description", "a" * 40, "description")],
        "expected_context": BuildContext.from_config(config).as_dict(),
    }


def test_assemble_aggregates_worker_receipts_including_empty_regions(tmp_path, monkeypatch):
    first, second = tmp_path / "w0" / "shards", tmp_path / "w1" / "shards"
    config = verified_shard(first)
    verified_shard(second, stem="beta", config=config)
    ShardStore(second, BuildContext.from_config(config)).write_outcome(
        "empty", pd.DataFrame(), RegionOutcome("empty")
    )
    # Receipt counters are authoritative; stale generic sidecars must never win.
    (first / "alpha.rejections.json").write_text('{"too_large": 999}')
    monkeypatch.setattr(cli.hub, "list_region_stems", lambda *a: ["alpha", "beta", "empty"])
    outcome = runner.invoke(cli.app, assemble_args(tmp_path, first, second))
    manifest = assembled_manifest(tmp_path)
    assert outcome.exit_code == 0, outcome.output
    assert _worker_receipt_summary(manifest) == {
        "processed_regions": ["alpha", "beta", "empty"],
        "complete": True,
        "examples": 4,
        "polygons_seen": 14,
        "text_rejections": {"empty_text": 2},
        "rejections": {"below_threshold": 6},
    }


def test_assemble_preserves_mixed_region_code_pins(tmp_path, monkeypatch):
    config, first, second = _mixed_receipt_workers(tmp_path, monkeypatch)
    unpinned = runner.invoke(cli.app, assemble_args(tmp_path, first, second))
    assert (
        unpinned.exit_code,
        "provide --legacy-code-revision" in unpinned.output,
        (tmp_path / "out").exists(),
    ) == (1, True, False)
    outcome = runner.invoke(
        cli.app,
        [
            *assemble_args(tmp_path, first, second),
            "--legacy-code-revision",
            "a" * 40,
        ],
    )
    ledger = assembled_manifest(tmp_path)["processing"]
    assert outcome.exit_code == 0, outcome.output
    assert _mixed_provenance_summary(ledger) == {
        "schema": 2,
        "provenance": [
            {
                "repository": config.as_manifest_settings()["code_repository"],
                "revision": "a" * 40,
                "regions": ["alpha"],
            },
            {
                "repository": config.as_manifest_settings()["code_repository"],
                "revision": "b" * 40,
                "regions": ["beta"],
            },
        ],
        "assembly": "c" * 40,
    }


@pytest.mark.parametrize(
    "option,value",
    [
        ("source", "wikidata"),
        ("threshold", "0.7"),
        ("revision", "b" * 40),
    ],
)
def test_assemble_refuses_explicit_conflicting_settings(tmp_path, monkeypatch, option, value):
    directory = tmp_path / "shards"
    verified_shard(directory)
    monkeypatch.setattr(cli.hub, "list_region_stems", lambda *a: pytest.fail("no network"))
    outcome = runner.invoke(cli.app, [*assemble_args(tmp_path, directory), f"--{option}", value])
    assert outcome.exit_code == 1
    assert "conflicts with verified receipt setting" in outcome.output
    assert not (tmp_path / "out").exists()


def test_assemble_accepts_matching_explicit_settings(tmp_path, monkeypatch):
    directory = tmp_path / "shards"
    verified_shard(directory)
    monkeypatch.setattr(cli.hub, "list_region_stems", lambda *a: ["alpha"])
    outcome = runner.invoke(
        cli.app,
        [
            *assemble_args(tmp_path, directory),
            "--source",
            "description",
            "--threshold",
            "0.8",
            "--revision",
            "a" * 40,
            "--dataset-version",
            "1.1.0",
        ],
    )
    assert outcome.exit_code == 0, outcome.output


@pytest.mark.parametrize(
    "change",
    [
        {"min_words": 10},
        {"source": "website"},
        {"source_revision": "b" * 40},
        {"max_polygon_area_m2": None},
        {"worldcover_year": 2020},
        {"split_seed": 27},
        {"extra": {"policy": "changed"}},
    ],
)
def test_assemble_refuses_workers_with_different_contexts(tmp_path, monkeypatch, change):
    first, second = tmp_path / "w0", tmp_path / "w1"
    config = verified_shard(first)
    from dataclasses import replace

    verified_shard(second, "beta", replace(config, **change))
    monkeypatch.setattr(cli.hub, "list_region_stems", lambda *a: pytest.fail("no network"))
    outcome = runner.invoke(cli.app, assemble_args(tmp_path, first, second))
    assert outcome.exit_code == 1
    assert "unverifiable region completion receipts" in outcome.output
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("repeat_directory", [False, True])
def test_assemble_rejects_duplicate_worker_assignments(tmp_path, monkeypatch, repeat_directory):
    first, second = tmp_path / "w0", tmp_path / "w1"
    verified_shard(first)
    verified_shard(second)
    monkeypatch.setattr(cli.hub, "list_region_stems", lambda *a: pytest.fail("no network"))
    outcome = runner.invoke(
        cli.app, assemble_args(tmp_path, first, first if repeat_directory else second)
    )
    assert outcome.exit_code == 1
    assert "duplicate region worker assignments" in outcome.output
    assert not (tmp_path / "out").exists()


def test_assemble_marks_subset_as_incomplete(tmp_path, monkeypatch):
    directory = tmp_path / "shards"
    verified_shard(directory)
    monkeypatch.setattr(cli.hub, "list_region_stems", lambda *a: ["alpha", "beta"])
    outcome = runner.invoke(cli.app, assemble_args(tmp_path, directory))
    ledger = assembled_manifest(tmp_path)["processing"]
    assert (
        outcome.exit_code,
        ledger["selected_complete"],
        ledger["full_source_complete"],
        ledger["missing_regions"],
        "incomplete source inventory" in outcome.output,
    ) == (0, True, False, ["beta"], True)


def test_assemble_rejects_regions_outside_pinned_inventory(tmp_path, monkeypatch):
    directory = tmp_path / "shards"
    verified_shard(directory)
    monkeypatch.setattr(cli.hub, "list_region_stems", lambda *a: ["other"])
    outcome = runner.invoke(cli.app, assemble_args(tmp_path, directory))
    assert outcome.exit_code == 1
    assert "unknown selected regions" in outcome.output


@pytest.mark.parametrize("damage", ["receipt", "shard", "bytes", "orphan", "schema"])
def test_assemble_rejects_damaged_or_incomplete_receipts(tmp_path, monkeypatch, damage):
    directory = tmp_path / "shards"
    verified_shard(directory)
    receipt = directory / "alpha.complete.json"
    shard = directory / "alpha.parquet"
    _damage_receipt(receipt, shard, directory, damage)
    monkeypatch.setattr(cli.hub, "list_region_stems", lambda *a: pytest.fail("no network"))
    outcome = runner.invoke(cli.app, assemble_args(tmp_path, directory))
    assert outcome.exit_code == 1
    assert "unverifiable" in outcome.output
    assert not (tmp_path / "out").exists()
