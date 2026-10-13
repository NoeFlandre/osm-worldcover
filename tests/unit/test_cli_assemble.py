"""Basic and legacy command-line assembly behavior."""

import json

import pandas as pd
import pytest
from tests.unit.cli_assemble_test_support import (
    _legacy_reuse_summary,
    _recovery_status,
    assemble_args,
    assembled_manifest,
    shard_file,
    verified_shard,
)
from typer.testing import CliRunner

from osm_worldcover import cli
from osm_worldcover.accounting import BuildContext
from osm_worldcover.config import Config

runner = CliRunner()


def test_assemble_turns_shards_into_a_dataset(tmp_path) -> None:
    shards = tmp_path / "shards"
    shards.mkdir()
    shard_file(shards / "a.parquet")
    outcome = runner.invoke(
        cli.app,
        [
            "assemble",
            "--allow-unverified-shards",
            str(shards),
            "--out",
            str(tmp_path / "out"),
            "--work",
            str(tmp_path / "work"),
        ],
    )
    assert outcome.exit_code == 0, outcome.output
    assert (tmp_path / "out" / "v1.1.0" / "train.parquet").exists()
    assert (tmp_path / "out" / "v1.1.0" / "manifest.json").exists()


def test_assemble_accepts_several_shard_directories(tmp_path) -> None:
    """A build split across workers leaves one shard directory per worker."""
    dirs = []
    for name in ("w0", "w1"):
        d = tmp_path / name
        d.mkdir()
        shard_file(d / f"{name}.parquet", n=2)
        dirs.append(str(d))
    outcome = runner.invoke(
        cli.app,
        [
            "assemble",
            "--allow-unverified-shards",
            *dirs,
            "--out",
            str(tmp_path / "out"),
            "--work",
            str(tmp_path / "work"),
        ],
    )
    assert outcome.exit_code == 0, outcome.output
    train = pd.read_parquet(tmp_path / "out" / "v1.1.0" / "train.parquet")
    assert len(train) >= 1


def test_assemble_fails_when_a_guarantee_breaks(tmp_path) -> None:
    shards = tmp_path / "shards"
    shards.mkdir()
    shard_file(shards / "a.parquet")
    frame = pd.read_parquet(shards / "a.parquet")
    frame["dominant_fraction"] = 0.1
    frame.to_parquet(shards / "a.parquet", index=False)
    outcome = runner.invoke(
        cli.app,
        [
            "assemble",
            "--allow-unverified-shards",
            str(shards),
            "--out",
            str(tmp_path / "out"),
            "--work",
            str(tmp_path / "work"),
        ],
    )
    assert outcome.exit_code == 1
    assert "  FAILED below_threshold: 3" in outcome.output.splitlines()


def test_assemble_refuses_an_empty_shard_directory(tmp_path) -> None:
    shards = tmp_path / "shards"
    shards.mkdir()
    outcome = runner.invoke(
        cli.app,
        ["assemble", "--allow-unverified-shards", str(shards), "--out", str(tmp_path / "out")],
    )
    assert outcome.exit_code == 1


def test_assemble_with_one_directory_uses_it_directly(tmp_path) -> None:
    """A single shard directory needs no combined copy."""
    shards = tmp_path / "shards"
    shards.mkdir()
    shard_file(shards / "a.parquet")
    outcome = runner.invoke(
        cli.app,
        [
            "assemble",
            "--allow-unverified-shards",
            str(shards),
            "--out",
            str(tmp_path / "out"),
            "--work",
            str(tmp_path / "work"),
        ],
    )
    assert outcome.exit_code == 0, outcome.output
    assert not (tmp_path / "work" / "shards").exists()


def test_assemble_does_not_inherit_a_previous_run(tmp_path) -> None:
    """Regression: the combined directory is shared, so it must be emptied."""
    work = tmp_path / "work"
    (work / "shards").mkdir(parents=True)
    (work / "shards" / "stale__old.parquet").write_bytes(b"not parquet")
    dirs = []
    for name in ("w0", "w1"):
        d = tmp_path / name
        d.mkdir()
        shard_file(d / f"{name}.parquet", n=2)
        dirs.append(str(d))
    outcome = runner.invoke(
        cli.app,
        [
            "assemble",
            "--allow-unverified-shards",
            *dirs,
            "--out",
            str(tmp_path / "out"),
            "--work",
            str(work),
        ],
    )
    assert outcome.exit_code == 0, outcome.output


def test_assemble_reports_rejections_recorded_by_the_builders(tmp_path) -> None:
    """A split build's counters live beside its shards; assembly must use them."""
    import json

    shards = tmp_path / "shards"
    shards.mkdir()
    shard_file(shards / "a.parquet")
    (shards / "a.rejections.json").write_text(json.dumps({"below_threshold": 12}))
    outcome = runner.invoke(
        cli.app,
        [
            "assemble",
            "--allow-unverified-shards",
            str(shards),
            "--out",
            str(tmp_path / "out"),
            "--work",
            str(tmp_path / "work"),
        ],
    )
    assert outcome.exit_code == 0, outcome.output
    manifest = json.loads((tmp_path / "out" / "v1.1.0" / "manifest.json").read_text())
    assert manifest["rejections"] == {"below_threshold": 12}


def test_assemble_sums_rejections_across_worker_directories(tmp_path) -> None:
    import json

    dirs = []
    for name, count in (("w0", 3), ("w1", 4)):
        d = tmp_path / name
        d.mkdir()
        shard_file(d / f"{name}.parquet", n=2)
        (d / f"{name}.rejections.json").write_text(json.dumps({"below_threshold": count}))
        dirs.append(str(d))
    outcome = runner.invoke(
        cli.app,
        [
            "assemble",
            "--allow-unverified-shards",
            *dirs,
            "--out",
            str(tmp_path / "out"),
            "--work",
            str(tmp_path / "work"),
        ],
    )
    assert outcome.exit_code == 0, outcome.output
    manifest = json.loads((tmp_path / "out" / "v1.1.0" / "manifest.json").read_text())
    assert manifest["rejections"] == {"below_threshold": 7}


def test_assemble_combines_directories_that_share_a_leaf_name(tmp_path) -> None:
    """Every worker's directory is called "shards", so the leaf cannot disambiguate.

    Regression: the combined view prefixed each file with its directory's leaf
    name, which is identical for every worker, so two files with the same name
    collided. Disjoint region sets hid it until the per-directory rejection
    counters -- all named alike -- made it fire.
    """
    import json

    dirs = []
    for name in ("group-0", "group-1"):
        d = tmp_path / name / "shards"
        d.mkdir(parents=True)
        shard_file(d / "same-region.parquet", n=2)
        (d / "_group-backfill.rejections.json").write_text(json.dumps({"too_large": 1}))
        dirs.append(str(d))
    outcome = runner.invoke(
        cli.app,
        [
            "assemble",
            "--allow-unverified-shards",
            *dirs,
            "--out",
            str(tmp_path / "out"),
            "--work",
            str(tmp_path / "work"),
        ],
    )
    assert outcome.exit_code == 0, outcome.output
    manifest = json.loads((tmp_path / "out" / "v1.1.0" / "manifest.json").read_text())
    assert manifest["rejections"] == {"too_large": 2}


def test_assemble_defaults_refuse_legacy_shards(tmp_path, monkeypatch) -> None:
    directory = tmp_path / "shards"
    directory.mkdir()
    shard_file(directory / "alpha.parquet")
    monkeypatch.setattr(cli.hub, "list_region_stems", lambda *a: pytest.fail("no network"))
    outcome = runner.invoke(cli.app, assemble_args(tmp_path, directory))
    assert outcome.exit_code == 1
    assert "unverifiable completion receipt" in outcome.output
    assert not (tmp_path / "out").exists()


def test_assemble_reuses_legacy_labeled_shards_for_new_finalization(tmp_path, monkeypatch):
    directory = tmp_path / "shards"
    config = Config(source="description", source_revision="a" * 40, dataset_version="1.0.0")
    verified_shard(directory, config=config)
    receipt = directory / "alpha.complete.json"
    document = json.loads(receipt.read_text())
    document["context"]["settings"].pop("deduplication_policy")
    document["context"]["settings"]["code_repository"] = (
        "https://github.com/NoeFlandre/osm-worldcover/tree/3ddd472e7deac10d116fd763cbf612e0d1a9c8db"
    )
    document["build_context_sha256"] = BuildContext.from_document(document["context"]).fingerprint
    receipt.write_text(json.dumps(document))
    monkeypatch.setattr(cli.hub, "list_region_stems", lambda *a: ["alpha"])

    outcome = runner.invoke(cli.app, assemble_args(tmp_path, directory))

    manifest = assembled_manifest(tmp_path, "1.1.0")
    assert outcome.exit_code == 0, outcome.output
    assert _legacy_reuse_summary(manifest) == {
        "policy": "polygon_id+normalized_text+worldcover_code",
        "source_dataset_version": "1.0.0",
        "legacy_policy_absent": True,
        "code_repository": "https://github.com/NoeFlandre/osm-worldcover/tree/3ddd472e7deac10d116fd763cbf612e0d1a9c8db",
    }


def test_legacy_recovery_is_explicitly_unpublishable_without_network(tmp_path, monkeypatch):
    directory = tmp_path / "shards"
    directory.mkdir()
    shard_file(directory / "alpha.parquet")
    monkeypatch.setattr(cli.hub, "list_region_stems", lambda *a: pytest.fail("no network"))
    outcome = runner.invoke(
        cli.app, [*assemble_args(tmp_path, directory), "--allow-unverified-shards"]
    )
    processing = assembled_manifest(tmp_path)["processing"]
    assert _recovery_status(outcome, processing) == (0, True, True, "unverified", False, False)
