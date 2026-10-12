"""Command-line assembly behavior."""

import json
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from osm_worldcover import cli
from osm_worldcover.accounting import BuildContext
from osm_worldcover.build import ShardStore
from osm_worldcover.config import Config
from osm_worldcover.pipeline import RegionOutcome

runner = CliRunner()


def shard_file(path, n=3):
    pd.DataFrame(
        {
            "polygon_id": [f"p{i}" for i in range(n)],
            "osm_type": ["way"] * n,
            "osm_id": list(range(n)),
            "region": ["r"] * n,
            "document_id": [f"d{i}" for i in range(n)],
            "language": ["en"] * n,
            "text": [" ".join(["word"] * 20) + f" {i}" for i in range(n)],
            "worldcover_code": [10] * n,
            "worldcover_label": ["Tree cover"] * n,
            "dominant_fraction": [0.95] * n,
            "lat": [49.6] * n,
            "lon": [6.1] * n,
        }
    ).to_parquet(path, index=False)


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


def verified_shard(directory, stem="alpha", config=None, count=2, receipt_context=None):
    """Write real completion receipts, including all pre-deduplication accounting."""
    config = config or Config(source="description", source_revision="a" * 40)
    directory.mkdir(parents=True, exist_ok=True)
    shard_file(directory / f"{stem}.parquet", n=count)
    rows = pd.read_parquet(directory / f"{stem}.parquet")
    rows["region"] = stem
    rows["polygon_id"] = [f"{stem}-{i}" for i in range(count)]
    outcome = RegionOutcome(
        stem,
        polygons_seen=count + 5,
        polygons_invalid=1,
        polygons_accepted=count + 1,
        polygons_with_examples=count,
        examples=count,
        source_links=count + 1,
        source_documents=count + 1,
        rejections=Counter({"below_threshold": 3}),
        text_rejections=Counter({"empty_text": 1}),
    )
    context = receipt_context or BuildContext.from_config(config)
    ShardStore(directory, context).write_outcome(stem, rows, outcome)
    return config


def assemble_args(tmp_path, *directories):
    return [
        "assemble",
        *(str(directory) for directory in directories),
        "--out",
        str(tmp_path / "out"),
        "--work",
        str(tmp_path / "work"),
    ]


def assembled_manifest(tmp_path, version="1.1.0"):
    return json.loads((tmp_path / "out" / f"v{version}" / "manifest.json").read_text())


def _legacy_reuse_summary(manifest: dict) -> dict:
    settings = manifest["processing"]["context"]["settings"]
    return {
        "policy": manifest["settings"]["deduplication_policy"],
        "source_dataset_version": settings["dataset_version"],
        "legacy_policy_absent": "deduplication_policy" not in settings,
        "code_repository": settings["code_repository"],
    }


def _recovery_status(outcome, processing: dict) -> tuple:
    return (
        outcome.exit_code,
        "UNVERIFIED" in outcome.output,
        "not publishable" in outcome.output,
        processing["scope"],
        processing["full_source_complete"],
        processing["complete"],
    )


def _derived_settings_summary(manifest: dict, config: Config, seen: list) -> dict:
    return {
        "settings": manifest["settings"],
        "min_words": manifest["settings"]["min_words"],
        "complete": manifest["processing"]["full_source_complete"],
        "context": manifest["processing"]["context"],
        "inventory_calls": seen,
        "expected_context": BuildContext.from_config(config).as_dict(),
    }


def _worker_receipt_summary(manifest: dict) -> dict:
    ledger = manifest["processing"]
    return {
        "processed_regions": ledger["processed_regions"],
        "complete": ledger["full_source_complete"],
        "examples": ledger["totals"]["examples"],
        "polygons_seen": ledger["totals"]["polygons_seen"],
        "text_rejections": ledger["totals"]["text_rejections"],
        "rejections": manifest["rejections"],
    }


def _mixed_receipt_workers(tmp_path, monkeypatch):
    config = Config(source="description", source_revision="a" * 40)
    legacy_document = BuildContext.from_config(config).as_dict()
    legacy_document.pop("code_revision", None)
    current_document = BuildContext.from_config(config).as_dict()
    current_document["code_revision"] = "b" * 40
    first, second = tmp_path / "w0", tmp_path / "w1"
    verified_shard(
        first,
        config=config,
        receipt_context=BuildContext.from_document(legacy_document),
    )
    verified_shard(
        second,
        stem="beta",
        config=config,
        receipt_context=BuildContext.from_document(current_document),
    )
    monkeypatch.setattr(
        "osm_worldcover.accounting.context._current_code_revision", lambda: "c" * 40
    )
    monkeypatch.setattr(cli.hub, "list_region_stems", lambda *a: ["alpha", "beta"])
    return config, first, second


def _mixed_provenance_summary(ledger: dict) -> dict:
    return {
        "schema": ledger["schema_version"],
        "provenance": ledger["code_provenance"],
        "assembly": ledger["assembly_code_revision"],
    }


def _damage_receipt(receipt: Path, shard: Path, directory: Path, damage: str) -> None:
    actions = {
        "receipt": lambda: receipt.unlink(),
        "shard": lambda: shard.unlink(),
        "bytes": lambda: shard.write_bytes(shard.read_bytes() + b"changed"),
        "orphan": lambda: (directory / "beta.complete.json").write_text(receipt.read_text()),
        "schema": lambda: _damage_receipt_schema(receipt),
    }
    actions[damage]()


def _damage_receipt_schema(receipt: Path) -> None:
    document = json.loads(receipt.read_text())
    document["context"]["pipeline_schema_version"] = 999
    receipt.write_text(json.dumps(document))


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
