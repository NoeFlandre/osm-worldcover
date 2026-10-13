"""Shared fixtures for CLI assembly tests."""

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from osm_worldcover import cli
from osm_worldcover.accounting import BuildContext
from osm_worldcover.build import ShardStore
from osm_worldcover.config import Config
from osm_worldcover.pipeline import RegionOutcome


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
