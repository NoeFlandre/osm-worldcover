"""Command line behaviour."""

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from osm_worldcover import accounting, cli
from osm_worldcover.accounting import BuildContext
from osm_worldcover.build import ShardStore
from osm_worldcover.config import Config
from osm_worldcover.domain.validation import Check, ValidationReport, Violation
from osm_worldcover.finalize import StreamedBuild
from osm_worldcover.pipeline import RegionOutcome

runner = CliRunner()

MANIFEST = {
    "counts": {"examples": {"train": 2, "validation": 1, "test": 1, "total": 4}},
    "class_distribution": [
        {"code": 10, "label": "Tree cover", "examples": 3, "share": 0.75},
        {"code": 50, "label": "Built-up", "examples": 1, "share": 0.25},
    ],
    "language_distribution": [{"language": "en", "examples": 4, "share": 1.0}],
}


def split_frames(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        name: frame[frame["split"] == name].reset_index(drop=True)
        for name in ("train", "validation", "test")
    }


def frame(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "polygon_id": [f"p{i}" for i in range(n)],
            "document_id": [f"d{i}" for i in range(n)],
            # Distinct text per row keeps this fixture free of duplicate records.
            "text": [" ".join(["word"] * 20) + f" {i}" for i in range(n)],
            "worldcover_code": [10] * n,
            "worldcover_label": ["Tree cover"] * n,
            "dominant_fraction": [0.95] * n,
            "split": ["train", "train", "validation", "test"][:n],
        }
    )


def built(tmp_path, report=None) -> StreamedBuild:
    """A StreamedBuild as run_build would return it, already written to disk."""
    target = tmp_path / "v1.0.0"
    target.mkdir(parents=True, exist_ok=True)
    paths = []
    for split in ("train", "validation", "test"):
        path = target / f"{split}.parquet"
        frame()[frame()["split"] == split].to_parquet(path, index=False)
        paths.append(path)
    paths.append(target / "manifest.json")
    paths[-1].write_text(json.dumps(MANIFEST))
    return StreamedBuild(4, paths, MANIFEST, report or ValidationReport(4))


def test_build_writes_a_dataset_and_reports_counts(tmp_path, monkeypatch) -> None:
    result = built(tmp_path)
    monkeypatch.setattr(cli, "run_build", lambda *a, **k: type("R", (), {"result": result})())
    outcome = runner.invoke(cli.app, ["build", "--out", str(tmp_path), "--cache", str(tmp_path)])
    assert outcome.exit_code == 0, outcome.output
    assert "examples: 4" in outcome.output
    assert (tmp_path / "v1.0.0" / "train.parquet").exists()


def test_build_fails_when_a_guarantee_is_broken(tmp_path, monkeypatch) -> None:
    report = ValidationReport(4, [Violation(Check.POLYGON_LEAKAGE, 2, ("p1",))])
    result = built(tmp_path, report)
    monkeypatch.setattr(cli, "run_build", lambda *a, **k: type("R", (), {"result": result})())
    outcome = runner.invoke(cli.app, ["build", "--out", str(tmp_path), "--cache", str(tmp_path)])
    assert outcome.exit_code == 1
    assert "polygon_leakage" in outcome.output


def test_verify_accepts_a_sound_build(tmp_path) -> None:
    build = tmp_path / "v1.0.0"
    build.mkdir(parents=True)
    for split in ("train", "validation", "test"):
        frame()[frame()["split"] == split].to_parquet(build / f"{split}.parquet", index=False)
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

    outcome = runner.invoke(cli.app, ["verify", str(build)])

    assert outcome.exit_code == 1
    assert "unusable_text" in outcome.output


def test_verify_rejects_a_build_that_breaks_a_guarantee(tmp_path) -> None:
    build = tmp_path / "v1.0.0"
    build.mkdir(parents=True)
    bad = frame(2)
    bad.loc[:, "dominant_fraction"] = 0.4
    bad.to_parquet(build / "train.parquet", index=False)
    outcome = runner.invoke(cli.app, ["verify", str(build)])
    assert outcome.exit_code == 1
    assert outcome.output.splitlines() == ["rows: 2", "FAILED below_threshold: 2 ('p0', 'p1')"]


def write_build(tmp_path, rows: pd.DataFrame, settings: dict | None = None) -> Path:
    build = tmp_path / "v1.0.0"
    build.mkdir(parents=True)
    # pyarrow keeps NaN as NaN; pandas' to_parquet would store it as null instead.
    table = pa.table({name: rows[name].tolist() for name in rows.columns})
    pq.write_table(table, build / "train.parquet")
    if settings is not None:
        (build / "manifest.json").write_text(json.dumps({"settings": settings}))
    return build


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
    outcome = runner.invoke(cli.app, ["verify", str(tmp_path)])
    assert outcome.exit_code == 1
    assert "no splits" in outcome.output


def test_audit_reports_warnings_and_writes_json_report(tmp_path, monkeypatch) -> None:
    report = SimpleNamespace(
        rows=4,
        warnings=[SimpleNamespace(code="identical_text_cross_split", count=2)],
        problems=[],
        ok=True,
        as_dict=lambda: {"ok": True, "rows": 4},
    )
    seen = {}

    def fake_audit(build_dir, **options):
        seen["build_dir"] = build_dir
        seen["options"] = options
        return report

    monkeypatch.setattr("osm_worldcover.adapters.audit.audit_build", fake_audit)
    build_dir = tmp_path / "release"
    report_file = tmp_path / "audit.json"
    outcome = runner.invoke(
        cli.app,
        [
            "audit",
            str(build_dir),
            "--require-complete",
            "--require-card",
            "--strict-text-leakage",
            "--report-file",
            str(report_file),
        ],
    )

    assert (
        outcome.exit_code,
        seen["build_dir"],
        seen["options"],
        "WARNING identical_text_cross_split: 2" in outcome.output,
        "independent release audit passed" in outcome.output,
        json.loads(report_file.read_text()),
    ) == (
        0,
        build_dir,
        {
            "require_complete": True,
            "require_card": True,
            "strict_text_leakage": True,
        },
        True,
        True,
        {"ok": True, "rows": 4},
    )


def test_audit_fails_when_report_contains_a_problem(monkeypatch, tmp_path) -> None:
    report = SimpleNamespace(
        rows=1,
        warnings=[],
        problems=[SimpleNamespace(code="invalid_text", count=1, examples=("p1",))],
        ok=False,
        as_dict=lambda: {"ok": False, "rows": 1},
    )
    monkeypatch.setattr(
        "osm_worldcover.adapters.audit.audit_build", lambda *_args, **_kwargs: report
    )

    outcome = runner.invoke(cli.app, ["audit", str(tmp_path / "release")])

    assert outcome.exit_code == 1
    assert "FAILED invalid_text: 1 ('p1',)" in outcome.output


def test_info_summarises_a_manifest(tmp_path) -> None:
    build = tmp_path / "v1.0.0"
    build.mkdir(parents=True)
    (build / "manifest.json").write_text(json.dumps(MANIFEST))
    outcome = runner.invoke(cli.app, ["info", str(build)])
    assert outcome.exit_code == 0
    assert "Tree cover" in outcome.output
    assert "examples: 4" in outcome.output


def test_build_reads_regions_from_a_file(tmp_path, monkeypatch) -> None:
    """A global run names hundreds of regions; a file beats a giant argv."""
    seen: dict[str, object] = {}
    result = built(tmp_path)

    def fake_build(config, regions=None, **kwargs):
        seen["regions"] = regions
        return type("R", (), {"result": result})()

    monkeypatch.setattr(cli, "run_build", fake_build)
    listing = tmp_path / "regions.txt"
    listing.write_text("alpha-latest\nbeta-latest\n\n# a comment\ngamma-latest\n")
    outcome = runner.invoke(
        cli.app,
        ["build", "--out", str(tmp_path), "--cache", str(tmp_path), "--regions-file", str(listing)],
    )
    assert outcome.exit_code == 0, outcome.output
    assert seen["regions"] == ["alpha-latest", "beta-latest", "gamma-latest"]


def test_regions_file_and_region_flags_combine(tmp_path, monkeypatch) -> None:
    seen: dict[str, object] = {}
    result = built(tmp_path)

    def fake_build(config, regions=None, **kwargs):
        seen["regions"] = regions
        return type("R", (), {"result": result})()

    monkeypatch.setattr(cli, "run_build", fake_build)
    listing = tmp_path / "regions.txt"
    listing.write_text("alpha-latest\n")
    runner.invoke(
        cli.app,
        [
            "build",
            "--out",
            str(tmp_path),
            "--cache",
            str(tmp_path),
            "--regions-file",
            str(listing),
            "--region",
            "beta-latest",
        ],
    )
    assert seen["regions"] == ["beta-latest", "alpha-latest"]


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
        ["assemble", "--allow-unverified-shards", str(shards), "--out", str(tmp_path / "out")],
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
        cli.app, ["assemble", "--allow-unverified-shards", *dirs, "--out", str(tmp_path / "out")]
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
        ["assemble", "--allow-unverified-shards", str(shards), "--out", str(tmp_path / "out")],
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


def test_regions_lists_the_pinned_sources_region_stems(monkeypatch) -> None:
    """Step one of a split run: learn which regions the pinned source has."""
    seen: dict[str, object] = {}

    def fake_list(repo_id, revision, source):
        seen["args"] = (repo_id, revision, source.name)
        return ["alpha-latest", "beta-latest"]

    monkeypatch.setattr(cli.hub, "list_region_stems", fake_list)
    outcome = runner.invoke(cli.app, ["regions", "--source", "description", "--revision", "abc"])
    assert outcome.exit_code == 0, outcome.output
    assert outcome.stdout.splitlines() == ["alpha-latest", "beta-latest"]
    assert seen["args"] == ("NoeFlandre/osm-polygon-description-tag", "abc", "description")


def test_regions_resolves_the_head_revision_when_none_is_pinned(monkeypatch) -> None:
    """An unpinned listing must still name the commit it described."""
    monkeypatch.setattr(cli.hub, "resolve_revision", lambda repo_id: "resolved-sha")
    monkeypatch.setattr(cli.hub, "list_region_stems", lambda *a, **k: ["alpha-latest"])
    outcome = runner.invoke(cli.app, ["regions", "--source", "website"])
    assert outcome.exit_code == 0, outcome.output
    assert outcome.stdout.splitlines() == ["alpha-latest"]
    # The sha goes to stderr, so a redirected listing stays a clean region file.
    assert "resolved-sha" in outcome.stderr


def test_regions_refuses_an_unknown_source() -> None:
    outcome = runner.invoke(cli.app, ["regions", "--source", "nonsense"])
    assert outcome.exit_code == 1
    assert "unknown source" in outcome.output


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
    monkeypatch.setattr(accounting, "_current_code_revision", lambda: "c" * 40)
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


def test_build_with_missing_regions_file_exits_cleanly(tmp_path) -> None:
    result = runner.invoke(cli.app, ["build", "--regions-file", str(tmp_path / "nope.txt")])
    assert result.exit_code == 1
    assert "error:" in result.output
    assert not isinstance(result.exception, OSError)


def test_build_with_unknown_source_exits_cleanly(tmp_path) -> None:
    result = runner.invoke(cli.app, ["build", "--source", "bogus", "--out", str(tmp_path)])
    assert result.exit_code == 1
    assert "error:" in result.output


def test_info_without_manifest_exits_cleanly(tmp_path) -> None:
    result = runner.invoke(cli.app, ["info", str(tmp_path / "missing")])
    assert result.exit_code == 1
    assert "error:" in result.output


def test_info_with_incomplete_manifest_exits_cleanly(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text("{}")
    result = runner.invoke(cli.app, ["info", str(tmp_path)])
    assert result.exit_code == 1
    assert "missing key" in result.output


def test_audit_os_error_exits_cleanly(tmp_path, monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise OSError("disk unreadable")

    monkeypatch.setattr("osm_worldcover.adapters.audit.audit_build", boom)
    result = runner.invoke(cli.app, ["audit", str(tmp_path / "missing")])
    assert result.exit_code == 1
    assert "error:" in result.output


def test_publish_on_missing_directory_exits_cleanly(tmp_path) -> None:
    result = runner.invoke(cli.app, ["publish", str(tmp_path / "missing"), "user/name"])
    assert result.exit_code == 1
    assert "error:" in result.output
