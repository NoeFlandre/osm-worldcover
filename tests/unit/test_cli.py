"""Command line behaviour."""

import json
import logging
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from osm_worldcover import cli
from osm_worldcover.domain.validation import Check, ValidationReport, Violation
from osm_worldcover.finalize import StreamedBuild
from osm_worldcover.release import release_lock

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


def _seed_release_lock(build_dir: Path) -> None:
    with release_lock(build_dir):
        pass


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
    _seed_release_lock(target)
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


def test_build_logs_run_settings_only_when_verbose(tmp_path, monkeypatch, caplog) -> None:
    result = built(tmp_path)
    monkeypatch.setattr(cli, "run_build", lambda *a, **k: type("R", (), {"result": result})())
    args = ["build", "--out", str(tmp_path), "--cache", str(tmp_path)]

    quiet = runner.invoke(cli.app, args)
    assert not [record for record in caplog.records if record.levelno == logging.INFO]

    with caplog.at_level(logging.INFO, logger="osm_worldcover"):
        loud = runner.invoke(cli.app, [*args, "--verbose"])
    assert loud.exit_code == 0, loud.output
    assert loud.stdout == quiet.stdout
    assert "build source=" in caplog.text


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


def write_build(tmp_path, rows: pd.DataFrame, settings: dict | None = None) -> Path:
    build = tmp_path / "v1.0.0"
    build.mkdir(parents=True)
    # pyarrow keeps NaN as NaN; pandas' to_parquet would store it as null instead.
    table = pa.table({name: rows[name].tolist() for name in rows.columns})
    pq.write_table(table, build / "train.parquet")
    if settings is not None:
        (build / "manifest.json").write_text(json.dumps({"settings": settings}))
    _seed_release_lock(build)
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
    _seed_release_lock(tmp_path)
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
    _seed_release_lock(build)
    outcome = runner.invoke(cli.app, ["info", str(build)])
    assert outcome.exit_code == 0
    assert "Tree cover" in outcome.output
    assert "examples: 4" in outcome.output


def test_migrate_release_lock_cli_is_idempotent(tmp_path) -> None:
    build = tmp_path / "v1.0.0"
    build.mkdir()
    for name in ("train.parquet", "validation.parquet", "test.parquet", "manifest.json"):
        (build / name).write_bytes(name.encode())
    command = ["migrate-release-lock", str(build)]

    first = runner.invoke(cli.app, command)
    second = runner.invoke(cli.app, command)

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert (tmp_path / ".v1.0.0.lock").is_file()


def test_migrate_release_lock_cli_rejects_an_incomplete_release(tmp_path) -> None:
    build = tmp_path / "v1.0.0"
    build.mkdir()
    (build / "manifest.json").write_text("{}")

    result = runner.invoke(cli.app, ["migrate-release-lock", str(build)])

    assert result.exit_code == 1
    assert "missing core file" in result.output
    assert not (tmp_path / ".v1.0.0.lock").exists()


def test_info_holds_a_release_read_lock_while_loading_the_manifest(tmp_path, monkeypatch) -> None:
    build = tmp_path / "v1.0.0"
    events = []

    @contextmanager
    def observed_lock(path):
        events.append(("lock-enter", path))
        yield
        events.append(("lock-exit", path))

    def observed_read(path):
        events.append(("read-manifest", path))
        return MANIFEST

    monkeypatch.setattr(cli, "release_read_lock", observed_lock)
    monkeypatch.setattr(cli, "read_manifest", observed_read)

    outcome = runner.invoke(cli.app, ["info", str(build)])

    assert outcome.exit_code == 0
    assert events == [
        ("lock-enter", build),
        ("read-manifest", build),
        ("lock-exit", build),
    ]


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
    _seed_release_lock(tmp_path)
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
