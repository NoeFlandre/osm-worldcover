"""Audit, release locking, and release summary CLI tests."""

import json
from contextlib import contextmanager
from types import SimpleNamespace

from tests.unit.cli_test_support import MANIFEST, _seed_release_lock
from typer.testing import CliRunner

from osm_worldcover import cli

runner = CliRunner()


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
