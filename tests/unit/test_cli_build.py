"""Build and region selection CLI tests."""

import logging

from tests.unit.cli_test_support import built
from typer.testing import CliRunner

from osm_worldcover import cli
from osm_worldcover.domain.validation import Check, ValidationReport, Violation

runner = CliRunner()


def test_assemble_callback_is_available_and_registered_on_the_cli_namespace() -> None:
    from osm_worldcover import cli_assemble

    assert cli.assemble.__module__ == cli_assemble.__name__
    assert any(command.callback is cli.assemble for command in cli.app.registered_commands)


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
