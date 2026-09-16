"""Command line behaviour."""

import json

import pandas as pd
from typer.testing import CliRunner

from osm_worldcover import cli
from osm_worldcover.domain.validation import Check, ValidationReport, Violation
from osm_worldcover.finalize import StreamedBuild

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
            # Distinct text per row: identical text under one label is a
            # duplicate, which verify is right to reject.
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


def test_verify_rejects_a_build_that_breaks_a_guarantee(tmp_path) -> None:
    build = tmp_path / "v1.0.0"
    build.mkdir(parents=True)
    bad = frame(2)
    bad.loc[:, "dominant_fraction"] = 0.4
    bad.to_parquet(build / "train.parquet", index=False)
    outcome = runner.invoke(cli.app, ["verify", str(build)])
    assert outcome.exit_code == 1
    assert "below_threshold" in outcome.output


def test_verify_refuses_a_directory_with_no_splits(tmp_path) -> None:
    outcome = runner.invoke(cli.app, ["verify", str(tmp_path)])
    assert outcome.exit_code == 1
    assert "no splits" in outcome.output


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
    outcome = runner.invoke(cli.app, ["assemble", str(shards), "--out", str(tmp_path / "out")])
    assert outcome.exit_code == 0, outcome.output
    assert (tmp_path / "out" / "v1.0.0" / "train.parquet").exists()
    assert (tmp_path / "out" / "v1.0.0" / "manifest.json").exists()


def test_assemble_accepts_several_shard_directories(tmp_path) -> None:
    """A build split across workers leaves one shard directory per worker."""
    dirs = []
    for name in ("w0", "w1"):
        d = tmp_path / name
        d.mkdir()
        shard_file(d / f"{name}.parquet", n=2)
        dirs.append(str(d))
    outcome = runner.invoke(cli.app, ["assemble", *dirs, "--out", str(tmp_path / "out")])
    assert outcome.exit_code == 0, outcome.output
    train = pd.read_parquet(tmp_path / "out" / "v1.0.0" / "train.parquet")
    assert len(train) >= 1


def test_assemble_fails_when_a_guarantee_breaks(tmp_path) -> None:
    shards = tmp_path / "shards"
    shards.mkdir()
    shard_file(shards / "a.parquet")
    frame = pd.read_parquet(shards / "a.parquet")
    frame["dominant_fraction"] = 0.1
    frame.to_parquet(shards / "a.parquet", index=False)
    outcome = runner.invoke(cli.app, ["assemble", str(shards), "--out", str(tmp_path / "out")])
    assert outcome.exit_code == 1
    assert "below_threshold" in outcome.output


def test_assemble_refuses_an_empty_shard_directory(tmp_path) -> None:
    shards = tmp_path / "shards"
    shards.mkdir()
    outcome = runner.invoke(cli.app, ["assemble", str(shards), "--out", str(tmp_path / "out")])
    assert outcome.exit_code == 1


def test_assemble_with_one_directory_uses_it_directly(tmp_path) -> None:
    """A single shard directory needs no combined copy."""
    shards = tmp_path / "shards"
    shards.mkdir()
    shard_file(shards / "a.parquet")
    outcome = runner.invoke(
        cli.app,
        ["assemble", str(shards), "--out", str(tmp_path / "out"), "--work", str(tmp_path / "work")],
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
        cli.app, ["assemble", *dirs, "--out", str(tmp_path / "out"), "--work", str(work)]
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
        ["assemble", str(shards), "--out", str(tmp_path / "out"), "--work", str(tmp_path / "work")],
    )
    assert outcome.exit_code == 0, outcome.output
    manifest = json.loads((tmp_path / "out" / "v1.0.0" / "manifest.json").read_text())
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
        ["assemble", *dirs, "--out", str(tmp_path / "out"), "--work", str(tmp_path / "work")],
    )
    assert outcome.exit_code == 0, outcome.output
    manifest = json.loads((tmp_path / "out" / "v1.0.0" / "manifest.json").read_text())
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
        ["assemble", *dirs, "--out", str(tmp_path / "out"), "--work", str(tmp_path / "work")],
    )
    assert outcome.exit_code == 0, outcome.output
    manifest = json.loads((tmp_path / "out" / "v1.0.0" / "manifest.json").read_text())
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
