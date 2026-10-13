"""Release transaction and streaming resource regression tests."""

import json

import pandas as pd
import pytest
from tests.unit.finalize_test_support import (
    _assert_empty_rerun_preserves_release,
    _assert_empty_result,
    _assert_failed_validation_result,
    _assert_old_release_unchanged_and_unstaged,
    _assert_release_recovered_and_clean,
    _assert_unchanged_rerun_result,
    _file_snapshot,
    _patch_release_name,
    _published_files,
    _recover_twice_and_assert_snapshot,
    _release_bytes,
    _release_entries,
    run,
    shard,
)
from tests.unit.finalize_test_support import (
    _clear_shards as _clear_shards,
)
from tests.unit.finalize_test_support import (
    fast_release_fsyncs as fast_release_fsyncs,
)
from tests.unit.finalize_test_support import (
    shards as shards,
)

from osm_worldcover.adapters.writer import write_batches
from osm_worldcover.config import Config
from osm_worldcover.domain import manifest as manifest_module
from osm_worldcover.finalize import _write_splits, finalize_shards
from osm_worldcover.release import paths as paths_module


def test_a_rerun_that_publishes_nothing_preserves_the_previous_release(shards, tmp_path) -> None:
    out = tmp_path / "out"
    shard(shards / "a.parquet", n=3)
    finalize_shards(shards, Config(), tmp_path / "w1", out)
    release = out / f"v{Config().dataset_version}"
    (release / "audit.json").write_text("local audit report")
    before = _release_bytes(out)

    empty = tmp_path / "empty"
    empty.mkdir()
    result = finalize_shards(empty, Config(), tmp_path / "w2", out)

    assert result.rows == 0
    assert result.paths == []
    assert _published_files(out) == sorted(before)
    assert _release_bytes(out) == before


def test_a_failed_rerun_leaves_the_previous_release_intact(shards, tmp_path, monkeypatch) -> None:
    out = tmp_path / "out"
    shard(shards / "a.parquet", n=3)
    finalize_shards(shards, Config(), tmp_path / "w1", out)
    before = {
        name: (out / f"v{Config().dataset_version}" / name).read_bytes()
        for name in _published_files(out)
    }

    later = tmp_path / "later"
    later.mkdir()
    shard(later / "b.parquet", n=5, start=100)
    calls = []

    def fail_second_split(reader, path):
        calls.append(path)
        if len(calls) == 2:
            raise OSError("disk full")
        return write_batches(reader, path)

    monkeypatch.setattr("osm_worldcover.finalize.write_batches", fail_second_split)
    with pytest.raises(OSError, match="disk full"):
        finalize_shards(later, Config(), tmp_path / "w2", out)

    assert _published_files(out) == sorted(before)
    assert {
        name: (out / f"v{Config().dataset_version}" / name).read_bytes() for name in before
    } == before
    assert sorted(p.name for p in out.iterdir()) == _release_entries()


def test_a_failed_first_run_publishes_nothing(shards, tmp_path, monkeypatch) -> None:
    out = tmp_path / "out"
    shard(shards / "a.parquet", n=3)

    def fail(reader, path):
        raise OSError("disk full")

    monkeypatch.setattr("osm_worldcover.finalize.write_batches", fail)
    with pytest.raises(OSError, match="disk full"):
        finalize_shards(shards, Config(), tmp_path / "w1", out)

    assert _published_files(out) == []
    assert sorted(p.name for p in out.iterdir()) == [f".v{Config().dataset_version}.lock"]


def test_a_successful_rerun_replaces_every_file_of_the_previous_release(shards, tmp_path) -> None:
    out = tmp_path / "out"
    shard(shards / "a.parquet", n=3)
    finalize_shards(shards, Config(), tmp_path / "w1", out)

    (shards / "a.parquet").unlink()
    shard(shards / "b.parquet", n=5, start=100)
    result = finalize_shards(shards, Config(), tmp_path / "w2", out)

    release = out / f"v{Config().dataset_version}"
    manifest = json.loads((release / "manifest.json").read_text())
    published = sum(
        len(pd.read_parquet(release / f"{s}.parquet")) for s in manifest_module.SPLIT_ORDER
    )
    assert result.rows == 5
    assert manifest["counts"]["examples"]["total"] == published == 5
    assert sorted(p.name for p in out.iterdir()) == _release_entries()


def test_split_writes_enable_large_arrow_string_buffers(tmp_path) -> None:
    import duckdb

    connection = duckdb.connect()
    try:
        connection.execute(
            "CREATE TEMP TABLE kept AS "
            "SELECT * FROM (VALUES ('train', 'p1', 'd1', 'key')) "
            "AS rows(split, polygon_id, document_id, _dedup_key)"
        )

        paths, rows = _write_splits(connection, tmp_path)

        assert rows == 1
        setting = connection.execute("SELECT current_setting('arrow_large_buffer_size')").fetchone()
        assert setting == (True,)
        assert {path.stem for path in paths} == {"train", "validation", "test"}
    finally:
        connection.close()


def test_shards_are_never_all_held_in_memory(shards, tmp_path, monkeypatch) -> None:
    """Guard the property that matters: one shard is read at a time."""
    import osm_worldcover.finalize as module

    live = 0
    peak = 0
    original = pd.read_parquet

    def counting_read(*args, **kwargs):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        frame = original(*args, **kwargs)
        live -= 1
        return frame

    monkeypatch.setattr(module.pd, "read_parquet", counting_read)
    for i in range(5):
        shard(shards / f"s{i}.parquet", n=2, start=i * 10)
    finalize_shards(shards, Config(), tmp_path / "work", tmp_path / "work" / "out")
    assert peak == 1


def test_an_empty_build_has_no_files_and_no_manifest(shards, tmp_path) -> None:
    result = run(shards, tmp_path)
    assert (result.rows, result.paths, result.manifest) == (0, [], {})


def test_empty_rerun_preserves_the_last_release_without_claiming_new_paths(
    shards, tmp_path
) -> None:
    shard(shards / "first.parquet", n=3)
    out = tmp_path / "out"
    target = out / f"v{Config().dataset_version}"
    first = finalize_shards(shards, Config(), tmp_path / "work", out)
    (target / "audit.json").write_text("local audit report")
    before = _file_snapshot(target)

    _clear_shards(shards)
    second = finalize_shards(shards, Config(), tmp_path / "work", out)

    _assert_empty_rerun_preserves_release(second, target, before)
    assert first.paths


def test_finalize_shards_accepts_a_symlinked_output_directory(shards, tmp_path):
    shard(shards / "first.parquet", n=3)
    real_out = tmp_path / "real-out"
    real_out.mkdir()
    output_alias = tmp_path / "out-link"
    output_alias.symlink_to(real_out, target_is_directory=True)

    result = finalize_shards(
        shards,
        Config(),
        tmp_path / "work",
        output_alias,
    )

    assert result.rows == 3
    assert {path.name for path in result.paths} == {
        "train.parquet",
        "validation.parquet",
        "test.parquet",
        "manifest.json",
    }
    assert all(path.is_file() for path in result.paths)
    assert list(real_out.glob(f"v{Config().dataset_version}"))


def test_empty_rerun_recovers_an_interrupted_promotion_before_counting_rows(
    shards, tmp_path, monkeypatch
) -> None:
    import osm_worldcover.release as release

    shard(shards / "first.parquet", n=3)
    out = tmp_path / "out"
    target = out / f"v{Config().dataset_version}"
    finalize_shards(shards, Config(), tmp_path / "work", out)
    before = release.inventory_release(target)
    original = paths_module._rename_directory

    def stop_after_backup(source, destination):
        original(source, destination)
        if source == target:
            raise SystemExit("simulated process interruption")

    _patch_release_name(monkeypatch, "_rename_directory", stop_after_backup)
    with release.release_lock(target):
        stage = release.create_stage(target)
        for name in release.CORE_FILES:
            (stage / name).write_bytes(f"uncommitted:{name}".encode())
        new = release.stage_inventory(stage)
        with pytest.raises(SystemExit, match="simulated process interruption"):
            release.commit_release(target, stage, new)
    _patch_release_name(monkeypatch, "_rename_directory", original)

    _clear_shards(shards)
    result = finalize_shards(shards, Config(), tmp_path / "work", out)

    _assert_empty_result(result)
    _assert_release_recovered_and_clean(target, before, out)


def test_failed_staged_validation_does_not_promote_or_replace_the_old_release(
    shards, tmp_path, monkeypatch
) -> None:
    import osm_worldcover.finalize as module

    shard(shards / "first.parquet", n=3)
    out = tmp_path / "out"
    target = out / f"v{Config().dataset_version}"
    finalize_shards(shards, Config(), tmp_path / "work", out)
    before = _file_snapshot(target)
    shard(shards / "second.parquet", n=5, start=100)
    invalid = module.ValidationReport(8, [module.validate([]).violations[0]])
    monkeypatch.setattr(module, "_validate_written", lambda *_args: invalid)

    result = finalize_shards(shards, Config(), tmp_path / "work", out)

    _assert_failed_validation_result(result)
    _assert_old_release_unchanged_and_unstaged(target, before, out)
    _recover_twice_and_assert_snapshot(target, before)


@pytest.mark.parametrize("failed_split", ["train", "validation", "test"])
def test_failed_split_write_never_changes_the_previous_release(
    shards, tmp_path, monkeypatch, failed_split
) -> None:
    import osm_worldcover.finalize as module

    shard(shards / "first.parquet", n=3)
    out = tmp_path / "out"
    target = out / f"v{Config().dataset_version}"
    finalize_shards(shards, Config(), tmp_path / "work", out)
    before = _file_snapshot(target)
    shard(shards / "second.parquet", n=5, start=100)
    original = module.write_batches

    def interrupted(reader, path):
        if path.name == f"{failed_split}.parquet":
            path.write_bytes(b"partial split")
            raise OSError("injected split write interruption")
        return original(reader, path)

    monkeypatch.setattr(module, "write_batches", interrupted)

    with pytest.raises(OSError, match="injected split write interruption"):
        finalize_shards(shards, Config(), tmp_path / "work", out)

    import osm_worldcover.release as release

    release.recover_release(target)
    release.recover_release(target)
    assert _file_snapshot(target) == before
    assert not list(out.glob(f".{target.name}.stage-*"))


def test_failed_manifest_write_never_changes_the_previous_release(
    shards, tmp_path, monkeypatch
) -> None:
    import osm_worldcover.finalize as module

    shard(shards / "first.parquet", n=3)
    out = tmp_path / "out"
    target = out / f"v{Config().dataset_version}"
    finalize_shards(shards, Config(), tmp_path / "work", out)
    before = _file_snapshot(target)
    shard(shards / "second.parquet", n=5, start=100)

    def interrupted(manifest, path):
        path.write_text("partial manifest")
        raise OSError("injected manifest write interruption")

    monkeypatch.setattr(module, "write_manifest", interrupted)

    with pytest.raises(OSError, match="injected manifest write interruption"):
        finalize_shards(shards, Config(), tmp_path / "work", out)

    import osm_worldcover.release as release

    release.recover_release(target)
    release.recover_release(target)
    assert _file_snapshot(target) == before
    assert not list(out.glob(f".{target.name}.stage-*"))


def test_unchanged_rerun_keeps_release_files_and_publication_sidecars_untouched(
    shards, tmp_path
) -> None:
    shard(shards / "first.parquet", n=3)
    out = tmp_path / "out"
    target = out / f"v{Config().dataset_version}"
    finalize_shards(shards, Config(), tmp_path / "work", out)
    (target / "README.md").write_text("publication sidecar")
    (target / "audit.json").write_text("local audit report")
    before = _file_snapshot(target)

    result = finalize_shards(shards, Config(), tmp_path / "work", out)

    _assert_unchanged_rerun_result(result)
    _assert_old_release_unchanged_and_unstaged(target, before, out)
    _recover_twice_and_assert_snapshot(target, before)
