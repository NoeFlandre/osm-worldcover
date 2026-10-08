import builtins
import importlib.util
import os
import sys
import threading
from pathlib import Path

import pytest

import osm_worldcover.release_commit as release
from osm_worldcover.adapters import writer as writer_adapter


@pytest.fixture(autouse=True)
def fast_successful_fsyncs(monkeypatch, request):
    if "real_release_fsync" in request.fixturenames:
        return
    monkeypatch.setattr(release, "fsync_file", lambda _path: None)
    monkeypatch.setattr(release, "fsync_directory", lambda _path: None)


@pytest.fixture
def real_release_fsync():
    pass


class SimulatedCrash(BaseException):
    pass


def _crash_at_parent_sync(target, boundary, message):
    original = release.fsync_directory
    calls = 0

    def interrupt(path):
        nonlocal calls
        if path == target.parent:
            calls += 1
            if calls == boundary + 1:  # Ignore the durable lock-entry sync.
                raise SimulatedCrash(message)
        original(path)

    return interrupt


def _fail_once_at_parent_sync(target, boundary):
    original = release.fsync_directory
    calls = 0

    def fail_once(path):
        nonlocal calls
        if path == target.parent:
            calls += 1
            if calls == boundary + 1:  # Ignore the durable lock-entry sync.
                raise OSError(f"injected promotion directory fsync {boundary}")
        original(path)

    return fail_once


def _release(directory, label, *, sidecars=()):
    directory.mkdir(parents=True, exist_ok=True)
    for name in release.CORE_FILES:
        (directory / name).write_bytes(f"{label}:{name}".encode())
    for name in sidecars:
        (directory / name).write_text(f"{label}:{name}")
    return release.inventory_release(directory)


def _prepared_stage(target, label):
    with release.release_lock(target):
        stage = release.create_stage(target)
        _release(stage, label)
        return stage, release.stage_inventory(stage)


def _journal_exists(target):
    return (target.parent / f".{target.name}.transaction.json").exists()


def _transaction_artifacts(target):
    return {
        path.name
        for path in target.parent.iterdir()
        if path.name.startswith(f".{target.name}.stage-")
        or path.name.startswith(f".{target.name}.backup-")
        or path.name == f".{target.name}.transaction.json"
    }


def _recover_twice(target):
    release.recover_release(target)
    release.recover_release(target)


def _pause_after_backup(target, promotion_gap, continue_promotion, rename_calls):
    original_rename = release._rename_directory

    def pause(source, destination):
        source = Path(source)
        destination = Path(destination)
        rename_calls.append((source, destination))
        original_rename(source, destination)
        if destination.name.startswith(f".{target.name}.backup-"):
            promotion_gap.set()
            if not continue_promotion.wait(timeout=30):
                raise TimeoutError("test did not resume the release promotion")

    return pause


def _promote_release(target, promotion_errors):
    try:
        with release.release_write_lock(target):
            stage = release.create_stage(target)
            _release(stage, "new")
            (stage / "manifest.json").write_text('{"generation": "new"}')
            new = release.stage_inventory(stage)
            assert target.is_dir()
            assert release._inventory_if_present(target) is not None
            release.commit_release(target, stage, new)
    except BaseException as error:
        promotion_errors.append(error)


def _read_manifest_in_thread(target, reader_started, reader_done, reader_errors, reader_values):
    reader_started.set()
    try:
        reader_values.append(writer_adapter.read_manifest(target))
    except BaseException as error:
        reader_errors.append(error)
    finally:
        reader_done.set()


def _assert_release_and_no_artifacts(target, expected):
    assert release.inventory_release(target) == expected
    assert _transaction_artifacts(target) == set()


def _assert_release_absent_and_no_artifacts(target):
    assert not target.exists()
    assert _transaction_artifacts(target) == set()


def _assert_legacy_migration_preserved(target, before):
    assert writer_adapter.read_manifest(target) == {"generation": "legacy"}
    assert {path.name: path.read_bytes() for path in target.iterdir()} == before


def _assert_migration_is_idempotent(target, lock_path, lock_inode):
    assert release.migrate_release_lock(target) == lock_path
    assert lock_path.stat().st_ino == lock_inode


def _assert_migration_started_and_waiting(migration_started, migration_finished):
    assert migration_started.wait(timeout=1)
    assert not migration_finished.wait(timeout=0.05)


def _assert_migration_threads_stopped(writer_thread, migration_thread):
    assert not writer_thread.is_alive()
    assert migration_thread is None or not migration_thread.is_alive()


def _assert_migration_succeeded(errors, migration_finished, lock_path):
    assert errors == []
    assert migration_finished.is_set()
    assert lock_path.is_file()


def _assert_reader_started_and_blocked(
    reader_started, reader_done, rename_calls, reader_errors, reader_values
):
    assert reader_started.wait(timeout=5)
    assert not reader_done.wait(timeout=1), (
        "direct manifest reads must wait while the promoted target is absent; "
        f"rename calls={rename_calls!r}; reader errors={reader_errors!r}; "
        f"reader values={reader_values!r}"
    )


def _join_reader_if_started(reader_thread):
    if reader_thread.ident is not None:
        reader_thread.join(timeout=30)


def _assert_promotion_threads_finished(promotion_thread, promotion_errors, reader_thread):
    assert not promotion_thread.is_alive()
    assert not reader_thread.is_alive()
    assert promotion_errors == []


def _assert_backup_rename(target, rename):
    source, destination = rename
    assert source == target
    assert destination.parent == target.parent
    assert destination.name.startswith(f".{target.name}.backup-")


def _assert_stage_rename(target, rename):
    source, destination = rename
    assert source.name.startswith(f".{target.name}.stage-")
    assert destination == target


def _assert_promotion_rename_sequence(target, rename_calls):
    assert len(rename_calls) == 2, rename_calls
    _assert_backup_rename(target, rename_calls[0])
    _assert_stage_rename(target, rename_calls[1])


def _assert_reader_received_new_manifest(reader_errors, reader_values):
    assert reader_errors == []
    assert reader_values == [{"generation": "new"}]


def _assert_interrupted_first_build_kept_journal(target, stage):
    assert not target.exists()
    assert not stage.exists()
    assert _journal_exists(target)


def _assert_failed_rollback_kept_inputs(target):
    assert not target.exists()
    assert len(list(target.parent.glob(f".{target.name}.backup-*"))) == 1
    assert len(list(target.parent.glob(f".{target.name}.stage-*"))) == 1
    assert _journal_exists(target)


def _assert_cleanup_failure_kept_journal(target, new):
    assert release.core_inventory(target) == new
    assert _journal_exists(target)
    assert list(target.parent.glob(f".{target.name}.backup-*"))


def _assert_partial_backup_cleanup_kept_journal(target, old, new, backup):
    assert release.core_inventory(target) == new
    assert _journal_exists(target)
    assert len(list(backup.iterdir())) < len(old.files)


def _assert_partial_stage_cleanup_kept_journal(target, stage, old):
    assert release.inventory_release(target) == old
    assert stage.is_dir()
    assert not (stage / "train.parquet").exists()
    assert _journal_exists(target)


def _assert_hash_mismatch_kept_candidates(target, stage, backup):
    assert _journal_exists(target)
    assert stage.is_dir()
    assert backup.is_dir()
    assert not target.exists()


def _restore_backup_files(backup, old):
    for entry in old.files:
        (backup / entry.name).write_bytes(f"old:{entry.name}".encode())


def test_directory_promotion_keeps_the_new_complete_inventory(tmp_path, real_release_fsync):
    target = tmp_path / "v1.0.0"
    old = _release(target, "old", sidecars=("README.md",))
    stage, new = _prepared_stage(target, "new")

    with release.release_lock(target):
        release.commit_release(target, stage, new)

    assert release.inventory_release(target) == new
    _recover_twice(target)
    assert release.inventory_release(target) == new
    assert {entry.name for entry in old.files} == set(release.CORE_FILES) | {"README.md"}
    assert _transaction_artifacts(target) == set()


def test_release_inventory_tracks_arbitrary_regular_auxiliary_files(tmp_path):
    target = tmp_path / "v1.0.0"
    core = _release(target, "old")
    (target / "audit.json").write_text('{"report": "local"}')

    assert release.core_inventory(target) == core
    assert {entry.name for entry in release.inventory_release(target).files} == {
        *release.CORE_FILES,
        "audit.json",
    }


def test_stage_inventory_rejects_auxiliary_files(tmp_path):
    stage = tmp_path / "stage"
    _release(stage, "candidate")
    (stage / "audit.json").write_text("must not enter the release stage")

    with pytest.raises(
        release.ReleaseCommitError,
        match="staged release is missing or has extra files",
    ):
        release.stage_inventory(stage)


def test_unrecorded_auxiliary_file_does_not_match_a_new_transaction_target(tmp_path):
    target = tmp_path / "v1.0.0"
    _release(target, "new")
    expected = release.core_inventory(target)
    assert expected is not None

    (target / "audit.json").write_text("unexpected during recovery")

    assert not release._target_matches_new(target, expected)


def test_failed_promotion_restores_arbitrary_auxiliary_files(tmp_path, monkeypatch):
    target = tmp_path / "v1.0.0"
    old = _release(target, "old", sidecars=("audit.json",))
    stage, new = _prepared_stage(target, "new")
    original = release._rename_directory

    def fail_new(source, destination):
        if source == stage:
            raise OSError("injected stage promotion failure")
        original(source, destination)

    monkeypatch.setattr(release, "_rename_directory", fail_new)
    with release.release_lock(target), pytest.raises(OSError, match="stage promotion"):
        release.commit_release(target, stage, new)

    _recover_twice(target)
    assert release.inventory_release(target) == old
    assert (target / "audit.json").read_text() == "old:audit.json"


def test_release_lock_creation_respects_umask_for_shared_readers(tmp_path):
    target = tmp_path / "v1.0.0"
    previous_umask = os.umask(0o027)
    try:
        with release.release_lock(target):
            pass
    finally:
        os.umask(previous_umask)

    lock_path = target.parent / f".{target.name}.lock"
    assert lock_path.stat().st_mode & 0o777 == 0o640


def test_exclusive_lock_syncs_its_parent_before_entering_writer(tmp_path, monkeypatch):
    target = tmp_path / "v1.0.0"
    events = []
    monkeypatch.setattr(release, "fsync_directory", lambda path: events.append(Path(path)))

    with release.release_lock(target):
        events.append("writer")

    assert events == [target.parent, "writer"]


def test_exclusive_lock_does_not_enter_writer_when_parent_sync_fails(tmp_path, monkeypatch):
    target = tmp_path / "v1.0.0"
    entered_writer = False

    def fail_parent_sync(path):
        assert path == target.parent
        raise OSError("injected lock parent fsync")

    monkeypatch.setattr(release, "fsync_directory", fail_parent_sync)
    with pytest.raises(OSError, match="lock parent fsync"), release.release_lock(target):
        entered_writer = True

    assert not entered_writer


def test_read_manifest_works_with_a_read_only_release_parent(tmp_path):
    target = tmp_path / "v1.0.0"
    with release.release_write_lock(target):
        stage = release.create_stage(target)
        _release(stage, "committed")
        (stage / "manifest.json").write_text('{"generation": "committed"}')
        release.commit_release(target, stage, release.stage_inventory(stage))

    lock_path = tmp_path / f".{target.name}.lock"
    parent_mode = tmp_path.stat().st_mode & 0o777
    lock_mode = lock_path.stat().st_mode & 0o777
    lock_path.chmod(0o444)
    tmp_path.chmod(0o555)
    try:
        assert writer_adapter.read_manifest(target) == {"generation": "committed"}
    finally:
        tmp_path.chmod(parent_mode)
        lock_path.chmod(lock_mode)


def test_read_manifest_fails_closed_when_the_release_lock_is_missing(tmp_path):
    target = tmp_path / "v1.0.0"
    target.mkdir()
    (target / "manifest.json").write_text('{"generation": "unlocked"}')

    with pytest.raises(release.ReleaseCommitError, match="lock file"):
        writer_adapter.read_manifest(target)


def test_migrate_release_lock_preserves_release_files_and_is_idempotent(tmp_path):
    target = tmp_path / "v1.0.0"
    _release(target, "legacy")
    (target / "manifest.json").write_text('{"generation": "legacy"}')
    before = {path.name: path.read_bytes() for path in target.iterdir()}

    with pytest.raises(release.ReleaseCommitError, match="lock file"):
        writer_adapter.read_manifest(target)

    lock_path = release.migrate_release_lock(target)
    lock_inode = lock_path.stat().st_ino

    assert lock_path == tmp_path / f".{target.name}.lock"
    _assert_legacy_migration_preserved(target, before)
    _assert_migration_is_idempotent(target, lock_path, lock_inode)


def test_migrate_release_lock_rejects_an_incomplete_release_without_a_sidecar(tmp_path):
    target = tmp_path / "v1.0.0"
    target.mkdir()
    (target / "manifest.json").write_text('{"generation": "incomplete"}')
    lock_path = tmp_path / f".{target.name}.lock"

    with pytest.raises(release.ReleaseCommitError, match="missing"):
        release.migrate_release_lock(target)

    assert not lock_path.exists()


def test_migrate_release_lock_refuses_pending_recovery_without_a_sidecar(tmp_path):
    target = tmp_path / "v1.0.0"
    _release(target, "legacy")
    journal = target.parent / f".{target.name}.transaction.json"
    journal.write_text("{}")
    lock_path = target.parent / f".{target.name}.lock"

    with pytest.raises(release.ReleaseCommitError, match="recovery is required"):
        release.migrate_release_lock(target)

    assert journal.exists()
    assert not lock_path.exists()


def test_migrate_release_lock_waits_for_a_modern_writer(tmp_path):
    target = tmp_path / "v1.0.0"
    _release(target, "legacy")
    writer_holds_lock = threading.Event()
    release_writer = threading.Event()
    migration_started = threading.Event()
    migration_finished = threading.Event()
    errors = []

    def writer():
        with release.release_lock(target):
            journal = target.parent / f".{target.name}.transaction.json"
            backup = target.parent / f".{target.name}.backup-active"
            journal.write_text("{}")
            target.rename(backup)
            writer_holds_lock.set()
            if not release_writer.wait(timeout=5):
                errors.append(TimeoutError("test did not release the modern writer"))
            backup.rename(target)
            journal.unlink()

    def migrate():
        migration_started.set()
        try:
            release.migrate_release_lock(target)
        except BaseException as error:
            errors.append(error)
        finally:
            migration_finished.set()

    writer_thread = threading.Thread(target=writer, name="modern-release-writer")
    writer_thread.start()
    migration_thread = None
    try:
        assert writer_holds_lock.wait(timeout=1)
        migration_thread = threading.Thread(target=migrate, name="release-lock-migration")
        migration_thread.start()
        _assert_migration_started_and_waiting(migration_started, migration_finished)
    finally:
        release_writer.set()
    writer_thread.join(timeout=1)
    if migration_thread is not None:
        migration_thread.join(timeout=1)

    _assert_migration_threads_stopped(writer_thread, migration_thread)
    _assert_migration_succeeded(errors, migration_finished, target.parent / f".{target.name}.lock")


def test_read_manifest_fails_closed_when_release_recovery_is_needed(tmp_path):
    target = tmp_path / "v1.0.0"
    with release.release_write_lock(target):
        target.mkdir()
        (target / "manifest.json").write_text('{"generation": "pending"}')
        journal = target.parent / f".{target.name}.transaction.json"
        journal.write_text("{}")

    with pytest.raises(release.ReleaseCommitError, match="recovery is required"):
        writer_adapter.read_manifest(target)

    assert journal.exists()


def test_direct_read_manifest_waits_for_release_promotion(tmp_path, monkeypatch):
    target = tmp_path / "v1.0.0"
    _release(target, "old")
    (target / "manifest.json").write_text('{"generation": "old"}')

    promotion_gap = threading.Event()
    continue_promotion = threading.Event()
    reader_started = threading.Event()
    reader_done = threading.Event()
    promotion_errors = []
    reader_errors = []
    reader_values = []
    rename_calls = []
    monkeypatch.setattr(
        release,
        "_rename_directory",
        _pause_after_backup(target, promotion_gap, continue_promotion, rename_calls),
    )
    promotion_thread = threading.Thread(
        target=_promote_release, args=(target, promotion_errors), name="release-promoter"
    )
    reader_thread = threading.Thread(
        target=_read_manifest_in_thread,
        args=(target, reader_started, reader_done, reader_errors, reader_values),
        name="manifest-reader",
    )
    promotion_thread.start()
    try:
        assert promotion_gap.wait(timeout=30), (
            f"promotion errors={promotion_errors!r}; rename calls={rename_calls!r}"
        )
        reader_thread.start()
        _assert_reader_started_and_blocked(
            reader_started, reader_done, rename_calls, reader_errors, reader_values
        )
    finally:
        continue_promotion.set()
        promotion_thread.join(timeout=30)
        _join_reader_if_started(reader_thread)

    _assert_promotion_threads_finished(promotion_thread, promotion_errors, reader_thread)
    _assert_promotion_rename_sequence(target, rename_calls)
    _assert_reader_received_new_manifest(reader_errors, reader_values)


def test_create_stage_requires_the_exclusive_target_lock(tmp_path):
    target = tmp_path / "v1.0.0"
    with release.release_lock(target):
        pass

    with pytest.raises(release.ReleaseCommitError, match="exclusive"):
        release.create_stage(target)

    with (
        release.release_read_lock(target),
        pytest.raises(release.ReleaseCommitError, match="exclusive"),
    ):
        release.create_stage(target)

    with release.release_write_lock(target):
        stage = release.create_stage(target)
        release.discard_stage(target, stage)

    assert not stage.exists()


def test_release_lock_reports_unsupported_platform_without_breaking_import(tmp_path, monkeypatch):
    target = tmp_path / "v1.0.0"
    monkeypatch.setattr(release, "fcntl", None)

    with (
        pytest.raises(release.ReleaseCommitError, match="macOS or Linux"),
        release.release_lock(target),
    ):
        pass

    assert not (target.parent / f".{target.name}.lock").exists()


def test_release_lock_fails_closed_when_fcntl_cannot_import(tmp_path, monkeypatch):
    module_name = "release_commit_without_fcntl"
    spec = importlib.util.spec_from_file_location(module_name, release.__file__)
    assert spec is not None and spec.loader is not None
    isolated_release = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = isolated_release
    original_import = builtins.__import__

    def import_without_fcntl(name, *args, **kwargs):
        if name == "fcntl":
            raise ImportError("simulated missing POSIX locks")
        return original_import(name, *args, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(builtins, "__import__", import_without_fcntl)
            spec.loader.exec_module(isolated_release)
    finally:
        sys.modules.pop(module_name, None)

    assert isolated_release.fcntl is None
    with (
        pytest.raises(isolated_release.ReleaseCommitError, match="macOS or Linux"),
        isolated_release.release_lock(tmp_path / "v1.0.0"),
    ):
        pass


def test_shared_lock_cannot_be_upgraded_to_a_writer(tmp_path):
    target = tmp_path / "v1.0.0"
    with release.release_lock(target):
        pass

    with (
        release.release_read_lock(target),
        pytest.raises(release.ReleaseCommitError, match="upgrade"),
        release.release_write_lock(target),
    ):
        pass


def test_commit_flushes_the_stage_before_installing_the_journal(tmp_path, monkeypatch):
    target = tmp_path / "v1.0.0"
    _release(target, "old")
    stage, new = _prepared_stage(target, "new")
    original = release.sync_stage
    calls = []

    def observe_sync(stage_path, parent):
        calls.append((stage_path, parent))
        original(stage_path, parent)

    monkeypatch.setattr(release, "sync_stage", observe_sync)

    with release.release_lock(target):
        release.commit_release(target, stage, new)

    assert calls == [(stage, target.parent)]
    assert release.inventory_release(target) == new


@pytest.mark.parametrize("failed_file", release.CORE_FILES)
def test_each_staged_file_fsync_failure_leaves_old_release_untouched(
    tmp_path, monkeypatch, failed_file
):
    target = tmp_path / "v1.0.0"
    old = _release(target, "old")
    stage, _new = _prepared_stage(target, "new")
    original = release.fsync_file

    def fail_selected(path):
        if path.parent == stage and path.name == failed_file:
            raise OSError(f"injected file fsync: {failed_file}")
        original(path)

    monkeypatch.setattr(release, "fsync_file", fail_selected)

    with release.release_lock(target), pytest.raises(OSError, match="injected file fsync"):
        release.sync_stage(stage, target.parent)

    assert release.inventory_release(target) == old
    assert not _journal_exists(target)


@pytest.mark.parametrize("failed_directory", ["stage", "parent"])
def test_staged_directory_fsync_failure_leaves_old_release_untouched(
    tmp_path, monkeypatch, failed_directory
):
    target = tmp_path / "v1.0.0"
    old = _release(target, "old")
    with release.release_lock(target):
        stage = release.create_stage(target)
        _release(stage, "new")
    original = release.fsync_directory

    def fail_selected(path):
        if path == (stage if failed_directory == "stage" else target.parent):
            raise OSError(f"injected directory fsync: {failed_directory}")
        original(path)

    with release.release_lock(target):
        monkeypatch.setattr(release, "fsync_directory", fail_selected)
        with pytest.raises(OSError, match="injected directory fsync"):
            release.sync_stage(stage, target.parent)

    assert release.inventory_release(target) == old
    assert not _journal_exists(target)


def test_journal_file_fsync_failure_is_recovered_twice(tmp_path, monkeypatch):
    target = tmp_path / "v1.0.0"
    old = _release(target, "old")
    stage, new = _prepared_stage(target, "new")
    original = release.fsync_file

    def fail_journal(path):
        if path.name.endswith(".transaction.json." + stage.name.rsplit("-", 1)[-1] + ".tmp"):
            raise OSError("injected journal file fsync")
        original(path)

    monkeypatch.setattr(release, "fsync_file", fail_journal)

    with release.release_lock(target), pytest.raises(OSError, match="journal file fsync"):
        release.commit_release(target, stage, new)

    monkeypatch.setattr(release, "fsync_file", original)
    _recover_twice(target)

    assert release.inventory_release(target) == old
    assert _transaction_artifacts(target) == set()


def test_journal_install_failure_does_not_replace_the_old_release(tmp_path, monkeypatch):
    target = tmp_path / "v1.0.0"
    old = _release(target, "old")
    stage, new = _prepared_stage(target, "new")
    original = release.os.replace

    def fail_journal(source, destination):
        if destination == target.parent / f".{target.name}.transaction.json":
            raise OSError("injected journal installation")
        original(source, destination)

    monkeypatch.setattr(release.os, "replace", fail_journal)

    with release.release_lock(target), pytest.raises(OSError, match="journal installation"):
        release.commit_release(target, stage, new)

    monkeypatch.setattr(release.os, "replace", original)
    _recover_twice(target)

    assert release.inventory_release(target) == old
    assert _transaction_artifacts(target) == set()


def test_interrupted_journal_install_without_old_release_aborts_candidate_twice(
    tmp_path, monkeypatch
):
    target = tmp_path / "v1.0.0"
    stage, _new = _prepared_stage(target, "new")
    original = release.fsync_directory
    monkeypatch.setattr(
        release,
        "fsync_directory",
        _crash_at_parent_sync(target, 2, "stopped after journal installation"),
    )
    with release.release_lock(target), pytest.raises(SimulatedCrash):
        release.commit_release(target, stage, release.stage_inventory(stage))

    monkeypatch.setattr(release, "fsync_directory", original)
    _recover_twice(target)

    assert not target.exists()
    assert _transaction_artifacts(target) == set()


def test_first_build_abort_after_stage_removal_finishes_after_restart(tmp_path, monkeypatch):
    target = tmp_path / "v1.0.0"
    stage, new = _prepared_stage(target, "new")
    original_fsync_directory = release.fsync_directory
    monkeypatch.setattr(
        release,
        "fsync_directory",
        _crash_at_parent_sync(target, 2, "process stopped after journal installation"),
    )
    with release.release_lock(target), pytest.raises(SimulatedCrash):
        release.commit_release(target, stage, new)

    monkeypatch.setattr(
        release,
        "fsync_directory",
        _crash_at_parent_sync(target, 2, "process stopped after stage removal"),
    )
    with pytest.raises(SimulatedCrash):
        release.recover_release(target)

    _assert_interrupted_first_build_kept_journal(target, stage)

    monkeypatch.setattr(release, "fsync_directory", original_fsync_directory)
    _recover_twice(target)

    _assert_release_absent_and_no_artifacts(target)


def test_recovery_replaces_a_stale_owned_journal_temporary(tmp_path, monkeypatch):
    target = tmp_path / "v1.0.0"
    old = _release(target, "old")
    stage, new = _prepared_stage(target, "new")
    monkeypatch.setattr(
        release,
        "fsync_directory",
        _crash_at_parent_sync(target, 2, "process stopped after journal installation"),
    )
    with release.release_lock(target), pytest.raises(SimulatedCrash):
        release.commit_release(target, stage, new)

    transaction_id = stage.name.rsplit("-", 1)[-1]
    temporary = target.parent / f".{target.name}.transaction.json.{transaction_id}.tmp"
    temporary.write_text("incomplete journal update")
    monkeypatch.setattr(release, "fsync_directory", lambda _path: None)
    _recover_twice(target)

    assert release.inventory_release(target) == old
    assert _transaction_artifacts(target) == set()
    assert not temporary.exists()


@pytest.mark.parametrize("failed_boundary", range(1, 8))
def test_each_promotion_directory_fsync_recovers_to_one_complete_inventory(
    tmp_path, monkeypatch, failed_boundary
):
    target = tmp_path / "v1.0.0"
    old = _release(target, "old")
    stage, new = _prepared_stage(target, "new")
    original = release.fsync_directory
    monkeypatch.setattr(
        release,
        "fsync_directory",
        _fail_once_at_parent_sync(target, failed_boundary),
    )
    with release.release_lock(target):
        try:
            release.commit_release(target, stage, new)
        except OSError as error:
            assert f"promotion directory fsync {failed_boundary}" in str(error)

    monkeypatch.setattr(release, "fsync_directory", original)
    _recover_twice(target)
    expected = old if failed_boundary <= 3 else new

    _assert_release_and_no_artifacts(target, expected)


@pytest.mark.parametrize("crash_after", ["target-to-backup", "stage-to-target"])
def test_crash_at_each_promotion_rename_recovers_twice_to_a_complete_release(
    tmp_path, monkeypatch, crash_after
):
    target = tmp_path / "v1.0.0"
    old = _release(target, "old")
    stage, new = _prepared_stage(target, "new")
    original = release._rename_directory

    def crash(source, destination):
        original(source, destination)
        did_crash = (crash_after == "target-to-backup" and source == target) or (
            crash_after == "stage-to-target" and source == stage
        )
        if did_crash:
            raise SimulatedCrash("process stopped after directory rename")

    monkeypatch.setattr(release, "_rename_directory", crash)
    with release.release_lock(target), pytest.raises(SimulatedCrash):
        release.commit_release(target, stage, new)

    monkeypatch.setattr(release, "_rename_directory", original)
    _recover_twice(target)
    assert release.inventory_release(target) == (old if crash_after == "target-to-backup" else new)
    assert _transaction_artifacts(target) == set()


def test_failed_promotion_rolls_back_before_reraising(tmp_path, monkeypatch):
    target = tmp_path / "v1.0.0"
    old = _release(target, "old")
    stage, new = _prepared_stage(target, "new")
    original = release._rename_directory

    def fail_new(source, destination):
        if source == stage:
            raise OSError("injected stage promotion failure")
        original(source, destination)

    monkeypatch.setattr(release, "_rename_directory", fail_new)
    with release.release_lock(target), pytest.raises(OSError, match="stage promotion"):
        release.commit_release(target, stage, new)

    _recover_twice(target)
    assert release.inventory_release(target) == old
    assert _transaction_artifacts(target) == set()


def test_failed_rollback_retains_all_recovery_inputs(tmp_path, monkeypatch):
    target = tmp_path / "v1.0.0"
    old = _release(target, "old")
    stage, new = _prepared_stage(target, "new")
    original = release._rename_directory

    def fail_promotion_and_rollback(source, destination):
        if source == stage:
            raise OSError("injected stage promotion failure")
        if source.name.startswith(f".{target.name}.backup-"):
            raise OSError("injected rollback failure")
        original(source, destination)

    monkeypatch.setattr(release, "_rename_directory", fail_promotion_and_rollback)
    with release.release_lock(target), pytest.raises(release.ReleaseCommitError, match="recovery"):
        release.commit_release(target, stage, new)

    _assert_failed_rollback_kept_inputs(target)

    monkeypatch.setattr(release, "_rename_directory", original)
    _recover_twice(target)

    _assert_release_and_no_artifacts(target, old)


def test_cleanup_failure_keeps_the_committed_release_and_journal_for_retry(tmp_path, monkeypatch):
    target = tmp_path / "v1.0.0"
    _release(target, "old")
    stage, new = _prepared_stage(target, "new")
    original = release.shutil.rmtree
    attempts = 0

    def fail_backup_once(path, *args, **kwargs):
        nonlocal attempts
        if Path(path).name.startswith(f".{target.name}.backup-"):
            attempts += 1
            if attempts <= 2:
                raise OSError("injected backup cleanup failure")
        original(path, *args, **kwargs)

    monkeypatch.setattr(release.shutil, "rmtree", fail_backup_once)
    with release.release_lock(target), pytest.raises(release.ReleaseCommitError, match="recovery"):
        release.commit_release(target, stage, new)

    _assert_cleanup_failure_kept_journal(target, new)

    monkeypatch.setattr(release.shutil, "rmtree", original)
    _recover_twice(target)

    _assert_release_and_no_artifacts(target, new)


def test_recovery_finishes_partial_backup_cleanup_by_hashing_remaining_files(tmp_path, monkeypatch):
    target = tmp_path / "v1.0.0"
    old = _release(target, "old", sidecars=("README.md",))
    stage, new = _prepared_stage(target, "new")
    original = release.shutil.rmtree
    attempts = 0

    def partially_remove_backup(path, *args, **kwargs):
        nonlocal attempts
        path = Path(path)
        if path.name.startswith(f".{target.name}.backup-"):
            attempts += 1
            if attempts <= 2:
                next(path.iterdir()).unlink()
                raise OSError("injected interruption during backup cleanup")
        original(path, *args, **kwargs)

    monkeypatch.setattr(release.shutil, "rmtree", partially_remove_backup)
    with release.release_lock(target), pytest.raises(release.ReleaseCommitError, match="recovery"):
        release.commit_release(target, stage, new)

    backup = next(target.parent.glob(f".{target.name}.backup-*"))
    _assert_partial_backup_cleanup_kept_journal(target, old, new, backup)

    monkeypatch.setattr(release.shutil, "rmtree", original)
    _recover_twice(target)

    _assert_release_and_no_artifacts(target, new)


def test_recovery_finishes_partial_stage_cleanup_by_hashing_remaining_files(tmp_path, monkeypatch):
    target = tmp_path / "v1.0.0"
    old = _release(target, "old")
    stage, new = _prepared_stage(target, "new")
    original_fsync_directory = release.fsync_directory
    monkeypatch.setattr(
        release,
        "fsync_directory",
        _crash_at_parent_sync(target, 2, "process stopped after journal installation"),
    )
    with release.release_lock(target), pytest.raises(SimulatedCrash):
        release.commit_release(target, stage, new)
    monkeypatch.setattr(release, "fsync_directory", original_fsync_directory)

    original_rmtree = release.shutil.rmtree

    def stop_during_stage_cleanup(path, *args, **kwargs):
        path = Path(path)
        if path == stage:
            (stage / "train.parquet").unlink()
            raise SimulatedCrash("process stopped during stage cleanup")
        original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(release.shutil, "rmtree", stop_during_stage_cleanup)
    with pytest.raises(SimulatedCrash):
        release.recover_release(target)

    _assert_partial_stage_cleanup_kept_journal(target, stage, old)

    monkeypatch.setattr(release.shutil, "rmtree", original_rmtree)
    _recover_twice(target)

    _assert_release_and_no_artifacts(target, old)


def test_recovery_rejects_a_hash_mismatch_without_deleting_candidates(tmp_path, monkeypatch):
    target = tmp_path / "v1.0.0"
    old = _release(target, "old")
    stage, new = _prepared_stage(target, "new")
    original = release._rename_directory

    def crash_after_backup(source, destination):
        original(source, destination)
        if source == target:
            raise SimulatedCrash("process stopped after backup rename")

    monkeypatch.setattr(release, "_rename_directory", crash_after_backup)
    with release.release_lock(target), pytest.raises(SimulatedCrash):
        release.commit_release(target, stage, new)
    monkeypatch.setattr(release, "_rename_directory", original)

    backup = next(target.parent.glob(f".{target.name}.backup-*"))
    (backup / "train.parquet").write_bytes(b"tampered")
    with pytest.raises(release.ReleaseCommitError, match="backup"):
        release.recover_release(target)
    _assert_hash_mismatch_kept_candidates(target, stage, backup)

    _restore_backup_files(backup, old)
    _recover_twice(target)
    _assert_release_and_no_artifacts(target, old)


def test_shared_reader_lock_blocks_finalization_for_the_entire_read(tmp_path):
    target = tmp_path / "v1.0.0"
    with release.release_lock(target):
        pass
    attempted = threading.Event()
    acquired = threading.Event()

    with release.release_read_lock(target):

        def writer():
            attempted.set()
            with release.release_lock(target):
                acquired.set()

        thread = threading.Thread(target=writer, name="release-writer")
        thread.start()
        assert attempted.wait(1)
        assert not acquired.wait(0.05)

    thread.join(1)
    assert acquired.is_set()
    assert (target.parent / f".{target.name}.lock").is_file()


def test_nested_reader_locks_reuse_the_pinned_context(tmp_path):
    target = tmp_path / "v1.0.0"
    with release.release_lock(target):
        pass

    with release.release_read_lock(target), release.release_read_lock(target):
        assert not _journal_exists(target)


def test_writer_lock_is_reentrant_for_repository_readers(tmp_path):
    target = tmp_path / "v1.0.0"

    with release.release_write_lock(target), release.release_read_lock(target):
        assert not _journal_exists(target)
