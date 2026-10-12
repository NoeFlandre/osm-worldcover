"""Recover an interrupted release transaction, and pin a release for writing."""

from __future__ import annotations

import re
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from osm_worldcover.release.durability import fsync_directory
from osm_worldcover.release.errors import ReleaseCommitError
from osm_worldcover.release.inventory import (
    ReleaseInventory,
    _matches_inventory,
    _matches_partial_inventory,
    _target_matches_new,
)
from osm_worldcover.release.journal import _Journal, _read_journal, _write_journal
from osm_worldcover.release.locks import (
    _held_locks,
    _release_lock_context,
    _require_exclusive_lock,
    release_lock,
)
from osm_worldcover.release.paths import (
    _absolute_target,
    _candidate_names,
    _journal_path,
    _lexists,
    _rename_directory,
)


@contextmanager
def release_write_lock(target: Path) -> Iterator[None]:
    """Recover, then pin a release against readers and other sidecar writers."""
    target = _absolute_target(target)
    mode = _held_locks().get(target)
    if mode is not None:
        if not mode:
            raise ReleaseCommitError("cannot upgrade a shared release lock to exclusive")
        yield
        return
    with release_lock(target):
        recover_release_locked(target)
        yield


def recover_release(target: Path) -> None:
    """Recover a pending transaction while holding the version's exclusive lock."""
    target = _absolute_target(target)
    with _release_lock_context(target):
        recover_release_locked(target)


def recover_release_locked(target: Path) -> None:
    """Recover a transaction; the caller must already hold the exclusive lock."""
    target = _absolute_target(target)
    _require_exclusive_lock(target)
    journal_path = _journal_path(target)
    if not _lexists(journal_path):
        _remove_orphan_stages(target)
        return
    journal = _read_journal(target, journal_path)
    state = _transaction_state(target, journal)
    _recover_transaction_state(target, journal_path, journal, state)
    _remove_journal_temporary_files(target)


def _recover_transaction_state(
    target: Path, journal_path: Path, journal: _Journal, state: str
) -> None:
    handlers = {
        "committed": _finish_committed,
        "old-at-target": _abort_unstarted,
        "old-at-backup": _restore_old,
        "no-old-stage": _abort_unstarted,
        "no-old-cleanup-complete": _abort_unstarted,
    }
    handler = handlers.get(state)
    if handler is None:
        raise ReleaseCommitError(f"unrecognized release transaction state for {target}")
    handler(target, journal_path, journal)


def _transaction_state(target: Path, journal: _Journal) -> str:
    stage, backup = _journal_candidates(target, journal)
    target_exists, stage_exists, backup_exists = map(_lexists, (target, stage, backup))
    state = _committed_state(
        target, stage, backup, target_exists, stage_exists, backup_exists, journal
    )
    if state is not None:
        return state
    state = _old_release_state(
        target, stage, backup, target_exists, stage_exists, backup_exists, journal
    )
    if state is not None:
        return state
    state = _no_old_release_state(stage, target_exists, stage_exists, backup_exists, journal)
    if state is not None:
        return state
    raise ReleaseCommitError(
        f"release candidates do not match the transaction journal for {target}"
    )


def _committed_state(
    target: Path,
    stage: Path,
    backup: Path,
    target_exists: bool,
    stage_exists: bool,
    backup_exists: bool,
    journal: _Journal,
) -> str | None:
    if not target_exists or not _target_matches_new(target, journal.new):
        return None
    _validate_committed_candidates(stage, stage_exists, backup, backup_exists, journal)
    return "committed"


def _old_release_state(
    target: Path,
    stage: Path,
    backup: Path,
    target_exists: bool,
    stage_exists: bool,
    backup_exists: bool,
    journal: _Journal,
) -> str | None:
    old = journal.old
    if old is None:
        return None
    state = _old_release_at_target(
        target, target_exists, stage, stage_exists, backup, backup_exists, old, journal
    )
    if state is not None:
        return state
    return _old_release_at_backup(
        target_exists, stage, stage_exists, backup, backup_exists, old, journal
    )


def _old_release_at_target(
    target: Path,
    target_exists: bool,
    stage: Path,
    stage_exists: bool,
    backup: Path,
    backup_exists: bool,
    old: ReleaseInventory,
    journal: _Journal,
) -> str | None:
    if not target_exists or not _matches_inventory(target, old):
        return None
    _validate_old_target_candidates(backup, backup_exists, stage, stage_exists, journal)
    return "old-at-target"


def _old_release_at_backup(
    target_exists: bool,
    stage: Path,
    stage_exists: bool,
    backup: Path,
    backup_exists: bool,
    old: ReleaseInventory,
    journal: _Journal,
) -> str | None:
    if target_exists or not backup_exists:
        return None
    if not _matches_inventory(backup, old):
        raise ReleaseCommitError(f"backup does not match the recorded old release: {backup}")
    _require_stage_candidate(stage, stage_exists, journal)
    return "old-at-backup"


def _no_old_release_state(
    stage: Path,
    target_exists: bool,
    stage_exists: bool,
    backup_exists: bool,
    journal: _Journal,
) -> str | None:
    if _transaction_has_old_release(target_exists, backup_exists, journal):
        return None
    if stage_exists:
        _require_stage_candidate(stage, True, journal)
        return "no-old-stage"
    if journal.cleanup_stage:
        return "no-old-cleanup-complete"
    return None


def _transaction_has_old_release(
    target_exists: bool, backup_exists: bool, journal: _Journal
) -> bool:
    return target_exists or backup_exists or journal.old is not None


def _validate_committed_candidates(
    stage: Path, stage_exists: bool, backup: Path, backup_exists: bool, journal: _Journal
) -> None:
    _require_stage_candidate(stage, stage_exists, journal)
    _require_old_backup(backup, backup_exists, journal.old, allow_partial=True)


def _validate_old_target_candidates(
    backup: Path,
    backup_exists: bool,
    stage: Path,
    stage_exists: bool,
    journal: _Journal,
) -> None:
    if backup_exists:
        raise ReleaseCommitError(f"unexpected backup beside the old release: {backup}")
    _require_stage_candidate(stage, stage_exists, journal)


def _validate_old_backup_candidates(
    backup: Path, stage: Path, stage_exists: bool, journal: _Journal
) -> None:
    old = journal.old
    if old is None or not _matches_inventory(backup, old):
        raise ReleaseCommitError(f"backup does not match the recorded old release: {backup}")
    _require_stage_candidate(stage, stage_exists, journal)


def _finish_committed(target: Path, journal_path: Path, journal: _Journal) -> None:
    stage, backup = _journal_candidates(target, journal)
    fsync_directory(target.parent)
    if _lexists(backup):
        _remove_committed_backup(backup, journal.old)
        fsync_directory(target.parent)
    if _lexists(stage):
        journal = _enable_stage_cleanup(journal_path, journal)
        _remove_partial_verified_tree(stage, journal.new)
        fsync_directory(target.parent)
    journal_path.unlink()
    fsync_directory(target.parent)


def _abort_unstarted(target: Path, journal_path: Path, journal: _Journal) -> None:
    stage, _backup = _journal_candidates(target, journal)
    if _lexists(stage):
        journal = _enable_stage_cleanup(journal_path, journal)
        _remove_partial_verified_tree(stage, journal.new)
        fsync_directory(target.parent)
    journal_path.unlink()
    fsync_directory(target.parent)


def _restore_old(target: Path, journal_path: Path, journal: _Journal) -> None:
    stage, backup = _journal_candidates(target, journal)
    _rename_directory(backup, target)
    fsync_directory(target.parent)
    if _lexists(stage):
        journal = _enable_stage_cleanup(journal_path, journal)
        _remove_partial_verified_tree(stage, journal.new)
        fsync_directory(target.parent)
    journal_path.unlink()
    fsync_directory(target.parent)


def _enable_stage_cleanup(journal_path: Path, journal: _Journal) -> _Journal:
    if journal.cleanup_stage:
        return journal
    journal = replace(journal, cleanup_stage=True)
    _write_journal(journal_path, journal)
    return journal


def _remove_partial_verified_tree(path: Path, expected: ReleaseInventory) -> None:
    if path.is_symlink() or not path.is_dir() or not _matches_partial_inventory(path, expected):
        raise ReleaseCommitError(f"transaction stage changed before cleanup: {path}")
    shutil.rmtree(path)


def _remove_committed_backup(path: Path, expected: ReleaseInventory | None) -> None:
    if not _matches_partial_inventory(path, expected):
        raise ReleaseCommitError(f"committed backup changed before cleanup: {path}")
    shutil.rmtree(path)


def _require_stage_candidate(path: Path, exists: bool, journal: _Journal) -> None:
    if not exists:
        return
    matches = (
        _matches_partial_inventory(path, journal.new)
        if journal.cleanup_stage
        else _matches_inventory(path, journal.new)
    )
    if not matches:
        raise ReleaseCommitError(f"stage does not match the recorded new release: {path}")


def _require_old_backup(
    backup: Path,
    exists: bool,
    expected: ReleaseInventory | None,
    *,
    allow_partial: bool = False,
) -> None:
    if not exists:
        return
    if not _old_backup_matches(backup, expected, allow_partial):
        raise ReleaseCommitError(f"backup does not match the recorded old release: {backup}")


def _old_backup_matches(
    backup: Path, expected: ReleaseInventory | None, allow_partial: bool
) -> bool:
    if expected is None:
        return False
    if allow_partial and _matches_partial_inventory(backup, expected):
        return True
    return _matches_inventory(backup, expected)


def _remove_orphan_stages(target: Path) -> None:
    parent = target.parent
    patterns = _orphan_patterns(target)
    for path in sorted(parent.iterdir(), key=lambda item: item.name):
        _clean_orphan_candidate(path, parent, patterns)


def _remove_journal_temporary_files(target: Path) -> None:
    parent = target.parent
    temporary_pattern = _orphan_patterns(target)[1]
    for path in sorted(parent.iterdir(), key=lambda item: item.name):
        if temporary_pattern.fullmatch(path.name):
            _remove_orphan_journal_temp(path, parent)


def _orphan_patterns(target: Path) -> tuple[re.Pattern[str], re.Pattern[str], re.Pattern[str]]:
    prefix = re.escape(target.name)
    stage = re.compile(rf"\.{prefix}\.stage-[0-9a-f]{{32}}\Z")
    temporary = re.compile(rf"\.{prefix}\.transaction\.json\.[0-9a-f]{{32}}\.tmp\Z")
    backup = re.compile(rf"\.{prefix}\.backup-[0-9a-f]{{32}}\Z")
    return stage, temporary, backup


def _clean_orphan_candidate(
    path: Path, parent: Path, patterns: tuple[re.Pattern[str], re.Pattern[str], re.Pattern[str]]
) -> None:
    stage_pattern, temporary_pattern, backup_pattern = patterns
    if backup_pattern.fullmatch(path.name):
        raise ReleaseCommitError(f"backup exists without a recovery journal: {path}")
    if stage_pattern.fullmatch(path.name):
        _remove_orphan_stage(path, parent)
    elif temporary_pattern.fullmatch(path.name):
        _remove_orphan_journal_temp(path, parent)


def _remove_orphan_stage(path: Path, parent: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ReleaseCommitError(f"orphan staging path is not a real directory: {path}")
    shutil.rmtree(path)
    fsync_directory(parent)


def _remove_orphan_journal_temp(path: Path, parent: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ReleaseCommitError(f"orphan journal temporary is not a regular file: {path}")
    path.unlink()
    fsync_directory(parent)


def _journal_candidates(target: Path, journal: _Journal) -> tuple[Path, Path]:
    stage_name, backup_name = _candidate_names(target, journal.transaction_id)
    if journal.target != target.name or (journal.stage, journal.backup) != (
        stage_name,
        backup_name,
    ):
        raise ReleaseCommitError(f"journal paths do not belong to {target}")
    return target.parent / stage_name, target.parent / backup_name
