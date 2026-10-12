"""Promote a validated stage over a release, journaling every step so it can recover."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from osm_worldcover.release.durability import fsync_directory, fsync_file
from osm_worldcover.release.errors import ReleaseCommitError
from osm_worldcover.release.inventory import (
    _CORE_FILE_SET,
    CORE_FILES,
    ReleaseInventory,
    _inventory_if_present,
    _matches_inventory,
    _target_matches_new,
    stage_inventory,
)
from osm_worldcover.release.journal import _Journal, _write_journal
from osm_worldcover.release.locks import _require_exclusive_lock
from osm_worldcover.release.paths import (
    _HEX_32,
    _absolute_stage,
    _absolute_target,
    _journal_path,
    _lexists,
    _rename_directory,
    _validate_stage_path,
    _validate_target,
)
from osm_worldcover.release.recovery import _finish_committed, recover_release_locked


def create_stage(target: Path) -> Path:
    """Create a unique staging directory beside the release target."""
    target = _absolute_target(target)
    _require_exclusive_lock(target)
    _validate_target(target)
    transaction_id = uuid.uuid4().hex
    stage = target.parent / f".{target.name}.stage-{transaction_id}"
    stage.mkdir()
    return stage


def discard_stage(target: Path, stage: Path) -> None:
    """Remove one private stage and durably record its removal."""
    target = _absolute_target(target)
    stage = Path(stage).parent.resolve() / Path(stage).name
    _require_exclusive_lock(target)
    _validate_stage_path(target, stage)
    if _lexists(stage):
        if stage.is_symlink() or not stage.is_dir():
            raise ReleaseCommitError(f"staging path is not a real directory: {stage}")
        shutil.rmtree(stage)
        fsync_directory(target.parent)


def sync_stage(stage: Path, parent: Path) -> None:
    """Flush every managed file and both directory entries before journaling."""
    stage = _absolute_stage(stage)
    parent = Path(parent).resolve()
    target = _target_for_stage(stage, parent)
    _require_exclusive_lock(target)
    _require_stage_directory(stage, parent)
    for name in CORE_FILES:
        fsync_file(stage / name)
    fsync_directory(stage)
    fsync_directory(parent)


def _target_for_stage(stage: Path, parent: Path) -> Path:
    marker = stage.name.rfind(".stage-")
    if marker < 2 or not _HEX_32.fullmatch(stage.name[marker + len(".stage-") :]):
        raise ReleaseCommitError(f"staging path is not a private release stage: {stage}")
    target = parent / stage.name[1:marker]
    _validate_target(target)
    return target


def _require_stage_directory(stage: Path, parent: Path) -> None:
    if stage.parent != parent or stage.is_symlink() or not stage.is_dir():
        raise ReleaseCommitError(f"invalid staging directory: {stage}")


def commit_release(target: Path, stage: Path, new: ReleaseInventory) -> None:
    """Flush and promote a validated stage under the exclusive version lock."""
    target = _absolute_target(target)
    stage = _absolute_stage(stage)
    _validate_target(target)
    _require_exclusive_lock(target)
    journal_path, backup, journal = _prepare_release_commit(target, stage, new)
    try:
        _promote_directories(target, stage, backup, journal_path, journal, journal.old)
    except Exception as error:
        if _recover_failed_promotion(target, journal_path, new, error):
            return
        raise


def _prepare_release_commit(
    target: Path, stage: Path, new: ReleaseInventory
) -> tuple[Path, Path, _Journal]:
    transaction_id = _validate_stage_path(target, stage)
    sync_stage(stage, target.parent)
    if stage_inventory(stage) != new or {entry.name for entry in new.files} != _CORE_FILE_SET:
        raise ReleaseCommitError(f"staged release inventory changed before commit: {stage}")
    journal_path = _journal_path(target)
    if _lexists(journal_path):
        raise ReleaseCommitError(f"an unrecovered release journal already exists: {journal_path}")
    old = _inventory_if_present(target)
    backup = target.parent / f".{target.name}.backup-{transaction_id}"
    journal = _Journal(
        transaction_id=transaction_id,
        target=target.name,
        stage=stage.name,
        backup=backup.name,
        old=old,
        new=new,
    )
    _write_journal(journal_path, journal)
    return journal_path, backup, journal


def _promote_directories(
    target: Path,
    stage: Path,
    backup: Path,
    journal_path: Path,
    journal: _Journal,
    old: ReleaseInventory | None,
) -> None:
    if old is not None:
        _rename_directory(target, backup)
        fsync_directory(target.parent)
    if not _matches_inventory(stage, journal.new):
        raise ReleaseCommitError(f"staged release changed before promotion: {stage}")
    _rename_directory(stage, target)
    fsync_directory(target.parent)
    if not _target_matches_new(target, journal.new):
        raise ReleaseCommitError(f"promoted release does not match its staged inventory: {target}")
    _finish_committed(target, journal_path, journal)


def _recover_failed_promotion(
    target: Path,
    journal_path: Path,
    new: ReleaseInventory,
    error: Exception,
) -> bool:
    try:
        recover_release_locked(target)
    except Exception as recovery_error:
        raise ReleaseCommitError(
            f"promotion failed and recovery is pending at {journal_path}: {recovery_error}"
        ) from error
    return _target_matches_new(target, new)
