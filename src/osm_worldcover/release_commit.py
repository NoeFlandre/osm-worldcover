from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any

try:
    import fcntl
except ImportError:
    fcntl = None

__all__ = [
    "CORE_FILES",
    "FileFingerprint",
    "ReleaseCommitError",
    "ReleaseInventory",
    "commit_release",
    "core_inventory",
    "create_stage",
    "discard_stage",
    "inventory_release",
    "migrate_release_lock",
    "recover_release",
    "recover_release_locked",
    "release_lock",
    "release_read_lock",
    "release_write_lock",
    "stage_inventory",
    "sync_stage",
    "transaction_pending",
]

CORE_FILES = ("train.parquet", "validation.parquet", "test.parquet", "manifest.json")
_CORE_FILE_SET = frozenset(CORE_FILES)
_SIDECAR_FILES = frozenset(
    {"README.md", "worldcover_centroids.png", "release_checksums.json", "publication_receipt.json"}
)
_KNOWN_NEW_FILE_SET = _CORE_FILE_SET | _SIDECAR_FILES
_HEX_32 = re.compile(r"[0-9a-f]{32}\Z")
_HELD_LOCKS: ContextVar[dict[Path, bool] | None] = ContextVar(
    "owc_held_release_locks", default=None
)


class ReleaseCommitError(OSError):
    """A release transaction cannot safely proceed or recover."""


@dataclass(frozen=True, slots=True, order=True)
class FileFingerprint:
    """Size and SHA-256 for one direct file in a release directory."""

    name: str
    size: int
    sha256: str

    def as_dict(self) -> dict[str, int | str]:
        return {"name": self.name, "size": self.size, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class ReleaseInventory:
    """A deterministic inventory of regular release files."""

    files: tuple[FileFingerprint, ...]

    def as_dict(self) -> dict[str, list[dict[str, int | str]]]:
        return {"files": [entry.as_dict() for entry in self.files]}


@dataclass(frozen=True, slots=True)
class _Journal:
    transaction_id: str
    target: str
    stage: str
    backup: str
    old: ReleaseInventory | None
    new: ReleaseInventory
    cleanup_stage: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "transaction_id": self.transaction_id,
            "target": self.target,
            "stage": self.stage,
            "backup": self.backup,
            "old": None if self.old is None else self.old.as_dict(),
            "new": self.new.as_dict(),
            "cleanup_stage": self.cleanup_stage,
        }


@contextmanager
def release_lock(target: Path, *, shared: bool = False) -> Iterator[None]:
    """Lock a version path while leaving the lock file in place permanently."""
    _require_posix_lock_support()
    target = _absolute_target(target)
    _validate_target(target)
    held = _held_locks()
    current_mode = held.get(target)
    if current_mode is not None:
        _require_reentrant_mode(current_mode, shared)
        yield
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.parent / f".{target.name}.lock"
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o666)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ReleaseCommitError(f"release lock is not a regular file: {lock_path}")
        with _lock_descriptor(descriptor, target, shared, held):
            if not shared:
                fsync_directory(target.parent)
            yield
    finally:
        os.close(descriptor)


def _require_reentrant_mode(current_mode: bool, shared: bool) -> None:
    if not current_mode and not shared:
        raise ReleaseCommitError("cannot upgrade a shared release lock to exclusive")


@contextmanager
def _lock_descriptor(
    descriptor: int, target: Path, shared: bool, held: dict[Path, bool]
) -> Iterator[None]:
    lock_module = _require_posix_lock_support()
    lock_module.flock(descriptor, lock_module.LOCK_SH if shared else lock_module.LOCK_EX)
    token = _HELD_LOCKS.set(held | {target: not shared})
    try:
        yield
    finally:
        _HELD_LOCKS.reset(token)
        lock_module.flock(descriptor, lock_module.LOCK_UN)


def _held_locks() -> dict[Path, bool]:
    return _HELD_LOCKS.get() or {}


@contextmanager
def _release_lock_context(target: Path, *, shared: bool = False) -> Iterator[None]:
    with release_lock(target, shared=shared):
        yield


@contextmanager
def _existing_shared_release_lock(target: Path) -> Iterator[None]:
    _require_posix_lock_support()
    target = _absolute_target(target)
    _validate_target(target)
    held = _held_locks()
    lock_path = target.parent / f".{target.name}.lock"
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags)
    except FileNotFoundError as error:
        raise ReleaseCommitError(f"release lock file is missing: {lock_path}") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ReleaseCommitError(f"release lock is not a regular file: {lock_path}")
        with _lock_descriptor(descriptor, target, True, held):
            yield
    finally:
        os.close(descriptor)


@contextmanager
def release_read_lock(target: Path) -> Iterator[None]:
    """Pin a coherent release view without changing the release directory."""
    target = _absolute_target(target)
    if target in _held_locks():
        yield
        return
    with _existing_shared_release_lock(target):
        if transaction_pending(target):
            raise ReleaseCommitError(f"release recovery is required before reading: {target}")
        yield


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


def migrate_release_lock(target: Path) -> Path:
    """Initialize the stable lock for an existing, structurally complete release."""
    target = _absolute_target(target)
    _validate_target(target)
    lock_path = target.parent / f".{target.name}.lock"
    _preflight_lock_migration(target, lock_path)
    with release_lock(target):
        _validate_lock_migration_target(target)
    return lock_path


def _preflight_lock_migration(target: Path, lock_path: Path) -> None:
    if _lexists(lock_path):
        return
    _require_no_unlocked_pending_migration(target, lock_path)
    try:
        _validate_lock_migration_target(target)
    except ReleaseCommitError:
        if not _lexists(lock_path):
            raise


def _require_no_unlocked_pending_migration(target: Path, lock_path: Path) -> None:
    if transaction_pending(target) and not _lexists(lock_path):
        raise ReleaseCommitError(f"release recovery is required before lock migration: {target}")


def _validate_lock_migration_target(target: Path) -> None:
    if transaction_pending(target):
        raise ReleaseCommitError(f"release recovery is required before lock migration: {target}")
    _require_real_directory(target)
    _require_core_release_files(target)


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


def inventory_release(directory: Path) -> ReleaseInventory:
    """Hash every direct regular file, rejecting symlinks and nested paths."""
    directory = Path(directory)
    _require_real_directory(directory)
    entries = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if path.is_symlink() or not path.is_file():
            raise ReleaseCommitError(f"unexpected release entry: {path}")
        entries.append(_fingerprint(path))
    return ReleaseInventory(tuple(entries))


def stage_inventory(stage: Path) -> ReleaseInventory:
    """Require and hash the exact four files owned by finalization."""
    inventory = inventory_release(stage)
    if {entry.name for entry in inventory.files} != _CORE_FILE_SET:
        raise ReleaseCommitError(f"staged release is missing or has extra files: {stage}")
    return inventory


def core_inventory(directory: Path) -> ReleaseInventory | None:
    """Return the four finalizer-owned hashes from a complete release."""
    directory = Path(directory)
    if not _lexists(directory):
        return None
    inventory = inventory_release(directory)
    files = {entry.name: entry for entry in inventory.files}
    if not files.keys() >= _CORE_FILE_SET:
        raise ReleaseCommitError(f"existing release is incomplete: {directory}")
    return ReleaseInventory(tuple(sorted(files[name] for name in _CORE_FILE_SET)))


def transaction_pending(target: Path) -> bool:
    """Whether a durable release journal is present for the version path."""
    return _lexists(_journal_path(_absolute_target(target)))


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


def _absolute_stage(stage: Path) -> Path:
    stage = Path(stage)
    return stage.parent.resolve() / stage.name


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


@contextmanager
def scratch_lock(directory: Path) -> Iterator[None]:
    """Serialise runs that share one scratch directory, since each rebuilds its contents."""
    lock_module = _require_posix_lock_support()
    directory.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(directory / ".scratch.lock", flags, 0o666)
    try:
        lock_module.flock(descriptor, lock_module.LOCK_EX)
        try:
            yield
        finally:
            lock_module.flock(descriptor, lock_module.LOCK_UN)
    finally:
        os.close(descriptor)


def _flush_to_media(descriptor: int) -> None:
    """Flush to the storage device, not just the OS cache.

    On macOS ``os.fsync`` stops at the drive cache, so ``F_FULLFSYNC`` is used when
    the platform has it. A filesystem that refuses it falls back to ``os.fsync``.
    """
    full_sync = getattr(fcntl, "F_FULLFSYNC", None)
    if fcntl is None or full_sync is None:
        os.fsync(descriptor)
        return
    try:
        fcntl.fcntl(descriptor, full_sync)
    except OSError:
        os.fsync(descriptor)


def fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ReleaseCommitError(f"release file is not regular: {path}")
        _flush_to_media(descriptor)
    finally:
        os.close(descriptor)


def fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ReleaseCommitError(f"release path is not a directory: {path}")
        _flush_to_media(descriptor)
    finally:
        os.close(descriptor)


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


def _target_matches_new(target: Path, expected: ReleaseInventory) -> bool:
    if not _lexists(target):
        return False
    actual = inventory_release(target)
    files = {entry.name: entry for entry in actual.files}
    names = set(files)
    return (
        _CORE_FILE_SET <= names <= _KNOWN_NEW_FILE_SET
        and ReleaseInventory(tuple(sorted(files[name] for name in CORE_FILES))) == expected
    )


def _matches_inventory(directory: Path, expected: ReleaseInventory) -> bool:
    return _lexists(directory) and inventory_release(directory) == expected


def _matches_partial_inventory(directory: Path, expected: ReleaseInventory | None) -> bool:
    if expected is None or not _lexists(directory):
        return False
    actual = inventory_release(directory)
    known = {entry.name: entry for entry in expected.files}
    return all(known.get(entry.name) == entry for entry in actual.files)


def _inventory_if_present(directory: Path) -> ReleaseInventory | None:
    if not _lexists(directory):
        return None
    inventory = inventory_release(directory)
    if not {entry.name for entry in inventory.files} >= _CORE_FILE_SET:
        raise ReleaseCommitError(f"existing release is incomplete: {directory}")
    return inventory


def _write_journal(path: Path, journal: _Journal) -> None:
    temporary = path.with_name(f"{path.name}.{journal.transaction_id}.tmp")
    payload = (json.dumps(journal.as_dict(), indent=2, sort_keys=True) + "\n").encode()
    try:
        if _lexists(temporary):
            _discard_stale_journal_temporary(temporary, path.parent)
        descriptor = os.open(
            temporary,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
        fsync_file(temporary)
        temporary.replace(path)
        fsync_directory(path.parent)
    except BaseException:
        if _lexists(temporary):
            with suppress(OSError):
                temporary.unlink()
        raise


def _discard_stale_journal_temporary(path: Path, parent: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ReleaseCommitError(f"journal temporary is not a regular file: {path}")
    path.unlink()
    fsync_directory(parent)


def _read_journal(target: Path, path: Path) -> _Journal:
    if path.is_symlink() or not path.is_file():
        raise ReleaseCommitError(f"release journal is not a regular file: {path}")
    try:
        return _journal_from_dict(target, json.loads(path.read_text()))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ReleaseCommitError(f"invalid release journal {path}: {error}") from error


def _journal_from_dict(target: Path, document: Any) -> _Journal:
    _require_journal_shape(document)
    transaction_id = _journal_transaction_id(document)
    stage, backup = _candidate_names(target, transaction_id)
    _validate_journal_paths(target, document, stage, backup)
    old = _optional_inventory_from_dict(document["old"])
    new = _inventory_from_dict(document["new"])
    cleanup_stage = _journal_cleanup_stage(document)
    _validate_journal_inventories(old, new)
    return _Journal(transaction_id, target.name, stage, backup, old, new, cleanup_stage)


def _require_journal_shape(document: Any) -> None:
    required_fields = {
        "format_version",
        "transaction_id",
        "target",
        "stage",
        "backup",
        "old",
        "new",
    }
    expected_fields = required_fields | {"cleanup_stage"}
    if not isinstance(document, dict) or set(document) not in (required_fields, expected_fields):
        raise ValueError("unexpected journal fields")


def _journal_transaction_id(document: dict[str, Any]) -> str:
    transaction_id = document["transaction_id"]
    if document["format_version"] != 1 or not isinstance(transaction_id, str):
        raise ValueError("unsupported journal version")
    return transaction_id


def _validate_journal_paths(
    target: Path, document: dict[str, Any], stage: str, backup: str
) -> None:
    paths = document["target"], document["stage"], document["backup"]
    if paths != (target.name, stage, backup):
        raise ValueError("journal paths do not match the target")


def _journal_cleanup_stage(document: dict[str, Any]) -> bool:
    cleanup_stage = document.get("cleanup_stage", False)
    if not isinstance(cleanup_stage, bool):
        raise TypeError("stage cleanup marker must be a boolean")
    return cleanup_stage


def _validate_journal_inventories(old: ReleaseInventory | None, new: ReleaseInventory) -> None:
    if {entry.name for entry in new.files} != _CORE_FILE_SET:
        raise ValueError("new inventory is not the complete core release")
    _validate_release_inventory(old)
    _validate_release_inventory(new, core_only=True)


def _optional_inventory_from_dict(document: Any) -> ReleaseInventory | None:
    return None if document is None else _inventory_from_dict(document)


def _inventory_from_dict(document: Any) -> ReleaseInventory:
    entries = _inventory_entries(document)
    files = [_fingerprint_from_dict(entry) for entry in entries]
    if _has_duplicate_inventory_files(files):
        raise ValueError("duplicate inventory file")
    return ReleaseInventory(tuple(sorted(files)))


def _inventory_entries(document: Any) -> list[Any]:
    if not isinstance(document, dict) or set(document) != {"files"}:
        raise ValueError("invalid inventory document")
    entries = document["files"]
    if not isinstance(entries, list):
        raise TypeError("inventory files must be a list")
    return entries


def _has_duplicate_inventory_files(files: list[FileFingerprint]) -> bool:
    return len({entry.name for entry in files}) != len(files)


def _fingerprint_from_dict(entry: Any) -> FileFingerprint:
    if not isinstance(entry, dict) or set(entry) != {"name", "size", "sha256"}:
        raise ValueError("invalid inventory entry")
    name, size, digest = entry["name"], entry["size"], entry["sha256"]
    _validate_fingerprint_name(name)
    _validate_fingerprint_size(size)
    _validate_fingerprint_digest(digest)
    return FileFingerprint(name, size, digest)


_UNSAFE_NAME_CHARACTERS = re.compile(r"[\x00-\x1f\x7f/\\]")


def _validate_fingerprint_name(name: Any) -> None:
    if not isinstance(name, str) or name in {"", ".", ".."} or _UNSAFE_NAME_CHARACTERS.search(name):
        raise ValueError("invalid inventory filename")


def _validate_fingerprint_size(size: Any) -> None:
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError("invalid inventory size")


def _validate_fingerprint_digest(digest: Any) -> None:
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("invalid inventory hash")


def _validate_release_inventory(
    inventory: ReleaseInventory | None, *, core_only: bool = False
) -> None:
    if inventory is None:
        return
    names = {entry.name for entry in inventory.files}
    _validate_release_layout(names)
    if core_only:
        _validate_core_release_layout(names)


def _validate_release_layout(names: set[str]) -> None:
    if not names >= _CORE_FILE_SET:
        raise ValueError("inventory does not contain a complete release layout")


def _validate_core_release_layout(names: set[str]) -> None:
    if names != _CORE_FILE_SET:
        raise ValueError("new inventory contains release sidecars")


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


def _rename_directory(source: Path, destination: Path) -> None:
    if source.parent != destination.parent or source.is_symlink() or not source.is_dir():
        raise ReleaseCommitError(f"directory rename must stay beside its source: {source}")
    if _lexists(destination):
        raise ReleaseCommitError(f"release rename destination already exists: {destination}")
    source.rename(destination)


def _fingerprint(path: Path) -> FileFingerprint:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ReleaseCommitError(f"release path is not a regular file: {path}")
        digest = hashlib.sha256()
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        after = os.fstat(descriptor)
        if _stat_key(before) != _stat_key(after):
            raise ReleaseCommitError(f"release file changed while hashing: {path}")
        return FileFingerprint(path.name, int(after.st_size), digest.hexdigest())
    finally:
        os.close(descriptor)


def _stat_key(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _require_real_directory(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ReleaseCommitError(f"release candidate is not a real directory: {path}")


def _require_core_release_files(directory: Path) -> None:
    missing, non_regular = _core_release_file_issues(directory)
    if missing:
        raise ReleaseCommitError(f"existing release is missing core file(s): {missing}")
    if non_regular:
        raise ReleaseCommitError(f"existing release has non-regular core file(s): {non_regular}")


def _core_release_file_issues(directory: Path) -> tuple[list[str], list[str]]:
    missing = []
    non_regular = []
    for name in CORE_FILES:
        path = directory / name
        file_type = _core_release_file_type(path)
        if file_type == "missing":
            missing.append(name)
        elif file_type == "non-regular":
            non_regular.append(name)
    return missing, non_regular


def _core_release_file_type(path: Path) -> str:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return "missing"
    return "regular" if stat.S_ISREG(mode) else "non-regular"


def _candidate_names(target: Path, transaction_id: str) -> tuple[str, str]:
    if not _HEX_32.fullmatch(transaction_id):
        raise ValueError("invalid transaction id")
    return (
        f".{target.name}.stage-{transaction_id}",
        f".{target.name}.backup-{transaction_id}",
    )


def _journal_candidates(target: Path, journal: _Journal) -> tuple[Path, Path]:
    stage_name, backup_name = _candidate_names(target, journal.transaction_id)
    if journal.target != target.name or (journal.stage, journal.backup) != (
        stage_name,
        backup_name,
    ):
        raise ReleaseCommitError(f"journal paths do not belong to {target}")
    return target.parent / stage_name, target.parent / backup_name


def _validate_stage_path(target: Path, stage: Path) -> str:
    if stage.parent != target.parent:
        raise ReleaseCommitError(
            "staging directory and release target must share a filesystem parent"
        )
    prefix = f".{target.name}.stage-"
    if not stage.name.startswith(prefix):
        raise ReleaseCommitError(f"staging path does not belong to {target}: {stage}")
    transaction_id = stage.name[len(prefix) :]
    _candidate_names(target, transaction_id)
    return transaction_id


def _journal_path(target: Path) -> Path:
    _validate_target(target)
    return target.parent / f".{target.name}.transaction.json"


def _validate_target(target: Path) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", target.name):
        raise ReleaseCommitError(f"unsafe release directory name: {target.name!r}")
    if target.is_symlink():
        raise ReleaseCommitError(f"release target must not be a symbolic link: {target}")


def _absolute_target(target: Path) -> Path:
    path = Path(target)
    return path.parent.resolve() / path.name


def _require_exclusive_lock(target: Path) -> None:
    target = _absolute_target(target)
    if _held_locks().get(target) is not True:
        raise ReleaseCommitError(f"an exclusive release lock is required for {target}")


def _require_posix_lock_support() -> ModuleType:
    if fcntl is None:
        raise ReleaseCommitError(
            "release finalization requires POSIX file locks and directory fsync (macOS or Linux)"
        )
    return fcntl


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)
