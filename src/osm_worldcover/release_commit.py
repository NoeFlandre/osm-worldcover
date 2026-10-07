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
_RELEASE_FILE_SET = _CORE_FILE_SET | _SIDECAR_FILES
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
        if not current_mode and not shared:
            raise ReleaseCommitError("cannot upgrade a shared release lock to exclusive")
        yield
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.parent / f".{target.name}.lock"
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ReleaseCommitError(f"release lock is not a regular file: {lock_path}")
        with _lock_descriptor(descriptor, target, shared, held):
            yield
    finally:
        os.close(descriptor)


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
    if not _lexists(lock_path):
        if transaction_pending(target) and not _lexists(lock_path):
            raise ReleaseCommitError(
                f"release recovery is required before lock migration: {target}"
            )
        try:
            _require_real_directory(target)
            _require_core_release_files(target)
        except ReleaseCommitError:
            if not _lexists(lock_path):
                raise

    with release_lock(target):
        if transaction_pending(target):
            raise ReleaseCommitError(
                f"release recovery is required before lock migration: {target}"
            )
        _require_real_directory(target)
        _require_core_release_files(target)
    return lock_path


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
    if state == "committed":
        _finish_committed(target, journal_path, journal)
    elif state == "old-at-target":
        _abort_unstarted(target, journal_path, journal)
    elif state == "old-at-backup":
        _restore_old(target, journal_path, journal)
    elif state in {"no-old-stage", "no-old-cleanup-complete"}:
        _abort_unstarted(target, journal_path, journal)
    else:
        raise ReleaseCommitError(f"unrecognized release transaction state for {target}")
    _remove_journal_temporary_files(target)


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
        if path.is_symlink() or not path.is_file() or path.name not in _RELEASE_FILE_SET:
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
    stage = Path(stage)
    parent = Path(parent).resolve()
    stage = stage.parent.resolve() / stage.name
    marker = stage.name.rfind(".stage-")
    if marker < 2 or not _HEX_32.fullmatch(stage.name[marker + len(".stage-") :]):
        raise ReleaseCommitError(f"staging path is not a private release stage: {stage}")
    target = parent / stage.name[1:marker]
    _validate_target(target)
    _require_exclusive_lock(target)
    if stage.parent != parent or stage.is_symlink() or not stage.is_dir():
        raise ReleaseCommitError(f"invalid staging directory: {stage}")
    for name in CORE_FILES:
        fsync_file(stage / name)
    fsync_directory(stage)
    fsync_directory(parent)


def commit_release(target: Path, stage: Path, new: ReleaseInventory) -> None:
    """Flush and promote a validated stage under the exclusive version lock."""
    target = _absolute_target(target)
    stage = Path(stage).parent.resolve() / Path(stage).name
    _validate_target(target)
    _require_exclusive_lock(target)
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
    try:
        _promote_directories(target, stage, backup, journal_path, journal, old)
    except Exception as error:
        if _recover_failed_promotion(target, journal_path, new, error):
            return
        raise


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


def fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ReleaseCommitError(f"release file is not regular: {path}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ReleaseCommitError(f"release path is not a directory: {path}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _transaction_state(target: Path, journal: _Journal) -> str:
    stage, backup = _journal_candidates(target, journal)
    target_exists, stage_exists, backup_exists = map(_lexists, (target, stage, backup))
    if target_exists and _target_matches_new(target, journal.new):
        _validate_committed_candidates(stage, stage_exists, backup, backup_exists, journal)
        return "committed"
    if target_exists and journal.old is not None and _matches_inventory(target, journal.old):
        _validate_old_target_candidates(backup, backup_exists, stage, stage_exists, journal)
        return "old-at-target"
    if not target_exists and journal.old is not None and backup_exists:
        _validate_old_backup_candidates(backup, stage, stage_exists, journal)
        return "old-at-backup"
    if not target_exists and journal.old is None and not backup_exists and stage_exists:
        _require_stage_candidate(stage, True, journal)
        return "no-old-stage"
    if (
        not target_exists
        and journal.old is None
        and not backup_exists
        and not stage_exists
        and journal.cleanup_stage
    ):
        return "no-old-cleanup-complete"
    raise ReleaseCommitError(
        f"release candidates do not match the transaction journal for {target}"
    )


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
    if expected is None:
        raise ReleaseCommitError(f"backup does not match the recorded old release: {backup}")
    matches = _matches_partial_inventory(backup, expected) if allow_partial else False
    matches = matches or _matches_inventory(backup, expected)
    if not matches:
        raise ReleaseCommitError(f"backup does not match the recorded old release: {backup}")


def _target_matches_new(target: Path, expected: ReleaseInventory) -> bool:
    if not _lexists(target):
        return False
    actual = inventory_release(target)
    files = {entry.name: entry for entry in actual.files}
    return (
        files.keys() >= _CORE_FILE_SET
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
    transaction_id = document["transaction_id"]
    if document["format_version"] != 1 or not isinstance(transaction_id, str):
        raise ValueError("unsupported journal version")
    stage, backup = _candidate_names(target, transaction_id)
    paths = document["target"], document["stage"], document["backup"]
    if paths != (target.name, stage, backup):
        raise ValueError("journal paths do not match the target")
    old = _optional_inventory_from_dict(document["old"])
    new = _inventory_from_dict(document["new"])
    cleanup_stage = document.get("cleanup_stage", False)
    if not isinstance(cleanup_stage, bool):
        raise TypeError("stage cleanup marker must be a boolean")
    if {entry.name for entry in new.files} != _CORE_FILE_SET:
        raise ValueError("new inventory is not the complete core release")
    _validate_release_inventory(old)
    _validate_release_inventory(new, core_only=True)
    return _Journal(transaction_id, target.name, stage, backup, old, new, cleanup_stage)


def _optional_inventory_from_dict(document: Any) -> ReleaseInventory | None:
    return None if document is None else _inventory_from_dict(document)


def _inventory_from_dict(document: Any) -> ReleaseInventory:
    if not isinstance(document, dict) or set(document) != {"files"}:
        raise ValueError("invalid inventory document")
    entries = document["files"]
    if not isinstance(entries, list):
        raise TypeError("inventory files must be a list")
    files = [_fingerprint_from_dict(entry) for entry in entries]
    if len({entry.name for entry in files}) != len(files):
        raise ValueError("duplicate inventory file")
    return ReleaseInventory(tuple(sorted(files)))


def _fingerprint_from_dict(entry: Any) -> FileFingerprint:
    if not isinstance(entry, dict) or set(entry) != {"name", "size", "sha256"}:
        raise ValueError("invalid inventory entry")
    name, size, digest = entry["name"], entry["size"], entry["sha256"]
    if not isinstance(name, str) or name not in _RELEASE_FILE_SET:
        raise ValueError("invalid inventory filename")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError("invalid inventory size")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("invalid inventory hash")
    return FileFingerprint(name, size, digest)


def _validate_release_inventory(
    inventory: ReleaseInventory | None, *, core_only: bool = False
) -> None:
    if inventory is None:
        return
    names = {entry.name for entry in inventory.files}
    if not names >= _CORE_FILE_SET or not names <= _RELEASE_FILE_SET:
        raise ValueError("inventory does not contain a complete release layout")
    if core_only and names != _CORE_FILE_SET:
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
    missing = []
    non_regular = []
    for name in CORE_FILES:
        path = directory / name
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError:
            missing.append(name)
            continue
        if not stat.S_ISREG(mode):
            non_regular.append(name)
    if missing:
        raise ReleaseCommitError(f"existing release is missing core file(s): {missing}")
    if non_regular:
        raise ReleaseCommitError(f"existing release has non-regular core file(s): {non_regular}")


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
