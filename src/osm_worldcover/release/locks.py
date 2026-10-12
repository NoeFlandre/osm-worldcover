"""Release locks: shared and exclusive locks on a version, and the scratch lock."""

from __future__ import annotations

import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from types import ModuleType

from osm_worldcover.release.durability import fsync_directory
from osm_worldcover.release.errors import ReleaseCommitError
from osm_worldcover.release.inventory import (
    _require_core_release_files,
    _require_real_directory,
)
from osm_worldcover.release.journal import transaction_pending
from osm_worldcover.release.paths import _absolute_target, _lexists, _validate_target

try:
    import fcntl
except ImportError:
    fcntl = None


_HELD_LOCKS: ContextVar[dict[Path, bool] | None] = ContextVar(
    "owc_held_release_locks", default=None
)


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
