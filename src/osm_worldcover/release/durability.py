"""Flush release files and directories to stable storage."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from osm_worldcover.release.errors import ReleaseCommitError

try:
    import fcntl
except ImportError:
    fcntl = None


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
