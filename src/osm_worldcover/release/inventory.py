"""Deterministic size and SHA-256 inventories of release directories."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osm_worldcover.release.errors import ReleaseCommitError
from osm_worldcover.release.paths import _lexists

CORE_FILES = ("train.parquet", "validation.parquet", "test.parquet", "manifest.json")
_CORE_FILE_SET = frozenset(CORE_FILES)
_SIDECAR_FILES = frozenset(
    {"README.md", "worldcover_centroids.png", "release_checksums.json", "publication_receipt.json"}
)
_KNOWN_NEW_FILE_SET = _CORE_FILE_SET | _SIDECAR_FILES


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
