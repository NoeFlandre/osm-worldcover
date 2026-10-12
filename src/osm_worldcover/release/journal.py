"""The durable transaction journal: its writes, and its validated reads."""

from __future__ import annotations

import json
import os
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osm_worldcover.release.durability import fsync_directory, fsync_file
from osm_worldcover.release.errors import ReleaseCommitError
from osm_worldcover.release.inventory import (
    _CORE_FILE_SET,
    FileFingerprint,
    ReleaseInventory,
    _validate_fingerprint_digest,
    _validate_fingerprint_name,
    _validate_fingerprint_size,
    _validate_release_inventory,
)
from osm_worldcover.release.paths import (
    _absolute_target,
    _candidate_names,
    _journal_path,
    _lexists,
)


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


def transaction_pending(target: Path) -> bool:
    """Whether a durable release journal is present for the version path."""
    return _lexists(_journal_path(_absolute_target(target)))


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
