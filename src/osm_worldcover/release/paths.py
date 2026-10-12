"""Names and path checks for a release, its stage, backup, lock and journal."""

from __future__ import annotations

import os
import re
from pathlib import Path

from osm_worldcover.release.errors import ReleaseCommitError

_HEX_32 = re.compile(r"[0-9a-f]{32}\Z")


def _absolute_stage(stage: Path) -> Path:
    stage = Path(stage)
    return stage.parent.resolve() / stage.name


def _rename_directory(source: Path, destination: Path) -> None:
    if source.parent != destination.parent or source.is_symlink() or not source.is_dir():
        raise ReleaseCommitError(f"directory rename must stay beside its source: {source}")
    if _lexists(destination):
        raise ReleaseCommitError(f"release rename destination already exists: {destination}")
    source.rename(destination)


def _candidate_names(target: Path, transaction_id: str) -> tuple[str, str]:
    if not _HEX_32.fullmatch(transaction_id):
        raise ValueError("invalid transaction id")
    return (
        f".{target.name}.stage-{transaction_id}",
        f".{target.name}.backup-{transaction_id}",
    )


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


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)
