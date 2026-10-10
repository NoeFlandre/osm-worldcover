"""Checks that the staged release matches the manifest and Parquet files written for it."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from osm_worldcover.release_commit import ReleaseCommitError


def _validate_staged_manifest(
    manifest_path: Path,
    split_paths: Sequence[Path],
    expected: dict[str, Any],
    rows: int,
) -> None:
    actual = _read_expected_staged_manifest(manifest_path, expected)
    examples = actual.get("counts", {}).get("examples", {})
    counts = _staged_split_counts(split_paths)
    _validate_staged_split_counts(examples, counts)
    _validate_staged_total(rows, examples, counts)


def _read_expected_staged_manifest(manifest_path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    actual = json.loads(manifest_path.read_text())
    if actual != expected:
        raise ReleaseCommitError("staged manifest changed after it was written")
    return actual


def _staged_split_counts(split_paths: Sequence[Path]) -> dict[str, int]:
    return {path.stem: int(pq.ParquetFile(path).metadata.num_rows) for path in split_paths}


def _validate_staged_split_counts(examples: dict[str, Any], counts: dict[str, int]) -> None:
    if any(examples.get(split) != count for split, count in counts.items()):
        raise ReleaseCommitError("staged manifest split counts do not match the Parquet files")


def _validate_staged_total(rows: int, examples: dict[str, Any], counts: dict[str, int]) -> None:
    if examples.get("total") != rows or sum(counts.values()) != rows:
        raise ReleaseCommitError("staged manifest total does not match the Parquet files")
