"""Shared fixtures for CLI tests."""

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tests.cli_test_helpers import frame

from osm_worldcover.domain.validation import ValidationReport
from osm_worldcover.finalize import StreamedBuild
from osm_worldcover.release import release_lock

MANIFEST = {
    "counts": {"examples": {"train": 2, "validation": 1, "test": 1, "total": 4}},
    "class_distribution": [
        {"code": 10, "label": "Tree cover", "examples": 3, "share": 0.75},
        {"code": 50, "label": "Built-up", "examples": 1, "share": 0.25},
    ],
    "language_distribution": [{"language": "en", "examples": 4, "share": 1.0}],
}


def split_frames(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        name: frame[frame["split"] == name].reset_index(drop=True)
        for name in ("train", "validation", "test")
    }


def _seed_release_lock(build_dir: Path) -> None:
    with release_lock(build_dir):
        pass


def built(tmp_path, report=None) -> StreamedBuild:
    """A StreamedBuild as run_build would return it, already written to disk."""
    target = tmp_path / "v1.0.0"
    target.mkdir(parents=True, exist_ok=True)
    paths = []
    for split in ("train", "validation", "test"):
        path = target / f"{split}.parquet"
        frame()[frame()["split"] == split].to_parquet(path, index=False)
        paths.append(path)
    paths.append(target / "manifest.json")
    paths[-1].write_text(json.dumps(MANIFEST))
    _seed_release_lock(target)
    return StreamedBuild(4, paths, MANIFEST, report or ValidationReport(4))


def write_build(tmp_path, rows: pd.DataFrame, settings: dict | None = None) -> Path:
    build = tmp_path / "v1.0.0"
    build.mkdir(parents=True)
    # pyarrow keeps NaN as NaN; pandas' to_parquet would store it as null instead.
    table = pa.table({name: rows[name].tolist() for name in rows.columns})
    pq.write_table(table, build / "train.parquet")
    if settings is not None:
        (build / "manifest.json").write_text(json.dumps({"settings": settings}))
    _seed_release_lock(build)
    return build
