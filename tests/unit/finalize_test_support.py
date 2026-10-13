"""Shared fixtures and helpers for finalization tests."""

import pandas as pd
import pytest

from osm_worldcover.config import Config
from osm_worldcover.finalize import finalize_shards
from osm_worldcover.release import commit as commit_module
from osm_worldcover.release import durability as durability_module
from osm_worldcover.release import inventory as inventory_module
from osm_worldcover.release import journal as journal_module
from osm_worldcover.release import locks as locks_module
from osm_worldcover.release import paths as paths_module
from osm_worldcover.release import recovery as recovery_module

RELEASE_MODULES = (
    commit_module,
    durability_module,
    inventory_module,
    journal_module,
    locks_module,
    paths_module,
    recovery_module,
)


def _patch_release_name(monkeypatch, name, replacement):
    """Patch a release name in every release module that binds it."""
    for module in RELEASE_MODULES:
        if hasattr(module, name):
            monkeypatch.setattr(module, name, replacement)


def written(result) -> pd.DataFrame:
    """Every published row, read back from the files that were written."""
    splits = [p for p in result.paths if p.suffix == ".parquet"]
    return pd.concat([pd.read_parquet(p) for p in splits], ignore_index=True)


@pytest.fixture(autouse=True)
def fast_release_fsyncs(monkeypatch):
    _patch_release_name(monkeypatch, "fsync_file", lambda _path: None)
    _patch_release_name(monkeypatch, "fsync_directory", lambda _path: None)


TEXT = " ".join(["word"] * 30)


def shard(
    path,
    n=1,
    start=0,
    region="luxembourg",
    code=10,
    text=None,
    text_words=None,
    lat=49.6,
    lon=6.1,
):
    pd.DataFrame(_shard_rows(n, start, region, code, text, text_words, lat, lon)).to_parquet(
        path, index=False
    )


def _shard_texts(text, ids: range) -> list[str]:
    return [text or f"{TEXT} {index}" for index in ids]


def _shard_text_words(text_words, texts: list[str]) -> list[int]:
    return [text_words or len(value.split()) for value in texts]


def _shard_rows(n, start, region, code, text, text_words, lat, lon) -> dict:
    ids = range(start, start + n)
    texts = _shard_texts(text, ids)
    return {
        "polygon_id": [f"{region}-latest:way:{index}" for index in ids],
        "osm_type": "way",
        "osm_id": list(ids),
        "region": region,
        "name": "N",
        "wikidata": "Q1",
        "document_id": [f"d{index}" for index in ids],
        "project": "wikipedia",
        "language": "en",
        "title": "T",
        "url": "u",
        "text": texts,
        "lead_text": "lead",
        "text_words": _shard_text_words(text_words, texts),
        "worldcover_code": code,
        "worldcover_label": {10: "Tree cover"}.get(code, "Built-up"),
        "dominant_fraction": 0.95,
        "observed_fraction": 1.0,
        "lat": lat,
        "lon": lon,
        "centroid_wkt": "POINT (6.1 49.6)",
        "polygon_area_m2": 1000.0,
        "source_pbf": f"{region}-latest.osm.pbf",
    }


@pytest.fixture
def shards(tmp_path):
    d = tmp_path / "shards"
    d.mkdir()
    return d


def _clear_shards(shards):
    for path in shards.glob("*.parquet"):
        path.unlink()


def _assert_empty_rerun_preserves_release(result, target, before):
    import hashlib

    assert result.rows == 0
    assert result.paths == []
    assert _file_snapshot(target) == before
    assert (
        hashlib.sha256((target / "manifest.json").read_bytes()).hexdigest()
        == before["manifest.json"][0]
    )


def _assert_empty_result(result):
    assert result.rows == 0
    assert result.paths == []


def _assert_release_recovered_and_clean(target, before, out):
    import osm_worldcover.release as release

    assert release.inventory_release(target) == before
    assert not list(out.glob(f".{target.name}.stage-*"))
    assert not list(out.glob(f".{target.name}.backup-*"))
    assert not (out / f".{target.name}.transaction.json").exists()


def _assert_failed_validation_result(result):
    assert result.rows == 8
    assert result.paths == []
    assert not result.report.ok


def _assert_old_release_unchanged_and_unstaged(target, before, out):
    assert _file_snapshot(target) == before
    assert not list(out.glob(f".{target.name}.stage-*"))


def _recover_twice_and_assert_snapshot(target, before):
    import osm_worldcover.release as release

    release.recover_release(target)
    release.recover_release(target)
    assert _file_snapshot(target) == before


def _assert_unchanged_rerun_result(result):
    assert result.rows == 3
    assert {path.name for path in result.paths} == {
        "train.parquet",
        "validation.parquet",
        "test.parquet",
        "manifest.json",
    }


def _published_files(out) -> list[str]:
    release = out / f"v{Config().dataset_version}"
    return sorted(p.name for p in release.iterdir()) if release.exists() else []


def _release_bytes(out) -> dict[str, bytes]:
    release = out / f"v{Config().dataset_version}"
    return {path.name: path.read_bytes() for path in release.iterdir()}


def _release_entries() -> list[str]:
    """The release directory and its permanent lock sidecar, as left in the output root."""
    version = f"v{Config().dataset_version}"
    return sorted([version, f".{version}.lock"])


def linked(path, polygon_ids, document_id, lats, lons, region="alpha", codes=None):
    """Rows sharing one document across several polygons.

    Rows share one document across polygons to exercise its one-home-split rule.
    Identical text and labels on distinct polygons remain separate records.
    """
    n = len(polygon_ids)
    codes = codes or [10] * n
    labels = {10: "Tree cover", 20: "Shrubland", 30: "Grassland", 50: "Built-up"}
    pd.DataFrame(
        {
            "polygon_id": [f"{region}-latest:way:{i}" for i in polygon_ids],
            "osm_type": ["way"] * n,
            "osm_id": list(polygon_ids),
            "region": [region] * n,
            "name": ["N"] * n,
            "wikidata": ["Q1"] * n,
            "document_id": [document_id] * n,
            "project": ["wikipedia"] * n,
            "language": ["en"] * n,
            "title": ["T"] * n,
            "url": ["u"] * n,
            "text": [f"{TEXT} {document_id}"] * n,
            "lead_text": ["lead"] * n,
            "text_words": [31] * n,
            "worldcover_code": list(codes),
            "worldcover_label": [labels[c] for c in codes],
            "dominant_fraction": [0.95] * n,
            "observed_fraction": [1.0] * n,
            "lat": list(lats),
            "lon": list(lons),
            "centroid_wkt": ["POINT (0 0)"] * n,
            "polygon_area_m2": [1000.0] * n,
            "source_pbf": [f"{region}-latest.osm.pbf"] * n,
        }
    ).to_parquet(path, index=False)


def run(shards, tmp_path, config=None):
    """Finalize ``shards`` under ``tmp_path``, returning the streamed build."""
    return finalize_shards(shards, config or Config(), tmp_path / "work", tmp_path / "work" / "out")


def with_columns(path, **columns) -> None:
    """Overwrite columns of a shard written by :func:`shard`."""
    frame = pd.read_parquet(path)
    for name, value in columns.items():
        frame[name] = value
    frame.to_parquet(path, index=False)


FIVE_PLACES = [  # Luxembourg, Tokyo, Sydney, Helsinki, Nairobi: five distinct H3 cells
    (49.6, 6.1),
    (35.7, 139.7),
    (-33.9, 151.2),
    (60.2, 24.9),
    (-1.3, 36.8),
]


@pytest.fixture
def five_places(shards):
    """One row per place, each in its own region, with distinct fractions, labels, languages."""
    fractions = [0.8, 0.85, 0.9000004, 0.95, 0.9876543217]
    languages = ["en", "fr", None, "en", "en"]
    codes = [10, 10, 50, 10, 50]
    for index, (lat, lon) in enumerate(FIVE_PLACES):
        path = shards / f"r{index}.parquet"
        shard(path, start=index, region=f"r{index}", code=codes[index], lat=lat, lon=lon)
        with_columns(path, dominant_fraction=fractions[index], language=languages[index])
    return shards


def repeated_polygon(path, n, start, text) -> None:
    """A shard whose ``n`` documents all describe one polygon with the same text."""
    shard(path, n=n, start=start, text=text)
    with_columns(path, polygon_id=f"p{start}", osm_id=start)


@pytest.fixture
def two_repeated_polygons(shards):
    """Two groups lose records: two removed at three words, one removed at twelve."""
    repeated_polygon(shards / "short.parquet", 3, 0, "one two three")
    twelve = "one two three four five six seven eight nine ten eleven twelve"
    repeated_polygon(shards / "long.parquet", 2, 10, twelve)
    return shards


def _file_snapshot(directory):
    import hashlib

    return {
        path.name: (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in sorted(directory.iterdir())
        if path.is_file()
    }
