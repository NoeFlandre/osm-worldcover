"""Centroid coverage-map generation."""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from pyogrio.errors import DataSourceError
from shapely.geometry import box

from osm_worldcover.adapters import coverage_map
from osm_worldcover.adapters.coverage_map import (
    CLASS_COLORS,
    MAP_FILENAME,
    CoverageMapError,
    centroids_from_build,
    write_coverage_map,
)
from osm_worldcover.domain.nomenclature import CLASS_LABELS
from osm_worldcover.release_commit import release_lock

_COLUMNS = [
    "polygon_id",
    "lat",
    "lon",
    "worldcover_code",
    "worldcover_label",
]


def _write_build(root: Path, rows: dict[str, list[dict[str, object]]]) -> Path:
    root.mkdir(parents=True)
    for split in ("train", "validation", "test"):
        frame = pd.DataFrame(rows.get(split, []), columns=_COLUMNS).astype(
            {
                "polygon_id": "string",
                "lat": "float64",
                "lon": "float64",
                "worldcover_code": "Int64",
                "worldcover_label": "string",
            }
        )
        frame.to_parquet(root / f"{split}.parquet", index=False)
    with release_lock(root):
        pass
    return root


def test_centroids_are_deduplicated_across_article_rows_and_splits(
    tmp_path: Path,
) -> None:
    build = _write_build(
        tmp_path / "build",
        {
            "train": [
                {
                    "polygon_id": "p2",
                    "lat": 48.8,
                    "lon": 2.3,
                    "worldcover_code": 50,
                    "worldcover_label": "Built-up",
                },
                {
                    "polygon_id": "p1",
                    "lat": 51.5,
                    "lon": -0.1,
                    "worldcover_code": 10,
                    "worldcover_label": "Tree cover",
                },
                {
                    "polygon_id": "p1",
                    "lat": 51.5,
                    "lon": -0.1,
                    "worldcover_code": 10,
                    "worldcover_label": "Tree cover",
                },
            ],
            "validation": [
                {
                    "polygon_id": "p2",
                    "lat": 48.8,
                    "lon": 2.3,
                    "worldcover_code": 50,
                    "worldcover_label": "Built-up",
                },
            ],
        },
    )

    result = centroids_from_build(build)

    assert result["polygon_id"].tolist() == ["p1", "p2"]
    assert result[["lat", "lon", "worldcover_code"]].to_dict("records") == [
        {"lat": 51.5, "lon": -0.1, "worldcover_code": 10},
        {"lat": 48.8, "lon": 2.3, "worldcover_code": 50},
    ]


def test_conflicting_labels_are_rejected(tmp_path: Path) -> None:
    build = _write_build(
        tmp_path / "build",
        {
            "train": [
                {
                    "polygon_id": "p",
                    "lat": 1,
                    "lon": 2,
                    "worldcover_code": 10,
                    "worldcover_label": "Tree cover",
                },
                {
                    "polygon_id": "p",
                    "lat": 1,
                    "lon": 2,
                    "worldcover_code": 50,
                    "worldcover_label": "Built-up",
                },
            ]
        },
    )

    with pytest.raises(CoverageMapError, match="conflicting"):
        centroids_from_build(build)


def test_a_build_directory_with_a_quote_in_its_name_is_read(tmp_path: Path) -> None:
    build = _write_build(
        tmp_path / "o'brien" / "build",
        {
            "train": [
                {
                    "polygon_id": "p",
                    "lat": 1,
                    "lon": 2,
                    "worldcover_code": 10,
                    "worldcover_label": "Tree cover",
                }
            ]
        },
    )

    assert centroids_from_build(build)["polygon_id"].tolist() == ["p"]


def test_code_label_mismatch_is_rejected(tmp_path: Path) -> None:
    build = _write_build(
        tmp_path / "build",
        {
            "train": [
                {
                    "polygon_id": "p",
                    "lat": 1,
                    "lon": 2,
                    "worldcover_code": 10,
                    "worldcover_label": "Built-up",
                }
            ]
        },
    )

    with pytest.raises(CoverageMapError, match="does not match"):
        centroids_from_build(build)


@pytest.mark.parametrize("lat, lon", [(91, 0), (0, 181)])
def test_out_of_range_coordinates_are_rejected(
    tmp_path: Path,
    lat: float,
    lon: float,
) -> None:
    build = _write_build(
        tmp_path / "build",
        {
            "train": [
                {
                    "polygon_id": "p",
                    "lat": lat,
                    "lon": lon,
                    "worldcover_code": 10,
                    "worldcover_label": "Tree cover",
                }
            ]
        },
    )

    with pytest.raises(CoverageMapError, match="coordinates"):
        centroids_from_build(build)


def test_unknown_worldcover_code_is_rejected(tmp_path: Path) -> None:
    build = _write_build(
        tmp_path / "build",
        {
            "train": [
                {
                    "polygon_id": "p",
                    "lat": 1,
                    "lon": 2,
                    "worldcover_code": 0,
                    "worldcover_label": "No data",
                }
            ]
        },
    )

    with pytest.raises(CoverageMapError, match="unknown WorldCover code"):
        centroids_from_build(build)


def test_palette_covers_the_existing_nomenclature() -> None:
    assert tuple(CLASS_COLORS) == tuple(CLASS_LABELS)


def test_write_coverage_map_creates_png(tmp_path: Path) -> None:
    build = _write_build(
        tmp_path / "build",
        {
            "train": [
                {
                    "polygon_id": "p",
                    "lat": 1,
                    "lon": 2,
                    "worldcover_code": 10,
                    "worldcover_label": "Tree cover",
                }
            ]
        },
    )
    land = gpd.GeoDataFrame(
        {"geometry": [box(-180, -90, 180, 90)]},
        crs="EPSG:4326",
    )

    count = write_coverage_map(build, build / MAP_FILENAME, land=land)

    assert count == 1
    assert (build / MAP_FILENAME).read_bytes().startswith(b"\x89PNG")


class TestLandOutline:
    """The base map is fetched from Natural Earth, so its failures matter."""

    def test_a_failed_fetch_is_reported_as_a_map_error(self, monkeypatch) -> None:
        def explode(*_args, **_kwargs):
            raise OSError("network is down")

        monkeypatch.setattr(coverage_map.gpd, "read_file", explode)
        with pytest.raises(coverage_map.CoverageMapError, match="Natural Earth"):
            coverage_map._load_land()

    @pytest.mark.parametrize(
        "failure", [OSError("down"), ValueError("bad"), DataSourceError("unreadable")]
    )
    def test_expected_read_failures_are_map_errors(self, monkeypatch, failure) -> None:
        def explode(*_args, **_kwargs):
            raise failure

        monkeypatch.setattr(coverage_map.gpd, "read_file", explode)
        with pytest.raises(coverage_map.CoverageMapError, match="Natural Earth"):
            coverage_map._load_land()

    def test_an_unrelated_exception_propagates(self, monkeypatch) -> None:
        def explode(*_args, **_kwargs):
            raise KeyError("bug")

        monkeypatch.setattr(coverage_map.gpd, "read_file", explode)
        with pytest.raises(KeyError):
            coverage_map._load_land()

    def test_an_empty_outline_is_refused(self, monkeypatch) -> None:
        """An empty outline would render a map with no land on it."""
        monkeypatch.setattr(
            coverage_map.gpd,
            "read_file",
            lambda *_a, **_k: gpd.GeoDataFrame({"geometry": []}, geometry="geometry"),
        )
        with pytest.raises(coverage_map.CoverageMapError, match="empty"):
            coverage_map._load_land()

    def test_a_usable_outline_is_returned(self, monkeypatch) -> None:
        outline = gpd.GeoDataFrame(
            {"geometry": [box(-1, -1, 1, 1)]}, geometry="geometry", crs="EPSG:4326"
        )
        monkeypatch.setattr(coverage_map.gpd, "read_file", lambda *_a, **_k: outline)
        assert len(coverage_map._load_land()) == 1

    def test_an_outline_without_a_crs_is_refused(self, tmp_path: Path) -> None:
        """Without a CRS the outline cannot be aligned with the centroids."""
        build = _write_build(
            tmp_path / "build",
            {
                "train": [
                    {
                        "polygon_id": "p1",
                        "lat": 48.8,
                        "lon": 2.3,
                        "worldcover_code": 50,
                        "worldcover_label": "Built-up",
                    }
                ]
            },
        )
        outline = gpd.GeoDataFrame({"geometry": [box(-1, -1, 1, 1)]}, geometry="geometry")
        with pytest.raises(coverage_map.CoverageMapError, match="coordinate reference system"):
            coverage_map.write_coverage_map(build, tmp_path / "map.png", land=outline)
