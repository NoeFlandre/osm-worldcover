from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
import shapely
from shapely.geometry import box

from osm_worldcover.adapters.worldcover import pool_attempts
from osm_worldcover.adapters.worldcover.pool_attempts import iter_sourcewide_coverage
from osm_worldcover.domain.tiling import Tile


class FakeTiles:
    def __init__(self, *, missing: set[str] | None = None, fail: set[str] | None = None) -> None:
        self.missing = missing or set()
        self.fail = fail or set()
        self.attempted: list[str] = []
        self.discarded: list[str] = []

    def ensure(self, tile: Tile) -> Path:
        self.attempted.append(tile.name)
        if tile.name in self.missing:
            raise pool_attempts.TileNotPublishedError(tile.name)
        if tile.name in self.fail:
            raise OSError("fixture read failure")
        return Path(f"/fixture/{tile.name}.tif")

    def discard(self, tile: Tile) -> None:
        self.discarded.append(tile.name)


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "polygon_id": "region:way/1",
                "geometry": shapely.to_geojson(box(2.9, 2.9, 3.1, 3.1)),
                "text_status": "missing",
            },
            {"polygon_id": "region:way/2", "geometry": "invalid GeoJSON"},
            {"polygon_id": None, "geometry": shapely.to_geojson(box(0.2, 0.2, 0.8, 0.8))},
        ]
    )


def test_sourcewide_adapter_attempts_all_raw_rows_and_records_missing_geometry(monkeypatch) -> None:
    fake_tiles = FakeTiles(missing={"N00E000"})
    coverage_calls: list[str] = []

    def fake_class_coverage(paths, frame: gpd.GeoDataFrame):
        coverage_calls.append(frame.geometry.iloc[0].wkt)
        assert len(paths) == 0 or all(path.name.endswith(".tif") for path in paths)
        return [{10: 0.8, 0: 0.2}]

    monkeypatch.setattr(pool_attempts, "class_coverage", fake_class_coverage)
    results = tuple(iter_sourcewide_coverage(_rows(), fake_tiles))

    assert len(results) == 3
    assert [row.source_row_ordinal for row in results] == [0, 1, 2]
    assert results[0].source_polygon_id == "region:way/1"
    assert results[0].class_fractions == ((0, 0.2), (10, 0.8))
    assert results[0].attempted_tiles == ("N00E000", "N00E003", "N03E000", "N03E003")
    assert results[0].missing_tiles == ("N00E000",)
    assert results[1].extraction_error == "geometry_unusable"
    assert results[2].extraction_error == "source_polygon_id_missing"
    assert coverage_calls == [box(2.9, 2.9, 3.1, 3.1).wkt]
    assert fake_tiles.attempted == ["N00E000", "N00E003", "N03E000", "N03E003"]
    assert fake_tiles.discarded == ["N00E000", "N00E003", "N03E000", "N03E003"]


def test_sourcewide_adapter_records_operational_failure_without_dropping_row(monkeypatch) -> None:
    fake_tiles = FakeTiles(fail={"N00E000"})
    monkeypatch.setattr(
        pool_attempts,
        "class_coverage",
        lambda _paths, _frame: pytest.fail("measurement should not use a failed tile"),
    )

    result = next(iter_sourcewide_coverage(_rows().iloc[[0]], fake_tiles))

    assert result.source_polygon_id == "region:way/1"
    assert result.class_fractions == ()
    assert result.attempted_tiles == ("N00E000", "N00E003", "N03E000", "N03E003")
    assert result.extraction_error == "tile_read_failed:OSError"
    assert fake_tiles.discarded == ["N00E000", "N00E003", "N03E000", "N03E003"]
