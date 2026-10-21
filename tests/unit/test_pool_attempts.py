from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import pytest
import rasterio
import shapely
from rasterio.transform import from_origin
from shapely.geometry import box

from osm_worldcover.adapters.worldcover import pool_attempts
from osm_worldcover.adapters.worldcover.geodesic_coverage import (
    AREA_MEASUREMENT,
    geodesic_class_coverage,
)
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

    def fake_geodesic_coverage(paths, geometry):
        coverage_calls.append(geometry.wkt)
        assert len(paths) == 0 or all(path.name.endswith(".tif") for path in paths)
        return 100.0, ((10, 80.0),)

    monkeypatch.setattr(pool_attempts, "geodesic_class_coverage", fake_geodesic_coverage)
    raw_rows = _rows()
    results = tuple(iter_sourcewide_coverage((raw_rows.iloc[:1], raw_rows.iloc[1:]), fake_tiles))

    assert len(results) == 3
    assert [row.source_row_ordinal for row in results] == [0, 1, 2]
    assert results[0].source_polygon_id == "region:way/1"
    assert results[0].polygon_area_m2 == 100.0
    assert results[0].class_areas_m2 == ((10, 80.0),)
    assert results[0].area_measurement == AREA_MEASUREMENT
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
        "geodesic_class_coverage",
        lambda _paths, _geometry: pytest.fail("measurement should not use a failed tile"),
    )

    result = next(iter_sourcewide_coverage(_rows().iloc[[0]], fake_tiles))

    assert result.source_polygon_id == "region:way/1"
    assert result.polygon_area_m2 is None
    assert result.class_areas_m2 == ()
    assert result.attempted_tiles == ("N00E000", "N00E003", "N03E000", "N03E003")
    assert result.extraction_error == "tile_read_failed:OSError"
    assert fake_tiles.discarded == ["N00E000", "N00E003", "N03E000", "N03E003"]


def _write_raster(path: Path, values: np.ndarray, transform) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype=values.dtype,
        crs="EPSG:4326",
        transform=transform,
        nodata=0,
    ) as dataset:
        dataset.write(values, 1)


def _geod_area(geometry) -> float:
    geod = pyproj.Geod(ellps="WGS84")
    oriented = shapely.orient_polygons(geometry, exterior_cw=False)
    return abs(geod.geometry_area_perimeter(oriented)[0])


def test_geodesic_coverage_weights_high_latitude_rows_physically(tmp_path: Path) -> None:
    raster = tmp_path / "high-latitude.tif"
    _write_raster(
        raster,
        np.array(
            [
                [
                    20,
                ],
                [
                    10,
                ],
            ],
            dtype=np.uint8,
        ),
        from_origin(0, 82, 1, 1),
    )
    polygon = box(0, 80, 1, 82)

    polygon_area, class_areas = geodesic_class_coverage([raster], polygon)
    by_code = dict(class_areas)
    expected_north = _geod_area(box(0, 81, 1, 82))
    expected_south = _geod_area(box(0, 80, 1, 81))

    assert polygon_area == pytest.approx(expected_north + expected_south, rel=1e-12)
    assert by_code[20] == pytest.approx(expected_north, rel=1e-12)
    assert by_code[10] == pytest.approx(expected_south, rel=1e-12)
    assert by_code[20] / polygon_area < 0.5


def test_geodesic_coverage_measures_boundary_and_multipart_geometry(tmp_path: Path) -> None:
    raster = tmp_path / "boundary-multipart.tif"
    _write_raster(
        raster,
        np.array([[10, 20], [10, 20]], dtype=np.uint8),
        from_origin(0, 2, 1, 1),
    )
    polygon = shapely.MultiPolygon(
        (
            box(0.15, 0.2, 0.75, 1.8),
            box(1.25, 0.1, 1.9, 1.5),
        )
    )

    polygon_area, class_areas = geodesic_class_coverage([raster], polygon)
    first_part, second_part = polygon.geoms

    assert dict(class_areas)[10] == pytest.approx(_geod_area(first_part), rel=1e-12)
    assert dict(class_areas)[20] == pytest.approx(_geod_area(second_part), rel=1e-12)
    assert polygon_area == pytest.approx(_geod_area(polygon), rel=1e-12)
