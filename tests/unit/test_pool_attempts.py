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
    geodesic_class_coverage_with_tile_evidence,
)
from osm_worldcover.adapters.worldcover.pool_attempts import (
    SourceWideCoverageAttempt,
    iter_sourcewide_coverage,
)
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


def _run_sourcewide_adapter(
    monkeypatch, raw_rows: pd.DataFrame, fake_tiles: FakeTiles
) -> tuple[tuple[SourceWideCoverageAttempt, ...], list[str]]:
    coverage_calls: list[str] = []

    def fake_geodesic_coverage(_tile_paths, geometry):
        coverage_calls.append(geometry.wkt)
        return 100.0, ((10, 80.0),), ((10, ("N00E003",)),)

    monkeypatch.setattr(
        pool_attempts, "geodesic_class_coverage_with_tile_evidence", fake_geodesic_coverage
    )
    results = tuple(iter_sourcewide_coverage((raw_rows.iloc[:1], raw_rows.iloc[1:]), fake_tiles))
    return results, coverage_calls


def test_sourcewide_adapter_attempts_all_raw_rows_before_filters(monkeypatch) -> None:
    fake_tiles = FakeTiles(missing={"N00E000"})
    results, coverage_calls = _run_sourcewide_adapter(monkeypatch, _rows(), fake_tiles)

    assert len(results) == 3
    assert tuple(row.source_row_ordinal for row in results) == (0, 1, 2)
    assert tuple(row.source_polygon_id for row in results) == (
        "region:way/1",
        "region:way/2",
        "row:2",
    )
    assert coverage_calls == [box(2.9, 2.9, 3.1, 3.1).wkt]


def test_sourcewide_adapter_preserves_full_area_tile_and_class_evidence(monkeypatch) -> None:
    fake_tiles = FakeTiles(missing={"N00E000"})
    results, _coverage_calls = _run_sourcewide_adapter(monkeypatch, _rows(), fake_tiles)
    row = results[0]

    assert (
        row.polygon_area_m2,
        row.class_areas_m2,
        row.area_measurement,
        row.attempted_tiles,
        row.missing_tiles,
        row.class_evidence_ids,
    ) == (
        100.0,
        ((10, 80.0),),
        AREA_MEASUREMENT,
        ("N00E000", "N00E003", "N03E000", "N03E003"),
        ("N00E000",),
        ((10, ("N00E003",)),),
    )
    assert fake_tiles.attempted == ["N00E000", "N00E003", "N03E000", "N03E003"]
    assert fake_tiles.discarded == fake_tiles.attempted


def test_sourcewide_adapter_records_missing_geometry_and_polygon_id(monkeypatch) -> None:
    fake_tiles = FakeTiles()
    results, coverage_calls = _run_sourcewide_adapter(monkeypatch, _rows(), fake_tiles)

    assert tuple(row.extraction_error for row in results) == (
        None,
        "geometry_unusable",
        "source_polygon_id_missing",
    )
    assert coverage_calls == [box(2.9, 2.9, 3.1, 3.1).wkt]


def test_sourcewide_adapter_records_operational_failure_without_dropping_row(monkeypatch) -> None:
    fake_tiles = FakeTiles(fail={"N00E000"})
    monkeypatch.setattr(
        pool_attempts,
        "geodesic_class_coverage_with_tile_evidence",
        lambda _tile_paths, _geometry: pytest.fail("measurement should not use a failed tile"),
    )

    result = next(iter_sourcewide_coverage(_rows().iloc[[0]], fake_tiles))

    assert (
        result.source_polygon_id,
        result.polygon_area_m2,
        result.class_areas_m2,
        result.attempted_tiles,
        result.extraction_error,
    ) == (
        "region:way/1",
        None,
        (),
        ("N00E000", "N00E003", "N03E000", "N03E003"),
        "tile_read_failed:OSError",
    )
    assert fake_tiles.discarded == ["N00E000", "N00E003", "N03E000", "N03E003"]


def test_polygon_decoder_accepts_geometry_and_rejects_non_geojson_values() -> None:
    polygon = box(0, 0, 1, 1)

    assert pool_attempts._decode_polygon(polygon) is polygon
    for raw_geometry in (None, 1, "", "invalid GeoJSON"):
        assert pool_attempts._decode_polygon(raw_geometry) is None


def test_sourcewide_adapter_accepts_one_frame_and_requires_raw_columns() -> None:
    rows = _rows().iloc[:1]
    fake_tiles = FakeTiles()
    result = tuple(iter_sourcewide_coverage(rows, fake_tiles))

    assert tuple(row.source_row_ordinal for row in result) == (0,)
    with pytest.raises(ValueError, match="polygon_id and geometry columns"):
        tuple(iter_sourcewide_coverage(pd.DataFrame({"geometry": ["{}"]}), fake_tiles))


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


def test_geodesic_coverage_reports_only_tiles_contributing_to_each_class(tmp_path: Path) -> None:
    west = tmp_path / "west.tif"
    east = tmp_path / "east.tif"
    _write_raster(west, np.array([[10]], dtype=np.uint8), from_origin(0, 1, 1, 1))
    _write_raster(east, np.array([[20]], dtype=np.uint8), from_origin(1, 1, 1, 1))

    polygon_area, class_areas, class_evidence = geodesic_class_coverage_with_tile_evidence(
        (("N00E000", west), ("N00E003", east)), box(0, 0, 2, 1)
    )

    assert polygon_area == pytest.approx(_geod_area(box(0, 0, 2, 1)), rel=1e-12)
    assert dict(class_areas)[10] > 0.0
    assert dict(class_areas)[20] > 0.0
    assert class_evidence == ((10, ("N00E000",)), (20, ("N00E003",)))


def test_geodesic_coverage_weights_interior_cells_by_physical_row_area(tmp_path: Path) -> None:
    raster = tmp_path / "interior-cells.tif"
    values = np.array(
        [
            [10, 10, 10, 10, 10],
            [10, 20, 20, 20, 10],
            [10, 20, 95, 20, 10],
            [10, 20, 20, 20, 10],
            [10, 10, 10, 10, 10],
        ],
        dtype=np.uint8,
    )
    _write_raster(raster, values, from_origin(0, 5, 1, 1))
    geometry = box(0, 0, 5, 5)

    polygon_area, class_areas = geodesic_class_coverage((raster,), geometry)

    assert polygon_area == pytest.approx(_geod_area(geometry), rel=1e-12)
    expected_class_20 = (
        3 * _geod_area(box(1, 3, 2, 4))
        + 2 * _geod_area(box(1, 2, 2, 3))
        + 3 * _geod_area(box(1, 1, 2, 2))
    )
    assert dict(class_areas)[20] == pytest.approx(expected_class_20, rel=1e-12)
    assert sum(area for _code, area in class_areas) <= polygon_area
