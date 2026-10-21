from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import pytest
import rasterio
import shapely
from rasterio.transform import from_origin
from shapely.geometry import box

from osm_worldcover.adapters.worldcover import geodesic_coverage as area_coverage
from osm_worldcover.adapters.worldcover import pool_attempts
from osm_worldcover.adapters.worldcover.geodesic_coverage import (
    AREA_MEASUREMENT,
    _equal_area_geometry,
    equal_area_class_coverage,
    equal_area_class_coverage_with_tile_evidence,
    equal_area_class_coverage_with_tile_evidence_detailed,
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
        return 100.0, ((10, 80.0),), ((10, ("N00E003",)),), None

    monkeypatch.setattr(
        pool_attempts,
        "equal_area_class_coverage_with_tile_evidence_detailed",
        fake_geodesic_coverage,
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
        "equal_area_class_coverage_with_tile_evidence_detailed",
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


def _equal_area_area(geometry) -> float:
    return float(shapely.area(_equal_area_geometry(geometry)))


def test_equal_area_coverage_weights_high_latitude_rows_physically(tmp_path: Path) -> None:
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

    polygon_area, class_areas = equal_area_class_coverage([raster], polygon)
    by_code = dict(class_areas)
    expected_north = _equal_area_area(box(0, 81, 1, 82))
    expected_south = _equal_area_area(box(0, 80, 1, 81))

    assert polygon_area == pytest.approx(expected_north + expected_south, rel=1e-12)
    assert by_code[20] == pytest.approx(expected_north, rel=1e-12)
    assert by_code[10] == pytest.approx(expected_south, rel=1e-12)
    assert by_code[20] / polygon_area < 0.5


def test_equal_area_coverage_measures_boundary_and_multipart_geometry(tmp_path: Path) -> None:
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

    polygon_area, class_areas = equal_area_class_coverage([raster], polygon)
    first_part, second_part = polygon.geoms

    assert dict(class_areas)[10] == pytest.approx(_equal_area_area(first_part), rel=1e-12)
    assert dict(class_areas)[20] == pytest.approx(_equal_area_area(second_part), rel=1e-12)
    assert polygon_area == pytest.approx(_equal_area_area(polygon), rel=1e-12)


def test_equal_area_coverage_reports_only_tiles_contributing_to_each_class(tmp_path: Path) -> None:
    west = tmp_path / "west.tif"
    east = tmp_path / "east.tif"
    _write_raster(west, np.array([[10]], dtype=np.uint8), from_origin(0, 1, 1, 1))
    _write_raster(east, np.array([[20]], dtype=np.uint8), from_origin(1, 1, 1, 1))

    polygon_area, class_areas, class_evidence = equal_area_class_coverage_with_tile_evidence(
        (("N00E000", west), ("N00E003", east)), box(0, 0, 2, 1)
    )

    assert polygon_area == pytest.approx(_equal_area_area(box(0, 0, 2, 1)), rel=1e-12)
    assert dict(class_areas)[10] > 0.0
    assert dict(class_areas)[20] > 0.0
    assert class_evidence == ((10, ("N00E000",)), (20, ("N00E003",)))


def test_equal_area_coverage_weights_interior_cells_by_physical_row_area(tmp_path: Path) -> None:
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

    polygon_area, class_areas = equal_area_class_coverage((raster,), geometry)

    assert polygon_area == pytest.approx(_equal_area_area(geometry), rel=1e-12)
    expected_class_20 = (
        3 * _equal_area_area(box(1, 3, 2, 4))
        + 2 * _equal_area_area(box(1, 2, 2, 3))
        + 3 * _equal_area_area(box(1, 1, 2, 2))
    )
    assert dict(class_areas)[20] == pytest.approx(expected_class_20, rel=1e-12)
    assert sum(area for _code, area in class_areas) <= polygon_area


def test_equal_area_coverage_is_additive_for_high_latitude_cell_mosaics(tmp_path: Path) -> None:
    raster = tmp_path / "homogeneous-high-latitude-mosaic.tif"
    _write_raster(
        raster,
        np.full((3, 3), 10, dtype=np.uint8),
        from_origin(0, 63, 1, 1),
    )
    polygon = box(0, 60, 3, 63)

    polygon_area, class_areas = equal_area_class_coverage((raster,), polygon)

    assert dict(class_areas)[10] == pytest.approx(polygon_area, rel=1e-12)
    assert dict(class_areas)[10] / polygon_area >= 0.8
    assert polygon_area == pytest.approx(_equal_area_area(polygon), rel=1e-12)


def test_equal_area_boundary_measurement_densifies_sloped_high_latitude_edges(
    tmp_path: Path, monkeypatch
) -> None:
    raster = tmp_path / "sloped-high-latitude.tif"
    _write_raster(
        raster,
        np.full((4, 4), 10, dtype=np.uint8),
        from_origin(0, 64, 1, 1),
    )
    polygon = shapely.Polygon(((0.2, 60.2), (3.4, 60.4), (3.1, 63.5), (0.4, 63.2), (0.2, 60.2)))
    monkeypatch.setattr(
        area_coverage,
        "_boundary_cell_ids",
        lambda *_args: np.array([], dtype=np.int64),
    )

    polygon_area, class_areas = equal_area_class_coverage((raster,), polygon)

    assert dict(class_areas)[10] == pytest.approx(polygon_area, rel=1e-12)
    assert polygon_area == pytest.approx(_equal_area_area(polygon), rel=1e-12)
    assert shapely.get_num_coordinates(_equal_area_geometry(polygon)) > (
        shapely.get_num_coordinates(polygon)
    )


def test_equal_area_mosaic_prevents_geodesic_false_positive_at_eighty_percent(
    tmp_path: Path,
) -> None:
    raster = tmp_path / "near-threshold-high-latitude.tif"
    _write_raster(
        raster,
        np.array([[10, 10, 10, 10, 0, 0]], dtype=np.uint8),
        from_origin(0, 61, 1, 1),
    )
    polygon = box(0, 60, 4 / 0.79975, 61)

    polygon_area, class_areas = equal_area_class_coverage((raster,), polygon)

    geod = pyproj.Geod(ellps="WGS84")
    old_geodesic_denominator = abs(geod.geometry_area_perimeter(polygon)[0])
    old_geodesic_numerator = sum(
        abs(geod.geometry_area_perimeter(box(column, 60, column + 1, 61))[0]) for column in range(4)
    )
    old_geodesic_ratio = old_geodesic_numerator / old_geodesic_denominator
    equal_area_ratio = dict(class_areas)[10] / polygon_area

    assert old_geodesic_ratio == pytest.approx(0.8003708190146516, rel=1e-12)
    assert old_geodesic_ratio >= 0.8
    assert equal_area_ratio == pytest.approx(0.79975, abs=1e-12)
    assert equal_area_ratio < 0.8


def test_equal_area_uses_one_projected_boundary_across_tile_splits(tmp_path: Path) -> None:
    triangle = shapely.Polygon(
        (
            (0, 48.6101096268196),
            (0.586477465912616, 50.83405932025402),
            (0.586477465912616, 50.83410816213973),
            (0, 48.6101096268196),
        )
    )
    midpoint = 0.293238732956308
    west = tmp_path / "west-half.tif"
    east = tmp_path / "east-half.tif"
    _write_raster(west, np.array([[10]], dtype=np.uint8), from_origin(0, 51, midpoint, 3))
    _write_raster(
        east,
        np.array([[10]], dtype=np.uint8),
        from_origin(midpoint, 51, midpoint, 3),
    )

    polygon_area, class_areas = equal_area_class_coverage((west, east), triangle)

    assert dict(class_areas)[10] == pytest.approx(polygon_area, rel=1e-8)


def test_equal_area_deduplicates_boundary_cells_across_forced_chunks(
    tmp_path: Path, monkeypatch
) -> None:
    triangle = shapely.Polygon(
        (
            (0, 48.6101096268196),
            (0.586477465912616, 50.83405932025402),
            (0.586477465912616, 50.83410816213973),
            (0, 48.6101096268196),
        )
    )
    raster = tmp_path / "single-cell.tif"
    _write_raster(raster, np.array([[10]], dtype=np.uint8), from_origin(0, 51, 1, 3))

    def forced_chunks(pixel_geometry, _width: int, height: int):
        yield shapely.intersection(pixel_geometry, box(-1, -1, 0.5, height + 1))
        yield shapely.intersection(pixel_geometry, box(0.5, -1, 2, height + 1))

    monkeypatch.setattr(area_coverage, "_bounded_pixel_chunks", forced_chunks)
    polygon_area, class_areas = equal_area_class_coverage((raster,), triangle)

    assert dict(class_areas)[10] == pytest.approx(polygon_area, rel=1e-8)


def test_equal_area_refinement_fails_closed_when_coverage_brackets_eighty_percent(
    tmp_path: Path,
) -> None:
    triangle = shapely.Polygon(
        (
            (0, 48.6101096268196),
            (0.586477465912616, 50.83405932025402),
            (0.586477465912616, 50.83410816213973),
            (0, 48.6101096268196),
        )
    )
    raster = tmp_path / "near-threshold-triangle.tif"
    _write_raster(
        raster,
        np.array([[10]], dtype=np.uint8),
        from_origin(0, 51, 0.5237018676079037, 3),
    )

    with pytest.raises(ValueError, match=r"coverage_numerically_unresolved_at_0\.8"):
        equal_area_class_coverage((raster,), triangle)


def test_equal_area_refinement_records_observed_ratio_range(tmp_path: Path) -> None:
    triangle = shapely.Polygon(
        (
            (0, 48.6101096268196),
            (0.586477465912616, 50.83405932025402),
            (0.586477465912616, 50.83410816213973),
            (0, 48.6101096268196),
        )
    )
    raster = tmp_path / "stable-refined-triangle.tif"
    _write_raster(
        raster,
        np.array([[10]], dtype=np.uint8),
        from_origin(0, 51, 0.5237, 3),
    )

    _area, _supports, evidence, diagnostics = equal_area_class_coverage_with_tile_evidence_detailed(
        (("fixture", raster),), triangle
    )

    assert evidence == ((10, ("fixture",)),)
    assert diagnostics is not None
    assert diagnostics.startswith("adaptive_cea_refinement;steps=0.01,0.001,0.0001;")
    assert "class=10,range=" in diagnostics
    assert "spread=" in diagnostics


@pytest.mark.parametrize(
    "polygon, message",
    (
        (shapely.Polygon(((179, 0), (-179, 0), (-179, 1), (179, 1))), "longitude_wrapping"),
        (box(-90, 80, 90, 85), "antimeridian"),
        (box(0, 89.999999, 1, 90), "pole_or_crs_edge"),
    ),
)
def test_equal_area_rejects_unsupported_global_wraps(polygon, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        equal_area_class_coverage((), polygon)
