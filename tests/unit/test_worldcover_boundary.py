"""WorldCover boundary correction tests."""

from collections import defaultdict

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import shapely
from shapely.geometry import Polygon
from tests.conftest import write_raster
from tests.unit.worldcover_test_support import (
    ExactExtractRecorder,
    ReadCounter,
    _boundary_ids,
    _halo_cells,
    one,
)

import osm_worldcover.adapters.worldcover as worldcover
from osm_worldcover.adapters.worldcover import class_coverage


class TestBoundaryCells:
    @pytest.mark.parametrize("chunk", [1, 2, 3, 256])
    @pytest.mark.parametrize(
        ("geometry", "width", "height"),
        [
            (shapely.box(1.5, 1.5, 3.5, 3.5), 7, 5),
            (shapely.Polygon([(0.3, 0.2), (9.4, 1.1), (4.2, 8.7)]), 11, 10),
            (shapely.box(-2.5, -2.5, 4.5, 3.5), 5, 4),
            (shapely.box(3.2, 1.2, 12.5, 9.5), 6, 7),
        ],
    )
    def test_rasterized_ids_match_the_cells_the_halo_touches(
        self, monkeypatch, chunk, geometry, width, height
    ) -> None:
        monkeypatch.setattr(worldcover.boundary, "_BOUNDARY_ROW_CHUNK", chunk)
        boundary = shapely.buffer(geometry.boundary, worldcover.boundary._BOUNDARY_HALO_PIXELS)
        min_x, min_y, max_x, max_y = geometry.bounds
        ids = worldcover.boundary._rasterized_boundary_ids(
            boundary,
            max(0, int(np.floor(min_x)) - 1),
            min(width, int(np.ceil(max_x)) + 1),
            max(0, int(np.floor(min_y)) - 1),
            min(height, int(np.ceil(max_y)) + 1),
            width,
            height,
        )
        assert ids.dtype == np.int64
        assert ids.tolist() == sorted(_halo_cells(geometry, width, height))

    def test_no_strip_hits_means_no_ids(self) -> None:
        boundary = shapely.buffer(shapely.box(20, 20, 21, 21).boundary, 0.01)
        ids = worldcover.boundary._rasterized_boundary_ids(boundary, 0, 5, 0, 5, 5, 5)
        assert ids.tolist() == []
        assert ids.dtype == np.int64

    def test_chunks_carry_exact_geos_areas_for_each_cell(self, monkeypatch) -> None:
        monkeypatch.setattr(worldcover.boundary, "_BOUNDARY_CELLS_PER_CHUNK", 3)
        chunks = list(
            worldcover.boundary._boundary_cell_chunks(shapely.box(1.5, 1.5, 3.5, 3.5), 7, 5)
        )
        assert [len(c[0]) for c in chunks] == [3, 3, 2]
        by_id = {}
        for ids, rows, columns, corrected in chunks:
            assert (rows * 7 + columns).tolist() == ids.tolist()
            by_id.update(zip(ids.tolist(), corrected.tolist(), strict=True))
        corner, edge = 0.25, 0.5
        assert by_id == {
            1 * 7 + 1: corner,
            1 * 7 + 2: edge,
            1 * 7 + 3: corner,
            2 * 7 + 1: edge,
            2 * 7 + 3: edge,
            3 * 7 + 1: corner,
            3 * 7 + 2: edge,
            3 * 7 + 3: corner,
        }

    @pytest.mark.parametrize(
        "geometry",
        [
            shapely.Polygon(),
            shapely.box(6.2, 0.5, 7.0, 1.5),
            shapely.box(0.5, 4.2, 1.5, 5.0),
            shapely.box(-9, 0.5, -8, 1.5),
        ],
    )
    def test_geometry_outside_the_raster_yields_no_chunks(self, geometry) -> None:
        assert list(worldcover.boundary._boundary_cell_chunks(geometry, 5, 4)) == []

    def test_geometry_touching_the_last_row_and_column_is_kept(self) -> None:
        geometry = shapely.box(3.5, 2.5, 4.5, 3.5)
        chunks = list(worldcover.boundary._boundary_cell_chunks(geometry, 5, 4))
        ids = np.concatenate([c[0] for c in chunks])
        assert ids.tolist() == sorted(_halo_cells(geometry, 5, 4))


class TestBoundaryCorrection:
    @pytest.fixture
    def uniform(self, tmp_path):
        path = write_raster(tmp_path / "ten.tif", np.full((6, 6), 10, dtype="uint8"), (0, 6))
        with rasterio.open(path) as dataset:
            yield ReadCounter(dataset)

    def test_boundary_cells_returned_by_exactextract_are_not_reread(self, uniform) -> None:
        geometry = shapely.box(1.5, 1.5, 4.5, 4.5)
        totals = defaultdict(float)
        excluded = worldcover.boundary._accumulate_boundary_cells(
            totals, np.arange(36), np.full(36, 10.0), geometry, uniform, 1.0
        )
        assert uniform.reads == 0
        assert excluded.sum() == 12
        assert dict(totals) == {10: pytest.approx(5.0)}

    def test_only_missing_cells_that_hold_polygon_area_are_read(self, uniform) -> None:
        # Cells outside the polygon touch its halo but contribute nothing to read.
        geometry = shapely.box(1, 1, 4, 4)
        totals = defaultdict(float)
        excluded = worldcover.boundary._accumulate_boundary_cells(
            totals, np.array([], dtype=np.int64), np.array([]), geometry, uniform, 0.5
        )
        assert uniform.reads == 1
        assert not excluded.any()
        assert dict(totals) == {10: pytest.approx(4.0)}

    def test_a_missing_cell_takes_its_class_from_the_raster(self, uniform) -> None:
        geometry = shapely.box(1.5, 1.5, 4.5, 4.5)
        ids = np.delete(np.arange(36), 1 * 6 + 1)  # drop the 0.25 corner cell
        totals = defaultdict(float)
        worldcover.boundary._accumulate_boundary_cells(
            totals, ids, np.full(35, 99.0), geometry, uniform, 1.0
        )
        assert uniform.reads == 1
        assert dict(totals) == {99: pytest.approx(4.75), 10: pytest.approx(0.25)}

    def test_missing_cells_are_read_in_one_window_per_row_band(self, tmp_path) -> None:
        values = np.arange(200 * 4, dtype="uint8").reshape(200, 4) % 200 + 1
        path = write_raster(tmp_path / "tall.tif", values, origin=(0, 200))
        rows = np.array([0, 1, 2, 150, 151, 152])
        columns = np.array([0, 1, 2, 1, 2, 3])
        with rasterio.open(path) as dataset:
            counted = ReadCounter(dataset)
            classes = np.full(len(rows), np.nan)
            worldcover.boundary._read_missing_classes(
                counted, rows, columns, classes, np.arange(len(rows))
            )
        assert counted.reads == 2
        assert classes.tolist() == values[rows, columns].astype(float).tolist()

    def test_stable_coverage_keeps_the_precision_of_float32_results(self, tmp_path) -> None:
        path = write_raster(tmp_path / "big.tif", np.full((20, 20), 10, dtype="uint8"), (0, 20))
        interior = np.array([r * 20 + c for r in range(1, 19) for c in range(1, 19)])
        fraction = np.float32(0.1)
        totals = defaultdict(float)
        with rasterio.open(path) as dataset:
            worldcover.boundary._accumulate_corrected(
                totals,
                interior,
                np.full(len(interior), fraction, dtype=np.float32),
                np.full(len(interior), 10, dtype="uint8"),
                shapely.box(0, 0, 20, 20),
                dataset,
                1.0,
                400.0,
            )
        assert totals[10] == pytest.approx((324 * float(fraction) + 76) / 400.0, rel=1e-12)

    def test_cells_missing_from_one_result_do_not_pair_with_the_wrong_class(self) -> None:
        class AllMasked:
            def read(self, *_args, **_kwargs):
                return np.ma.masked_all((1, 1))

        classes = np.full(2, np.nan)
        for rows, columns, positions in [([0, 1], [0, 1], [0]), ([0], [0, 1], [0, 1])]:
            with pytest.raises(ValueError, match="same length"):
                worldcover.boundary._read_missing_classes(
                    AllMasked(), np.array(rows), np.array(columns), classes, np.array(positions)
                )

    def test_extraction_results_must_match_the_features(self, half_and_half, monkeypatch) -> None:
        empty = {"cell_id": [], "coverage": [], "values": []}
        monkeypatch.setattr(worldcover.extract, "exact_extract", lambda *_a, **_k: empty)
        with pytest.raises(ValueError, match="zip"):
            class_coverage([half_and_half], one(shapely.box(0, 0, 2, 2)))

    @pytest.mark.parametrize(
        "geometry",
        [
            shapely.box(0.5, 0.5, 2.5, 2.5),
            shapely.box(3.5, 0.2, 5.5, 3.8),
            shapely.box(0.2, 4.2, 3.9, 7.5),
            shapely.Polygon([(0.3, 0.2), (7.4, 1.1), (4.2, 6.7)]),
            shapely.box(-2.5, -2.5, 4.5, 3.5),
            shapely.box(2.2, 2.2, 9.5, 9.5),
        ],
    )
    def test_boundary_chunks_cover_exactly_the_cells_the_halo_touches(self, geometry) -> None:
        ids = _boundary_ids(geometry, 8, 8)
        assert ids.tolist() == sorted(_halo_cells(geometry, 8, 8))

    def test_distant_parts_are_all_found_when_strips_in_between_are_empty(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(worldcover.boundary, "_BOUNDARY_ROW_CHUNK", 2)
        geometry = shapely.MultiPolygon(
            [shapely.box(1.5, 1.5, 3.5, 2.5), shapely.box(1.5, 12.5, 3.5, 13.5)]
        )
        assert _boundary_ids(geometry, 6, 16).tolist() == sorted(_halo_cells(geometry, 6, 16))

    def test_rasterization_is_strip_wise_and_sized_by_the_strip_not_its_offset(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(worldcover.boundary, "_BOUNDARY_ROW_CHUNK", 4)
        shapes = []
        real = worldcover.boundary.rasterize

        def spy(*args, **kwargs):
            shapes.append(kwargs["out_shape"])
            return real(*args, **kwargs)

        monkeypatch.setattr(worldcover.boundary, "rasterize", spy)
        geometry = shapely.box(60.5, 60.5, 63.5, 75.5)
        ids = _boundary_ids(geometry, 100, 100)
        assert ids.tolist() == sorted(_halo_cells(geometry, 100, 100))
        assert len(shapes) == 5
        assert all(height <= 8 and width <= 6 for height, width in shapes)

    def test_batches_follow_the_raster_envelope_budget(self, half_and_half, monkeypatch) -> None:
        frame = gpd.GeoDataFrame(
            {"polygon_id": ["left", "right", "across"]},
            geometry=[
                shapely.box(0.5, 0.5, 1.5, 1.5),
                shapely.box(2.5, 0.5, 3.5, 1.5),
                shapely.box(1.5, 2.5, 2.5, 3.5),
            ],
            crs="EPSG:4326",
        )
        recorder = ExactExtractRecorder()
        monkeypatch.setattr(worldcover.extract, "exact_extract", recorder)
        monkeypatch.setattr(worldcover.geometry, "_MAX_BATCH_PIXEL_BBOX_CELLS", 12)

        coverage = class_coverage([half_and_half], frame)

        assert [len(call[1]) for call in recorder.calls] == [1, 1, 1]
        assert coverage == [
            {10: pytest.approx(1.0)},
            {50: pytest.approx(1.0)},
            {10: pytest.approx(0.5), 50: pytest.approx(0.5)},
        ]

    def test_classes_are_reported_in_ascending_order(self, tmp_path) -> None:
        high = write_raster(tmp_path / "high.tif", np.full((4, 2), 50, dtype="uint8"), (2, 4))
        low = write_raster(tmp_path / "low.tif", np.full((4, 2), 10, dtype="uint8"), (0, 4))
        across = one(Polygon([(1, 0), (1, 4), (3, 4), (3, 0)]))
        assert list(class_coverage([high, low], across)[0]) == [10, 50]
