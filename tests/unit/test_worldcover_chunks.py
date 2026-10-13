"""WorldCover geometry splitting and bounded extraction tests."""

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import shapely
from tests.unit.worldcover_test_support import _bounds

import osm_worldcover.adapters.worldcover as worldcover


class TestSplitAxisSelection:
    @pytest.mark.parametrize(
        ("minimum", "maximum", "expected"),
        [
            (0.0, 4.0, 2.0),
            (0.0, 3.0, 1.0),
            (0.0, 1.5, 1.0),
            (2.5, 3.5, 3.0),
            (-3.0, -1.0, -2.0),
            (-1.5, -0.5, -1.0),
            (0.0, 1.0, None),
            (0.2, 0.8, None),
            (3.0, 3.0, None),
        ],
    )
    def test_midpoint_is_an_interior_integer_boundary(self, minimum, maximum, expected) -> None:
        assert worldcover.geometry._pixel_aligned_midpoint(minimum, maximum) == expected

    @pytest.mark.parametrize(
        ("x_span", "y_span", "x_mid", "y_mid", "expected"),
        [
            (2.0, 2.0, None, None, True),
            (3.0, 2.0, None, None, True),
            (1.0, 2.0, None, None, False),
            (9.0, 1.0, None, 1.0, False),
            (1.0, 9.0, 1.0, None, True),
            (2.0, 2.0, 1.0, 1.0, True),
            (3.0, 2.0, 1.0, 1.0, True),
            (1.0, 2.0, 1.0, 1.0, False),
        ],
    )
    def test_axis_prefers_integer_splits_then_the_longest_span(
        self, x_span, y_span, x_mid, y_mid, expected
    ) -> None:
        assert worldcover.geometry._choose_split_axis(x_span, y_span, x_mid, y_mid) is expected


class TestSplitOnce:
    @pytest.mark.parametrize(
        ("geometry", "first", "second"),
        [
            (shapely.box(0, 0, 8, 4), (0, 0, 4, 4), (4, 0, 8, 4)),
            (shapely.box(0, 0, 4, 8), (0, 0, 4, 4), (0, 4, 4, 8)),
            (shapely.box(0, 0, 4, 4), (0, 0, 2, 4), (2, 0, 4, 4)),
            (shapely.box(1, 1, 8, 4), (1, 1, 4, 4), (4, 1, 8, 4)),
            (shapely.box(0.2, 0.1, 0.8, 0.3), (0.2, 0.1, 0.5, 0.3), (0.5, 0.1, 0.8, 0.3)),
            (shapely.box(0.2, 0.1, 0.4, 0.9), (0.2, 0.1, 0.4, 0.5), (0.2, 0.5, 0.4, 0.9)),
            (shapely.box(0.2, 0.0, 0.8, 4.0), (0.2, 0.0, 0.8, 2.0), (0.2, 2.0, 0.8, 4.0)),
            (shapely.box(0.0, 0.2, 9.0, 0.8), (0.0, 0.2, 4.0, 0.8), (4.0, 0.2, 9.0, 0.8)),
            # Offsets from the origin must not change which axis is longest.
            (shapely.box(10, 10, 12, 14), (10, 10, 12, 12), (10, 12, 12, 14)),
            (shapely.box(10, 10, 14, 12), (10, 10, 12, 12), (12, 10, 14, 12)),
        ],
    )
    def test_geometry_is_cut_on_pixel_lines_or_the_plain_midpoint(
        self, geometry, first, second
    ) -> None:
        pieces = worldcover.geometry._split_pixel_geometry_once(geometry)
        assert _bounds(pieces) == [first, second]
        assert sum(p.area for p in pieces) == pytest.approx(geometry.area)

    @pytest.mark.parametrize(
        ("first", "second", "expected"),
        [
            (shapely.Polygon(), shapely.Polygon(), True),
            ("same", "same", True),
            ("same", shapely.Polygon(), False),
            (shapely.Polygon(), "same", False),
            ("same", shapely.box(0, 0, 1, 1), False),
            (shapely.box(0, 0, 1, 1), "same", False),
            (shapely.box(0, 0, 1, 1), shapely.box(1, 0, 2, 1), False),
        ],
    )
    def test_a_split_is_stalled_only_when_it_makes_no_progress(
        self, first, second, expected
    ) -> None:
        geometry = shapely.box(0, 0, 2, 1)
        first = geometry if isinstance(first, str) else first
        second = geometry if isinstance(second, str) else second
        assert worldcover.geometry._split_is_stalled(geometry, first, second) is expected


class TestBoundedChunks:
    @pytest.mark.parametrize(
        ("geometry", "width", "height", "expected"),
        [
            (shapely.Polygon(), 5, 5, 0),
            (shapely.box(2.5, 2.5, 3.5, 3.5), 10, 10, 16),
            (shapely.box(0, 0, 1, 1), 10, 10, 4),
            (shapely.box(2, 0, 3, 1), 3, 7, 2 * 2),
            (shapely.box(5, 4, 6, 5), 6, 9, 2 * 3),
            (shapely.box(1, 5, 2, 6), 10, 6, 3 * 2),
            (shapely.box(1, 1, 2, 2), 2, 3, 2 * 3),
            (shapely.box(50, 0, 51, 1), 10, 10, 0),
            (shapely.box(0, 50, 1, 51), 10, 10, 0),
        ],
    )
    def test_envelope_cells_are_clamped_to_the_raster(
        self, geometry, width, height, expected
    ) -> None:
        assert worldcover.geometry._pixel_bbox_cells(geometry, width, height) == expected

    @pytest.mark.parametrize(
        ("cells", "coordinates", "bounded"),
        [(9, 5, True), (8, 5, False), (9, 4, False), (10, 6, True)],
    )
    def test_geometry_is_bounded_inclusive_of_both_limits(
        self, monkeypatch, cells, coordinates, bounded
    ) -> None:
        monkeypatch.setattr(worldcover.geometry, "_MAX_BATCH_PIXEL_BBOX_CELLS", cells)
        monkeypatch.setattr(worldcover.geometry, "_MAX_GEOMETRY_COORDINATES", coordinates)
        square = shapely.box(0, 0, 2, 2)  # 9 envelope cells, 5 coordinates
        assert bool(worldcover.geometry._geometry_is_bounded(square, 10, 10)) is bounded
        assert bool(worldcover.geometry._needs_spatial_chunks(square, 10, 10)) is (not bounded)

    def test_chunks_split_on_pixel_lines_and_clip_to_the_raster(self, monkeypatch) -> None:
        monkeypatch.setattr(worldcover.geometry, "_MAX_BATCH_PIXEL_BBOX_CELLS", 15)
        chunks = list(worldcover.geometry._bounded_pixel_chunks(shapely.box(0, 0, 8, 2), 10, 10))
        assert _bounds(chunks) == [(0, 0, 4, 2), (4, 0, 6, 2), (6, 0, 8, 2)]
        clipped = list(worldcover.geometry._bounded_pixel_chunks(shapely.box(-5, 0, 3, 2), 10, 10))
        assert _bounds(clipped) == [(0, 0, 3, 2)]
        assert (
            list(worldcover.geometry._bounded_pixel_chunks(shapely.box(20, 20, 21, 21), 10, 10))
            == []
        )

    def test_a_bounded_geometry_is_returned_whole(self) -> None:
        geometry = shapely.box(0, 0, 3, 3)
        (chunk,) = worldcover.geometry._split_pixel_geometry(geometry, 10, 10, depth=0)
        assert chunk.equals(geometry)

    def test_splitting_stops_at_the_depth_limit(self, monkeypatch) -> None:
        monkeypatch.setattr(worldcover.geometry, "_MAX_BATCH_PIXEL_BBOX_CELLS", 15)
        geometry = shapely.box(0, 0, 8, 2)
        monkeypatch.setattr(worldcover.geometry, "_MAX_GEOMETRY_SPLIT_DEPTH", 2)
        assert len(list(worldcover.geometry._bounded_pixel_chunks(geometry, 10, 10))) == 3
        monkeypatch.setattr(worldcover.geometry, "_MAX_GEOMETRY_SPLIT_DEPTH", 1)
        message = "^could not bound a WorldCover geometry after 1 spatial splits$"
        with pytest.raises(ValueError, match=message):
            list(worldcover.geometry._bounded_pixel_chunks(geometry, 10, 10))
        with pytest.raises(ValueError, match=message):
            list(worldcover.geometry._split_pixel_geometry(geometry, 10, 10, depth=1))
        monkeypatch.setattr(worldcover.geometry, "_MAX_GEOMETRY_SPLIT_DEPTH", 4)
        assert len(list(worldcover.geometry._split_pixel_geometry(geometry, 10, 10, depth=2))) == 3
        monkeypatch.setattr(worldcover.geometry, "_MAX_GEOMETRY_SPLIT_DEPTH", 3)
        with pytest.raises(ValueError, match="after 3 spatial splits"):
            list(worldcover.geometry._split_pixel_geometry(geometry, 10, 10, depth=2))

    @pytest.mark.parametrize("pieces", [("empty", "empty"), ("same", "same")])
    def test_a_stalled_split_is_an_error_not_an_endless_recursion(
        self, monkeypatch, pieces
    ) -> None:
        geometry = shapely.box(0, 0, 8, 2)
        monkeypatch.setattr(worldcover.geometry, "_MAX_BATCH_PIXEL_BBOX_CELLS", 1)
        stalled = tuple(geometry if p == "same" else shapely.Polygon() for p in pieces)
        monkeypatch.setattr(worldcover.geometry, "_split_pixel_geometry_once", lambda _g: stalled)
        with pytest.raises(
            ValueError, match=r"^WorldCover geometry split did not reduce its spatial extent$"
        ):
            list(worldcover.geometry._split_pixel_geometry(geometry, 10, 10, depth=0))

    def test_empty_halves_are_skipped(self) -> None:
        piece = shapely.box(0, 0, 1, 1)
        kept = list(
            worldcover.geometry._nonempty_pixel_children(shapely.Polygon(), piece, 10, 10, 1)
        )
        assert _bounds(kept) == [(0, 0, 1, 1)]
        kept = list(
            worldcover.geometry._nonempty_pixel_children(piece, shapely.Polygon(), 10, 10, 1)
        )
        assert _bounds(kept) == [(0, 0, 1, 1)]


class TestFeatureBatching:
    @pytest.mark.parametrize(
        ("args", "kwargs", "expected"),
        [
            ((127, 0.0, 0.0), {}, False),
            ((128, 0.0, 0.0), {}, True),
            ((1, 1_999_999_999.0, 1.0), {}, False),
            ((1, 1_999_999_999.0, 2.0), {}, True),
            ((1, 0.0, 2_000_000_001.0), {}, True),
            ((1, 0.0, 0.0, 1_999_999, 1), {}, False),
            ((1, 0.0, 0.0, 1_999_999, 2), {}, True),
            ((1, 0.0, 0.0, 0, 2_000_001), {}, True),
            # Unstated pixel-cell counts default to zero.
            ((1, 0.0, 0.0), {"next_pixel_cells": 2_000_000}, False),
            ((1, 0.0, 0.0), {"batch_pixel_cells": 2_000_000}, False),
        ],
    )
    def test_limits_are_inclusive(self, args, kwargs, expected) -> None:
        assert worldcover.geometry._batch_exceeds_limits(*args, **kwargs) is expected

    @pytest.mark.parametrize(
        ("setting", "areas", "cells", "expected"),
        [
            (None, [1, 1, 1], [0, 0, 0], [(0, 3)]),
            ("_MAX_FEATURES_PER_BATCH", [1, 1, 1], [0, 0, 0], [(0, 2), (2, 3)]),
            ("_MAX_BATCH_AREA_M2", [3, 2, 1], [0, 0, 0], [(0, 2), (2, 3)]),
            ("_MAX_BATCH_AREA_M2", [100, 1, 1], [0, 0, 0], [(0, 1), (1, 3)]),
            ("_MAX_BATCH_AREA_M2", [3, 3, 3], [0, 0, 0], [(0, 1), (1, 2), (2, 3)]),
            ("_MAX_BATCH_AREA_M2", [3, 1, 1, 1, 3], [0] * 5, [(0, 3), (3, 5)]),
            ("_MAX_BATCH_AREA_M2", [3, 3, 1, 1, 1], [0] * 5, [(0, 1), (1, 4), (4, 5)]),
            ("_MAX_BATCH_PIXEL_BBOX_CELLS", [0] * 4, [6, 3, 3, 3], [(0, 2), (2, 4)]),
            ("_MAX_BATCH_PIXEL_BBOX_CELLS", [0] * 3, [6, 4, 1], [(0, 2), (2, 3)]),
            ("_MAX_BATCH_PIXEL_BBOX_CELLS", [0] * 3, [6, 6, 4], [(0, 1), (1, 3)]),
            ("_MAX_BATCH_PIXEL_BBOX_CELLS", [0] * 2, [99, 1], [(0, 1), (1, 2)]),
            (None, [], [], []),
        ],
    )
    def test_ranges_respect_each_budget(self, monkeypatch, setting, areas, cells, expected) -> None:
        limits = {
            "_MAX_FEATURES_PER_BATCH": 2,
            "_MAX_BATCH_AREA_M2": 5.0,
            "_MAX_BATCH_PIXEL_BBOX_CELLS": 10,
        }
        if setting:
            monkeypatch.setattr(worldcover.geometry, setting, limits[setting])
        got = list(worldcover.geometry._yield_feature_batches(np.array(areas), np.array(cells)))
        assert got == expected

    def test_area_and_cell_arrays_must_align(self) -> None:
        with pytest.raises(ValueError, match="zip"):
            list(
                worldcover.geometry._yield_feature_batches(np.zeros(2), np.zeros(3, dtype=np.int64))
            )

    @staticmethod
    def _frame(**columns) -> gpd.GeoDataFrame:
        geoms = [shapely.box(0.5, 0.5, 1.5, 1.5), shapely.box(5, 5, 6, 6)]
        return gpd.GeoDataFrame(columns, geometry=geoms, crs="EPSG:4326")

    def test_areas_are_read_from_the_frame_as_floats(self) -> None:
        areas = worldcover.geometry._feature_areas(self._frame(area_m2=[3, 4]))
        assert (areas.tolist(), areas.dtype) == ([3.0, 4.0], np.float64)

    def test_the_area_column_is_borrowed_not_duplicated(self) -> None:
        frame = self._frame(area_m2=[3.0, 4.0])
        assert np.shares_memory(
            worldcover.geometry._feature_areas(frame), frame["area_m2"].to_numpy()
        )

    def test_areas_default_to_zero_without_the_column(self) -> None:
        areas = worldcover.geometry._feature_areas(self._frame())
        assert (areas.tolist(), areas.dtype) == ([0.0, 0.0], np.float64)

    @pytest.mark.parametrize("missing", ["transform", "width", "height", "all"])
    def test_pixel_cells_are_zero_without_a_complete_raster_description(self, missing) -> None:
        args = {"transform": rasterio.Affine(1, 0, 0, 0, -1, 10), "width": 10, "height": 10}
        for key in args if missing == "all" else [missing]:
            args[key] = None
        cells = worldcover.geometry._feature_pixel_cells(self._frame(), **args)
        assert (cells.tolist(), cells.dtype) == ([0, 0], np.int64)

    def test_pixel_cells_bound_each_features_raster_envelope(self) -> None:
        transform = rasterio.Affine(1, 0, 0, 0, -1, 10)
        cells = worldcover.geometry._feature_pixel_cells(self._frame(), transform, 10, 10)
        assert (cells.tolist(), cells.dtype) == ([9, 9], np.int64)

    def test_feature_batches_split_on_raster_envelope_cells(self, monkeypatch) -> None:
        geoms = [shapely.box(0.5, 0.5, 1.5, 1.5), shapely.box(5, 5, 6, 6)]
        frame = gpd.GeoDataFrame(geometry=geoms, crs="EPSG:4326")
        transform = rasterio.Affine(1, 0, 0, 0, -1, 10)
        assert list(worldcover.geometry._feature_batches(frame)) == [(0, 2)]
        assert list(worldcover.geometry._feature_batches(frame, transform, 10, 10)) == [(0, 2)]
        monkeypatch.setattr(worldcover.geometry, "_MAX_BATCH_PIXEL_BBOX_CELLS", 17)
        assert list(worldcover.geometry._feature_batches(frame, transform, 10, 10)) == [
            (0, 1),
            (1, 2),
        ]


class TestGeometryTransforms:
    def test_pixel_and_map_coordinates_round_trip_through_a_sheared_transform(self) -> None:
        transform = rasterio.Affine(0.5, 0.125, 10.0, 0.25, -0.5, 20.0)
        pixel = worldcover.geometry._pixel_geometry(shapely.Point(11.0, 18.0), transform)
        assert transform * (pixel.x, pixel.y) == pytest.approx((11.0, 18.0))
        back = worldcover.geometry._map_geometry(pixel, transform)
        assert (back.x, back.y) == pytest.approx((11.0, 18.0))
        mapped = worldcover.geometry._map_geometry(shapely.Point(2.0, 3.0), transform)
        assert (mapped.x, mapped.y) == pytest.approx((10.0 + 1.0 + 0.375, 20.0 + 0.5 - 1.5))
