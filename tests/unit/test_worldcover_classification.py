"""WorldCover candidate lookup and class accumulation tests."""

from collections import defaultdict

import numpy as np
import pytest
import rasterio
import shapely
from tests.conftest import write_raster

import osm_worldcover.adapters.worldcover as worldcover


class TestCandidateLookup:
    def test_sorted_ids_are_searched_in_place(self) -> None:
        present, positions = worldcover.boundary._find_candidates(
            np.array([1, 3, 5], dtype=np.int64), np.array([0, 3, 5, 6], dtype=np.int64)
        )
        assert present.tolist() == [False, True, True, False]
        assert positions[present].tolist() == [1, 2]

    def test_unsorted_ids_are_matched_chunk_by_chunk(self, monkeypatch) -> None:
        monkeypatch.setattr(worldcover.boundary, "_ACCUMULATION_CELL_CHUNK", 2)
        cell_ids = np.array([9, 2, 6, 1, 7, 99, 4], dtype=np.int64)
        candidates = np.array([1, 2, 3, 4, 7, 8, 9, 30], dtype=np.int64)
        present, positions = worldcover.boundary._find_candidates(cell_ids, candidates)
        assert present.tolist() == [True, True, False, True, True, False, True, False]
        assert positions[present].tolist() == [3, 1, 6, 4, 0]
        assert present.dtype == np.bool_
        assert positions.dtype == np.int64

    @pytest.mark.parametrize(("cell_ids", "candidates"), [([], [1, 2]), ([1, 2], []), ([], [])])
    def test_empty_inputs_find_nothing(self, cell_ids, candidates) -> None:
        present, positions = worldcover.boundary._find_candidates(
            np.array(cell_ids, dtype=np.int64), np.array(candidates, dtype=np.int64)
        )
        assert present.tolist() == [False] * len(candidates)
        assert positions.tolist() == [0] * len(candidates)
        assert present.dtype == np.bool_
        assert positions.dtype == np.int64

    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            ([], True),
            ([5], True),
            ([1, 2, 3, 4, 5], True),
            ([1, 2, 2, 3], True),
            ([2, 2, 2, 2, 2], True),
            ([1, 2, 3, 2], False),
            ([1, 5, 3, 4], False),
            ([1, 3, 2, 4], False),
            ([1, 2, 1, 4, 5], False),
            ([3, 2, 1], False),
            ([2, 1], False),
            ([1, 0, 5, 6], False),
        ],
    )
    def test_sortedness_is_checked_within_and_across_slices(
        self, monkeypatch, values, expected
    ) -> None:
        array = np.array(values, dtype=np.int64)
        assert worldcover.boundary._is_sorted(array) is expected  # one slice
        monkeypatch.setattr(worldcover.boundary, "_ACCUMULATION_CELL_CHUNK", 2)
        assert worldcover.boundary._is_sorted(array) is expected  # several slices

    @pytest.mark.parametrize(
        ("previous", "chunk", "expected"),
        [
            (None, [1, 2], True),
            (1, [1, 2], True),
            (3, [2, 4], False),
            (None, [2, 1], False),
            (0, [2, 1], False),
            (None, [4], True),
            (None, [1, 3, 2], False),
        ],
    )
    def test_chunk_order_is_checked_against_its_predecessor(
        self, previous, chunk, expected
    ) -> None:
        assert bool(worldcover.boundary._chunk_follows(previous, np.array(chunk))) is expected


class TestClassAccumulation:
    def test_per_class_areas_are_scaled_and_added_to_existing_totals(self) -> None:
        totals = defaultdict(float, {10: 1.0})
        values = np.array([10, 10, 50, np.nan, 60, 70, 80])
        coverage = np.array([0.5, 0.25, 1.0, 1.0, np.nan, 0.0, -1.0])
        worldcover.boundary._add_class_coverage(totals, values, coverage, 2.0)
        assert dict(totals) == {10: pytest.approx(2.5), 50: pytest.approx(2.0)}

    def test_infinite_values_or_coverage_never_become_a_class_share(self) -> None:
        totals = defaultdict(float)
        values = np.array([np.inf, 10.0, 20.0])
        coverage = np.array([1.0, np.inf, 0.5])
        worldcover.boundary._add_class_coverage(totals, values, coverage, 1.0)
        assert dict(totals) == {20: 0.5}

    def test_a_zero_scale_adds_nothing(self) -> None:
        totals = defaultdict(float)
        worldcover.boundary._add_class_coverage(totals, np.array([10.0]), np.array([1.0]), 0.0)
        assert dict(totals) == {}

    def test_stable_cells_skip_excluded_invalid_and_empty_cells_across_slices(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(worldcover.boundary, "_ACCUMULATION_CELL_CHUNK", 3)
        classes = np.array([10, 10, 50, 50, np.nan, 50, 10, 10], dtype=float)
        fractions = np.array([0.5, 0.0, 0.25, 1.0, 1.0, np.nan, 0.5, 0.25])
        excluded = np.array([False, False, False, True, False, False, False, True])
        totals = defaultdict(float)
        worldcover.boundary._accumulate_stable_cells(totals, classes, fractions, excluded, 2.0)
        assert dict(totals) == {10: pytest.approx(2.0), 50: pytest.approx(0.5)}

    def test_candidate_classes_follow_their_matched_positions(self) -> None:
        got = worldcover.boundary._matched_candidate_classes(
            np.arange(4),
            np.array([7.0, 8.0]),
            np.array([True, False, True, False]),
            np.array([1, 0, 0, 0]),
        )
        assert got[[0, 2]].tolist() == [8.0, 7.0]
        assert np.isnan(got[[1, 3]]).all()

    def test_missing_classes_are_read_from_the_raster_and_masked_pixels_stay_empty(
        self, tmp_path
    ) -> None:
        values = np.arange(1, 13, dtype="uint8").reshape(3, 4)
        values[1, 1] = 0
        path = write_raster(tmp_path / "ids.tif", values, origin=(0, 3))
        classes = np.full(4, np.nan)
        with rasterio.open(path) as dataset:
            worldcover.boundary._read_missing_classes(
                dataset,
                np.array([1, 2, 1]),
                np.array([2, 3, 1]),
                classes,
                np.array([3, 0, 1]),
            )
        assert classes[[0, 3]].tolist() == [12.0, 7.0]
        assert np.isnan(classes[[1, 2]]).all()

    @pytest.mark.parametrize("area", [0.0, -1.0, float("nan"), float("inf")])
    def test_a_polygon_without_a_usable_area_contributes_nothing(self, area) -> None:
        totals = defaultdict(float)
        worldcover.boundary._accumulate_corrected(
            totals, [], [], [], shapely.box(0, 0, 1, 1), None, 1.0, area
        )
        assert dict(totals) == {}
