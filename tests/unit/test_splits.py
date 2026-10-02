"""Geographically disjoint split assignment via H3 cells."""

import pytest

from osm_worldcover.domain.splits import (
    DEFAULT_RATIOS,
    DEFAULT_RESOLUTION,
    Split,
    SplitRatios,
    assign_cell,
    cell_for,
)

ANDORRA = (42.4954, 1.5169)


def test_cell_for_is_deterministic() -> None:
    assert cell_for(*ANDORRA) == cell_for(*ANDORRA)


def test_nearby_points_share_a_cell_and_therefore_a_split() -> None:
    a = cell_for(42.4954, 1.5169)
    b = cell_for(42.4960, 1.5175)  # ~80 m away
    assert a == b
    assert assign_cell(a) is assign_cell(b)


def test_distant_points_do_not_share_a_cell() -> None:
    assert cell_for(*ANDORRA) != cell_for(52.52, 13.405)  # Berlin


def test_assignment_is_deterministic_for_a_given_seed() -> None:
    cell = cell_for(*ANDORRA)
    assert assign_cell(cell, seed=7) is assign_cell(cell, seed=7)


def test_seed_changes_the_partition() -> None:
    cells = [cell_for(40 + i * 0.5, 2 + i * 0.5) for i in range(60)]
    assert [assign_cell(c, seed=1) for c in cells] != [assign_cell(c, seed=2) for c in cells]


def test_ratios_are_approximately_honoured_over_many_cells() -> None:
    cells = [cell_for(36 + (i % 180) * 0.15, -9 + (i // 180) * 0.15) for i in range(4000)]
    splits = [assign_cell(c) for c in set(cells)]
    n = len(splits)
    train = splits.count(Split.TRAIN) / n
    assert DEFAULT_RATIOS.train - 0.05 < train < DEFAULT_RATIOS.train + 0.05
    assert set(splits) == set(Split)


def test_default_resolution_and_ratios_are_declared() -> None:
    assert DEFAULT_RESOLUTION == 5
    assert DEFAULT_RATIOS.train + DEFAULT_RATIOS.validation + DEFAULT_RATIOS.test == 1.0


def test_ratios_must_sum_to_one() -> None:
    with pytest.raises(ValueError):
        SplitRatios(0.5, 0.2, 0.2)


@pytest.mark.parametrize(("lat", "lon"), [(91.0, 0.0), (-91.0, 0.0), (0.0, 181.0), (0.0, -181.0)])
def test_out_of_range_coordinates_are_rejected(lat: float, lon: float) -> None:
    with pytest.raises(ValueError):
        cell_for(lat, lon)


def test_nan_coordinates_are_rejected() -> None:
    with pytest.raises(ValueError):
        cell_for(float("nan"), 0.0)


class TestCoordinateBoundaries:
    """Mutation testing found the range edges and the finiteness check untested."""

    @pytest.mark.parametrize(
        ("lat", "lon"), [(90.0, 0.0), (-90.0, 0.0), (0.0, 180.0), (0.0, -180.0)]
    )
    def test_the_extremes_of_the_globe_are_valid(self, lat: float, lon: float) -> None:
        assert cell_for(lat, lon)

    @pytest.mark.parametrize(("lat", "lon"), [(float("nan"), 0.0), (0.0, float("nan"))])
    def test_nan_is_rejected_as_non_finite_not_as_out_of_range(self, lat, lon) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            cell_for(lat, lon)

    @pytest.mark.parametrize(("lat", "lon"), [(float("inf"), 0.0), (0.0, float("-inf"))])
    def test_infinity_is_rejected_as_non_finite(self, lat, lon) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            cell_for(lat, lon)

    def test_out_of_range_latitude_says_so(self) -> None:
        with pytest.raises(ValueError, match="latitude out of range"):
            cell_for(91.0, 0.0)

    def test_out_of_range_longitude_says_so(self) -> None:
        with pytest.raises(ValueError, match="longitude out of range"):
            cell_for(0.0, 181.0)


class TestSplitRatioValidation:
    @pytest.mark.parametrize(
        "ratios",
        [(-0.1, 0.6, 0.5), (0.6, -0.1, 0.5), (0.6, 0.5, -0.1)],
    )
    def test_any_negative_share_is_rejected(self, ratios) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            SplitRatios(*ratios)

    def test_a_zero_share_is_allowed(self) -> None:
        """A build may legitimately want no validation split."""
        assert SplitRatios(0.9, 0.0, 0.1).validation == 0.0

    def test_ratios_summing_to_one_within_float_error_are_accepted(self) -> None:
        third = 1.0 / 3.0
        assert SplitRatios(third, third, third).train == third

    def test_a_sum_that_is_wrong_says_so(self) -> None:
        with pytest.raises(ValueError, match=r"must sum to 1\.0"):
            SplitRatios(0.5, 0.2, 0.2)


@pytest.mark.parametrize(
    ("position", "expected"),
    [
        (0.49, Split.TRAIN),
        (0.5, Split.VALIDATION),
        (0.74, Split.VALIDATION),
        (0.75, Split.TEST),
        (0.99, Split.TEST),
    ],
)
def test_split_boundaries_belong_to_the_upper_split(monkeypatch, position, expected) -> None:
    from osm_worldcover.domain import splits

    monkeypatch.setattr(splits, "_unit_hash", lambda cell, seed: position)

    assert assign_cell("cell", SplitRatios(0.5, 0.25, 0.25)) is expected
