"""Polygon validity screening."""

import warnings

import pytest
from shapely import wkt
from shapely.errors import ShapelyError
from shapely.geometry import GeometryCollection, LineString, Point, Polygon

from osm_worldcover.domain.geometry import _all_finite, is_usable_polygon

SQUARE = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
BOWTIE = wkt.loads("POLYGON ((0 0, 1 1, 1 0, 0 1, 0 0))")


def test_a_simple_square_is_usable() -> None:
    assert is_usable_polygon(SQUARE)


def test_none_is_not_usable() -> None:
    assert not is_usable_polygon(None)


def test_empty_geometry_is_not_usable() -> None:
    assert not is_usable_polygon(Polygon())
    assert not is_usable_polygon(GeometryCollection())


@pytest.mark.parametrize("geom", [Point(0, 0), LineString([(0, 0), (1, 1)])])
def test_non_areal_geometry_is_not_usable(geom) -> None:
    assert not is_usable_polygon(geom)


def test_self_intersecting_polygon_is_not_usable() -> None:
    assert not is_usable_polygon(BOWTIE)


@pytest.mark.parametrize("error", [AttributeError, ValueError, ShapelyError])
def test_a_geometry_whose_bounds_cannot_be_read_is_not_finite(error) -> None:
    class Broken:
        @property
        def bounds(self):
            raise error("no bounds")

    assert not _all_finite(Broken())  # ty: ignore[invalid-argument-type]


def test_an_unexpected_bounds_error_propagates() -> None:
    class Buggy:
        @property
        def bounds(self):
            raise RuntimeError("bug")

    with pytest.raises(RuntimeError, match="bug"):
        _all_finite(Buggy())  # ty: ignore[invalid-argument-type]


def test_a_valid_polygon_whose_area_underflows_to_zero_is_not_usable() -> None:
    sliver = Polygon([(0, 0), (1e-200, 0), (1e-200, 1e-200), (0, 1e-200)])

    assert sliver.is_valid
    assert sliver.area == 0.0
    assert not is_usable_polygon(sliver)


def test_zero_area_polygon_is_not_usable() -> None:
    assert not is_usable_polygon(Polygon([(0, 0), (1, 1), (2, 2), (0, 0)]))


def test_non_finite_coordinates_are_not_usable() -> None:
    # Building the NaN polygon is itself what warns, on some platforms, and
    # that is shapely's construction rather than the code under test.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        nan_polygon = Polygon([(0, 0), (0, 1), (float("nan"), 1), (1, 0)])
    assert not is_usable_polygon(nan_polygon)


def test_multipolygon_is_usable() -> None:
    from shapely.geometry import MultiPolygon

    assert is_usable_polygon(MultiPolygon([SQUARE, Polygon([(2, 2), (2, 3), (3, 3)])]))


def test_an_invalid_polygon_with_positive_area_is_rejected() -> None:
    """Validity and area are both required, not either.

    Mutation testing found `and` could become `or` undetected: the bow-tie case
    has zero area, so it never distinguished the two. A hole outside its shell
    is invalid *and* has positive area, which does.
    """
    hole_outside = wkt.loads("POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0), (3 3, 4 3, 4 4, 3 3))")
    assert hole_outside.area > 0
    assert not hole_outside.is_valid
    assert not is_usable_polygon(hole_outside)
