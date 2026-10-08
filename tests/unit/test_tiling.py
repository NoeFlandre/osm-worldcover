"""ESA WorldCover's own 3-degree tile grid."""

import pytest

from osm_worldcover.domain.tiling import (
    TILE_DEGREES,
    Tile,
    tile_bounds,
    tile_for,
    tiles_for_bbox,
)


def test_tiles_are_three_degrees() -> None:
    assert TILE_DEGREES == 3


def test_tile_name_matches_the_esa_filename_convention() -> None:
    # Tiles are named by their south-west corner, zero padded: N48E006.
    assert tile_for(6.13, 49.61).name == "N48E006"


def test_tile_name_uses_south_and_west_hemispheres() -> None:
    assert tile_for(-3.0, -6.0).name == "S06W003"


def test_tile_name_at_the_origin() -> None:
    assert tile_for(0.0, 0.0).name == "N00E000"


def test_points_in_the_same_tile_share_it() -> None:
    assert tile_for(6.01, 48.01) == tile_for(8.99, 50.99)


def test_points_across_a_boundary_differ() -> None:
    assert tile_for(5.9, 49.0) != tile_for(6.1, 49.0)


def test_tile_bounds_contain_their_own_point() -> None:
    tile = tile_for(6.13, 49.61)
    minx, miny, maxx, maxy = tile_bounds(tile)
    assert (minx, miny, maxx, maxy) == (6.0, 48.0, 9.0, 51.0)


def test_negative_coordinates_floor_downwards() -> None:
    assert tile_bounds(tile_for(-0.1, -0.1))[:2] == (-3.0, -3.0)


def test_a_bbox_inside_one_tile_needs_one_tile() -> None:
    assert tiles_for_bbox((6.1, 49.1, 6.2, 49.2)) == [tile_for(6.1, 49.1)]


def test_a_bbox_spanning_tiles_returns_every_covering_tile() -> None:
    tiles = tiles_for_bbox((5.9, 47.9, 6.1, 48.1))
    assert len(tiles) == 4
    assert tiles == sorted(tiles)


def test_a_bbox_ending_exactly_on_a_boundary_does_not_pull_in_the_next_tile() -> None:
    assert tiles_for_bbox((6.0, 48.0, 9.0, 51.0)) == [tile_for(6.0, 48.0)]


def test_degenerate_bbox_still_yields_its_tile() -> None:
    assert tiles_for_bbox((6.13, 49.61, 6.13, 49.61)) == [tile_for(6.13, 49.61)]


def test_degenerate_bbox_on_a_boundary_still_yields_its_tile() -> None:
    assert tiles_for_bbox((6.0, 48.0, 6.0, 48.0)) == [tile_for(6.0, 48.0)]


@pytest.mark.parametrize(
    ("bbox", "expected_corners"),
    [
        # Two corner tiles alone span this box, so only the full set proves coverage.
        ((0.0, 0.0, 9.0, 9.0), [(lat, lon) for lat in (0, 3, 6) for lon in (0, 3, 6)]),
        # Zero height on a grid line: the tiles of that row only.
        ((0.0, 0.0, 9.0, 0.0), [(0, 0), (0, 3), (0, 6)]),
        # Zero width on a meridian: the two tiles it straddles.
        ((3.0, 0.0, 3.0, 6.0), [(0, 3), (3, 3)]),
        # Crossing the origin with negative coordinates.
        ((-4.0, -4.0, 4.0, -2.0), [(lat, lon) for lat in (-6, -3) for lon in (-6, -3, 0, 3)]),
    ],
)
def test_tiles_for_bbox_returns_exactly_the_tiles_it_touches(
    bbox: tuple[float, float, float, float], expected_corners: list[tuple[int, int]]
) -> None:
    assert tiles_for_bbox(bbox) == [Tile(lat, lon) for lat, lon in expected_corners]


def test_a_subnormal_negative_coordinate_belongs_to_the_tile_below_zero() -> None:
    # value / 3 underflows to -0.0 here, and flooring that gave tile 0 instead of -3.
    assert tile_for(0.0, -5e-324) == Tile(-3, 0)


def test_a_bbox_ending_at_zero_after_a_subnormal_negative_is_not_empty() -> None:
    assert tiles_for_bbox((0.0, -5e-324, 0.0, 0.0)) == [Tile(-3, 0)]


def test_inverted_bbox_is_rejected() -> None:
    with pytest.raises(ValueError):
        tiles_for_bbox((9.0, 48.0, 6.0, 51.0))


def test_tiles_for_bbox_is_deterministically_ordered() -> None:
    bbox = (2.0, 44.0, 11.0, 53.0)
    assert tiles_for_bbox(bbox) == sorted(set(tiles_for_bbox(bbox)))


def test_every_tile_name_round_trips_through_its_bounds() -> None:
    for lon in range(-180, 180, 3):
        for lat in range(-60, 60, 3):
            tile = tile_for(lon + 0.5, lat + 0.5)
            assert tile_for(*tile_bounds(tile)[:2]) == tile
