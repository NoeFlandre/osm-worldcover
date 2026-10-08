"""Invariants of splits, tiling and text identity across generated inputs."""

import hashlib
import math

import h3
from hypothesis import given, settings
from hypothesis import strategies as st

from osm_worldcover.domain.splits import Split, SplitRatios, assign_cell, cell_for
from osm_worldcover.domain.text import dedup_key, normalise
from osm_worldcover.domain.tiling import (
    TILE_DEGREES,
    Tile,
    tile_bounds,
    tile_for,
    tiles_for_bbox,
)

PROFILE = settings(max_examples=100, deadline=None, derandomize=True)

lats = st.floats(min_value=-90.0, max_value=90.0, allow_nan=False, allow_infinity=False)
lons = st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False)
seeds = st.integers(min_value=0, max_value=2**32)
shares = st.tuples(st.integers(0, 10), st.integers(0, 10)).map(
    lambda p: SplitRatios(min(p) / 10, (max(p) - min(p)) / 10, (10 - max(p)) / 10)
)
whitespace = st.text(
    alphabet=[" ", "\t", "\n", "\r", chr(0xA0), chr(0x2003)], min_size=1, max_size=4
)
words = st.text(
    alphabet=st.characters(blacklist_categories=("Cs", "Z", "Cc")), min_size=1, max_size=8
)


@PROFILE
@given(lat=lats, lon=lons, seed=seeds, ratios=shares)
def test_split_is_shared_by_every_coordinate_in_a_cell(lat, lon, seed, ratios):
    """A point and its cell's centre are distinct coordinates in one H3 cell."""
    cell = cell_for(lat, lon)
    centre_lat, centre_lon = h3.cell_to_latlng(cell)
    assert cell_for(centre_lat, centre_lon) == cell
    assert assign_cell(cell_for(lat, lon), ratios, seed) == assign_cell(
        cell_for(centre_lat, centre_lon), ratios, seed
    )


@PROFILE
@given(lat=lats, lon=lons, seed=seeds, ratios=shares)
def test_split_matches_the_documented_hash_thresholds(lat, lon, seed, ratios):
    """Independent oracle: SHA-256 of 'seed:cell' against cumulative ratio thresholds.

    A constant or otherwise wrong assignment cannot satisfy this for varied inputs.
    """
    cell = cell_for(lat, lon)
    digest = hashlib.sha256(f"{seed}:{cell}".encode()).digest()
    position = int.from_bytes(digest[:8], "big") / float(1 << 64)
    if position < ratios.train:
        expected = Split.TRAIN
    elif position < ratios.train + ratios.validation:
        expected = Split.VALIDATION
    else:
        expected = Split.TEST
    assert assign_cell(cell, ratios, seed) is expected


@PROFILE
@given(lat=lats, lon=lons, seed=seeds)
def test_an_all_one_ratio_puts_every_cell_in_that_split(lat, lon, seed):
    cell = cell_for(lat, lon)
    assert assign_cell(cell, SplitRatios(1.0, 0.0, 0.0), seed) is Split.TRAIN
    assert assign_cell(cell, SplitRatios(0.0, 1.0, 0.0), seed) is Split.VALIDATION
    assert assign_cell(cell, SplitRatios(0.0, 0.0, 1.0), seed) is Split.TEST


@PROFILE
@given(lon=lons, lat=lats)
def test_tile_for_falls_inside_tile_bounds(lon, lat):
    minx, miny, maxx, maxy = tile_bounds(tile_for(lon, lat))
    assert minx <= lon < maxx
    assert miny <= lat < maxy
    assert maxx - minx == maxy - miny == TILE_DEGREES


@PROFILE
@given(a=lons, b=lons, c=lats, d=lats)
def test_tiles_for_bbox_contains_the_south_west_tile_and_its_envelope(a, b, c, d):
    bbox = (min(a, b), min(c, d), max(a, b), max(c, d))
    tiles = tiles_for_bbox(bbox)
    assert tiles == sorted(set(tiles))
    assert tile_for(bbox[0], bbox[1]) in tiles
    envelope = _envelope(tiles)
    assert all(u <= v for u, v in zip(envelope[:2], bbox[:2], strict=True))
    assert all(u >= v for u, v in zip(envelope[2:], bbox[2:], strict=True))


@PROFILE
@given(a=lons, b=lons, c=lats, d=lats)
def test_tiles_for_bbox_is_exactly_the_tiles_the_bbox_touches(a, b, c, d):
    bbox = (min(a, b), min(c, d), max(a, b), max(c, d))
    assert tiles_for_bbox(bbox) == _expected_tiles(bbox)


def _envelope(tiles):
    """The bounding box of the tiles' extents. This is not a geometric union."""
    mins_x, mins_y, maxs_x, maxs_y = zip(*(tile_bounds(t) for t in tiles), strict=True)
    return (min(mins_x), min(mins_y), max(maxs_x), max(maxs_y))


def _expected_tiles(bbox):
    """Independent oracle: every grid tile whose half-open extent meets the bbox.

    A tile [g, g + 3) counts when it overlaps the bbox with positive length, or when
    the bbox is a single point and the tile contains it.
    """
    minx, miny, maxx, maxy = bbox
    return [Tile(lat, lon) for lat in _grid_lines(miny, maxy) for lon in _grid_lines(minx, maxx)]


def _grid_lines(low, high):
    """Grid lines of the tiles that meet the interval [low, high]."""
    first = math.floor(low / TILE_DEGREES) - 1
    last = math.floor(high / TILE_DEGREES) + 1
    return [
        g
        for g in range(first * TILE_DEGREES, last * TILE_DEGREES + 1, TILE_DEGREES)
        if _meets(g, low, high)
    ]


def _meets(g, low, high):
    if low == high:
        return g <= low < g + TILE_DEGREES
    return g < high and g + TILE_DEGREES > low


@PROFILE
@given(text=st.text())
def test_normalise_is_idempotent(text):
    once = normalise(text)
    assert normalise(once) == once


@PROFILE
@given(parts=st.lists(words, min_size=1, max_size=6), label=st.text(max_size=8), data=st.data())
def test_dedup_key_stable_under_whitespace(parts, label, data):
    plain = " ".join(parts)
    padded = data.draw(whitespace) + "".join(part + data.draw(whitespace) for part in parts)
    assert dedup_key(padded, label) == dedup_key(plain, label)
