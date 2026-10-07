"""Invariants of splits, tiling and text identity across generated inputs."""

from hypothesis import given, settings
from hypothesis import strategies as st

from osm_worldcover.domain.splits import SplitRatios, assign_cell, cell_for
from osm_worldcover.domain.text import dedup_key, normalise
from osm_worldcover.domain.tiling import TILE_DEGREES, tile_bounds, tile_for, tiles_for_bbox

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
def test_same_cell_gives_same_split(lat, lon, seed, ratios):
    cell = cell_for(lat, lon)
    assert assign_cell(cell, ratios, seed) == assign_cell(cell, ratios, seed)
    assert cell_for(lat, lon) == cell


@PROFILE
@given(lon=lons, lat=lats)
def test_tile_for_falls_inside_tile_bounds(lon, lat):
    minx, miny, maxx, maxy = tile_bounds(tile_for(lon, lat))
    assert minx <= lon < maxx
    assert miny <= lat < maxy
    assert maxx - minx == maxy - miny == TILE_DEGREES


@PROFILE
@given(a=lons, b=lons, c=lats, d=lats)
def test_tiles_for_bbox_covers_bbox(a, b, c, d):
    bbox = (min(a, b), min(c, d), max(a, b), max(c, d))
    tiles = tiles_for_bbox(bbox)
    assert tiles == sorted(set(tiles))
    bounds = [tile_bounds(t) for t in tiles]
    assert min(b_[0] for b_ in bounds) <= bbox[0]
    assert min(b_[1] for b_ in bounds) <= bbox[1]
    assert max(b_[2] for b_ in bounds) >= bbox[2]
    assert max(b_[3] for b_ in bounds) >= bbox[3]
    assert tile_for(bbox[0], bbox[1]) in tiles


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
