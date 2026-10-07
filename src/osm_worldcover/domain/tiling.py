"""ESA WorldCover's 3-degree tile grid.

WorldCover ships as 3x3 degree GeoTIFFs named after their south-west corner
(``ESA_WorldCover_10m_2021_v200_N48E006_Map.tif``). Reusing the product's own
grid means a tile name is both the cache key and the remote filename, so a
re-run asks for exactly the same tiles and nothing has to be re-derived.
"""

import math
from dataclasses import dataclass
from typing import Final

__all__ = ["TILE_DEGREES", "Bbox", "Tile", "tile_bounds", "tile_for", "tiles_for_bbox"]

TILE_DEGREES: Final[int] = 3

#: ``(min_lon, min_lat, max_lon, max_lat)`` in WGS84 degrees.
Bbox = tuple[float, float, float, float]


@dataclass(frozen=True, order=True, slots=True)
class Tile:
    """One 3-degree cell, identified by the integer degrees of its SW corner."""

    lat: int
    lon: int

    @property
    def name(self) -> str:
        """The ESA tile name, e.g. ``N48E006`` -- also the remote filename stem."""
        ns = "N" if self.lat >= 0 else "S"
        ew = "E" if self.lon >= 0 else "W"
        return f"{ns}{abs(self.lat):02d}{ew}{abs(self.lon):03d}"


def tile_for(lon: float, lat: float) -> Tile:
    """Return the tile containing ``(lon, lat)``."""
    return Tile(
        _floor_to_grid(lat),
        _floor_to_grid(lon),
    )


def tile_bounds(tile: Tile) -> Bbox:
    """Return the WGS84 bounds of ``tile``."""
    return (
        float(tile.lon),
        float(tile.lat),
        float(tile.lon + TILE_DEGREES),
        float(tile.lat + TILE_DEGREES),
    )


def tiles_for_bbox(bbox: Bbox) -> list[Tile]:
    """Return every tile covering ``bbox``, sorted for reproducible iteration."""
    minx, miny, maxx, maxy = bbox
    if maxx < minx or maxy < miny:
        raise ValueError(f"inverted bbox: {bbox!r}")
    lo = tile_for(minx, miny)
    # An edge landing exactly on a tile boundary touches the next tile without
    # entering it, so the upper tile is taken from just inside that edge.
    hi = Tile(_upper_grid(miny, maxy), _upper_grid(minx, maxx))
    # Iterated in Tile's own field order so the result is already sorted.
    return [
        Tile(lat, lon)
        for lat in range(lo.lat, hi.lat + 1, TILE_DEGREES)
        for lon in range(lo.lon, hi.lon + 1, TILE_DEGREES)
    ]


def _floor_to_grid(value: float) -> int:
    # Float floor division, not value / TILE_DEGREES: a subnormal negative value
    # divides to -0.0 and would floor to the wrong tile.
    return int(math.floor(value // TILE_DEGREES) * TILE_DEGREES)


def _upper_grid(low: float, high: float) -> int:
    """The grid line of the last tile actually entered between ``low`` and ``high``."""
    grid = _floor_to_grid(high)
    if high == grid and high > low:
        return grid - TILE_DEGREES
    return grid
