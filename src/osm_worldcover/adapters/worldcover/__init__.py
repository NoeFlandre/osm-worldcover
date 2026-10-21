"""Fetch ESA WorldCover tiles and measure class coverage per polygon.

Coverage is expressed as a share of the **polygon's own area**, never of the
area that happened to be observed. WorldCover marks unobserved pixels as
no-data, and renormalising over the observed ones would label a polygon that
is mostly ocean, or mostly off the edge of a tile, with full confidence. Here
the unobserved part simply goes unattributed, so the shares sum to less than
one and the dominance rule can refuse the polygon.

Because shares are relative to the polygon, contributions from several tiles
add up directly: a polygon straddling a tile boundary needs no mosaic, just
the sum over the tiles it touches.

``class_coverage`` and published dataset builders keep their degree-based
legacy semantics. The source-wide polygon-pool adapter uses
``geodesic_class_coverage`` to report physical WGS84 square-metre evidence;
those values must not be confused with existing released fractions.
"""

from .extract import class_coverage
from .geodesic_coverage import AREA_MEASUREMENT, AREA_UNIT, geodesic_class_coverage
from .pool_attempts import SourceWideCoverageAttempt, iter_sourcewide_coverage
from .tiles import (
    DEFAULT_BASE_URL,
    DEFAULT_CACHED_TILES,
    TileNotPublishedError,
    TileSource,
    WorldCoverTiles,
)

__all__ = [
    "AREA_MEASUREMENT",
    "AREA_UNIT",
    "DEFAULT_BASE_URL",
    "DEFAULT_CACHED_TILES",
    "SourceWideCoverageAttempt",
    "TileNotPublishedError",
    "TileSource",
    "WorldCoverTiles",
    "class_coverage",
    "geodesic_class_coverage",
    "iter_sourcewide_coverage",
]
