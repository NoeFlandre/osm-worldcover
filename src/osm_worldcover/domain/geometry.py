"""Polygon validity screening.

Invalid polygons are dropped rather than repaired. Repair (``make_valid``)
would silently alter the very shape whose area fraction becomes the label, so
the dataset prefers a smaller, trustworthy set of polygons over a larger one
built on geometries the source did not actually assert.
"""

import math

from shapely.geometry.base import BaseGeometry

__all__ = ["AREAL_TYPES", "is_usable_polygon"]

AREAL_TYPES = frozenset({"Polygon", "MultiPolygon"})


def is_usable_polygon(geom: BaseGeometry | None) -> bool:
    """Return whether ``geom`` is an areal, simple, finite, non-degenerate polygon.

    The order matters: emptiness and finiteness are checked before validity and
    area, because those two are unreliable on a geometry carrying NaN.
    """
    if geom is None or geom.is_empty:
        return False
    return geom.geom_type in AREAL_TYPES and _is_measurable(geom)


def _is_measurable(geom: BaseGeometry) -> bool:
    """Whether an area fraction can be computed from ``geom``.

    Finiteness is checked first: ``is_valid`` and ``area`` are both unreliable
    on a geometry carrying NaN.
    """
    return _all_finite(geom) and geom.is_valid and geom.area > 0.0


def _all_finite(geom: BaseGeometry) -> bool:
    """Return whether every coordinate is finite.

    NaN coordinates make ``is_valid`` and ``area`` unreliable, so they are
    screened out before either is consulted.
    """
    try:
        bounds = geom.bounds
    except Exception:
        return False
    return all(math.isfinite(value) for value in bounds)
