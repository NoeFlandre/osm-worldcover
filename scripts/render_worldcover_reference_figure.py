#!/usr/bin/env python3
"""Render one bounded WorldCover label example from the repository fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import shapely
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patheffects import Stroke, withStroke
from pyproj import Transformer
from pyproj import network as pyproj_network
from rasterio.enums import Resampling
from rasterio.warp import array_bounds, calculate_default_transform, reproject
from shapely import wkt
from shapely.ops import transform as transform_geometry

pyproj_network.set_network_enabled(False)

ROOT = Path(__file__).resolve().parents[1]
RASTER = ROOT / "tests/fixtures/worldcover-boundary/netherlands-window.tif"
OSM_WKT = ROOT / "tests/fixtures/worldcover-boundary/netherlands.wkt"
CRS_METRIC = "EPSG:3035"
CLASSES = {
    10: ("Tree cover", "#006400"),
    30: ("Grassland", "#ffff4c"),
    40: ("Cropland", "#f096ff"),
    50: ("Built-up", "#fa0000"),
    80: ("Permanent water bodies", "#0064c8"),
    90: ("Herbaceous wetland", "#0096a0"),
}
BACKGROUND = "#f5f6f3"
INK = "#1d2933"
MUTED = "#53616b"
GRID = "#dfe4e2"
OSM_EDGE = "#202b35"


def to_metric(geometry: shapely.Geometry) -> shapely.Geometry:
    transformer = Transformer.from_crs("EPSG:4326", CRS_METRIC, always_xy=True)
    return transform_geometry(transformer.transform, geometry)


def to_local_km(geometry: shapely.Geometry, center: tuple[float, float]) -> shapely.Geometry:
    return transform_geometry(
        lambda x, y, _z=None: ((x - center[0]) / 1000, (y - center[1]) / 1000),
        geometry,
    )


def polygons(geometry: shapely.Geometry):
    if geometry.geom_type == "Polygon":
        yield geometry
    elif geometry.geom_type in {"MultiPolygon", "GeometryCollection"}:
        for child in geometry.geoms:
            yield from polygons(child)


def add_osm_outline(
    axis: plt.Axes, geometry: shapely.Geometry, center: tuple[float, float]
) -> None:
    for polygon in polygons(to_local_km(geometry, center)):
        for ring in (polygon.exterior, *polygon.interiors):
            (line,) = axis.plot(
                *ring.xy,
                color=OSM_EDGE,
                linewidth=2.1,
                zorder=9,
                solid_joinstyle="round",
            )
            line.set_path_effects([Stroke(linewidth=4.6, foreground="white", alpha=0.9)])


def add_scale(axis: plt.Axes, length_km: float) -> None:
    xmin, xmax = axis.get_xlim()
    ymin, ymax = axis.get_ylim()
    x0 = xmin + (xmax - xmin) * 0.055
    y0 = ymin + (ymax - ymin) * 0.065
    axis.plot([x0, x0 + length_km], [y0, y0], color="white", linewidth=5, zorder=12)
    axis.plot([x0, x0 + length_km], [y0, y0], color=INK, linewidth=2.2, zorder=13)
    axis.plot([x0, x0], [y0 - 0.035, y0 + 0.035], color=INK, linewidth=1, zorder=13)
    axis.plot(
        [x0 + length_km, x0 + length_km],
        [y0 - 0.035, y0 + 0.035],
        color=INK,
        linewidth=1,
        zorder=13,
    )
    axis.text(
        x0 + length_km / 2,
        y0 + 0.04,
        f"{length_km:g} km",
        ha="center",
        va="bottom",
        color=INK,
        fontsize=9,
        weight="bold",
        zorder=14,
        path_effects=[withStroke(linewidth=2.5, foreground="white")],
    )


def render(output: Path) -> dict[str, object]:
    if not RASTER.is_file() or not OSM_WKT.is_file():
        raise FileNotFoundError("Run this script from a complete osm-worldcover checkout.")

    polygon_ll = wkt.loads(OSM_WKT.read_text(encoding="utf-8").strip())
    polygon_m = to_metric(polygon_ll)
    center = (polygon_m.centroid.x, polygon_m.centroid.y)

    with rasterio.open(RASTER) as source:
        values = source.read(1)
        dst_transform, dst_width, dst_height = calculate_default_transform(
            source.crs,
            CRS_METRIC,
            source.width,
            source.height,
            *source.bounds,
            resolution=10,
        )
        projected = np.zeros((dst_height, dst_width), dtype=np.uint8)
        reproject(
            source=values,
            destination=projected,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=0,
            dst_transform=dst_transform,
            dst_crs=CRS_METRIC,
            dst_nodata=0,
            resampling=Resampling.nearest,
        )

    left, bottom, right, top = array_bounds(dst_height, dst_width, dst_transform)
    display = np.ma.masked_equal(projected, 0)
    codes = [code for code in CLASSES if np.any(projected == code)]
    colors = [CLASSES[code][1] for code in codes]
    cmap = ListedColormap(colors)
    cmap.set_bad("#edf1ee")
    norm = BoundaryNorm([code - 5 for code in codes] + [codes[-1] + 15], cmap.N)

    from osm_worldcover.adapters.worldcover import class_coverage
    from osm_worldcover.domain.dominance import decide

    frame = gpd.GeoDataFrame({"geometry": [polygon_ll]}, crs="EPSG:4326")
    shares = class_coverage([RASTER], frame)[0]
    verdict = decide(shares, polygon_area=1.0, threshold=0.80)

    fig = plt.figure(figsize=(13.2, 8.7), dpi=180, facecolor=BACKGROUND)
    fig.text(
        0.055,
        0.947,
        "REFERENCE LABELS  /  LOCAL VIEW",
        color=MUTED,
        fontsize=10,
        weight="bold",
        family="DejaVu Sans",
    )
    fig.text(
        0.055,
        0.892,
        "WorldCover pixels inside one mapped area",
        color=INK,
        fontsize=22,
        weight="bold",
        family="DejaVu Sans",
    )
    fig.text(
        0.055,
        0.850,
        "ESA WorldCover 2021 v200  ·  10 m nominal  ·  11-class land-cover scheme",
        color=MUTED,
        fontsize=11.5,
        family="DejaVu Sans",
    )
    axis = fig.add_axes([0.055, 0.145, 0.625, 0.66], facecolor="#edf1ee")
    side = fig.add_axes([0.715, 0.145, 0.235, 0.66], facecolor="white")
    side.set_axis_off()
    side.add_patch(
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor="white",
            edgecolor="#e1e5e3",
            linewidth=1,
            transform=side.transAxes,
            zorder=-1,
        )
    )

    axis.imshow(
        display,
        origin="upper",
        extent=(
            (left - center[0]) / 1000,
            (right - center[0]) / 1000,
            (bottom - center[1]) / 1000,
            (top - center[1]) / 1000,
        ),
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        zorder=1,
    )
    add_osm_outline(axis, polygon_m, center)
    bounds = (
        (left - center[0]) / 1000,
        (bottom - center[1]) / 1000,
        (right - center[0]) / 1000,
        (top - center[1]) / 1000,
    )
    minx, miny, maxx, maxy = bounds
    width = (maxx - minx) * 1.14
    height = (maxy - miny) * 1.14
    ax_box = axis.get_position()
    target_ratio = (fig.get_figwidth() * ax_box.width) / (fig.get_figheight() * ax_box.height)
    if width / height < target_ratio:
        width = height * target_ratio
    else:
        height = width / target_ratio
    midx = (minx + maxx) / 2
    midy = (miny + maxy) / 2
    axis.set_xlim(midx - width / 2, midx + width / 2)
    axis.set_ylim(midy - height / 2, midy + height / 2)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Easting offset from local view centre (km)", color=MUTED, fontsize=9)
    axis.set_ylabel("Northing offset from local view centre (km)", color=MUTED, fontsize=9)
    axis.tick_params(colors=MUTED, labelsize=8, length=0, pad=5)
    axis.grid(color=GRID, linewidth=0.6, zorder=0)
    for spine in axis.spines.values():
        spine.set_color("#cbd3d0")
    add_scale(axis, 0.5)
    axis.annotate(
        "N",
        xy=(0.94, 0.92),
        xytext=(0.94, 0.84),
        xycoords="axes fraction",
        ha="center",
        va="center",
        color=INK,
        fontsize=9,
        weight="bold",
        arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.25},
    )
    longitude, latitude = Transformer.from_crs(CRS_METRIC, "EPSG:4326", always_xy=True).transform(
        *center
    )
    axis.set_title(
        f"{latitude:.3f}°N, {longitude:.3f}°E   ·   original OSM footprint",
        loc="left",
        color=MUTED,
        fontsize=9.5,
        pad=8,
    )

    method_note = (
        "Union same-class pixel support, intersect it with the original OSM polygon, "
        "then divide by that polygon's full area. NoData and uncovered area stay "
        "in the denominator."
    )
    side.text(
        0.045,
        0.95,
        "LAND-COVER PIXELS",
        transform=side.transAxes,
        color=INK,
        fontsize=10,
        weight="bold",
        va="top",
    )
    y = 0.88
    for code in codes:
        name, color = CLASSES[code]
        side.add_patch(
            plt.Rectangle(
                (0.035, y - 0.018),
                0.052,
                0.037,
                facecolor=color,
                edgecolor="#56636c",
                linewidth=0.55,
                transform=side.transAxes,
            )
        )
        side.text(
            0.11,
            y,
            f"{code}  {name}",
            transform=side.transAxes,
            color=INK,
            fontsize=8.5,
            weight="bold",
            va="center",
        )
        y -= 0.066
    side.plot(
        [0.035, 0.088],
        [y, y],
        color=OSM_EDGE,
        linewidth=2.2,
        transform=side.transAxes,
        solid_capstyle="round",
    )
    side.text(
        0.11,
        y,
        "OSM polygon boundary",
        transform=side.transAxes,
        color=INK,
        fontsize=8.5,
        weight="bold",
        va="center",
    )

    class_name = CLASSES.get(verdict.code, (str(verdict.code), ""))[0]
    side.text(
        0.045,
        0.34,
        "FULL-POLYGON AREA",
        transform=side.transAxes,
        color=MUTED,
        fontsize=9,
        weight="bold",
        va="top",
    )
    side.text(
        0.045,
        0.305,
        f"{class_name}: {verdict.fraction * 100:.1f}%",
        transform=side.transAxes,
        color=INK,
        fontsize=12,
        weight="bold",
        va="top",
    )
    side.text(
        0.045,
        0.265,
        "passes ≥80%" if verdict.accepted else "does not pass ≥80%",
        transform=side.transAxes,
        color="#2f6c58" if verdict.accepted else "#9a4b3e",
        fontsize=10,
        weight="bold",
        va="top",
    )
    side.text(
        0.045,
        0.23,
        "HOW THE POLYGON LABEL IS GATED",
        transform=side.transAxes,
        color=MUTED,
        fontsize=9,
        weight="bold",
        va="top",
    )
    side.text(
        0.045,
        0.20,
        method_note,
        transform=side.transAxes,
        color=INK,
        fontsize=8.5,
        va="top",
        linespacing=1.35,
        wrap=True,
    )

    fig.text(
        0.055,
        0.055,
        "Drunen area, Netherlands  ·  OSM way 15304016  ·  © ESA WorldCover / "
        "Copernicus Sentinel data 2021  ·  CC BY 4.0",
        color=MUTED,
        fontsize=8,
        family="DejaVu Sans",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor=fig.get_facecolor(), bbox_inches=None)
    plt.close(fig)

    return {
        "output": str(output),
        "dominant_class": verdict.code,
        "dominant_fraction": verdict.fraction,
        "accepted_at_80_percent": verdict.accepted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs/assets/worldcover-reference-labels.png",
    )
    args = parser.parse_args()
    print(json.dumps(render(args.output), indent=2))


if __name__ == "__main__":
    main()
