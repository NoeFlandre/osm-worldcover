"""WorldCover tile addressing and class-coverage extraction."""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import shapely
from rasterio.windows import Window
from shapely.geometry import MultiPolygon, Polygon
from tests.conftest import write_raster

from osm_worldcover.adapters.worldcover import WorldCoverTiles, class_coverage
from osm_worldcover.domain.tiling import Tile

BOUNDARY_FIXTURES = Path(__file__).parents[1] / "fixtures" / "worldcover-boundary"


def one(geom) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"polygon_id": ["p"]}, geometry=[geom], crs="EPSG:4326")


class TestTileAddressing:
    def test_url_follows_the_published_naming_scheme(self, tmp_path) -> None:
        tiles = WorldCoverTiles(tmp_path)
        assert tiles.url_for(Tile(48, 6)).endswith(
            "/v200/2021/map/ESA_WorldCover_10m_2021_v200_N48E006_Map.tif"
        )

    def test_version_and_year_are_reflected_in_the_url(self, tmp_path) -> None:
        tiles = WorldCoverTiles(tmp_path, version="v100", year=2020)
        url = tiles.url_for(Tile(-3, -6))
        assert "/v100/2020/map/" in url
        assert url.endswith("ESA_WorldCover_10m_2020_v100_S03W006_Map.tif")

    def test_cache_path_is_derived_from_the_tile_name(self, tmp_path) -> None:
        tiles = WorldCoverTiles(tmp_path)
        assert tiles.path_for(Tile(48, 6)).parent == tmp_path
        assert "N48E006" in tiles.path_for(Tile(48, 6)).name


class TestClassCoverage:
    def test_a_polygon_split_between_two_classes(self, half_and_half, square) -> None:
        assert class_coverage([half_and_half], square) == [
            {10: pytest.approx(0.5), 50: pytest.approx(0.5)}
        ]

    def test_a_polygon_inside_one_class_is_wholly_that_class(self, half_and_half) -> None:
        left = one(Polygon([(0, 0), (0, 4), (2, 4), (2, 0)]))
        assert class_coverage([half_and_half], left) == [{10: pytest.approx(1.0)}]

    def test_partial_pixels_are_weighted_by_area_not_counted_whole(self, half_and_half) -> None:
        # Spans 1.5 columns of class 10 and 0.5 of class 50.
        strip = one(Polygon([(0.5, 0), (0.5, 4), (2.5, 4), (2.5, 0)]))
        assert class_coverage([half_and_half], strip) == [
            {10: pytest.approx(0.75), 50: pytest.approx(0.25)}
        ]

    def test_nodata_is_absent_and_leaves_the_polygon_only_half_covered(
        self, with_nodata, square
    ) -> None:
        """No-data is not a class, so it is reported as missing coverage.

        Renormalising over observed pixels would label a half-unobserved
        polygon with full confidence; leaving the gap lets dominance refuse it.
        """
        coverage = class_coverage([with_nodata], square)[0]
        assert coverage == {10: pytest.approx(0.5)}
        assert sum(coverage.values()) == pytest.approx(0.5)

    def test_a_polygon_outside_the_raster_has_no_coverage(self, half_and_half) -> None:
        far = one(Polygon([(50, 50), (50, 51), (51, 51), (51, 50)]))
        assert class_coverage([half_and_half], far) == [{}]

    def test_coverage_beyond_the_raster_edge_is_not_counted_as_observed(
        self, half_and_half
    ) -> None:
        """Half the polygon lies off the raster, so classes must cover only half of it.

        Reporting 1.0 here would let an unobserved polygon pass the dominance
        test on the strength of the sliver that happened to be on the tile.
        """
        overhang = one(Polygon([(2, 0), (2, 4), (6, 4), (6, 0)]))
        assert sum(class_coverage([half_and_half], overhang)[0].values()) == pytest.approx(0.5)

    def test_several_polygons_keep_their_input_order(self, half_and_half) -> None:
        frame = gpd.GeoDataFrame(
            {"polygon_id": ["a", "b"]},
            geometry=[
                Polygon([(0, 0), (0, 4), (2, 4), (2, 0)]),
                Polygon([(2, 0), (2, 4), (4, 4), (4, 0)]),
            ],
            crs="EPSG:4326",
        )
        assert class_coverage([half_and_half], frame) == [
            {10: pytest.approx(1.0)},
            {50: pytest.approx(1.0)},
        ]

    def test_an_empty_frame_yields_no_rows(self, half_and_half) -> None:
        empty = gpd.GeoDataFrame({"polygon_id": []}, geometry=[], crs="EPSG:4326")
        assert class_coverage([half_and_half], empty) == []

    def test_a_polygon_with_a_hole_keeps_only_observed_pixel_area(self, half_and_half) -> None:
        shell = [(0, 0), (0, 4), (4, 4), (4, 0)]
        hole = [[(1, 1), (1, 3), (3, 3), (3, 1)]]
        coverage = class_coverage([half_and_half], one(Polygon(shell, holes=hole)))[0]
        assert coverage == {10: pytest.approx(0.5), 50: pytest.approx(0.5)}

    def test_a_multipolygon_keeps_each_component(self, half_and_half) -> None:
        left = Polygon([(0, 0), (0, 2), (1, 2), (1, 0)])
        right = Polygon([(3, 2), (3, 4), (4, 4), (4, 2)])
        coverage = class_coverage([half_and_half], one(MultiPolygon([left, right])))[0]
        assert coverage == {10: pytest.approx(0.5), 50: pytest.approx(0.5)}

    def test_southern_hemisphere_polygon_crosses_two_raster_tiles(self, tmp_path) -> None:
        left = write_raster(
            tmp_path / "south-west.tif",
            np.full((4, 2), 10, dtype="uint8"),
            origin=(0, -2),
        )
        right = write_raster(
            tmp_path / "south-east.tif",
            np.full((4, 2), 50, dtype="uint8"),
            origin=(2, -2),
        )
        across_seam = one(Polygon([(1, -6), (1, -2), (3, -2), (3, -6)]))
        coverage = class_coverage([left, right], across_seam)[0]
        assert coverage == {10: pytest.approx(0.5), 50: pytest.approx(0.5)}


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            "peru",
            {
                30: 0.03163706888209,
                40: 0.00005711142495,
                80: 0.90592590549736,
                90: 0.06237991419558,
            },
        ),
        (
            "netherlands",
            {
                10: 0.92213518778889,
                30: 0.03879794117976,
                40: 0.00004881067089,
                50: 0.00052447539422,
                80: 0.02518630617598,
                90: 0.01330727879024,
            },
        ),
    ],
)
def test_real_raster_corner_cases_match_geos_coverage(name, expected, tmp_path) -> None:
    metadata = json.loads((BOUNDARY_FIXTURES / f"{name}-metadata.json").read_text())
    geometry = shapely.from_wkt((BOUNDARY_FIXTURES / f"{name}.wkt").read_text())
    raster_path = _sparse_full_grid_fixture(name, metadata, tmp_path)

    coverage = class_coverage([raster_path], one(geometry))[0]

    assert coverage == pytest.approx(expected, abs=1e-9)
    assert sum(coverage.values()) == pytest.approx(1.0, abs=1e-9)
    assert max(coverage, key=coverage.get) in {10, 80}


def _sparse_full_grid_fixture(name: str, metadata: dict, tmp_path: Path) -> Path:
    """Restore a small tile window at its full WorldCover grid origin."""
    window_path = BOUNDARY_FIXTURES / f"{name}-window.tif"
    with rasterio.open(window_path) as window:
        values = window.read(1)
    output = tmp_path / f"{name}-full-grid.tif"
    with rasterio.open(
        output,
        "w",
        driver="GTiff",
        width=metadata["width"],
        height=metadata["height"],
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=rasterio.Affine(*metadata["transform"]),
        nodata=0,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress="DEFLATE",
        SPARSE_OK="TRUE",
    ) as dataset:
        dataset.write(
            values,
            1,
            window=Window(metadata["col0"], metadata["row0"], values.shape[1], values.shape[0]),
        )
    return output
