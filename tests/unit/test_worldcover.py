"""WorldCover tile addressing and class-coverage extraction."""

import json
import urllib.error
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import shapely
from rasterio.windows import Window
from shapely.geometry import MultiPolygon, Polygon
from tests.conftest import write_raster

import osm_worldcover.adapters.worldcover as worldcover
from osm_worldcover.adapters.worldcover import (
    TileNotPublishedError,
    WorldCoverTiles,
    class_coverage,
)
from osm_worldcover.domain.dominance import decide
from osm_worldcover.domain.tiling import Tile

BOUNDARY_FIXTURES = Path(__file__).parents[1] / "fixtures" / "worldcover-boundary"


def one(geom) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"polygon_id": ["p"]}, geometry=[geom], crs="EPSG:4326")


class ExactExtractRecorder:
    """Capture the real adapter call and outputs for raster contract checks."""

    def __init__(self) -> None:
        self.original = worldcover.exact_extract
        self.calls = []
        self.values = []

    def __call__(self, raster, features, operations, **kwargs):
        result = self.original(raster, features, operations, **kwargs)
        self.calls.append((raster, features.copy(), operations, kwargs))
        self.values.extend(result["values"])
        return result


def _assert_nodata_values_are_unmasked(recorder: ExactExtractRecorder) -> None:
    assert all(not np.any(np.ma.getmaskarray(values)) for values in recorder.values)


def _assert_nodata_calls_omit_default(recorder: ExactExtractRecorder) -> None:
    assert all(call[2] == ["cell_id", "coverage", "values"] for call in recorder.calls)
    assert all("default_value" not in call[3] for call in recorder.calls)


def _assert_nodata_coverage(coverage: dict[int, float]) -> None:
    assert coverage == {10: pytest.approx(0.5)}
    assert sum(coverage.values()) == pytest.approx(0.5)


def _recorded_pixel_bounds(recorder: ExactExtractRecorder) -> list[tuple[int, int]]:
    bounds = []
    for raster, features, _, _ in recorder.calls:
        with rasterio.open(raster) as dataset:
            for geometry in features.geometry:
                pixel_geometry = worldcover._pixel_geometry(geometry, dataset.transform)
                bounds.append(
                    (
                        worldcover._pixel_bbox_cells(pixel_geometry, dataset.width, dataset.height),
                        shapely.get_num_coordinates(pixel_geometry),
                    )
                )
    return bounds


def _assert_pixel_bounds(bounds: list[tuple[int, int]], minimum: int) -> None:
    assert len(bounds) > minimum
    assert all(cells <= 9 and coordinates <= 12 for cells, coordinates in bounds)


def _assert_accepted_chunk_label(actual, expected) -> None:
    assert (actual.code, actual.accepted) == (expected.code, True)
    assert actual.code == 50
    assert actual.fraction == pytest.approx(expected.fraction, abs=1e-12)


def _assert_rejected_chunk_label(actual, expected) -> None:
    assert (actual.code, actual.accepted) == (expected.code, False)
    assert actual.code == 50
    assert actual.fraction == pytest.approx(expected.fraction, abs=1e-12)


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

    def test_a_frame_in_a_different_crs_than_the_raster_is_rejected(self, tmp_path) -> None:
        values = np.full((4, 4), 10, dtype="uint8")
        mercator = write_raster(tmp_path / "mercator.tif", values, crs="EPSG:3857")
        with pytest.raises(ValueError, match="CRS"):
            class_coverage([mercator], one(shapely.box(0, 0, 1, 1)))

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
        self, with_nodata, square, monkeypatch
    ) -> None:
        """No-data is not a class, so it is reported as missing coverage.

        Renormalising over observed pixels would label a half-unobserved
        polygon with full confidence; leaving the gap lets dominance refuse it.
        """
        recorder = ExactExtractRecorder()
        monkeypatch.setattr(worldcover, "exact_extract", recorder)
        coverage = class_coverage([with_nodata], square)[0]

        _assert_nodata_values_are_unmasked(recorder)
        _assert_nodata_calls_omit_default(recorder)
        _assert_nodata_coverage(coverage)

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

    def test_feature_batches_preserve_coverage_and_bound_extraction(
        self, half_and_half, monkeypatch
    ) -> None:
        frame = gpd.GeoDataFrame(
            {"polygon_id": ["left", "right", "across"], "area_m2": [6.0, 3.0, 3.0]},
            geometry=[
                Polygon([(0, 0), (0, 4), (2, 4), (2, 0)]),
                Polygon([(2, 0), (2, 4), (4, 4), (4, 0)]),
                Polygon([(1, 0), (1, 4), (3, 4), (3, 0)]),
            ],
            crs="EPSG:4326",
        )
        monkeypatch.setattr(worldcover, "_MAX_FEATURES_PER_BATCH", 2)
        monkeypatch.setattr(worldcover, "_MAX_BATCH_AREA_M2", 8.0)
        monkeypatch.setattr(worldcover, "_ACCUMULATION_CELL_CHUNK", 2)
        original_extract = worldcover.exact_extract
        batch_sizes = []

        def recording_extract(raster, features, operations, **kwargs):
            batch_sizes.append((len(features), kwargs["max_cells_in_memory"]))
            return original_extract(raster, features, operations, **kwargs)

        monkeypatch.setattr(worldcover, "exact_extract", recording_extract)

        coverage = class_coverage([half_and_half], frame)

        assert batch_sizes == [
            (1, worldcover._MAX_CELLS_IN_MEMORY),
            (2, worldcover._MAX_CELLS_IN_MEMORY),
        ]
        assert coverage == [
            {10: pytest.approx(1.0)},
            {50: pytest.approx(1.0)},
            {10: pytest.approx(0.5), 50: pytest.approx(0.5)},
        ]

    def test_boundary_candidate_lookup_handles_unsorted_cell_ids(self) -> None:
        cell_ids = np.array([9, 2, 6, 1], dtype=np.int64)
        candidates = np.array([1, 6, 8, 9], dtype=np.int64)

        present, positions = worldcover._find_candidates(cell_ids, candidates)

        assert present.tolist() == [True, True, False, True]
        assert cell_ids[positions[present]].tolist() == candidates[present].tolist()

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

    def test_spatially_chunked_geometry_preserves_coverage_and_label(
        self, half_and_half, monkeypatch
    ) -> None:
        left = Polygon([(0.2, 0.2), (0.2, 0.8), (0.8, 0.8), (0.8, 0.2)])
        right = Polygon([(2.1, 0.25), (2.1, 3.75), (3.8, 3.75), (3.8, 0.25)])
        frame = one(MultiPolygon([left, right]))
        expected = class_coverage([half_and_half], frame)[0]
        expected_label = decide(expected, polygon_area=1.0, threshold=0.8)
        recorder = ExactExtractRecorder()

        with monkeypatch.context() as bounded:
            bounded.setattr(worldcover, "_MAX_BATCH_PIXEL_BBOX_CELLS", 9)
            bounded.setattr(worldcover, "_MAX_GEOMETRY_COORDINATES", 12)
            bounded.setattr(worldcover, "_BOUNDARY_CELLS_PER_CHUNK", 1)
            bounded.setattr(worldcover, "exact_extract", recorder)
            actual = class_coverage([half_and_half], frame)[0]

        actual_label = decide(actual, polygon_area=1.0, threshold=0.8)
        assert actual == pytest.approx(expected, abs=1e-12)
        _assert_pixel_bounds(_recorded_pixel_bounds(recorder), minimum=1)
        _assert_accepted_chunk_label(actual_label, expected_label)

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

    def test_spatially_chunked_polygon_preserves_tile_boundary_nodata_and_purity(
        self, tmp_path, monkeypatch
    ) -> None:
        left = write_raster(
            tmp_path / "west.tif",
            np.full((6, 4), 10, dtype="uint8"),
            origin=(0, -2),
        )
        east_values = np.full((6, 4), 50, dtype="uint8")
        east_values[:, 1:] = 0
        right = write_raster(
            tmp_path / "east.tif",
            east_values,
            origin=(4, -2),
        )
        geometry = Polygon([(3.25, -7.5), (3.25, -2.5), (5.5, -2.5), (5.5, -7.5)])
        frame = one(geometry)
        expected = class_coverage([left, right], frame)[0]
        expected_label = decide(expected, polygon_area=1.0, threshold=0.45)
        recorder = ExactExtractRecorder()

        with monkeypatch.context() as bounded:
            bounded.setattr(worldcover, "_MAX_BATCH_PIXEL_BBOX_CELLS", 9)
            bounded.setattr(worldcover, "_MAX_GEOMETRY_COORDINATES", 12)
            bounded.setattr(worldcover, "_BOUNDARY_CELLS_PER_CHUNK", 1)
            bounded.setattr(worldcover, "exact_extract", recorder)
            actual = class_coverage([left, right], frame)[0]

        actual_label = decide(actual, polygon_area=1.0, threshold=0.45)
        assert expected == {10: pytest.approx(1 / 3), 50: pytest.approx(4 / 9)}
        assert sum(expected.values()) == pytest.approx(7 / 9)
        assert actual == pytest.approx(expected, abs=1e-12)
        _assert_pixel_bounds(_recorded_pixel_bounds(recorder), minimum=2)
        _assert_rejected_chunk_label(actual_label, expected_label)
        assert actual_label.fraction == pytest.approx(4 / 9)


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


def _bounds(geometries) -> list[tuple[float, ...]]:
    return [tuple(g.bounds) for g in geometries]


class TestSplitAxisSelection:
    @pytest.mark.parametrize(
        ("minimum", "maximum", "expected"),
        [
            (0.0, 4.0, 2.0),
            (0.0, 3.0, 1.0),
            (0.0, 1.5, 1.0),
            (2.5, 3.5, 3.0),
            (-3.0, -1.0, -2.0),
            (-1.5, -0.5, -1.0),
            (0.0, 1.0, None),
            (0.2, 0.8, None),
            (3.0, 3.0, None),
        ],
    )
    def test_midpoint_is_an_interior_integer_boundary(self, minimum, maximum, expected) -> None:
        assert worldcover._pixel_aligned_midpoint(minimum, maximum) == expected

    @pytest.mark.parametrize(
        ("x_span", "y_span", "x_mid", "y_mid", "expected"),
        [
            (2.0, 2.0, None, None, True),
            (3.0, 2.0, None, None, True),
            (1.0, 2.0, None, None, False),
            (9.0, 1.0, None, 1.0, False),
            (1.0, 9.0, 1.0, None, True),
            (2.0, 2.0, 1.0, 1.0, True),
            (3.0, 2.0, 1.0, 1.0, True),
            (1.0, 2.0, 1.0, 1.0, False),
        ],
    )
    def test_axis_prefers_integer_splits_then_the_longest_span(
        self, x_span, y_span, x_mid, y_mid, expected
    ) -> None:
        assert worldcover._choose_split_axis(x_span, y_span, x_mid, y_mid) is expected


class TestSplitOnce:
    @pytest.mark.parametrize(
        ("geometry", "first", "second"),
        [
            (shapely.box(0, 0, 8, 4), (0, 0, 4, 4), (4, 0, 8, 4)),
            (shapely.box(0, 0, 4, 8), (0, 0, 4, 4), (0, 4, 4, 8)),
            (shapely.box(0, 0, 4, 4), (0, 0, 2, 4), (2, 0, 4, 4)),
            (shapely.box(1, 1, 8, 4), (1, 1, 4, 4), (4, 1, 8, 4)),
            (shapely.box(0.2, 0.1, 0.8, 0.3), (0.2, 0.1, 0.5, 0.3), (0.5, 0.1, 0.8, 0.3)),
            (shapely.box(0.2, 0.1, 0.4, 0.9), (0.2, 0.1, 0.4, 0.5), (0.2, 0.5, 0.4, 0.9)),
            (shapely.box(0.2, 0.0, 0.8, 4.0), (0.2, 0.0, 0.8, 2.0), (0.2, 2.0, 0.8, 4.0)),
            (shapely.box(0.0, 0.2, 9.0, 0.8), (0.0, 0.2, 4.0, 0.8), (4.0, 0.2, 9.0, 0.8)),
            # Offsets from the origin must not change which axis is longest.
            (shapely.box(10, 10, 12, 14), (10, 10, 12, 12), (10, 12, 12, 14)),
            (shapely.box(10, 10, 14, 12), (10, 10, 12, 12), (12, 10, 14, 12)),
        ],
    )
    def test_geometry_is_cut_on_pixel_lines_or_the_plain_midpoint(
        self, geometry, first, second
    ) -> None:
        pieces = worldcover._split_pixel_geometry_once(geometry)
        assert _bounds(pieces) == [first, second]
        assert sum(p.area for p in pieces) == pytest.approx(geometry.area)

    @pytest.mark.parametrize(
        ("first", "second", "expected"),
        [
            (shapely.Polygon(), shapely.Polygon(), True),
            ("same", "same", True),
            ("same", shapely.Polygon(), False),
            (shapely.Polygon(), "same", False),
            ("same", shapely.box(0, 0, 1, 1), False),
            (shapely.box(0, 0, 1, 1), "same", False),
            (shapely.box(0, 0, 1, 1), shapely.box(1, 0, 2, 1), False),
        ],
    )
    def test_a_split_is_stalled_only_when_it_makes_no_progress(
        self, first, second, expected
    ) -> None:
        geometry = shapely.box(0, 0, 2, 1)
        first = geometry if isinstance(first, str) else first
        second = geometry if isinstance(second, str) else second
        assert worldcover._split_is_stalled(geometry, first, second) is expected


class TestBoundedChunks:
    @pytest.mark.parametrize(
        ("geometry", "width", "height", "expected"),
        [
            (shapely.Polygon(), 5, 5, 0),
            (shapely.box(2.5, 2.5, 3.5, 3.5), 10, 10, 16),
            (shapely.box(0, 0, 1, 1), 10, 10, 4),
            (shapely.box(2, 0, 3, 1), 3, 7, 2 * 2),
            (shapely.box(5, 4, 6, 5), 6, 9, 2 * 3),
            (shapely.box(1, 5, 2, 6), 10, 6, 3 * 2),
            (shapely.box(1, 1, 2, 2), 2, 3, 2 * 3),
            (shapely.box(50, 0, 51, 1), 10, 10, 0),
            (shapely.box(0, 50, 1, 51), 10, 10, 0),
        ],
    )
    def test_envelope_cells_are_clamped_to_the_raster(
        self, geometry, width, height, expected
    ) -> None:
        assert worldcover._pixel_bbox_cells(geometry, width, height) == expected

    @pytest.mark.parametrize(
        ("cells", "coordinates", "bounded"),
        [(9, 5, True), (8, 5, False), (9, 4, False), (10, 6, True)],
    )
    def test_geometry_is_bounded_inclusive_of_both_limits(
        self, monkeypatch, cells, coordinates, bounded
    ) -> None:
        monkeypatch.setattr(worldcover, "_MAX_BATCH_PIXEL_BBOX_CELLS", cells)
        monkeypatch.setattr(worldcover, "_MAX_GEOMETRY_COORDINATES", coordinates)
        square = shapely.box(0, 0, 2, 2)  # 9 envelope cells, 5 coordinates
        assert bool(worldcover._geometry_is_bounded(square, 10, 10)) is bounded
        assert bool(worldcover._needs_spatial_chunks(square, 10, 10)) is (not bounded)

    def test_chunks_split_on_pixel_lines_and_clip_to_the_raster(self, monkeypatch) -> None:
        monkeypatch.setattr(worldcover, "_MAX_BATCH_PIXEL_BBOX_CELLS", 15)
        chunks = list(worldcover._bounded_pixel_chunks(shapely.box(0, 0, 8, 2), 10, 10))
        assert _bounds(chunks) == [(0, 0, 4, 2), (4, 0, 6, 2), (6, 0, 8, 2)]
        clipped = list(worldcover._bounded_pixel_chunks(shapely.box(-5, 0, 3, 2), 10, 10))
        assert _bounds(clipped) == [(0, 0, 3, 2)]
        assert list(worldcover._bounded_pixel_chunks(shapely.box(20, 20, 21, 21), 10, 10)) == []

    def test_a_bounded_geometry_is_returned_whole(self) -> None:
        geometry = shapely.box(0, 0, 3, 3)
        (chunk,) = worldcover._split_pixel_geometry(geometry, 10, 10, depth=0)
        assert chunk.equals(geometry)

    def test_splitting_stops_at_the_depth_limit(self, monkeypatch) -> None:
        monkeypatch.setattr(worldcover, "_MAX_BATCH_PIXEL_BBOX_CELLS", 15)
        geometry = shapely.box(0, 0, 8, 2)
        monkeypatch.setattr(worldcover, "_MAX_GEOMETRY_SPLIT_DEPTH", 2)
        assert len(list(worldcover._bounded_pixel_chunks(geometry, 10, 10))) == 3
        monkeypatch.setattr(worldcover, "_MAX_GEOMETRY_SPLIT_DEPTH", 1)
        message = "^could not bound a WorldCover geometry after 1 spatial splits$"
        with pytest.raises(ValueError, match=message):
            list(worldcover._bounded_pixel_chunks(geometry, 10, 10))
        with pytest.raises(ValueError, match=message):
            list(worldcover._split_pixel_geometry(geometry, 10, 10, depth=1))
        monkeypatch.setattr(worldcover, "_MAX_GEOMETRY_SPLIT_DEPTH", 4)
        assert len(list(worldcover._split_pixel_geometry(geometry, 10, 10, depth=2))) == 3
        monkeypatch.setattr(worldcover, "_MAX_GEOMETRY_SPLIT_DEPTH", 3)
        with pytest.raises(ValueError, match="after 3 spatial splits"):
            list(worldcover._split_pixel_geometry(geometry, 10, 10, depth=2))

    @pytest.mark.parametrize("pieces", [("empty", "empty"), ("same", "same")])
    def test_a_stalled_split_is_an_error_not_an_endless_recursion(
        self, monkeypatch, pieces
    ) -> None:
        geometry = shapely.box(0, 0, 8, 2)
        monkeypatch.setattr(worldcover, "_MAX_BATCH_PIXEL_BBOX_CELLS", 1)
        stalled = tuple(geometry if p == "same" else shapely.Polygon() for p in pieces)
        monkeypatch.setattr(worldcover, "_split_pixel_geometry_once", lambda _g: stalled)
        with pytest.raises(
            ValueError, match=r"^WorldCover geometry split did not reduce its spatial extent$"
        ):
            list(worldcover._split_pixel_geometry(geometry, 10, 10, depth=0))

    def test_empty_halves_are_skipped(self) -> None:
        piece = shapely.box(0, 0, 1, 1)
        kept = list(worldcover._nonempty_pixel_children(shapely.Polygon(), piece, 10, 10, 1))
        assert _bounds(kept) == [(0, 0, 1, 1)]
        kept = list(worldcover._nonempty_pixel_children(piece, shapely.Polygon(), 10, 10, 1))
        assert _bounds(kept) == [(0, 0, 1, 1)]


class TestFeatureBatching:
    @pytest.mark.parametrize(
        ("args", "kwargs", "expected"),
        [
            ((127, 0.0, 0.0), {}, False),
            ((128, 0.0, 0.0), {}, True),
            ((1, 1_999_999_999.0, 1.0), {}, False),
            ((1, 1_999_999_999.0, 2.0), {}, True),
            ((1, 0.0, 2_000_000_001.0), {}, True),
            ((1, 0.0, 0.0, 1_999_999, 1), {}, False),
            ((1, 0.0, 0.0, 1_999_999, 2), {}, True),
            ((1, 0.0, 0.0, 0, 2_000_001), {}, True),
            # Unstated pixel-cell counts default to zero.
            ((1, 0.0, 0.0), {"next_pixel_cells": 2_000_000}, False),
            ((1, 0.0, 0.0), {"batch_pixel_cells": 2_000_000}, False),
        ],
    )
    def test_limits_are_inclusive(self, args, kwargs, expected) -> None:
        assert worldcover._batch_exceeds_limits(*args, **kwargs) is expected

    @pytest.mark.parametrize(
        ("setting", "areas", "cells", "expected"),
        [
            (None, [1, 1, 1], [0, 0, 0], [(0, 3)]),
            ("_MAX_FEATURES_PER_BATCH", [1, 1, 1], [0, 0, 0], [(0, 2), (2, 3)]),
            ("_MAX_BATCH_AREA_M2", [3, 2, 1], [0, 0, 0], [(0, 2), (2, 3)]),
            ("_MAX_BATCH_AREA_M2", [100, 1, 1], [0, 0, 0], [(0, 1), (1, 3)]),
            ("_MAX_BATCH_AREA_M2", [3, 3, 3], [0, 0, 0], [(0, 1), (1, 2), (2, 3)]),
            ("_MAX_BATCH_AREA_M2", [3, 1, 1, 1, 3], [0] * 5, [(0, 3), (3, 5)]),
            ("_MAX_BATCH_AREA_M2", [3, 3, 1, 1, 1], [0] * 5, [(0, 1), (1, 4), (4, 5)]),
            ("_MAX_BATCH_PIXEL_BBOX_CELLS", [0] * 4, [6, 3, 3, 3], [(0, 2), (2, 4)]),
            ("_MAX_BATCH_PIXEL_BBOX_CELLS", [0] * 3, [6, 4, 1], [(0, 2), (2, 3)]),
            ("_MAX_BATCH_PIXEL_BBOX_CELLS", [0] * 3, [6, 6, 4], [(0, 1), (1, 3)]),
            ("_MAX_BATCH_PIXEL_BBOX_CELLS", [0] * 2, [99, 1], [(0, 1), (1, 2)]),
            (None, [], [], []),
        ],
    )
    def test_ranges_respect_each_budget(self, monkeypatch, setting, areas, cells, expected) -> None:
        limits = {
            "_MAX_FEATURES_PER_BATCH": 2,
            "_MAX_BATCH_AREA_M2": 5.0,
            "_MAX_BATCH_PIXEL_BBOX_CELLS": 10,
        }
        if setting:
            monkeypatch.setattr(worldcover, setting, limits[setting])
        got = list(worldcover._yield_feature_batches(np.array(areas), np.array(cells)))
        assert got == expected

    def test_area_and_cell_arrays_must_align(self) -> None:
        with pytest.raises(ValueError, match="zip"):
            list(worldcover._yield_feature_batches(np.zeros(2), np.zeros(3, dtype=np.int64)))

    @staticmethod
    def _frame(**columns) -> gpd.GeoDataFrame:
        geoms = [shapely.box(0.5, 0.5, 1.5, 1.5), shapely.box(5, 5, 6, 6)]
        return gpd.GeoDataFrame(columns, geometry=geoms, crs="EPSG:4326")

    def test_areas_are_read_from_the_frame_as_floats(self) -> None:
        areas = worldcover._feature_areas(self._frame(area_m2=[3, 4]))
        assert (areas.tolist(), areas.dtype) == ([3.0, 4.0], np.float64)

    def test_the_area_column_is_borrowed_not_duplicated(self) -> None:
        frame = self._frame(area_m2=[3.0, 4.0])
        assert np.shares_memory(worldcover._feature_areas(frame), frame["area_m2"].to_numpy())

    def test_areas_default_to_zero_without_the_column(self) -> None:
        areas = worldcover._feature_areas(self._frame())
        assert (areas.tolist(), areas.dtype) == ([0.0, 0.0], np.float64)

    @pytest.mark.parametrize("missing", ["transform", "width", "height", "all"])
    def test_pixel_cells_are_zero_without_a_complete_raster_description(self, missing) -> None:
        args = {"transform": rasterio.Affine(1, 0, 0, 0, -1, 10), "width": 10, "height": 10}
        for key in args if missing == "all" else [missing]:
            args[key] = None
        cells = worldcover._feature_pixel_cells(self._frame(), **args)
        assert (cells.tolist(), cells.dtype) == ([0, 0], np.int64)

    def test_pixel_cells_bound_each_features_raster_envelope(self) -> None:
        transform = rasterio.Affine(1, 0, 0, 0, -1, 10)
        cells = worldcover._feature_pixel_cells(self._frame(), transform, 10, 10)
        assert (cells.tolist(), cells.dtype) == ([9, 9], np.int64)

    def test_feature_batches_split_on_raster_envelope_cells(self, monkeypatch) -> None:
        geoms = [shapely.box(0.5, 0.5, 1.5, 1.5), shapely.box(5, 5, 6, 6)]
        frame = gpd.GeoDataFrame(geometry=geoms, crs="EPSG:4326")
        transform = rasterio.Affine(1, 0, 0, 0, -1, 10)
        assert list(worldcover._feature_batches(frame)) == [(0, 2)]
        assert list(worldcover._feature_batches(frame, transform, 10, 10)) == [(0, 2)]
        monkeypatch.setattr(worldcover, "_MAX_BATCH_PIXEL_BBOX_CELLS", 17)
        assert list(worldcover._feature_batches(frame, transform, 10, 10)) == [(0, 1), (1, 2)]


class TestGeometryTransforms:
    def test_pixel_and_map_coordinates_round_trip_through_a_sheared_transform(self) -> None:
        transform = rasterio.Affine(0.5, 0.125, 10.0, 0.25, -0.5, 20.0)
        pixel = worldcover._pixel_geometry(shapely.Point(11.0, 18.0), transform)
        assert transform * (pixel.x, pixel.y) == pytest.approx((11.0, 18.0))
        back = worldcover._map_geometry(pixel, transform)
        assert (back.x, back.y) == pytest.approx((11.0, 18.0))
        mapped = worldcover._map_geometry(shapely.Point(2.0, 3.0), transform)
        assert (mapped.x, mapped.y) == pytest.approx((10.0 + 1.0 + 0.375, 20.0 + 0.5 - 1.5))


class TestCandidateLookup:
    def test_sorted_ids_are_searched_in_place(self) -> None:
        present, positions = worldcover._find_candidates(
            np.array([1, 3, 5], dtype=np.int64), np.array([0, 3, 5, 6], dtype=np.int64)
        )
        assert present.tolist() == [False, True, True, False]
        assert positions[present].tolist() == [1, 2]

    def test_unsorted_ids_are_matched_chunk_by_chunk(self, monkeypatch) -> None:
        monkeypatch.setattr(worldcover, "_ACCUMULATION_CELL_CHUNK", 2)
        cell_ids = np.array([9, 2, 6, 1, 7, 99, 4], dtype=np.int64)
        candidates = np.array([1, 2, 3, 4, 7, 8, 9, 30], dtype=np.int64)
        present, positions = worldcover._find_candidates(cell_ids, candidates)
        assert present.tolist() == [True, True, False, True, True, False, True, False]
        assert positions[present].tolist() == [3, 1, 6, 4, 0]
        assert present.dtype == np.bool_
        assert positions.dtype == np.int64

    @pytest.mark.parametrize(("cell_ids", "candidates"), [([], [1, 2]), ([1, 2], []), ([], [])])
    def test_empty_inputs_find_nothing(self, cell_ids, candidates) -> None:
        present, positions = worldcover._find_candidates(
            np.array(cell_ids, dtype=np.int64), np.array(candidates, dtype=np.int64)
        )
        assert present.tolist() == [False] * len(candidates)
        assert positions.tolist() == [0] * len(candidates)
        assert present.dtype == np.bool_
        assert positions.dtype == np.int64

    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            ([], True),
            ([5], True),
            ([1, 2, 3, 4, 5], True),
            ([1, 2, 2, 3], True),
            ([2, 2, 2, 2, 2], True),
            ([1, 2, 3, 2], False),
            ([1, 5, 3, 4], False),
            ([1, 3, 2, 4], False),
            ([1, 2, 1, 4, 5], False),
            ([3, 2, 1], False),
            ([2, 1], False),
            ([1, 0, 5, 6], False),
        ],
    )
    def test_sortedness_is_checked_within_and_across_slices(
        self, monkeypatch, values, expected
    ) -> None:
        array = np.array(values, dtype=np.int64)
        assert worldcover._is_sorted(array) is expected  # one slice
        monkeypatch.setattr(worldcover, "_ACCUMULATION_CELL_CHUNK", 2)
        assert worldcover._is_sorted(array) is expected  # several slices

    @pytest.mark.parametrize(
        ("previous", "chunk", "expected"),
        [
            (None, [1, 2], True),
            (1, [1, 2], True),
            (3, [2, 4], False),
            (None, [2, 1], False),
            (0, [2, 1], False),
            (None, [4], True),
            (None, [1, 3, 2], False),
        ],
    )
    def test_chunk_order_is_checked_against_its_predecessor(
        self, previous, chunk, expected
    ) -> None:
        assert bool(worldcover._chunk_follows(previous, np.array(chunk))) is expected


class TestClassAccumulation:
    def test_per_class_areas_are_scaled_and_added_to_existing_totals(self) -> None:
        totals = worldcover.defaultdict(float, {10: 1.0})
        values = np.array([10, 10, 50, np.nan, 60, 70, 80])
        coverage = np.array([0.5, 0.25, 1.0, 1.0, np.nan, 0.0, -1.0])
        worldcover._add_class_coverage(totals, values, coverage, 2.0)
        assert dict(totals) == {10: pytest.approx(2.5), 50: pytest.approx(2.0)}

    def test_infinite_values_or_coverage_never_become_a_class_share(self) -> None:
        totals = worldcover.defaultdict(float)
        values = np.array([np.inf, 10.0, 20.0])
        coverage = np.array([1.0, np.inf, 0.5])
        worldcover._add_class_coverage(totals, values, coverage, 1.0)
        assert dict(totals) == {20: 0.5}

    def test_a_zero_scale_adds_nothing(self) -> None:
        totals = worldcover.defaultdict(float)
        worldcover._add_class_coverage(totals, np.array([10.0]), np.array([1.0]), 0.0)
        assert dict(totals) == {}

    def test_stable_cells_skip_excluded_invalid_and_empty_cells_across_slices(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(worldcover, "_ACCUMULATION_CELL_CHUNK", 3)
        classes = np.array([10, 10, 50, 50, np.nan, 50, 10, 10], dtype=float)
        fractions = np.array([0.5, 0.0, 0.25, 1.0, 1.0, np.nan, 0.5, 0.25])
        excluded = np.array([False, False, False, True, False, False, False, True])
        totals = worldcover.defaultdict(float)
        worldcover._accumulate_stable_cells(totals, classes, fractions, excluded, 2.0)
        assert dict(totals) == {10: pytest.approx(2.0), 50: pytest.approx(0.5)}

    def test_candidate_classes_follow_their_matched_positions(self) -> None:
        got = worldcover._matched_candidate_classes(
            np.arange(4),
            np.array([7.0, 8.0]),
            np.array([True, False, True, False]),
            np.array([1, 0, 0, 0]),
        )
        assert got[[0, 2]].tolist() == [8.0, 7.0]
        assert np.isnan(got[[1, 3]]).all()

    def test_missing_classes_are_read_from_the_raster_and_masked_pixels_stay_empty(
        self, tmp_path
    ) -> None:
        values = np.arange(1, 13, dtype="uint8").reshape(3, 4)
        values[1, 1] = 0
        path = write_raster(tmp_path / "ids.tif", values, origin=(0, 3))
        classes = np.full(4, np.nan)
        with rasterio.open(path) as dataset:
            worldcover._read_missing_classes(
                dataset,
                np.array([1, 2, 1]),
                np.array([2, 3, 1]),
                classes,
                np.array([3, 0, 1]),
            )
        assert classes[[0, 3]].tolist() == [12.0, 7.0]
        assert np.isnan(classes[[1, 2]]).all()

    @pytest.mark.parametrize("area", [0.0, -1.0, float("nan"), float("inf")])
    def test_a_polygon_without_a_usable_area_contributes_nothing(self, area) -> None:
        totals = worldcover.defaultdict(float)
        worldcover._accumulate_corrected(
            totals, [], [], [], shapely.box(0, 0, 1, 1), None, 1.0, area
        )
        assert dict(totals) == {}


def _halo_cells(geometry, width, height) -> set[int]:
    """Independent oracle: cells whose square meets the buffered boundary."""
    halo = shapely.buffer(geometry.boundary, worldcover._BOUNDARY_HALO_PIXELS)
    return {
        row * width + col
        for row in range(height)
        for col in range(width)
        if halo.intersects(shapely.box(col, row, col + 1, row + 1))
    }


class TestBoundaryCells:
    @pytest.mark.parametrize("chunk", [1, 2, 3, 256])
    @pytest.mark.parametrize(
        ("geometry", "width", "height"),
        [
            (shapely.box(1.5, 1.5, 3.5, 3.5), 7, 5),
            (shapely.Polygon([(0.3, 0.2), (9.4, 1.1), (4.2, 8.7)]), 11, 10),
            (shapely.box(-2.5, -2.5, 4.5, 3.5), 5, 4),
            (shapely.box(3.2, 1.2, 12.5, 9.5), 6, 7),
        ],
    )
    def test_rasterized_ids_match_the_cells_the_halo_touches(
        self, monkeypatch, chunk, geometry, width, height
    ) -> None:
        monkeypatch.setattr(worldcover, "_BOUNDARY_ROW_CHUNK", chunk)
        boundary = shapely.buffer(geometry.boundary, worldcover._BOUNDARY_HALO_PIXELS)
        min_x, min_y, max_x, max_y = geometry.bounds
        ids = worldcover._rasterized_boundary_ids(
            boundary,
            max(0, int(np.floor(min_x)) - 1),
            min(width, int(np.ceil(max_x)) + 1),
            max(0, int(np.floor(min_y)) - 1),
            min(height, int(np.ceil(max_y)) + 1),
            width,
            height,
        )
        assert ids.dtype == np.int64
        assert ids.tolist() == sorted(_halo_cells(geometry, width, height))

    def test_no_strip_hits_means_no_ids(self) -> None:
        boundary = shapely.buffer(shapely.box(20, 20, 21, 21).boundary, 0.01)
        ids = worldcover._rasterized_boundary_ids(boundary, 0, 5, 0, 5, 5, 5)
        assert ids.tolist() == []
        assert ids.dtype == np.int64

    def test_chunks_carry_exact_geos_areas_for_each_cell(self, monkeypatch) -> None:
        monkeypatch.setattr(worldcover, "_BOUNDARY_CELLS_PER_CHUNK", 3)
        chunks = list(worldcover._boundary_cell_chunks(shapely.box(1.5, 1.5, 3.5, 3.5), 7, 5))
        assert [len(c[0]) for c in chunks] == [3, 3, 2]
        by_id = {}
        for ids, rows, columns, corrected in chunks:
            assert (rows * 7 + columns).tolist() == ids.tolist()
            by_id.update(zip(ids.tolist(), corrected.tolist(), strict=True))
        corner, edge = 0.25, 0.5
        assert by_id == {
            1 * 7 + 1: corner,
            1 * 7 + 2: edge,
            1 * 7 + 3: corner,
            2 * 7 + 1: edge,
            2 * 7 + 3: edge,
            3 * 7 + 1: corner,
            3 * 7 + 2: edge,
            3 * 7 + 3: corner,
        }

    @pytest.mark.parametrize(
        "geometry",
        [
            shapely.Polygon(),
            shapely.box(6.2, 0.5, 7.0, 1.5),
            shapely.box(0.5, 4.2, 1.5, 5.0),
            shapely.box(-9, 0.5, -8, 1.5),
        ],
    )
    def test_geometry_outside_the_raster_yields_no_chunks(self, geometry) -> None:
        assert list(worldcover._boundary_cell_chunks(geometry, 5, 4)) == []

    def test_geometry_touching_the_last_row_and_column_is_kept(self) -> None:
        geometry = shapely.box(3.5, 2.5, 4.5, 3.5)
        chunks = list(worldcover._boundary_cell_chunks(geometry, 5, 4))
        ids = np.concatenate([c[0] for c in chunks])
        assert ids.tolist() == sorted(_halo_cells(geometry, 5, 4))


class FakeDownload:
    """Stand in for ``urlretrieve``: write bytes, or fail with an HTTP status."""

    def __init__(self, status: int | None = None) -> None:
        self.status = status
        self.calls: list[tuple[str, Path]] = []

    def __call__(self, url: str, target: Path) -> None:
        self.calls.append((url, Path(target)))
        Path(target).write_bytes(b"tile")
        if self.status is not None:
            raise urllib.error.HTTPError(url, self.status, "boom", None, None)  # type: ignore[arg-type]


class TestTileCache:
    def test_defaults_describe_the_published_product(self, tmp_path) -> None:
        tiles = WorldCoverTiles(str(tmp_path), base_url="https://example.test/rootX//")
        name = "ESA_WorldCover_10m_2021_v200_N48E006_Map.tif"
        assert (
            tiles.cache_dir,
            tiles.version,
            tiles.year,
            tiles.max_cached_tiles,
            tiles.url_for(Tile(48, 6)),
            tiles.path_for(Tile(48, 6)),
        ) == (
            tmp_path,
            "v200",
            2021,
            8,
            f"https://example.test/rootX/v200/2021/map/{name}",
            tmp_path / name,
        )

    def test_download_lands_atomically_in_the_cache(self, tmp_path, monkeypatch) -> None:
        fake = FakeDownload()
        monkeypatch.setattr(worldcover.urllib.request, "urlretrieve", fake)
        tiles = WorldCoverTiles(tmp_path / "nested" / "cache")
        tile = Tile(48, 6)

        path = tiles.ensure(tile)

        assert (path, path.read_bytes(), tiles._in_use) == (
            tiles.path_for(tile),
            b"tile",
            {tile.name},
        )
        assert fake.calls == [(tiles.url_for(tile), path.with_name(path.name + ".part"))]
        assert [p.name for p in path.parent.iterdir()] == [path.name]

    @pytest.mark.parametrize("cached", [b"cached", b"1"])
    def test_a_cached_tile_is_not_downloaded_again(self, tmp_path, monkeypatch, cached) -> None:
        fake = FakeDownload()
        monkeypatch.setattr(worldcover.urllib.request, "urlretrieve", fake)
        tiles = WorldCoverTiles(tmp_path)
        tiles.path_for(Tile(48, 6)).write_bytes(cached)
        assert (tiles.ensure(Tile(48, 6)).read_bytes(), fake.calls) == (cached, [])

    def test_an_empty_cached_tile_is_downloaded_again(self, tmp_path, monkeypatch) -> None:
        fake = FakeDownload()
        monkeypatch.setattr(worldcover.urllib.request, "urlretrieve", fake)
        tiles = WorldCoverTiles(tmp_path)
        tiles.path_for(Tile(48, 6)).write_bytes(b"")
        assert tiles.ensure(Tile(48, 6)).read_bytes() == b"tile"
        assert len(fake.calls) == 1

    def test_an_absent_tile_is_reported_as_unpublished_and_leaves_no_partial_file(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(worldcover.urllib.request, "urlretrieve", FakeDownload(status=404))
        tiles = WorldCoverTiles(tmp_path)
        with pytest.raises(TileNotPublishedError, match=r"^N48E006 is not published$") as caught:
            tiles.ensure(Tile(48, 6))
        assert isinstance(caught.value.__cause__, urllib.error.HTTPError)
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize("status", [403, 500])
    def test_other_http_errors_propagate_and_clean_up(self, tmp_path, monkeypatch, status) -> None:
        monkeypatch.setattr(worldcover.urllib.request, "urlretrieve", FakeDownload(status=status))
        with pytest.raises(urllib.error.HTTPError) as caught:
            WorldCoverTiles(tmp_path).ensure(Tile(48, 6))
        assert caught.value.code == status
        assert not isinstance(caught.value, TileNotPublishedError)
        assert list(tmp_path.iterdir()) == []

    def test_an_ensured_tile_survives_the_release_of_others(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(worldcover.urllib.request, "urlretrieve", FakeDownload())
        tiles = WorldCoverTiles(tmp_path, max_cached_tiles=1)
        first, second = Tile(1, 1), Tile(2, 2)
        first_path = tiles.ensure(first)
        tiles.discard(first)
        assert first_path.exists()
        tiles.ensure(first)
        tiles.discard(second)
        assert second.name in tiles._released
        assert first.name not in tiles._released
        assert first_path.exists()

    def test_least_recently_released_tiles_are_evicted_first(self, tmp_path) -> None:
        tiles = WorldCoverTiles(tmp_path, max_cached_tiles=2)
        a, b, c = Tile(1, 1), Tile(2, 2), Tile(3, 3)
        for tile in (a, b, c):
            tiles.path_for(tile).write_bytes(b"x")
            tiles._in_use.add(tile.name)
        for released in (a, b, a, c):  # re-releasing a refreshes it: b is now the oldest
            tiles.discard(released)
        on_disk = [tiles.path_for(tile).exists() for tile in (a, b, c)]
        assert (on_disk, list(tiles._released), tiles._in_use) == (
            [True, False, True],
            [a.name, c.name],
            set(),
        )

    def test_evicting_a_tile_already_gone_from_disk_is_harmless(self, tmp_path) -> None:
        tiles = WorldCoverTiles(tmp_path, max_cached_tiles=0)
        tiles.discard(Tile(1, 1))
        assert tiles._released == {}

    def test_the_cache_holds_exactly_its_capacity(self, tmp_path) -> None:
        tiles = WorldCoverTiles(tmp_path, max_cached_tiles=1)
        a, b = Tile(1, 1), Tile(2, 2)
        for tile in (a, b):
            tiles.path_for(tile).write_bytes(b"x")
        tiles.discard(a)
        assert tiles.path_for(a).exists()
        tiles.discard(b)
        assert not tiles.path_for(a).exists()
        assert tiles.path_for(b).exists()


class ReadCounter:
    """Wrap a raster dataset and count pixel reads, delegating all else."""

    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.reads = 0

    def read(self, *args, **kwargs):
        self.reads += 1
        return self.dataset.read(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.dataset, name)


def _boundary_ids(geometry, width, height) -> np.ndarray:
    chunks = list(worldcover._boundary_cell_chunks(geometry, width, height))
    return np.concatenate([c[0] for c in chunks]) if chunks else np.array([], dtype=np.int64)


class TestBoundaryCorrection:
    @pytest.fixture
    def uniform(self, tmp_path):
        path = write_raster(tmp_path / "ten.tif", np.full((6, 6), 10, dtype="uint8"), (0, 6))
        with rasterio.open(path) as dataset:
            yield ReadCounter(dataset)

    def test_boundary_cells_returned_by_exactextract_are_not_reread(self, uniform) -> None:
        geometry = shapely.box(1.5, 1.5, 4.5, 4.5)
        totals = worldcover.defaultdict(float)
        excluded = worldcover._accumulate_boundary_cells(
            totals, np.arange(36), np.full(36, 10.0), geometry, uniform, 1.0
        )
        assert uniform.reads == 0
        assert excluded.sum() == 12
        assert dict(totals) == {10: pytest.approx(5.0)}

    def test_only_missing_cells_that_hold_polygon_area_are_read(self, uniform) -> None:
        # Cells outside the polygon touch its halo but contribute nothing to read.
        geometry = shapely.box(1, 1, 4, 4)
        totals = worldcover.defaultdict(float)
        excluded = worldcover._accumulate_boundary_cells(
            totals, np.array([], dtype=np.int64), np.array([]), geometry, uniform, 0.5
        )
        assert uniform.reads == 1
        assert not excluded.any()
        assert dict(totals) == {10: pytest.approx(4.0)}

    def test_a_missing_cell_takes_its_class_from_the_raster(self, uniform) -> None:
        geometry = shapely.box(1.5, 1.5, 4.5, 4.5)
        ids = np.delete(np.arange(36), 1 * 6 + 1)  # drop the 0.25 corner cell
        totals = worldcover.defaultdict(float)
        worldcover._accumulate_boundary_cells(
            totals, ids, np.full(35, 99.0), geometry, uniform, 1.0
        )
        assert uniform.reads == 1
        assert dict(totals) == {99: pytest.approx(4.75), 10: pytest.approx(0.25)}

    def test_missing_cells_are_read_in_one_window_per_row_band(self, tmp_path) -> None:
        values = np.arange(200 * 4, dtype="uint8").reshape(200, 4) % 200 + 1
        path = write_raster(tmp_path / "tall.tif", values, origin=(0, 200))
        rows = np.array([0, 1, 2, 150, 151, 152])
        columns = np.array([0, 1, 2, 1, 2, 3])
        with rasterio.open(path) as dataset:
            counted = ReadCounter(dataset)
            classes = np.full(len(rows), np.nan)
            worldcover._read_missing_classes(counted, rows, columns, classes, np.arange(len(rows)))
        assert counted.reads == 2
        assert classes.tolist() == values[rows, columns].astype(float).tolist()

    def test_stable_coverage_keeps_the_precision_of_float32_results(self, tmp_path) -> None:
        path = write_raster(tmp_path / "big.tif", np.full((20, 20), 10, dtype="uint8"), (0, 20))
        interior = np.array([r * 20 + c for r in range(1, 19) for c in range(1, 19)])
        fraction = np.float32(0.1)
        totals = worldcover.defaultdict(float)
        with rasterio.open(path) as dataset:
            worldcover._accumulate_corrected(
                totals,
                interior,
                np.full(len(interior), fraction, dtype=np.float32),
                np.full(len(interior), 10, dtype="uint8"),
                shapely.box(0, 0, 20, 20),
                dataset,
                1.0,
                400.0,
            )
        assert totals[10] == pytest.approx((324 * float(fraction) + 76) / 400.0, rel=1e-12)

    def test_cells_missing_from_one_result_do_not_pair_with_the_wrong_class(self) -> None:
        class AllMasked:
            def read(self, *_args, **_kwargs):
                return np.ma.masked_all((1, 1))

        classes = np.full(2, np.nan)
        for rows, columns, positions in [([0, 1], [0, 1], [0]), ([0], [0, 1], [0, 1])]:
            with pytest.raises(ValueError, match="same length"):
                worldcover._read_missing_classes(
                    AllMasked(), np.array(rows), np.array(columns), classes, np.array(positions)
                )

    def test_extraction_results_must_match_the_features(self, half_and_half, monkeypatch) -> None:
        empty = {"cell_id": [], "coverage": [], "values": []}
        monkeypatch.setattr(worldcover, "exact_extract", lambda *_a, **_k: empty)
        with pytest.raises(ValueError, match="zip"):
            class_coverage([half_and_half], one(shapely.box(0, 0, 2, 2)))

    @pytest.mark.parametrize(
        "geometry",
        [
            shapely.box(0.5, 0.5, 2.5, 2.5),
            shapely.box(3.5, 0.2, 5.5, 3.8),
            shapely.box(0.2, 4.2, 3.9, 7.5),
            shapely.Polygon([(0.3, 0.2), (7.4, 1.1), (4.2, 6.7)]),
            shapely.box(-2.5, -2.5, 4.5, 3.5),
            shapely.box(2.2, 2.2, 9.5, 9.5),
        ],
    )
    def test_boundary_chunks_cover_exactly_the_cells_the_halo_touches(self, geometry) -> None:
        ids = _boundary_ids(geometry, 8, 8)
        assert ids.tolist() == sorted(_halo_cells(geometry, 8, 8))

    def test_distant_parts_are_all_found_when_strips_in_between_are_empty(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(worldcover, "_BOUNDARY_ROW_CHUNK", 2)
        geometry = shapely.MultiPolygon(
            [shapely.box(1.5, 1.5, 3.5, 2.5), shapely.box(1.5, 12.5, 3.5, 13.5)]
        )
        assert _boundary_ids(geometry, 6, 16).tolist() == sorted(_halo_cells(geometry, 6, 16))

    def test_rasterization_is_strip_wise_and_sized_by_the_strip_not_its_offset(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(worldcover, "_BOUNDARY_ROW_CHUNK", 4)
        shapes = []
        real = worldcover.rasterize

        def spy(*args, **kwargs):
            shapes.append(kwargs["out_shape"])
            return real(*args, **kwargs)

        monkeypatch.setattr(worldcover, "rasterize", spy)
        geometry = shapely.box(60.5, 60.5, 63.5, 75.5)
        ids = _boundary_ids(geometry, 100, 100)
        assert ids.tolist() == sorted(_halo_cells(geometry, 100, 100))
        assert len(shapes) == 5
        assert all(height <= 8 and width <= 6 for height, width in shapes)

    def test_batches_follow_the_raster_envelope_budget(self, half_and_half, monkeypatch) -> None:
        frame = gpd.GeoDataFrame(
            {"polygon_id": ["left", "right", "across"]},
            geometry=[
                shapely.box(0.5, 0.5, 1.5, 1.5),
                shapely.box(2.5, 0.5, 3.5, 1.5),
                shapely.box(1.5, 2.5, 2.5, 3.5),
            ],
            crs="EPSG:4326",
        )
        recorder = ExactExtractRecorder()
        monkeypatch.setattr(worldcover, "exact_extract", recorder)
        monkeypatch.setattr(worldcover, "_MAX_BATCH_PIXEL_BBOX_CELLS", 12)

        coverage = class_coverage([half_and_half], frame)

        assert [len(call[1]) for call in recorder.calls] == [1, 1, 1]
        assert coverage == [
            {10: pytest.approx(1.0)},
            {50: pytest.approx(1.0)},
            {10: pytest.approx(0.5), 50: pytest.approx(0.5)},
        ]

    def test_classes_are_reported_in_ascending_order(self, tmp_path) -> None:
        high = write_raster(tmp_path / "high.tif", np.full((4, 2), 50, dtype="uint8"), (2, 4))
        low = write_raster(tmp_path / "low.tif", np.full((4, 2), 10, dtype="uint8"), (0, 4))
        across = one(Polygon([(1, 0), (1, 4), (3, 4), (3, 0)]))
        assert list(class_coverage([high, low], across)[0]) == [10, 50]
