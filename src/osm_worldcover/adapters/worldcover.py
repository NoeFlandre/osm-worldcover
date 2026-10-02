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
"""

import math
import urllib.error
import urllib.request
from collections import OrderedDict, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Final, Protocol

import geopandas as gpd
import numpy as np
import rasterio
import shapely
from exactextract import exact_extract
from rasterio.features import rasterize
from shapely.affinity import affine_transform
from shapely.geometry import box

from osm_worldcover.domain.tiling import Tile

__all__ = [
    "DEFAULT_BASE_URL",
    "TileNotPublishedError",
    "TileSource",
    "WorldCoverTiles",
    "class_coverage",
]

DEFAULT_BASE_URL: Final[str] = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"

#: Released tiles kept on disk against a neighbouring group wanting them again.
#: Eight tiles is roughly 750 MB, a good trade against a 94 MB re-download.
DEFAULT_CACHED_TILES: Final[int] = 8

#: exactextract operations needed to correct pixel-boundary coverage.
_OPS: Final[Sequence[str]] = ("cell_id", "coverage", "values")

#: A small pixel-space halo makes all-touched rasterization retain cells that
#: meet a polygon boundary at a single grid corner. GEOS measures the original
#: polygon against those cells; the halo only selects candidates for checking.
_BOUNDARY_HALO_PIXELS: Final[float] = 0.01
_BOUNDARY_ROW_CHUNK: Final[int] = 256

# exactextract returns one cell-level result array per feature. Bound each
# batch by both feature count and polygon area so one busy tile cannot make a
# whole region's result arrays resident at once. A 2,000 km² batch covers at
# most about 20 million 10 m cells; larger polygons are processed alone.
_MAX_FEATURES_PER_BATCH: Final[int] = 128
_MAX_BATCH_AREA_M2: Final[float] = 2_000_000_000.0
_MAX_CELLS_IN_MEMORY: Final[int] = 2_000_000
_MAX_BATCH_PIXEL_BBOX_CELLS: Final[int] = 2_000_000
_MAX_GEOMETRY_COORDINATES: Final[int] = 25_000
_MAX_GEOMETRY_SPLIT_DEPTH: Final[int] = 32
_BOUNDARY_CELLS_PER_CHUNK: Final[int] = 16_384
_ACCUMULATION_CELL_CHUNK: Final[int] = 1_000_000


class TileNotPublishedError(FileNotFoundError):
    """Raised for a tile the product does not publish (open ocean, mostly)."""


class TileSource(Protocol):
    """What the pipeline needs of a tile store: get one, then let it go.

    Stated as a protocol so the pipeline depends on the capability rather than
    on :class:`WorldCoverTiles` itself, and a stand-in can satisfy it honestly
    instead of merely happening to have the right methods.
    """

    def ensure(self, tile: Tile) -> Path:
        """Return a local path to ``tile``, fetching it if absent."""
        ...

    def discard(self, tile: Tile) -> None:
        """Release ``tile``; the store decides when to delete it."""
        ...


class WorldCoverTiles:
    """Addresses, downloads and caches WorldCover tiles.

    Tiles are ~94 MB each and a global run touches thousands of them, so they
    cannot all be kept. They also cannot be deleted the moment one group of
    polygons is done, because neighbouring groups usually want the same tile
    and re-downloading 94 MB is pure waste.

    ``discard`` therefore *releases* a tile rather than deleting it, and the
    least recently used tiles are evicted once more than ``max_cached_tiles``
    are held. A released tile that is asked for again is simply reused.
    """

    def __init__(
        self,
        cache_dir: Path,
        version: str = "v200",
        year: int = 2021,
        base_url: str = DEFAULT_BASE_URL,
        max_cached_tiles: int = DEFAULT_CACHED_TILES,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.version = version
        self.year = year
        self.base_url = base_url.rstrip("/")
        self.max_cached_tiles = max_cached_tiles
        # Insertion-ordered: the oldest released tile is evicted first.
        self._released: OrderedDict[str, Tile] = OrderedDict()
        self._in_use: set[str] = set()

    def filename_for(self, tile: Tile) -> str:
        """The product's own filename for ``tile``."""
        return f"ESA_WorldCover_10m_{self.year}_{self.version}_{tile.name}_Map.tif"

    def url_for(self, tile: Tile) -> str:
        """The published URL for ``tile``."""
        return f"{self.base_url}/{self.version}/{self.year}/map/{self.filename_for(tile)}"

    def path_for(self, tile: Tile) -> Path:
        """Where ``tile`` is cached locally."""
        return self.cache_dir / self.filename_for(tile)

    def ensure(self, tile: Tile) -> Path:
        """Return a local path to ``tile``, downloading it if absent.

        Raises :class:`TileNotPublishedError` for tiles the product omits.
        """
        self._in_use.add(tile.name)
        self._released.pop(tile.name, None)
        path = self.path_for(tile)
        if path.exists() and path.stat().st_size > 0:
            return path
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Download beside the target and rename, so an interrupted run never
        # leaves a truncated tile that a later run would trust.
        partial = path.with_suffix(path.suffix + ".part")
        try:
            urllib.request.urlretrieve(self.url_for(tile), partial)
        except urllib.error.HTTPError as exc:
            partial.unlink(missing_ok=True)
            if exc.code == 404:
                raise TileNotPublishedError(f"{tile.name} is not published") from exc
            raise
        partial.rename(path)
        return path

    def discard(self, tile: Tile) -> None:
        """Release ``tile``, deleting it only once the cache is over capacity."""
        self._in_use.discard(tile.name)
        self._released[tile.name] = tile
        self._released.move_to_end(tile.name)
        self._evict()

    def _evict(self) -> None:
        """Delete released tiles, oldest first, until the cache fits."""
        while len(self._released) > self.max_cached_tiles:
            _, evicted = self._released.popitem(last=False)
            self.path_for(evicted).unlink(missing_ok=True)


def class_coverage(raster_paths: Iterable[Path], frame: gpd.GeoDataFrame) -> list[dict[int, float]]:
    """Return, per row of ``frame``, each class's share of that row's area.

    Shares are relative to the polygon's own area, so they sum to at most one
    and fall short wherever the polygon was not observed. Contributions from
    multiple non-overlapping rasters are summed.
    """
    if len(frame) == 0:
        return []

    totals: list[defaultdict[int, float]] = [defaultdict(float) for _ in range(len(frame))]
    # Areas are measured in the raster's own planar units, for both the cells
    # and the polygons. Only their ratio is used, so it is exact even though
    # the units are degrees and a degree is not a constant ground distance.
    areas = shapely.area(frame.geometry.to_numpy())

    for path in raster_paths:
        _add_raster(totals, path, frame, areas)

    return [dict(sorted(total.items())) for total in totals]


def _add_raster(
    totals: list[defaultdict[int, float]],
    path: Path,
    frame: gpd.GeoDataFrame,
    areas: np.ndarray,
) -> None:
    """Add one raster's contribution to every polygon's running totals."""
    with rasterio.open(path) as dataset:
        transform = dataset.transform
        cell_area = abs(transform.a * transform.e - transform.b * transform.d)
        for start, stop in _feature_batches(frame, transform, dataset.width, dataset.height):
            ordinary, spatial = _partition_feature_batch(
                frame, start, stop, transform, dataset.width, dataset.height
            )
            _add_ordinary_features(totals, path, frame, ordinary, dataset, cell_area, areas)
            for index, pixel_geometry in spatial:
                _add_spatial_feature(
                    totals,
                    path,
                    frame.crs,
                    index,
                    pixel_geometry,
                    transform,
                    dataset,
                    cell_area,
                    areas[index],
                )


def _partition_feature_batch(
    frame: gpd.GeoDataFrame,
    start: int,
    stop: int,
    transform: rasterio.Affine,
    width: int,
    height: int,
) -> tuple[list[int], list[tuple[int, shapely.Geometry]]]:
    """Separate ordinary features from geometries requiring spatial chunks."""
    ordinary = []
    spatial = []
    for index in range(start, stop):
        pixel_geometry = _pixel_geometry(frame.geometry.iloc[index], transform)
        if _needs_spatial_chunks(pixel_geometry, width, height):
            spatial.append((index, pixel_geometry))
        else:
            ordinary.append(index)
    return ordinary, spatial


def _add_ordinary_features(
    totals: list[defaultdict[int, float]],
    path: Path,
    frame: gpd.GeoDataFrame,
    indices: list[int],
    dataset: rasterio.DatasetReader,
    cell_area: float,
    areas: np.ndarray,
) -> None:
    """Extract one bounded feature batch and accumulate its class coverage."""
    if indices:
        _extract_and_accumulate(
            totals,
            path,
            frame.iloc[indices],
            indices,
            list(frame.geometry.iloc[indices]),
            dataset,
            cell_area,
            [float(areas[index]) for index in indices],
        )


def _add_spatial_feature(
    totals: list[defaultdict[int, float]],
    path: Path,
    crs,
    index: int,
    pixel_geometry: shapely.Geometry,
    transform: rasterio.Affine,
    dataset: rasterio.DatasetReader,
    cell_area: float,
    polygon_area: float,
) -> None:
    """Extract bounded spatial pieces while retaining the original area denominator."""
    for pixel_chunk in _bounded_pixel_chunks(pixel_geometry, dataset.width, dataset.height):
        geometry_chunk = _map_geometry(pixel_chunk, transform)
        chunk_frame = gpd.GeoDataFrame(geometry=[geometry_chunk], crs=crs)
        _extract_and_accumulate(
            totals,
            path,
            chunk_frame,
            [index],
            [geometry_chunk],
            dataset,
            cell_area,
            [polygon_area],
        )


def _extract_and_accumulate(
    totals: list[defaultdict[int, float]],
    path: Path,
    frame: gpd.GeoDataFrame,
    indices: list[int],
    geometries: list[shapely.Geometry],
    dataset: rasterio.DatasetReader,
    cell_area: float,
    polygon_areas: Sequence[float],
) -> None:
    """Keep exactextract output batches small and accumulate each feature in order."""
    result = exact_extract(
        str(path),
        frame,
        list(_OPS),
        max_cells_in_memory=_MAX_CELLS_IN_MEMORY,
        output="pandas",
    )
    for index, geometry, polygon_area, cell_ids, coverage, values in zip(
        indices,
        geometries,
        polygon_areas,
        result["cell_id"],
        result["coverage"],
        result["values"],
        strict=True,
    ):
        _accumulate_corrected(
            totals[index],
            cell_ids,
            coverage,
            values,
            geometry,
            dataset,
            cell_area,
            polygon_area,
        )


def _feature_batches(
    frame: gpd.GeoDataFrame,
    transform: rasterio.Affine | None = None,
    width: int | None = None,
    height: int | None = None,
) -> Iterator[tuple[int, int]]:
    """Yield feature ranges bounded by count, source area, and raster envelope."""
    areas = _feature_areas(frame)
    pixel_cells = _feature_pixel_cells(frame, transform, width, height)
    yield from _yield_feature_batches(areas, pixel_cells)


def _feature_pixel_cells(
    frame: gpd.GeoDataFrame,
    transform: rasterio.Affine | None,
    width: int | None,
    height: int | None,
) -> np.ndarray:
    """Return each feature's raster-envelope cell bound, or zero without a raster."""
    if transform is None or width is None or height is None:
        return np.zeros(len(frame), dtype=np.int64)
    return np.fromiter(
        (
            _pixel_bbox_cells(_pixel_geometry(geometry, transform), width, height)
            for geometry in frame.geometry
        ),
        dtype=np.int64,
        count=len(frame),
    )


def _yield_feature_batches(areas: np.ndarray, pixel_cells: np.ndarray) -> Iterator[tuple[int, int]]:
    """Yield consecutive ranges within both per-batch budgets."""
    start = 0
    batch_area = 0.0
    batch_pixel_cells = 0
    for index, (value, cells) in enumerate(zip(areas, pixel_cells, strict=True)):
        area = float(value)
        feature_count = index - start
        if feature_count and _batch_exceeds_limits(
            feature_count, batch_area, area, batch_pixel_cells, int(cells)
        ):
            yield start, index
            start = index
            batch_area = 0.0
            batch_pixel_cells = 0
        batch_area += area
        batch_pixel_cells += int(cells)
    if start < len(areas):
        yield start, len(areas)


def _pixel_bbox_cells(geometry: shapely.Geometry, width: int, height: int) -> int:
    """Bound all candidate cells in a geometry's raster-space envelope."""
    if geometry.is_empty:
        return 0
    min_x, min_y, max_x, max_y = geometry.bounds
    left = max(0, math.floor(min_x) - 1)
    right = min(width, math.ceil(max_x) + 1)
    top = max(0, math.floor(min_y) - 1)
    bottom = min(height, math.ceil(max_y) + 1)
    return max(0, right - left) * max(0, bottom - top)


def _needs_spatial_chunks(geometry: shapely.Geometry, width: int, height: int) -> bool:
    """Whether one feature can exceed the per-result cell or coordinate budget."""
    return (
        _pixel_bbox_cells(geometry, width, height) > _MAX_BATCH_PIXEL_BBOX_CELLS
        or shapely.get_num_coordinates(geometry) > _MAX_GEOMETRY_COORDINATES
    )


def _bounded_pixel_chunks(
    geometry: shapely.Geometry, width: int, height: int
) -> Iterator[shapely.Geometry]:
    """Yield raster-clipped pieces with bounded raster envelopes and coordinates."""
    clipped = shapely.intersection(geometry, box(0, 0, width, height))
    if not clipped.is_empty:
        yield from _split_pixel_geometry(clipped, width, height, depth=0)


def _split_pixel_geometry(
    geometry: shapely.Geometry, width: int, height: int, depth: int
) -> Iterator[shapely.Geometry]:
    """Recursively split a large feature on pixel-aligned lines."""
    if _geometry_is_bounded(geometry, width, height):
        yield geometry
        return
    if depth >= _MAX_GEOMETRY_SPLIT_DEPTH:
        raise ValueError(
            "could not bound a WorldCover geometry after "
            f"{_MAX_GEOMETRY_SPLIT_DEPTH} spatial splits"
        )
    first, second = _split_pixel_geometry_once(geometry)
    if _split_is_stalled(geometry, first, second):
        raise ValueError("WorldCover geometry split did not reduce its spatial extent")
    yield from _nonempty_pixel_children(first, second, width, height, depth + 1)


def _geometry_is_bounded(geometry: shapely.Geometry, width: int, height: int) -> bool:
    """Return whether one geometry fits both spatial memory budgets."""
    pixel_cells = _pixel_bbox_cells(geometry, width, height)
    coordinates = shapely.get_num_coordinates(geometry)
    return pixel_cells <= _MAX_BATCH_PIXEL_BBOX_CELLS and coordinates <= _MAX_GEOMETRY_COORDINATES


def _split_pixel_geometry_once(
    geometry: shapely.Geometry,
) -> tuple[shapely.Geometry, shapely.Geometry]:
    """Bisect a geometry along its longest pixel-space axis."""
    min_x, min_y, max_x, max_y = geometry.bounds
    x_midpoint = _pixel_aligned_midpoint(min_x, max_x)
    y_midpoint = _pixel_aligned_midpoint(min_y, max_y)
    split_x = _choose_split_axis(max_x - min_x, max_y - min_y, x_midpoint, y_midpoint)
    if split_x:
        middle = x_midpoint if x_midpoint is not None else (min_x + max_x) / 2.0
        first_box = box(min_x - 1.0, min_y - 1.0, middle, max_y + 1.0)
        second_box = box(middle, min_y - 1.0, max_x + 1.0, max_y + 1.0)
    else:
        middle = y_midpoint if y_midpoint is not None else (min_y + max_y) / 2.0
        first_box = box(min_x - 1.0, min_y - 1.0, max_x + 1.0, middle)
        second_box = box(min_x - 1.0, middle, max_x + 1.0, max_y + 1.0)
    first = shapely.intersection(geometry, first_box)
    second = shapely.intersection(geometry, second_box)
    return first, second


def _pixel_aligned_midpoint(minimum: float, maximum: float) -> float | None:
    """Return an interior integer pixel boundary near the midpoint, if one exists."""
    midpoint = (minimum + maximum) / 2.0
    for candidate in (math.floor(midpoint), math.ceil(midpoint)):
        if minimum < candidate < maximum:
            return float(candidate)
    return None


def _choose_split_axis(
    x_span: float,
    y_span: float,
    x_midpoint: float | None,
    y_midpoint: float | None,
) -> bool:
    """Prefer the longest axis with an integer pixel split; fall back if needed."""
    if x_midpoint is None and y_midpoint is None:
        return x_span >= y_span
    if x_midpoint is None:
        return False
    if y_midpoint is None:
        return True
    return x_span >= y_span


def _split_is_stalled(
    geometry: shapely.Geometry,
    first: shapely.Geometry,
    second: shapely.Geometry,
) -> bool:
    """Detect a split that fails to create any smaller geometry."""
    return (first.is_empty and second.is_empty) or (
        first.equals(geometry) and second.equals(geometry)
    )


def _nonempty_pixel_children(
    first: shapely.Geometry,
    second: shapely.Geometry,
    width: int,
    height: int,
    depth: int,
) -> Iterator[shapely.Geometry]:
    """Yield bounded chunks from the nonempty halves of one spatial split."""
    for piece in (first, second):
        if not piece.is_empty:
            yield from _split_pixel_geometry(piece, width, height, depth)


def _map_geometry(geometry: shapely.Geometry, transform: rasterio.Affine) -> shapely.Geometry:
    """Return a pixel-space geometry to the raster's map coordinates."""
    return affine_transform(
        geometry,
        [transform.a, transform.b, transform.d, transform.e, transform.c, transform.f],
    )


def _feature_areas(frame: gpd.GeoDataFrame) -> np.ndarray:
    """Return source areas, or zeros when only the feature-count limit applies."""
    if "area_m2" in frame:
        return frame["area_m2"].to_numpy(dtype=float, copy=False)
    return np.zeros(len(frame), dtype=float)


def _batch_exceeds_limits(
    feature_count: int,
    batch_area: float,
    next_area: float,
    batch_pixel_cells: int = 0,
    next_pixel_cells: int = 0,
) -> bool:
    """Whether adding another feature would exceed either extraction bound."""
    return (
        feature_count >= _MAX_FEATURES_PER_BATCH
        or batch_area + next_area > _MAX_BATCH_AREA_M2
        or batch_pixel_cells + next_pixel_cells > _MAX_BATCH_PIXEL_BBOX_CELLS
    )


def _accumulate_corrected(
    into: defaultdict[int, float],
    cell_ids: Iterable[int],
    coverage: Iterable[float],
    values: Iterable[float],
    geometry: shapely.Geometry,
    dataset: rasterio.DatasetReader,
    cell_area: float,
    polygon_area: float,
) -> None:
    """Replace exactextract's boundary-cell values with independent GEOS areas."""
    if polygon_area <= 0.0 or not math.isfinite(polygon_area):
        return
    ids = np.asarray(cell_ids, dtype=np.int64)
    fractions = np.asarray(coverage, dtype=float)
    # exactextract omits NoData cells by default; this call never sets default_value.
    classes = np.asarray(values)
    pixel_geometry = _pixel_geometry(geometry, dataset.transform)
    scale = cell_area / polygon_area
    excluded = _accumulate_boundary_cells(into, ids, classes, pixel_geometry, dataset, scale)
    _accumulate_stable_cells(into, classes, fractions, excluded, scale)


def _accumulate_boundary_cells(
    into: defaultdict[int, float],
    ids: np.ndarray,
    classes: np.ndarray,
    pixel_geometry: shapely.Geometry,
    dataset: rasterio.DatasetReader,
    scale: float,
) -> np.ndarray:
    """Replace candidate boundary pixels and return ids excluded from stable sums."""
    excluded = np.zeros(len(ids), dtype=bool)
    for candidate_ids, rows, columns, corrected in _boundary_cell_chunks(
        pixel_geometry, dataset.width, dataset.height
    ):
        present, positions = _find_candidates(ids, candidate_ids)
        excluded[positions[present]] = True
        candidate_classes = _matched_candidate_classes(candidate_ids, classes, present, positions)
        missing = ~present & (corrected > 0.0)
        _read_missing_classes(
            dataset,
            rows[missing],
            columns[missing],
            candidate_classes,
            np.flatnonzero(missing),
        )
        _add_class_coverage(into, candidate_classes, corrected, scale)
    return excluded


def _matched_candidate_classes(
    candidate_ids: np.ndarray,
    classes: np.ndarray,
    present: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    """Map exactextract values onto candidate cells, leaving unmatched cells empty."""
    candidate_classes = np.full(len(candidate_ids), np.nan)
    matched_candidates = np.flatnonzero(present)
    matched_positions = positions[matched_candidates]
    candidate_classes[matched_candidates] = classes[matched_positions]
    return candidate_classes


def _pixel_geometry(geometry: shapely.Geometry, transform: rasterio.Affine) -> shapely.Geometry:
    """Express a raster-CRS geometry in pixel coordinates for stable overlay."""
    inverse = ~transform
    return affine_transform(
        geometry,
        [inverse.a, inverse.b, inverse.d, inverse.e, inverse.c, inverse.f],
    )


def _boundary_cell_chunks(
    geometry: shapely.Geometry, width: int, height: int
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield boundary-cell coverage in bounded candidate arrays."""
    if geometry.is_empty:
        return
    min_x, min_y, max_x, max_y = geometry.bounds
    col_start = max(0, math.floor(min_x) - 1)
    col_stop = min(width, math.ceil(max_x) + 1)
    row_start = max(0, math.floor(min_y) - 1)
    row_stop = min(height, math.ceil(max_y) + 1)
    if col_start >= col_stop or row_start >= row_stop:
        return
    boundary = shapely.buffer(geometry.boundary, _BOUNDARY_HALO_PIXELS)
    candidate_ids = _rasterized_boundary_ids(
        boundary, col_start, col_stop, row_start, row_stop, width, height
    )
    for start in range(0, len(candidate_ids), _BOUNDARY_CELLS_PER_CHUNK):
        cell_ids = candidate_ids[start : start + _BOUNDARY_CELLS_PER_CHUNK]
        rows, columns = cell_ids // width, cell_ids % width
        cells = shapely.box(columns, rows, columns + 1, rows + 1)
        corrected = np.asarray(shapely.area(shapely.intersection(geometry, cells)), dtype=float)
        yield cell_ids, rows, columns, corrected


def _rasterized_boundary_ids(
    boundary: shapely.Geometry,
    col_start: int,
    col_stop: int,
    row_start: int,
    row_stop: int,
    width: int,
    height: int,
) -> np.ndarray:
    """Rasterize boundary strips with bounded memory, then return unique IDs."""
    ids: list[np.ndarray] = []
    for row in range(row_start, row_stop, _BOUNDARY_ROW_CHUNK):
        strip_top = max(0, row - 1)
        strip_bottom = min(height, row + _BOUNDARY_ROW_CHUNK + 1)
        part = shapely.intersection(boundary, box(col_start, strip_top, col_stop, strip_bottom))
        if part.is_empty:
            continue
        min_x, min_y, max_x, max_y = part.bounds
        left = max(col_start, math.floor(min_x) - 1)
        right = min(col_stop, math.ceil(max_x) + 1)
        top = max(0, math.floor(min_y) - 1)
        bottom = min(height, math.ceil(max_y) + 1)
        mask = rasterize(
            [(part, 1)],
            out_shape=(bottom - top, right - left),
            transform=rasterio.Affine.translation(left, top),
            all_touched=True,
            dtype="uint8",
        )
        rows, columns = np.nonzero(mask)
        ids.append((rows + top) * width + columns + left)
    if not ids:
        return np.array([], dtype=np.int64)
    return np.unique(np.concatenate(ids).astype(np.int64, copy=False))


def _find_candidates(cell_ids: np.ndarray, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find result positions without reordering the potentially huge cell array."""
    if not len(cell_ids) or not len(candidates):
        return np.zeros(len(candidates), dtype=bool), np.zeros(len(candidates), dtype=np.int64)
    found = np.zeros(len(candidates), dtype=bool)
    positions = np.zeros(len(candidates), dtype=np.int64)
    if _is_sorted(cell_ids):
        positions = np.searchsorted(cell_ids, candidates)
        in_range = positions < len(cell_ids)
        found[in_range] = cell_ids[positions[in_range]] == candidates[in_range]
        return found, positions
    for start in range(0, len(cell_ids), _ACCUMULATION_CELL_CHUNK):
        stop = min(start + _ACCUMULATION_CELL_CHUNK, len(cell_ids))
        chunk = cell_ids[start:stop]
        offsets = np.searchsorted(candidates, chunk)
        in_range = offsets < len(candidates)
        matched = np.zeros(len(chunk), dtype=bool)
        matched[in_range] = candidates[offsets[in_range]] == chunk[in_range]
        found[offsets[matched]] = True
        positions[offsets[matched]] = start + np.flatnonzero(matched)
    return found, positions


def _is_sorted(values: np.ndarray) -> bool:
    """Check monotonic order in bounded slices, avoiding a full-size mask."""
    previous: int | None = None
    for start in range(0, len(values), _ACCUMULATION_CELL_CHUNK):
        chunk = values[start : start + _ACCUMULATION_CELL_CHUNK]
        if not _chunk_follows(previous, chunk):
            return False
        previous = int(chunk[-1])
    return True


def _chunk_follows(previous: int | None, chunk: np.ndarray) -> bool:
    """Check sorted order within a chunk and across its leading boundary."""
    starts_after_previous = previous is None or previous <= chunk[0]
    return starts_after_previous and not np.any(chunk[1:] < chunk[:-1])


def _accumulate_stable_cells(
    into: defaultdict[int, float],
    classes: np.ndarray,
    fractions: np.ndarray,
    excluded: np.ndarray,
    scale: float,
) -> None:
    """Accumulate non-boundary cells in bounded slices of exactextract output."""
    for start in range(0, len(classes), _ACCUMULATION_CELL_CHUNK):
        stop = min(start + _ACCUMULATION_CELL_CHUNK, len(classes))
        valid = ~excluded[start:stop]
        valid &= np.isfinite(classes[start:stop]) & np.isfinite(fractions[start:stop])
        valid &= fractions[start:stop] > 0.0
        _add_class_coverage(into, classes[start:stop][valid], fractions[start:stop][valid], scale)


def _read_missing_classes(
    dataset: rasterio.DatasetReader,
    rows: np.ndarray,
    columns: np.ndarray,
    classes: np.ndarray,
    positions: np.ndarray,
) -> None:
    """Read valid pixel classes for boundary cells omitted by exactextract."""
    for row, column, position in zip(rows, columns, positions, strict=True):
        value = dataset.read(
            1,
            window=((int(row), int(row) + 1), (int(column), int(column) + 1)),
            masked=True,
        )[0, 0]
        if not np.ma.is_masked(value):
            classes[position] = float(value)


def _add_class_coverage(
    into: defaultdict[int, float],
    values: np.ndarray,
    coverage: np.ndarray,
    scale: float,
) -> None:
    """Add valid per-cell areas to class totals."""
    valid = np.isfinite(values) & np.isfinite(coverage) & (coverage > 0.0)
    for value in np.unique(values[valid]):
        share = float(coverage[valid & (values == value)].sum()) * scale
        if share > 0.0:
            into[int(value)] += share
