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
        for start, stop in _feature_batches(frame):
            batch = frame.iloc[start:stop]
            result = exact_extract(
                str(path),
                batch,
                list(_OPS),
                max_cells_in_memory=_MAX_CELLS_IN_MEMORY,
                output="pandas",
            )
            for index, (cell_ids, coverage, values) in enumerate(
                zip(result["cell_id"], result["coverage"], result["values"], strict=True),
                start=start,
            ):
                _accumulate_corrected(
                    totals[index],
                    cell_ids,
                    coverage,
                    values,
                    frame.geometry.iloc[index],
                    dataset,
                    cell_area,
                    areas[index],
                )


def _feature_batches(frame: gpd.GeoDataFrame) -> Iterator[tuple[int, int]]:
    """Yield feature ranges with bounded count and total source area."""
    areas = _feature_areas(frame)
    start = 0
    batch_area = 0.0
    for index, value in enumerate(areas):
        area = float(value)
        feature_count = index - start
        if feature_count and _batch_exceeds_limits(feature_count, batch_area, area):
            yield start, index
            start = index
            batch_area = 0.0
        batch_area += area
    if start < len(frame):
        yield start, len(frame)


def _feature_areas(frame: gpd.GeoDataFrame) -> np.ndarray:
    """Return source areas, or zeros when only the feature-count limit applies."""
    if "area_m2" in frame:
        return frame["area_m2"].to_numpy(dtype=float, copy=False)
    return np.zeros(len(frame), dtype=float)


def _batch_exceeds_limits(feature_count: int, batch_area: float, next_area: float) -> bool:
    """Whether adding another feature would exceed either extraction bound."""
    return feature_count >= _MAX_FEATURES_PER_BATCH or batch_area + next_area > _MAX_BATCH_AREA_M2


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
    raw_values = np.ma.asarray(values)
    classes = np.asarray(raw_values.data)
    mask = None if raw_values.mask is np.ma.nomask else np.asarray(raw_values.mask, dtype=bool)
    pixel_geometry = _pixel_geometry(geometry, dataset.transform)
    candidate_ids, rows, columns, corrected = _boundary_cells(
        pixel_geometry, dataset.width, dataset.height
    )
    present, positions = _find_candidates(ids, candidate_ids)
    candidate_positions = np.sort(positions[present])
    scale = cell_area / polygon_area
    candidate_classes = np.full(len(candidate_ids), np.nan)
    matched_candidates = np.flatnonzero(present)
    matched_positions = positions[matched_candidates]
    if mask is None:
        candidate_classes[matched_candidates] = classes[matched_positions]
    else:
        valid_candidates = ~mask[matched_positions]
        candidate_classes[matched_candidates[valid_candidates]] = classes[
            matched_positions[valid_candidates]
        ]
    _accumulate_stable_cells(into, classes, mask, fractions, candidate_positions, scale)
    _read_missing_classes(
        dataset,
        rows[~present & (corrected > 0.0)],
        columns[~present & (corrected > 0.0)],
        candidate_classes,
        np.flatnonzero(~present & (corrected > 0.0)),
    )
    _add_class_coverage(into, candidate_classes, corrected, scale)


def _pixel_geometry(geometry: shapely.Geometry, transform: rasterio.Affine) -> shapely.Geometry:
    """Express a raster-CRS geometry in pixel coordinates for stable overlay."""
    inverse = ~transform
    return affine_transform(
        geometry,
        [inverse.a, inverse.b, inverse.d, inverse.e, inverse.c, inverse.f],
    )


def _boundary_cells(
    geometry: shapely.Geometry, width: int, height: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return raster cells near the polygon boundary and GEOS coverage for each."""
    if geometry.is_empty:
        empty = np.array([], dtype=np.int64)
        return empty, empty, empty, empty.astype(float)
    min_x, min_y, max_x, max_y = geometry.bounds
    col_start = max(0, math.floor(min_x) - 1)
    col_stop = min(width, math.ceil(max_x) + 1)
    row_start = max(0, math.floor(min_y) - 1)
    row_stop = min(height, math.ceil(max_y) + 1)
    if col_start >= col_stop or row_start >= row_stop:
        empty = np.array([], dtype=np.int64)
        return empty, empty, empty, empty.astype(float)
    boundary = shapely.buffer(geometry.boundary, _BOUNDARY_HALO_PIXELS)
    cell_ids = _rasterized_boundary_ids(
        boundary, col_start, col_stop, row_start, row_stop, width, height
    )
    rows, columns = cell_ids // width, cell_ids % width
    cells = shapely.box(columns, rows, columns + 1, rows + 1)
    corrected = np.asarray(shapely.area(shapely.intersection(geometry, cells)), dtype=float)
    return cell_ids, rows, columns, corrected


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
    mask: np.ndarray | None,
    fractions: np.ndarray,
    excluded: np.ndarray,
    scale: float,
) -> None:
    """Accumulate non-boundary cells in bounded slices of exactextract output."""
    for start in range(0, len(classes), _ACCUMULATION_CELL_CHUNK):
        stop = min(start + _ACCUMULATION_CELL_CHUNK, len(classes))
        stable = np.ones(stop - start, dtype=bool)
        left = np.searchsorted(excluded, start, side="left")
        right = np.searchsorted(excluded, stop, side="left")
        stable[excluded[left:right] - start] = False
        valid = stable & np.isfinite(classes[start:stop]) & np.isfinite(fractions[start:stop])
        valid &= fractions[start:stop] > 0.0
        if mask is not None:
            valid &= ~mask[start:stop]
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
