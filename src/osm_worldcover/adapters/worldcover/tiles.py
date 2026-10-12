import shutil
import urllib.error
import urllib.request
from collections import OrderedDict
from pathlib import Path
from typing import Final, Protocol

from osm_worldcover.domain.tiling import Tile

DEFAULT_BASE_URL: Final[str] = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"

#: Released tiles kept on disk against a neighbouring group wanting them again.
#: Eight tiles is roughly 750 MB, a good trade against a 94 MB re-download.
DEFAULT_CACHED_TILES: Final[int] = 8

#: Seconds a tile download may wait on one socket read before it is abandoned.
#: This bounds each stall, not the whole ~94 MB transfer.
DOWNLOAD_TIMEOUT_SECONDS: Final[float] = 60.0


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


def _download(url: str, target: Path) -> None:
    """Stream ``url`` into ``target``, failing on a stall or a short body.

    ``urlretrieve`` offers no timeout, so the request is made with ``urlopen``.
    The body is checked against ``Content-Length`` as ``urlretrieve`` did, so a
    truncated tile is never renamed into the cache.
    """
    with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
        expected = response.headers.get("Content-Length")
        with target.open("wb") as out:
            shutil.copyfileobj(response, out)
            written = out.tell()
    if expected is not None and written != int(expected):
        raise urllib.error.ContentTooShortError(
            f"retrieved {written} of {expected} bytes", (str(target), response.headers)
        )


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
        # leaves a truncated tile that a later run would trust. The partial file
        # is removed on every failure, not only HTTP errors.
        partial = path.with_suffix(path.suffix + ".part")
        try:
            _download(self.url_for(tile), partial)
            partial.rename(path)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise TileNotPublishedError(f"{tile.name} is not published") from exc
            raise
        finally:
            partial.unlink(missing_ok=True)
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
