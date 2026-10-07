"""Write a finished build to disk.

Output is versioned: each build lands in its own ``v<version>`` directory
alongside its manifest, so an older dataset is never silently overwritten by a
newer one and both can be compared.

Splits are written as a stream of Arrow batches, because a split is routinely
larger than memory. Pandas' own index bookkeeping is kept out of the file, so
rebuilding the same data yields byte-identical output -- the cheapest possible
check that a pipeline change altered nothing it should not have.
"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from osm_worldcover.release_commit import release_read_lock

__all__ = [
    "MANIFEST_NAME",
    "PARQUET_ROW_GROUP_SIZE",
    "read_manifest",
    "write_batches",
    "write_manifest",
]

MANIFEST_NAME = "manifest.json"
PARQUET_ROW_GROUP_SIZE = 25_000


def read_manifest(build_dir: Path) -> dict:
    """Read the manifest written beside a build."""
    with release_read_lock(build_dir):
        path = Path(build_dir) / MANIFEST_NAME
        if not path.exists():
            raise FileNotFoundError(f"no manifest in {build_dir}")
        return json.loads(path.read_text())


def write_batches(reader: pa.RecordBatchReader, path: Path) -> int:
    """Stream ``reader`` to a Parquet file, returning the rows written.

    Batches are written as they arrive, so a split larger than memory costs no
    more than one batch. An empty result still produces a file with the right
    schema, because a consumer expecting three splits should find three.
    """
    rows = 0
    with pq.ParquetWriter(
        path,
        reader.schema,
        compression="zstd",
        store_schema=False,
        write_page_index=True,
    ) as writer:
        for batch in reader:
            writer.write_batch(batch, row_group_size=PARQUET_ROW_GROUP_SIZE)
            rows += batch.num_rows
    return rows


def write_manifest(manifest: Mapping[str, Any], path: Path) -> Path:
    """Write ``manifest`` as readable, stable JSON."""
    path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")
    return path
