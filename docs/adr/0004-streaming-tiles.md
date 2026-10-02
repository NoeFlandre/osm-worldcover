# ADR 0004: Stream WorldCover tiles rather than stage them

**Status:** accepted — 2026-09-13

## Context

WorldCover ships as 3°x3° GeoTIFFs of about 94 MB each. The source polygons touch
**2,499** of them. This is about 229 GB. If the build stages them first, this
step uses most of the runtime and the disk budget before the build labels one
polygon.

## Decision

Group the polygons by the tiles they touch. For each group: download the tiles,
label every polygon over them, and discard the tiles immediately. Write the
region shards to disk when they are complete.

## Consequences

- The peak disk use stays near one tile and not 229 GB. The 229 GB is
  bandwidth and not storage.
- Grouping fetches each tile one time for all the polygons over it. It does not
  fetch the tile for each polygon.
- The shards of each region limit the memory to one region. They make the build
  resumable. A run that stops skips the regions that are finished. A region
  that gives no examples still writes a shard. This is a finished result. Without
  the shard, each restart tries the empty regions again without end.
- The build deletes the downloaded source tables when it writes the shard of a
  region. The full source snapshot is about 21 GB. The build never needs it
  twice.
- `--keep-tiles` keeps the tiles. Use it when you rebuild one region many times.
