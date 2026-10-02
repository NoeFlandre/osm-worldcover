# ADR 0005: Refuse continent-scale polygons

**Status:** accepted — 2026-09-13

## Context

The first global build stopped at Angola. One polygon caused the problem:
Angola itself, 1.25 million km². At 10 m, this is about 10¹⁰ pixels in about
20 WorldCover tiles. It needs about 2 GB of download and minutes of computation
for **one row**. The largest polygon in the source is 10.2 million km². It needs
about 100 tiles and about 10¹¹ pixels.

The measured cost is low up to a point. Then the number of tiles dominates it:

| polygon | pixels | zonal-stats time |
| --- | --- | --- |
| 100 km² | 10⁶ | 0.05 s |
| 1,000 km² | 10⁷ | 0.17 s |
| 10,000 km² | 10⁸ | 1.38 s |

The cost is linear in area. A first cap of 100,000 km² was still too high. The
run became CPU-bound on the provinces of Algeria at about 14 s for each polygon.
The network was idle at 5% of its capacity.

## Decision

Refuse the polygons whose source `area_m2` is more than **10¹⁰ m² (10,000
km²)**. Do this check *before* the build fetches any tile. You can configure it
with `--max-area-km2`. The value `None` disables it.

## Consequences

- This excludes **1,099 of 1,259,424** polygons (0.087%).
- It limits the worst-case polygon to about 10⁸ pixels, about 1.4 s.
- It limits the cost of each polygon to a few tiles. This removes the stop.
- The build counts the rejections as `too_large` and reports them in the
  manifest. The exclusion is therefore visible and not silent.
- The decision is also correct for the content. A polygon of this size is a
  country or a continent. Its article describes history and governance and not
  the ground below it. No single land-cover class can cover 80% of it.

## Alternatives considered

- **Read from the overviews of the rasters** for large polygons. This is correct
  in principle and keeps those rows. But `exactextract` does not expose overview
  selection. The project would need to write the decimated reads by hand. It
  would also need a second code path with lower fidelity for 0.019% of the data.
- **No cap.** The project rejected this. It needs about 450 GB of extra download
  for 238 rows. Almost all of them fail the dominance test.
