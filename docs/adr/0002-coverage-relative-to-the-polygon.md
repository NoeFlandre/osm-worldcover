# ADR 0002: Coverage is a share of the polygon, not of the observed area

**Status:** accepted — 2026-09-13

## Context

For each polygon, `exactextract` reports the distinct raster values under the
polygon and the share of each value. It normalises the shares over the cells
that it actually read. The shares therefore always add to 1.0. This is true also
when the build never observed most of the polygon. The tool does not read cells
outside the raster. WorldCover marks unobserved ground with a no-data value,
and the tool masks this value.

Take a polygon that is 95% over open ocean and 5% over a wooded islet. If the
build uses the raw shares, it reports 100% `Tree cover`. The polygon then passes
the 80% filter.

## Decision

Scale each share by the fraction of the polygon that the raster observed:

```
share_of_polygon = share_of_observed x (observed_cells x cell_area / polygon_area)
```

The shares then add to **at most** one. The shortfall is exactly the unobserved
part. The build never emits no-data as a class.

## Consequences

- The build refuses a polygon that is mostly unobserved because it fails the
  dominance test. This is the correct result. The evidence for a label does not
  exist.
- The shares are relative to the polygon. The contributions from different tiles
  **add**. A polygon on a tile boundary needs no mosaic and no VRT. It needs only
  a sum over the tiles that it touches. This removed a whole class of machinery.
- The parts cannot be larger than the whole. If the coverage adds to more than
  1.0, the caller counted a part twice. This is a real bug.
  `OverlappingCoverageError` fails with a clear error and does not clamp. This is
  how the build found the duplicate-raster defect in `_fetch`.
- The build measures the areas of the cells and of the polygon in the planar
  units of the raster. It uses only their ratio. The result is exact although
  the units are degrees.
