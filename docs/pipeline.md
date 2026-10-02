# How it works

## Structure

```text
cli  ->  build  ->  finalize  ->  pipeline  ->  adapters  ->  domain
```

The dependencies go in one direction only. `lint-imports` checks this on every
run. `domain/` is pure. It does not use the network, the filesystem or the
clock. The code in `adapters/` does all the work that touches the outside
world.

## One region at a time

A global run uses 2,499 WorldCover tiles. This is about 229 GB. The build does
not stage the tiles first (ADR 0004).

1. Download the polygons and the source-specific text fields of one region.
2. Parse the geometries. Drop the invalid geometries and count them.
3. Group the polygons by the tiles they touch.
4. For each group: fetch the tiles, measure the coverage, and discard the tiles.
5. Keep the polygons that one class dominates. Join these polygons to their
   source text.
6. Write the shard of the region. Then delete the downloaded tables.

The peak disk use stays near the size of one tile. A run that stops can
continue. It skips the regions that already have a shard.

## Measuring coverage

`exactextract` gives the values and the coverage of each cell. The build
selects the cells near the polygon boundary in pixel coordinates. GEOS then
computes the areas of these cells again. It uses the original polygon and unit
pixel squares. This corrects the corner cases where `exactextract` can report a
full exterior cell or omit an interior cell. The small pixel-space halo only
selects cells for the check. It does not change the polygon or its area. The
build divides the corrected class areas by the area of the polygon. Because of
this, no-data and raster gaps stay unobserved (ADR 0002).

Coverage is additive across tiles. A polygon on a tile boundary needs only a
sum. It does not need a mosaic. The build never clamps or renormalizes the
shares. If the shares add to more than one, validation fails.
`OverlappingCoverageError` reports this error.

## Deciding the label

`domain/dominance.py` contains the complete rule.

- The build computes the shares against the area of the **polygon**. It never
  uses the observed area.
- No-data is not a class. It can never win. It still uses part of the polygon.
  The build therefore refuses a polygon that is mostly unobserved.
- The highest share wins. A tie goes to the lowest class code. The result then
  never depends on the iteration order.
- The build accepts a polygon only at or above the threshold. The default is
  0.8.

## Source-specific text eligibility

Description tags are short labels and descriptions. They are not articles. The
default minimum is **1 whitespace-separated word**. This keeps useful short base
descriptions and localized descriptions. It also keeps scripts that do not use
spaces between words. An empty value or a whitespace-only value is never an
example.

Wikipedia/Wikivoyage text and website text keep the **10-word** minimum. A
build can override the policy of its recipe. Use a positive integer
`Config.min_words` or the `min_words` YAML setting. The value `null` selects the
default of the recipe. The manifest records the resolved integer as
`settings.min_words`. The build uses it to select examples and to validate
them. The build does not expand a text to reach the threshold. The
normalization only trims and collapses whitespace.

## Assembling the dataset

The build solves three different problems. Do not treat them as one problem.
Otherwise at least one result is wrong.

1. Geofabrik extracts overlap. One OSM object can then have several
   `polygon_id` values. The build chooses one region for each object.
2. The build removes a repeated row only when the stable polygon identity, the
   normalized text and the WorldCover label all match. Different polygons that
   have identical text and labels stay separate examples.
3. One article can describe several distant places. These places are in
   different cells and so in different splits. The split that has most rows of
   the document keeps its rows. The build drops the other rows. Moving them
   would break the geographic blocking.

The manifest and the generated card report the exact same-polygon removals and
their word counts. They also report repeated normalized text across polygons
and splits as diagnostics. The build keeps identical text on different
polygons. Cross-split text collisions do not cause a broad deletion of rows.
The audit gives a warning when such collisions span splits. Polygon leakage,
document leakage and H3 leakage stay failures.

### The build never holds the whole dataset in memory

A global build is several times larger than the memory of the machine. The
measured size is 4.9 KB for each row. Therefore:

- The build does the row-local work (H3 cell, split, duplicate keys) **for one
  shard at a time**. One region limits the size.
- **DuckDB over files** does the global work (de-duplication and ordering).
- The build **streams** each split from DuckDB to Parquet in Arrow batches.
- Validation reads the written files **back** one batch at a time. It then
  checks the files that the build actually published.

A rebuild with the same inputs produces byte-identical files. This is the
cheapest check that these methods did not change the data.
