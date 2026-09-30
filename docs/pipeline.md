# How it works

## Shape

```text
cli  ->  build  ->  finalize  ->  pipeline  ->  adapters  ->  domain
```

Dependencies point one way only, and that is checked automatically by
`lint-imports` on every run. `domain/` is pure: no network, no filesystem, no
clock. Everything that touches the outside world lives in `adapters/`.

## A region at a time

A global run touches 2,499 WorldCover tiles, about 229 GB. Nothing is staged up
front (ADR 0004):

1. Download one region's polygons and source-specific text fields.
2. Parse the geometries; drop the invalid ones and count them.
3. Group polygons by the tiles they touch.
4. For each group: fetch the tiles, measure coverage, discard the tiles.
5. Keep polygons one class dominates; join those to their source text.
6. Write the region's shard, then delete its downloaded tables.

Peak disk stays near a single tile. An interrupted run resumes by skipping
regions whose shard already exists.

## Measuring coverage

`exactextract` supplies per-cell values and coverage. Cells near each polygon
boundary are selected in pixel coordinates, then their areas are recomputed by
GEOS against the original polygon and unit pixel squares. This corrects corner
cases where exactextract can report a full exterior cell or omit an interior
cell. The small pixel-space halo only selects cells for checking; it does not
change the polygon or its area. The corrected class areas are divided by the
polygon's own area, so no-data and raster gaps remain unobserved (ADR 0002).

Coverage is additive across tiles, so a polygon straddling a tile boundary is
just a sum — no mosaic. Shares are never clamped or renormalized. Shares
summing past one remain a validation failure, and `OverlappingCoverageError`
reports them.

## Deciding the label

`domain/dominance.py` is the whole rule:

- Shares are taken against the **polygon's** area, never the observed area.
- No-data is not a class and can never win — but it still consumes the polygon,
  so a mostly-unobserved polygon is refused.
- The highest share wins; ties break on the lowest class code, so the result
  never depends on iteration order.
- Accepted only at or above the threshold (default 0.8).

## Source-specific text eligibility

Description tags are concise labels and descriptions, not articles. Their
default minimum is **1 whitespace-separated word**, preserving useful short
base and localized descriptions, including scripts that do not separate words
with spaces. Empty and whitespace-only values are never examples.

Wikipedia/Wikivoyage and website text retain the **10-word** minimum. A build
can override its recipe's policy with a positive integer `Config.min_words`
(or the `min_words` YAML setting); `null` selects the recipe default. The
resolved integer is recorded as `settings.min_words` in the manifest and used
for both example selection and validation. No text is expanded to meet the
threshold; normalization only trims and collapses whitespace.

## Assembling the dataset

Three different problems are resolved, and conflating them would get at least
one wrong:

1. Geofabrik extracts overlap, so one OSM object appears under several
   `polygon_id`s. One region is chosen per object.
2. Repeated rows are removed only when stable polygon identity, normalized text
   and WorldCover label all match. Different polygons with identical text and
   labels remain separate examples.
3. One article can describe several distant places, which fall in different
   cells and so different splits. The split holding most of that document's
   rows keeps them; the rest are dropped, because moving them would break the
   geographic blocking.

The manifest and generated card report exact same-polygon record removals and
their word counts. They also report repeated normalized text across polygons
and splits as diagnostics. Identical text on distinct polygons is retained;
cross-split text collisions do not trigger broad row deletion. The audit warns
when those collisions span splits, while polygon, document and H3 leakage
remain failures.

### Nothing is held whole

A global build is several times the memory of the machine that produces it —
measured at 4.9 KB per row. So:

- row-local work (H3 cell, split, duplicate keys) happens **one shard at a
  time**, bounded by a single region;
- the global work — de-duplication and ordering — is left to **DuckDB over
  files**;
- each split is **streamed** from DuckDB to Parquet in Arrow batches;
- validation reads the written files **back** a batch at a time, so what is
  checked is what was actually published.

Rebuilding the same inputs still produces byte-identical files, which is the
cheapest check that none of this changed the data.
