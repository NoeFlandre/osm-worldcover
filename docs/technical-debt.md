# Technical debt and known weaknesses

Recorded deliberately, with why each exists and how it would be cleaned up.

## The one-off parallel release helper is retired

The temporary `scripts/parallel_release.py` used while preparing the
description-tag release is not part of the tracked repository after PR #5.
It hard-coded one source revision and output root, duplicated the tested CLI
workflow, and its shard-combining step was not safe to rerun. Region discovery,
split builds, and assembly now have supported paths:

```bash
SOURCE_REVISION=5c8e56a50b5679118a28aef057af002209f80a5e
uv run owc regions --source website --revision "$SOURCE_REVISION" > regions.txt
# Deterministically partition the pinned listing into the two worker files.
awk 'NF { output = "regions-" ((count++ % 2) ? "b" : "a") ".txt"; print > output }' regions.txt
uv run owc build --source website --revision "$SOURCE_REVISION" --regions-file regions-a.txt --cache data/w0 --out data/w0/out
uv run owc build --source website --revision "$SOURCE_REVISION" --regions-file regions-b.txt --cache data/w1 --out data/w1/out
uv run owc assemble data/w0/shards data/w1/shards --source website --revision "$SOURCE_REVISION" --out data/out
```

The retained ignored release scratch at
`data/releases/description/parallel` is recovery evidence, not repository
source. Keep it while a resume, verification, or publication check may still
depend on it. A later cleanup may remove it only after the public release has
been independently verified and no release worker is active; this issue does
not delete that scratch or any published files. A stale local copy of the old
helper likewise needs explicit local cleanup outside this worktree.

## The label describes the place, not the feature

A 10 m pixel is 100 m². Polygons below that — 7.6% of the source — are smaller
than a single pixel, and their label is effectively "whatever covers the ground
here". Even at the median (1,301 m², ~13 pixels) a church is labelled
`Built-up` because its surroundings are, not because the building was
classified as such.

*Why it exists:* inherent to raster land cover at any resolution the source
publishes. **Not** fixable with better data.

*Mitigation in place:* every row carries `polygon_area_m2` and
`observed_fraction`, and the manifest reports the `dominant_fraction`
distribution, so a consumer can filter to polygons where dominance was a real
test.

*Cleanup path:* publish a recommended `polygon_area_m2 >= 2500` subset (25+
pixels) as a named config alongside the full dataset.

## Built-up dominates the class distribution

Wikidata-linked polygons are overwhelmingly buildings and settlements, so
`Built-up` can swamp the other ten classes. The balance is source-dependent;
description and website recipes should be reported separately.

*Why it exists:* a property of what people write encyclopaedia articles about,
compounded by WorldCover collapsing all settlement into one class (ADR 0001).

*Cleanup path:* ship a class-balanced subset, or report per-class metrics and
macro-averages rather than accuracy. Adding CORINE as a second label column
would restore the urban distinctions; see ADR 0001.

## Very large polygons are excluded

Polygons above 10,000 km² are refused (ADR 0005) — 1,099 rows, 0.087%.

*Why it exists:* zonal-statistics cost is linear in area, and without a cap a
single continent-scale polygon needs ~100 tiles and hours of computation.

*Cleanup path:* read those polygons from the rasters' overview pyramids
instead of at full resolution. `exactextract` does not expose overview
selection, so this needs a second, decimated code path — worth it only if
those 1,099 rows are wanted.

## Documents describing several distant places lose rows

One source document can describe many places — a river, a mountain range, a
chain of monuments. Those polygons fall in different H3 cells and therefore different
splits, so the document would appear in train *and* test.

*Why it exists:* the alternative is moving every row of that document into one
split, which breaks the geographic blocking the splits exist to provide. The
rows in the minority splits are dropped instead.

*Mitigation in place:* the count is reported as `documents_split_across_splits`
so the loss is visible. On a partial global build it was 30 rows.

*Cleanup path:* group cells into connected components joined by shared
documents and split per component. Risky: one article about a continent could
chain most of the world into a single component, so measure component sizes
before adopting it.

## Split boundaries are cell edges, not buffers

H3 blocking guarantees that everything *within* a cell shares a split, but two
polygons a metre apart on opposite sides of a cell edge can still be separated
(ADR 0003).

*Why it exists:* true buffered blocking means discarding a margin around every
boundary, which costs data and complicates reproducibility.

*Cleanup path:* drop examples within a fixed distance of a cell boundary, or
group cells into buffered super-cells. Quantify the affected share first — it
is a perimeter effect and may not be worth the loss.

## Invalid geometries are dropped, not repaired

`domain/geometry.py` refuses self-intersecting polygons rather than running
`make_valid`.

*Why it exists:* repair alters the very shape whose area fraction becomes the
label. A smaller trustworthy dataset was preferred to a larger one resting on
geometries the source never asserted.

*Cleanup path:* if the discarded share proves material, repair *and* record a
`geometry_repaired` flag so consumers can exclude those rows.

## Split ratios are approximate

80/10/10 applies to H3 cells, not rows. Realised row counts drift — the
Luxembourg smoke build came out 76/10/14 because a small country spans few
cells.

*Why it exists:* the alternative is packing cells to hit row targets, which
makes assignment depend on the whole dataset instead of on the cell id alone,
costing reproducibility.

*Cleanup path:* accept it, and report realised counts in the manifest — which
it does.

## Only the first document language is normalised

`language` is taken from the document, falling back to the link table. No
attempt is made to reconcile disagreements between the two.

*Why it exists:* they disagree rarely, and the document is authoritative.

*Cleanup path:* count the disagreements; if non-trivial, record both.
