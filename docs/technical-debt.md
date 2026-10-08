# Technical debt and known weaknesses

This page records each weakness on purpose. For each weakness, it gives the
reason and the way to remove it.

## The one-off parallel release helper is retired

The temporary `scripts/parallel_release.py` prepared the description-tag
release. After PR #5, it is not in the tracked repository. It had one source
revision and one output root in the code. It duplicated the tested CLI workflow.
Its shard-combining step was not safe to run again. The project now has
supported paths for region discovery, split builds and assembly:

```bash
SOURCE_REVISION=5c8e56a50b5679118a28aef057af002209f80a5e
uv run owc regions --source website --revision "$SOURCE_REVISION" > regions.txt
# Deterministically partition the pinned listing into the two worker files.
awk 'NF { output = "regions-" ((count++ % 2) ? "b" : "a") ".txt"; print > output }' regions.txt
uv run owc build --source website --revision "$SOURCE_REVISION" --regions-file regions-a.txt --cache data/w0 --out data/w0/out
uv run owc build --source website --revision "$SOURCE_REVISION" --regions-file regions-b.txt --cache data/w1 --out data/w1/out
uv run owc assemble data/w0/shards data/w1/shards --source website --revision "$SOURCE_REVISION" --out data/out
```

The README's [Use section](https://github.com/NoeFlandre/osm-worldcover#use) is the canonical copy of these commands.

The ignored release scratch at `data/releases/description/parallel` is recovery
evidence. It is not repository source. Keep it while a resume, a verification or
a publication check can still need it. Remove it only after two conditions are
true. First, a person has independently verified the public release. Second, no
release worker is active. This issue does not delete this scratch or any
published file. A stale local copy of the old helper needs a separate local
cleanup outside this worktree.

## The label describes the place, not the feature

A 10 m pixel is 100 m². 7.6% of the source polygons are smaller than one pixel.
For these polygons, the label is the class of the ground around them. The median
polygon is 1,301 m² (about 13 pixels). A church of this size gets the label
`Built-up` because its surroundings are built-up. The classifier did not
classify the building itself.

*Why it exists:* All raster land cover has this limit at the resolution that
the source publishes. Better data does **not** remove it.

*Mitigation in place:* Each row has `polygon_area_m2` and `observed_fraction`.
The manifest reports the distribution of `dominant_fraction`. A consumer can
then filter to the polygons where dominance was a real test.

*Cleanup path:* Publish a recommended subset with `polygon_area_m2 >= 2500`
(25 pixels or more). Publish it as a named config next to the full dataset.

## Built-up dominates the class distribution

The Wikidata-linked polygons are mostly buildings and settlements. The class
`Built-up` can then be much larger than the other ten classes. The balance
depends on the source. Report the description recipe and the website recipe
separately.

*Why it exists:* People write encyclopaedia articles mainly about built places.
WorldCover also puts all settlement into one class (ADR 0001).

*Cleanup path:* Ship a class-balanced subset. Or report per-class metrics and
macro-averages and not accuracy. A second label column with CORINE restores the
urban distinctions. See ADR 0001.

## Very large polygons are excluded

The build refuses polygons above 10,000 km² (ADR 0005). This removes 1,099 rows
(0.087%).

*Why it exists:* The cost of zonal statistics is linear in area. Without a cap,
one continent-scale polygon needs about 100 tiles and hours of computation.

*Cleanup path:* Read these polygons from the overview pyramids of the rasters
and not at full resolution. `exactextract` does not expose overview selection.
This change therefore needs a second, decimated code path. Do it only if the
project needs these 1,099 rows.

## Documents that describe several distant places lose rows

One source document can describe many places, for example a river, a mountain
range or a chain of monuments. These polygons are in different H3 cells and so
in different splits. The document would then be in train *and* in test.

*Why it exists:* The alternative is to move all rows of the document into one
split. This breaks the geographic blocking that the splits must give. The build
drops the rows in the minority splits.

*Mitigation in place:* The build reports the count as
`documents_split_across_splits`. The loss is then visible. A partial global
build lost 30 rows.

*Cleanup path:* Group the cells into connected components that shared documents
join. Then split for each component. This is risky. One article about a
continent can chain most of the world into one component. Measure the component
sizes before you adopt it.

## Split boundaries are cell edges, not buffers

H3 blocking guarantees that all rows **inside** a cell share a split. Two
polygons that are one metre apart on opposite sides of a cell edge can still be
in different splits (ADR 0003).

*Why it exists:* True buffered blocking discards a margin around each boundary.
This costs data and makes reproducibility more complex.

*Cleanup path:* Drop the examples within a fixed distance of a cell boundary.
Or group the cells into buffered super-cells. First quantify the affected share.
It is a perimeter effect and the loss can be too large.

## Invalid geometries are dropped, not repaired

`domain/geometry.py` refuses self-intersecting polygons. It does not run
`make_valid`.

*Why it exists:* A repair changes the shape whose area fraction becomes the
label. The project prefers a smaller dataset that it can trust. It does not
want a larger dataset that depends on geometries the source never asserted.

*Cleanup path:* If the discarded share is large, repair the geometries **and**
record a `geometry_repaired` flag. A consumer can then exclude these rows.

## Split ratios are approximate

The ratio 80/10/10 applies to H3 cells and not to rows. The realised row counts
differ from it. The Luxembourg smoke build gave 76/10/14 because a small
country has few cells.

*Why it exists:* The alternative is to pack cells to reach row targets. Then the
assignment depends on the whole dataset and not only on the cell id. This
reduces reproducibility.

*Cleanup path:* Accept it. Report the realised counts in the manifest. The
manifest already does this.

## Only the first document language is normalised

The build takes `language` from the document. If the document has no language,
it uses the link table. The build does not try to resolve a disagreement between
the two.

*Why it exists:* They rarely disagree. The document is the authority.

*Cleanup path:* Count the disagreements. If the count is significant, record
both values.
