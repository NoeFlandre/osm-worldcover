# osm-worldcover

Build supervised **text → land-cover** datasets. Each example pairs the text
of an OpenStreetMap polygon with the [ESA
WorldCover](https://esa-worldcover.org/) class that dominates that polygon.

The same pipeline supports three recipes:

| recipe | source | output |
| --- | --- | --- |
| `wikidata` | [`NoeFlandre/osm-polygon-wikidata-and-wikipedia`](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-and-wikipedia) | [`NoeFlandre/osm-wikidata-worldcover`](https://huggingface.co/datasets/NoeFlandre/osm-wikidata-worldcover) (unchanged) |
| `description` | [`NoeFlandre/osm-polygon-description-tag`](https://huggingface.co/datasets/NoeFlandre/osm-polygon-description-tag) | [`NoeFlandre/osm-polygon-description-tag-worldcover`](https://huggingface.co/datasets/NoeFlandre/osm-polygon-description-tag-worldcover) |
| `website` | [`NoeFlandre/osm-polygon-website-tag`](https://huggingface.co/datasets/NoeFlandre/osm-polygon-website-tag) | [`NoeFlandre/osm-polygon-website-tag-worldcover`](https://huggingface.co/datasets/NoeFlandre/osm-polygon-website-tag-worldcover) |

This project builds datasets only. It does not contain pretraining or
fine-tuning code.

See the [glossary](docs/glossary.md) for the project terms.

## How the pipeline makes an example

1. Measure the share of each class in the polygon. Recompute the cells that
   touch the boundary with GEOS in pixel coordinates. This corrects raster-grid
   corner errors.
2. Keep the polygon only if **one class covers at least 80%** of it.
3. Emit one example for each `(polygon, source text)` pair.
4. Keep description tags that have at least 1 word. Require at least 10 words
   for Wikipedia/Wikivoyage text and website text. Remove a repeated row only
   when the polygon identity, the normalized text and the WorldCover label all
   match. Keep identical text on different polygons. Report the collisions
   between splits.
5. Split on H3 cells. Nearby places then stay in one of train, validation or
   test.

## Use

```bash
uv sync

# Use --source description or --source website for the other recipes.
uv run owc build --source wikidata --region luxembourg-latest --out data/out
uv run owc verify data/out/v1.1.0
uv run owc info data/out/v1.1.0
uv run owc publish data/out/v1.1.0 NoeFlandre/osm-wikidata-worldcover
```

For a global run, give each process a separate list of regions. Then assemble
the shards one time:

```bash
# Region stems go to stdout; pin the source revision for a reproducible list.
SOURCE_REVISION=5c8e56a50b5679118a28aef057af002209f80a5e
uv run owc regions --source website --revision "$SOURCE_REVISION" > regions.txt
# Deterministically partition the pinned listing into the two worker files.
awk 'NF { output = "regions-" ((count++ % 2) ? "b" : "a") ".txt"; print > output }' regions.txt

uv run owc build --source website --revision "$SOURCE_REVISION" --regions-file regions-a.txt --cache data/w0 --out data/w0/out
uv run owc build --source website --revision "$SOURCE_REVISION" --regions-file regions-b.txt --cache data/w1 --out data/w1/out
uv run owc assemble data/w0/shards data/w1/shards --source website --revision "$SOURCE_REVISION" --out data/out
```

If you do not give `--revision`, `owc regions` uses the current Hub commit. It
prints that commit to stderr. The region file stays clean for redirection. Give
each worker its own cache directory and output directory. `owc assemble`
combines the shards. You can safely run it again.

Docker:

```bash
docker build -t osm-worldcover .
mkdir -p data
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/data:/data" osm-worldcover \
  build --source description --out /data/out --cache /data/cache
```

The image runs as the non-root user `owc` (UID 10001). The `--user` flag makes
the output files owned by you on the host. Without it, the container cannot
write to a host directory that you own.

## Guarantees

The build checks each published release against these rules. The build fails if
one rule is broken.

- No polygon or source document is in more than one split.
- Each label is one of the 11 real WorldCover classes. No-data is never a label.
- The dominant class of each row covers at least the configured threshold.
- No two rows have the same polygon identity, normalized text and label.
- Identical text on different polygons can remain, also across splits. The
  build reports these cross-split collisions as diagnostics.
- A rebuild with the same inputs produces byte-identical files.

## Design

`domain/` is pure. It does not use the network, the filesystem or the clock.
The adapters contain the source-specific schema handling. The recipes share the
WorldCover labelling, the splitting, the validation, the cards and the
publication.

The design decisions and trade-offs are in [`docs/adr/`](docs/adr). The known
weaknesses are in [`docs/technical-debt.md`](docs/technical-debt.md).

## Development

```bash
uv run ruff check . && uv run ruff format --check .
uv run ty check src/
uv run lint-imports
uv run pytest
uv run pytest --cov --cov-report=json -q && uv run python scripts/crap.py src scripts tests
uv run mutmut run
uv run mkdocs build --strict
```

CI runs the same quality gates and the Docker smoke test.

## Licence

The code uses the MIT licence. Each dataset card describes the licence of the
source text and geometry. ESA WorldCover uses CC BY 4.0.
