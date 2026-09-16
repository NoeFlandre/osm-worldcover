# osm-worldcover

Build supervised **text → land-cover** datasets: each example pairs text
associated with an OpenStreetMap polygon with the [ESA
WorldCover](https://esa-worldcover.org/) class that dominates that polygon.

The same pipeline supports three source recipes:

| recipe | source | output |
| --- | --- | --- |
| `wikidata` | [`NoeFlandre/osm-polygon-wikidata-and-wikipedia`](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-and-wikipedia) | [`NoeFlandre/osm-wikidata-worldcover`](https://huggingface.co/datasets/NoeFlandre/osm-wikidata-worldcover) (unchanged) |
| `description` | [`NoeFlandre/osm-polygon-description-tag`](https://huggingface.co/datasets/NoeFlandre/osm-polygon-description-tag) | [`NoeFlandre/osm-polygon-description-tag-worldcover`](https://huggingface.co/datasets/NoeFlandre/osm-polygon-description-tag-worldcover) |
| `website` | [`NoeFlandre/osm-polygon-website-tag`](https://huggingface.co/datasets/NoeFlandre/osm-polygon-website-tag) | [`NoeFlandre/osm-polygon-website-tag-worldcover`](https://huggingface.co/datasets/NoeFlandre/osm-polygon-website-tag-worldcover) |

Scope is **dataset construction only** — no pretraining or fine-tuning lives
here.

## How an example is made

1. Measure what share of each OSM polygon every WorldCover class covers.
2. Keep the polygon only if **one class covers at least 80%** of it.
3. Emit one example per `(polygon, source text)` pair.
4. Drop empty, very short, and exactly duplicated text.
5. Split on H3 cells so nearby places never straddle train/validation/test.

## Use

```bash
uv sync

# Use --source description or --source website for the other recipes.
uv run owc build --source wikidata --region luxembourg-latest --out data/out
uv run owc verify data/out/v1.0.0
uv run owc info data/out/v1.0.0
uv run owc publish data/out/v1.0.0 NoeFlandre/osm-wikidata-worldcover
```

For a global run, split disjoint region lists across processes and assemble
the shards once:

```bash
# Region stems go to stdout; pin the source revision for a reproducible list.
SOURCE_REVISION=5c8e56a50b5679118a28aef057af002209f80a5e
uv run owc regions --source website --revision "$SOURCE_REVISION" > regions.txt

uv run owc build --source website --revision "$SOURCE_REVISION" --regions-file regions-a.txt --cache data/w0 --out data/w0/out
uv run owc build --source website --revision "$SOURCE_REVISION" --regions-file regions-b.txt --cache data/w1 --out data/w1/out
uv run owc assemble data/w0/shards data/w1/shards --source website --revision "$SOURCE_REVISION" --out data/out
```

Without `--revision`, `owc regions` resolves the current Hub commit and prints
that commit to stderr, keeping the region file clean for redirection. Give each
worker its own cache and output directory; `owc assemble` can combine their
shards and is safe to rerun.

Docker:

```bash
docker build -t osm-worldcover .
docker run --rm -v "$PWD/data:/data" osm-worldcover \
  build --source description --out /data/out --cache /data/cache
```

## Guarantees

Every published build is checked against these, and fails if any breaks:

- No polygon or source document appears in more than one split.
- Every label is one of the 11 real WorldCover classes — never no-data.
- Every row's dominant class covers at least the configured threshold.
- No two rows share both text and label.
- Rebuilding the same inputs produces byte-identical files.

## Design

`domain/` is pure — no network, filesystem, or clock. Source-specific schema
handling is isolated in adapters; WorldCover labelling, splitting, validation,
cards, and publication are shared.

Decisions and trade-offs are in [`docs/adr/`](docs/adr); known weaknesses are
in [`docs/technical-debt.md`](docs/technical-debt.md).

## Development

```bash
uv run ruff check . && uv run ruff format --check .
uv run ty check src/
uv run lint-imports
uv run pytest
uv run pytest --cov --cov-report=json -q && uv run python scripts/crap.py src
uv run mutmut run
uv run mkdocs build --strict
```

CI runs the same quality gates and the Docker smoke test.

## Licence

Code MIT. Source-specific text and geometry licensing is documented in each
dataset card; ESA WorldCover is CC BY 4.0.
