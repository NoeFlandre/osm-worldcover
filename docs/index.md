# osm-worldcover

`osm-worldcover` converts the text of OSM polygons into reproducible ESA
WorldCover land-cover datasets. One shared pipeline supports three text
sources: Wikipedia/Wikivoyage text linked through Wikidata, OSM `description`
tags and OSM website tags.

The Wikidata release stays at
[`NoeFlandre/osm-wikidata-worldcover`](https://huggingface.co/datasets/NoeFlandre/osm-wikidata-worldcover).
The description recipe and the website recipe publish to their own Hugging Face
dataset repositories.

See the [glossary](glossary.md) for the project terms.

## Quick start

```bash
uv sync
uv run owc build --source wikidata --region luxembourg-latest --out data/out
uv run owc verify data/out/v1.1.0
uv run owc info data/out/v1.1.0
```

To select another source, use `--source description` or `--source website`.

## Global builds

A global run uses the CPU heavily and uses one thread. Give each process a
separate list of regions and its own cache. Then assemble the shards one time.

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

The README's [Use section](https://github.com/NoeFlandre/osm-worldcover#use) is the canonical copy of these commands.

If you do not give `--revision`, `owc regions` uses the current Hub commit. It
prints that commit to stderr. The region file stays clean for redirection. Give
each worker its own cache directory and output directory. `owc assemble`
combines the shards. You can safely run it again.

## Label meaning

WorldCover classifies the ground in 10 m pixels. The label gives the class of
the area that **contains** a feature. It does not give the material of the
feature. For the known limits, see [Technical debt](technical-debt.md).
