"""Render the Hugging Face dataset card from a build manifest.

Pure: the card is a function of the manifest alone, so the published
description can never drift from the data it describes.
"""

from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["render"]


def render(manifest: Mapping[str, Any]) -> str:
    """Return the dataset card for ``manifest``."""
    counts = manifest["counts"]
    settings = manifest.get("settings", {})
    threshold_pct = round(float(settings.get("dominance_threshold", 0.8)) * 100)

    return "".join(
        [
            _header(settings),
            _intro(counts, threshold_pct, settings),
            _coverage_map(counts),
            _example_polygons(manifest.get("example_polygons", [])),
            _caveat(),
            _splits(counts),
            _table(
                "Classes",
                ("code", "class", "examples", "share"),
                [
                    (
                        str(row["code"]),
                        row["label"],
                        f"{row['examples']:,}",
                        f"{row['share'] * 100:.1f}%",
                    )
                    for row in manifest.get("class_distribution", [])
                ],
            ),
            _table(
                "Languages (top 15)",
                ("language", "examples", "share"),
                [
                    (
                        row["language"] or "(unspecified)",
                        f"{row['examples']:,}",
                        f"{row['share'] * 100:.1f}%",
                    )
                    for row in manifest.get("language_distribution", [])[:15]
                ],
            ),
            _coverage(manifest.get("geographic_coverage", {})),
            _guarantees(threshold_pct),
            _schema(),
            _provenance(
                settings,
                manifest.get("rejections", {}),
                manifest.get("deduplication", {}),
                manifest.get("deduplication_analysis", {}),
            ),
        ]
    )


def _header(settings: Mapping[str, Any]) -> str:
    """Render Hub metadata from the selected source recipe."""
    source = settings.get("source", "wikidata")
    tags = ["land-cover", "openstreetmap"]
    if source == "wikidata":
        tags.append("wikipedia")
    if source in {"description", "website"}:
        tags.append(f"osm-{source}")
    tags.extend(("geospatial", "text-classification"))
    tag_lines = "\n".join(f"  - {tag}" for tag in tags)
    license_name = settings.get("dataset_license", "cc-by-sa-4.0")
    return f"""---
license: {license_name}
language:
  - multilingual
tags:
{tag_lines}
task_categories:
  - text-classification
configs:
  - config_name: default
    data_files:
      - split: train
        path: train.parquet
      - split: validation
        path: validation.parquet
      - split: test
        path: test.parquet
---
"""


def _intro(counts: Mapping[str, Any], threshold_pct: int, settings: Mapping[str, Any]) -> str:
    total = counts["examples"]["total"]
    output_dataset = settings.get("output_dataset", "NoeFlandre/osm-wikidata-worldcover")
    title = str(output_dataset).rsplit("/", 1)[-1]
    # Used verbatim: every recipe phrases this to read after "pairs", and
    # lowercasing the first letter mangled names like OpenStreetMap.
    source_text = str(settings.get("source_text_description", "a Wikipedia or Wikivoyage article"))
    return f"""
# {title}

A supervised **text to land-cover** dataset. Each example pairs {source_text}
with the [ESA WorldCover](https://esa-worldcover.org/) class that covers at
least {threshold_pct}% of the OpenStreetMap polygon the text describes.

**{total:,} examples**, {counts["polygons"]["total"]:,} distinct polygons,
{counts["documents"]["total"]:,} distinct documents.

```python
from datasets import load_dataset

ds = load_dataset("{output_dataset}")
print(ds["train"][0]["text"][:200], ds["train"][0]["worldcover_label"])
```
"""


def _caveat() -> str:
    return """
## What the label means

WorldCover classifies the ground in 10 m pixels. The label therefore describes
the land cover of the area **containing** a feature, not the feature itself: a
church is `Built-up` because its surroundings are, not because the building was
classified. Treat this as place-context classification.

Each row carries `polygon_area_m2` and `observed_fraction` so you can restrict
to polygons large enough for the dominance test to have been a real filter
(`polygon_area_m2 >= 2500` is 25+ pixels).
"""


def _coverage_map(counts: Mapping[str, Any]) -> str:
    polygons = counts["polygons"]["total"]
    return f"""
## Geographic coverage map

![World map of ESA-labelled polygon centroids](worldcover_centroids.png)

The map shows the centroids of all **{polygons:,} distinct polygons** represented
in this release and carrying an ESA WorldCover label. Each point is colored by
its WorldCover class. Points are centroids, not polygon boundaries; use
`lat`, `lon`, and `centroid_wkt` for the tabular locations.
"""


def _example_polygons(examples: Sequence[Mapping[str, Any]]) -> str:
    """Show deterministic named examples and their ESA labels."""
    if not examples:
        return ""
    rows = [
        (
            _markdown_cell(example["name"]),
            _markdown_cell(example["worldcover_label"]),
        )
        for example in examples
    ]
    return (
        _table("Example labelled polygons", ("polygon name", "ESA WorldCover class"), rows)
        + "\nOne deterministic named polygon is shown for each ESA WorldCover class "
        "represented in this release.\n"
    )


def _markdown_cell(value: Any) -> str:
    """Keep generated table cells valid when source names contain Markdown syntax."""
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _splits(counts: Mapping[str, Any]) -> str:
    rows = [
        (
            name,
            f"{counts['examples'][name]:,}",
            f"{counts['polygons'][name]:,}",
            f"{counts['documents'][name]:,}",
        )
        for name in ("train", "validation", "test")
    ]
    return _table("Splits", ("split", "examples", "polygons", "documents"), rows)


def _coverage(coverage: Mapping[str, Any]) -> str:
    if not coverage:
        return ""
    bbox = coverage.get("bbox", {})
    return (
        "\n## Geographic coverage\n\n"
        f"- H3 cells (resolution 5): {coverage.get('h3_cells', 0):,}\n"
        f"- Source regions: {coverage.get('regions', 0):,}\n"
        f"- Bounding box: {bbox.get('min_lon')}, {bbox.get('min_lat')}"
        f" to {bbox.get('max_lon')}, {bbox.get('max_lat')}\n"
    )


def _guarantees(threshold_pct: int) -> str:
    return f"""
## Guarantees

Every build is checked against these and fails if any breaks:

- **No polygon leakage** — no polygon appears in more than one split.
- **No document leakage** — no document appears in more than one split.
- **Geographic blocking** — splits are assigned per H3 cell, so nearby places
  cannot straddle train, validation and test.
- **Valid labels** — every label is one of the 11 real WorldCover classes;
  no-data is never a label.
- **Dominance** — every row's class covers at least {threshold_pct}% of its polygon.
- **No exact duplicates** — no two rows share both text and label.
- **Deterministic** — rebuilding the same inputs yields byte-identical files.
"""


def _schema() -> str:
    return """
## Columns

| column | meaning |
| --- | --- |
| `text` | Full source text |
| `worldcover_code` / `worldcover_label` | The target class |
| `dominant_fraction` | Share of the polygon covered by that class |
| `observed_fraction` | Share of the polygon observed at all |
| `polygon_id`, `osm_type`, `osm_id` | The OpenStreetMap object |
| `lat`, `lon`, `centroid_wkt` | Polygon centroid |
| `polygon_area_m2` | Polygon area |
| `language`, `project`, `title`, `url` | The article |
| `split`, `h3_cell` | Split and the cell it was decided on |
"""


def _provenance(
    settings: Mapping[str, Any],
    rejections: Mapping[str, int],
    deduplication: Mapping[str, int],
    deduplication_analysis: Mapping[str, Any],
) -> str:
    source_dataset = settings.get("source_dataset")
    source_url = settings.get("source_url", f"https://huggingface.co/datasets/{source_dataset}")
    code_repository = settings.get(
        "code_repository", "https://github.com/NoeFlandre/osm-worldcover"
    )
    text_license = settings.get(
        "text_license", "Article text is CC BY-SA 4.0 (Wikipedia/Wikivoyage)"
    )
    lines = [
        "\n## Provenance\n",
        f"- Source: [`{source_dataset}`]({source_url})"
        f" at revision `{settings.get('source_revision')}`\n",
        f"- Land cover: ESA WorldCover {settings.get('worldcover_year')}"
        f" {settings.get('worldcover_version')} (10 m)\n",
        f"- Dominance threshold: {settings.get('dominance_threshold')}\n",
        f"- Minimum source text length: {settings.get('min_words')} words\n",
        f"- Maximum polygon area: {settings.get('max_polygon_area_m2')} m2\n",
        f"- Split seed: {settings.get('split_seed')},"
        f" H3 resolution {settings.get('h3_resolution')}\n",
        f"\nCode: [{code_repository.removeprefix('https://')}]({code_repository})\n",
    ]
    lines.extend(_rejection_table(rejections))
    lines.extend(_deduplication_table(deduplication))
    lines.extend(_deduplication_analysis(deduplication_analysis))
    lines.append(
        "\n## Licence\n\n"
        f"{str(text_license).rstrip('.')}. OpenStreetMap geometry is ODbL. "
        "ESA WorldCover is CC BY 4.0.\n"
    )
    return "".join(lines)


def _rejection_table(rejections: Mapping[str, int]) -> list[str]:
    if not rejections:
        return []
    return [
        "\n### Polygons refused\n\n",
        "| reason | polygons |\n| --- | --- |\n",
        *[f"| `{key}` | {count:,} |\n" for key, count in sorted(rejections.items())],
    ]


def _deduplication_table(deduplication: Mapping[str, int]) -> list[str]:
    if not deduplication:
        return []
    return [
        "\n### Rows removed after labelling\n\n",
        "| reason | rows |\n| --- | --- |\n",
        *[f"| `{key}` | {count:,} |\n" for key, count in sorted(deduplication.items())],
    ]


def _deduplication_analysis(analysis: Mapping[str, Any]) -> list[str]:
    """Make the repeated-text loss and split tradeoff visible in the card."""
    if not analysis:
        return []
    by_length = analysis["duplicate_rows_removed_by_text_words"]
    short_summary = ", ".join(_short_word_counts(by_length)) or "none"
    short_total = _short_word_total(by_length)
    return [
        "\n### Repeated text and split tradeoff\n\n",
        f"The build collapsed {analysis['duplicate_text_label_groups']:,} exact text-label "
        f"groups, removing {analysis['duplicate_rows_removed']:,} rows. Of those, "
        f"{short_total:,} had fewer than 10 words ({short_summary}). "
        f"{analysis['duplicate_groups_crossing_splits']:,} duplicate groups crossed "
        "the spatial split assignments, accounting for "
        f"{analysis['duplicate_rows_removed_from_cross_split_groups']:,} removed rows.\n\n",
        "This can remove valid repeated short descriptions on distinct polygons; keeping "
        "all of them could expose identical text and labels across splits. Spatial and "
        "document split assignments are retained, and the complete counts are in "
        "`manifest.json`.\n",
    ]


def _short_word_counts(by_length: Mapping[str, int]) -> list[str]:
    return [
        f"{words} word: {count:,}" for words, count in by_length.items() if words != "10+" and count
    ]


def _short_word_total(by_length: Mapping[str, int]) -> int:
    return sum(count for words, count in by_length.items() if words != "10+")


def _table(title: str, header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    if not rows:
        return ""
    out = [f"\n## {title}\n\n", "| " + " | ".join(header) + " |\n"]
    out.append("| " + " | ".join("---" for _ in header) + " |\n")
    out += ["| " + " | ".join(r) + " |\n" for r in rows]
    return "".join(out)
