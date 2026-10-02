# ADR 0006: Source recipes at the adapter boundary

## Status

Accepted

## Context

The first release used only the Wikidata/Wikipedia source layout. The
description-tag source and the website-tag source publish different Parquet
layouts, geometry encodings and text fields. The WorldCover labelling, the
split, the validation, the manifest, the map and the publication steps must
stay identical.

## Decision

Keep one canonical `RegionTables` contract in the pipeline. Register one source
recipe for each input: `wikidata`, `description` and `website`. Each recipe owns
the source-repository paths and the text-eligibility defaults. It also owns the
normalization into the canonical polygon, link and document tables. The domain
and the downstream pipeline do not branch on source-specific columns.

The Wikidata-derived HF dataset stays the existing
`NoeFlandre/osm-wikidata-worldcover` repository. The new recipes publish to
source-specific derived repositories. Every manifest and card records the source
revisions and the recipe metadata.

## Consequences

- To add a source, write an adapter and deterministic fixtures. Do not change
  the WorldCover algorithm or the split rules.
- Description text keeps the exact base values and the exact localized tag
  values. If no detector label is available, the build keeps the localized
  suffix as an opaque language value.
- Description tags need only one whitespace-separated word by default.
  Wikipedia/Wikivoyage text and website text need ten words. The build keeps
  the explicit positive-integer overrides. It records the resolved threshold in
  the manifest.
- Website text and contact-website text stay separate documents. The card must
  state that the source sites own the copyright and the reuse conditions.
- The name of the repository and package becomes `osm-worldcover`. The
  historical HF Wikidata output name stays on purpose as a publication target.

## Known weakness and cleanup path

The description source rows do not always have a language for the base
`description=*` values. The adapter keeps this value as null. It does not guess.
If a future release needs complete language coverage, join the `language-v1`
annotation of the source. Add it as a versioned and tested enrichment step.
