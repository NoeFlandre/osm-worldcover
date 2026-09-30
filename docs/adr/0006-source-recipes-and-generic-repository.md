# ADR 0006: Source recipes at the adapter boundary

## Status

Accepted

## Context

The first release was built only from the Wikidata/Wikipedia source layout.
The description-tag and website-tag sources publish different Parquet layouts,
geometry encodings, and text fields. The WorldCover labelling, split, validation,
manifest, map, and publication steps should nevertheless remain identical.

## Decision

Keep one canonical `RegionTables` contract inside the pipeline and register one
source recipe for each input: `wikidata`, `description`, and `website`. Each
recipe owns source-repository paths, text-eligibility defaults, and normalization
into canonical polygon, link, and document tables. The domain and downstream
pipeline do not branch on source-specific columns.

The Wikidata-derived HF dataset remains the existing
`NoeFlandre/osm-wikidata-worldcover` repository. The new recipes publish to
source-specific derived repositories. Source revisions and recipe metadata are
recorded in every manifest and card.

## Consequences

- Adding a source requires an adapter and deterministic fixtures, not changes to
  the WorldCover algorithm or split rules.
- Description text keeps exact base and localized tag values; localized suffixes
  are preserved as opaque language values when no detector label is available.
- Description tags require only one whitespace-separated word by default;
  Wikipedia/Wikivoyage and website text require ten. Explicit positive-integer
  overrides are retained and the resolved threshold is recorded in the manifest.
- Website and contact-website text remain separate documents, and the card must
  state that their copyright and reuse conditions belong to the source sites.
- The repository/package name becomes `osm-worldcover`; the historical HF
  Wikidata output name is intentionally retained as a publication target.

## Known weakness and cleanup path

Description source rows do not carry a guaranteed language for base
`description=*` values. The adapter preserves that value as null rather than
guessing. If a future release requires complete language coverage, join the
source's `language-v1` annotation as a versioned, tested enrichment step.
