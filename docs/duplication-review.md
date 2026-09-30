# Duplication review — 2026-09-30

## Scope and method

This review used the exact merged tree at `3ddd472e7deac10d116fd763cbf612e0d1a9c8db`.
It scanned all 61 Python files under `src/` and `tests/` with Pylint 4.1.1's
`duplicate-code` checker at its four-line minimum, then manually reviewed shared
fixtures and the reported production matches. This is a targeted clone scan;
it does not establish that the repository contains no duplication.

The baseline scan found five clone reports. Two were repeated canonical table
column declarations, and one manual test-setup review found the same fixed
raster tile source defined in three test modules. The follow-up centralizes the
table contract in `adapters.source_profiles` and shares a single recording
`FixedTiles` test double in `tests/conftest.py`. The normalized document table
still appends its `project` column, and the Wikidata document reader still
returns the same base columns before its caller adds that value.

The same scan now reports three matches, all retained as independent reader /
writer contracts described below. The test double consolidation removes the
three local copies without changing what `ensure` returns or the cache calls
that unit tests observe.

## Intentional remaining matches

- `pipeline.OUTPUT_COLUMNS` and `pipeline.TEXT_COLUMNS` overlap with the
  independently declared release schema in `adapters.audit`. The auditor reads
  the finalized files and must be able to catch a producer schema regression;
  it also checks provenance, split, and H3 columns that are added after the
  pipeline output.
- `accounting.COUNT_FIELDS` and `adapters.audit._OUTCOME_COUNTS` name the same
  serialized counters. The audit validates the receipt written by accounting,
  so keeping its expected field list independent lets it detect a missing or
  changed writer field.

These are deliberate duplicated contracts, not shared processing logic. The
remaining clone reports are limited to those declarations. No zero-duplication
claim is made.

Test setup that remains local is scenario-specific: the acceptance `World`
builder and its polygon/document helpers assemble BDD cases, while source-recipe
tests write distinct upstream row layouts. The unit `polygons_frame` helper
creates canonical pipeline inputs and is not equivalent to those source files.

## Cross-repository candidates and ownership plan

No sibling repository was inspected for this review. The following are
candidates inferred from this repository's source recipes and adapters, not
confirmed cross-repository duplicates:

- OSM polygon identity construction and the canonical polygon/link/document
  contract.
- Source text cleanup and conversion of description, website, and article
  fields into canonical documents.
- Region inventory pinning and provenance receipts around per-region builds.

The proposed ownership boundary is that each source repository owns extraction
and the meaning of its raw fields; `osm-worldcover` owns canonicalization after
ingest, WorldCover labeling, geographic splits, and release auditing. Before
any sibling-repository comparison, the parent should approve a read-only review
and identify the maintainers for those source contracts. If comparison proves
that an algorithm is genuinely shared, its owners should agree on one existing
canonical owner and compatibility tests before any code is moved. No new
package should be introduced without that ownership decision, a stable API, and
an agreed release/versioning plan. The geoparser remains out of scope.

## Focused follow-up boundary

The source-column and fixture changes are follow-ups to the already merged PR
#8. They do not change PR #8's WorldCover algorithm, thresholds, source policy,
or release flow. The active data build continues to use the exact merged commit
above for provenance. Any further production cleanup should be a separate,
focused change after this review, with existing tests, CRAP `< 6`, and the
mutation floor preserved.

## Verification

- Full test suite with coverage: 572 passed.
- Ruff lint and formatting, `ty check src/`, and import-boundary checks passed.
- CRAP: 370 blocks measured, none at or above 6.0; the highest score was 5.93.
- Mutation gate: 84.4%, above the existing 80% floor.
- `mkdocs build --strict` passed. MkDocs reported pre-existing unlisted planning
  pages and the Material for MkDocs notice; neither caused the strict build to
  fail.
