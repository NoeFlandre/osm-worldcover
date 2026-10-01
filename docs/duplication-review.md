# Duplication review — 2026-10-01

## Scope and method

This refresh scanned all 62 Python files under `src/` and `tests/` with
Pylint 4.1.1's `duplicate-code` checker at its four-line minimum. It also
reviewed test setup and the production matches. This is a targeted clone scan;
it does not establish that the repository contains no duplication.

The 2026-09-30 baseline on merged main `3ddd472e7deac10d116fd763cbf612e0d1a9c8db`
found five clone reports and three local copies of the fixed-tile test source.
The follow-up centralizes the repeated table contract in
`adapters.source_profiles` and shares a recording `FixedTiles` test double in
`tests/conftest.py`. The normalized document table still appends its `project`
column, and the Wikidata document reader still returns the same base columns
before its caller adds that value.

The current PR13 scan reports exactly three clone matches, all independent
reader/writer contracts below. It found no duplicated test setup. The test
helpers that remain local assemble different fixtures: the acceptance `World`
builder and its polygon/document helpers create BDD cases, while source-recipe
tests write different upstream row layouts. The unit `polygons_frame` helper
creates canonical pipeline inputs and is not equivalent to those source files.

## Intentional remaining matches

The three reports comprise two schema overlaps and one counter-list overlap:

- `pipeline.OUTPUT_COLUMNS` and `pipeline.TEXT_COLUMNS` overlap with the
  independently declared `_STRINGS` release schema in `adapters.audit`. The
  auditor reads finalized files and must detect a producer schema regression;
  it also checks provenance, split, and H3 columns added after pipeline output.
- `accounting.COUNT_FIELDS` and `adapters.audit._OUTCOME_COUNTS` name the same
  serialized counters. The audit validates the receipt written by accounting,
  so its expected field list stays independent and can detect a missing or
  changed writer field.

These are deliberate duplicated contracts, not shared processing logic. No
zero-duplication claim is made.

## Cross-repository candidates and ownership plan

No sibling repository was inspected for this review. These are candidates
inferred from this repository's source recipes and adapters, not confirmed
cross-repository duplicates:

- OSM polygon identity construction and the canonical polygon/link/document
  contract.
- Source text cleanup and conversion of description, website, and article
  fields into canonical documents.
- Region inventory pinning and provenance receipts around per-region builds.

Each source repository should own extraction and the meaning of its raw
fields. `osm-worldcover` owns canonicalization after ingest, WorldCover
labeling, geographic splits, and release auditing. Before comparing sibling
repositories, the parent should approve a read-only review and identify the
maintainers for those source contracts. If an algorithm is genuinely shared,
its owners should agree on one existing canonical owner and compatibility
tests before code moves. No new package should be introduced without that
ownership decision, a stable API, and an agreed release/versioning plan. The
geoparser remains out of scope.

## Recursive callable and CRAP inventory

Radon's JSON output omits methods inside nested classes and classes local to
functions. `scripts/crap.py` now walks each Python AST and measures each
`FunctionDef` and `AsyncFunctionDef` once, including nested functions and
methods in nested/local classes. Radon measures each callable independently;
coverage for an enclosing callable excludes child callable and local-class
bodies, so nested code cannot dilute its score. Empty inventories still fail.

The full `src/`, `scripts/`, and `tests/` inventory measured 1,053 callables.
The highest CRAP score was 5.58; zero scores were at or above 6, with no
allowlisted exceptions.

## Verification

- Full unit, property, and acceptance test suite passed with coverage enabled.
- Ruff lint and formatting, `ty check src/`, and import-boundary checks passed.
- Strict CRAP gate passed across `src/`, `scripts/`, and `tests/` as reported
  above.
- `mkdocs build --strict` passed. It reported the existing three unlisted
  planning pages and the Material for MkDocs notice; neither caused the build
  to fail.
- Mutation scope and the 80% floor remain configured in CI; the PR check run
  is the source of truth for each candidate commit.
