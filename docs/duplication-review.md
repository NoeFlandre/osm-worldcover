# Duplication review — 2026-10-01

## Scope and method

This review used the exact merged tree at
`3ddd472e7deac10d116fd763cbf612e0d1a9c8db`. The review scanned all 61 Python
files under `src/` and `tests/`. It used the `duplicate-code` checker of Pylint
4.1.1 with a minimum of four lines. Then a person reviewed the shared fixtures
and the reported production matches. This is a targeted clone scan. It does not
prove that the repository has no duplication.

The baseline scan found five clone reports. Two reports were repeated canonical
table column declarations. A manual review of the test setup found the same
fixed raster tile source in three test modules. The follow-up moves the table
contract to `adapters.source_profiles`. It also shares one recording `FixedTiles`
test double in `tests/conftest.py`. The normalized document table still appends
its `project` column. The Wikidata document reader still returns the same base
columns before its caller adds that value.

The same scan now reports three matches. The project keeps all three as
independent reader and writer contracts. The sections below describe them. The
consolidation of the test double removes the three local copies. It does not
change what `ensure` returns. It does not change the cache calls that the unit
tests observe.

## Intentional remaining matches

- `pipeline.OUTPUT_COLUMNS` and `pipeline.TEXT_COLUMNS` overlap with the release
  schema that `adapters.audit` declares separately. The auditor reads the
  finalized files. It must be able to catch an error in the producer schema. It
  also checks the provenance columns, the split columns and the H3 columns. The
  build adds these columns after the pipeline output.
- `accounting.COUNT_FIELDS` and `adapters.audit._OUTCOME_COUNTS` name the same
  serialized counters. The audit validates the receipt that accounting writes.
  Its expected field list stays independent. The audit can then detect a
  missing or changed writer field.

These matches are deliberate duplicated contracts. They are not shared
processing logic. The remaining clone reports are only these declarations. The
project does not claim zero duplication.

Some test setup stays local because it is specific to a scenario. The
acceptance `World` builder and its polygon and document helpers assemble BDD
cases. The source-recipe tests write different upstream row layouts. The unit
`polygons_frame` helper creates canonical pipeline inputs. It is not equivalent
to those source files.

## Cross-repository candidates and ownership plan

No person inspected a sibling repository for this review. The items below are
candidates. They come from the source recipes and the adapters of this
repository. They are not confirmed cross-repository duplicates:

- The construction of the OSM polygon identity, and the canonical
  polygon/link/document contract.
- The cleanup of source text, and the conversion of description, website and
  article fields into canonical documents.
- The pinning of the region inventory, and the provenance receipts around the
  builds of each region.

The proposed ownership boundary is as follows. Each source repository owns the
extraction and the meaning of its raw fields. `osm-worldcover` owns the
canonicalization after ingest, the WorldCover labelling, the geographic splits
and the release audit. Before any comparison with a sibling repository, the
parent must approve a read-only review. The parent must also name the
maintainers of those source contracts. If a comparison proves that an algorithm
is really shared, the owners must agree on one existing canonical owner. They
must also agree on compatibility tests. Only then can they move any code. Do not
add a new package without three items: an ownership decision, a stable API and
an agreed release and versioning plan. The geoparser stays out of scope.

## Recursive callable and CRAP inventory

Radon's JSON output omits methods inside nested classes and classes local to
functions. `scripts/crap.py` now walks each Python AST and measures each
`FunctionDef` and `AsyncFunctionDef` once, including nested functions and
methods in nested/local classes. Radon measures each callable independently;
coverage for an enclosing callable excludes child callable and local-class
bodies, so nested code cannot dilute its score. Empty inventories still fail.

The full `src/`, `scripts/`, and `tests/` inventory measured 1,346 callables.
The highest CRAP score was 5.58; zero scores were at or above 6, with no
allowlisted exceptions.

A refresh of the clone scan on the PR13 tree reports the same three matches.
It found no new duplication.

## Verification

- Full unit, property, and acceptance test suite passed with coverage enabled.
- Ruff lint and formatting, `ty check src/`, and import-boundary checks passed.
- Strict CRAP gate passed across `src/`, `scripts/`, and `tests/` as reported
  above.
- `mkdocs build --strict` passed. It reported the existing three unlisted
  planning pages and the Material for MkDocs notice; neither caused the build
  to fail.
- Mutation gate: a full local run generated 4,969 mutants. It killed 4,529
  (91.1%), above the 80% floor. Every mutant had a result. The CI check run
  is the source of truth for each candidate commit.
- Mutation scope and the 80% floor remain configured in CI; the PR check run
  is the source of truth for each candidate commit.
