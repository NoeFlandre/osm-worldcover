# Duplication review — 2026-09-30

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

## Focused follow-up boundary

The changes to the source columns and fixtures are follow-ups to the merged PR
#8. They do not change the WorldCover algorithm, the thresholds, the source
policy or the release flow of PR #8. The active data build continues to use the
exact merged commit above for provenance. Make any further production cleanup
in a separate, focused change after this review. Keep the existing tests, CRAP
`< 6` and the mutation floor.

## Verification

- Full test suite with coverage: 572 passed.
- Ruff lint and formatting, `ty check src/`, and import-boundary checks passed.
- CRAP: the check measured 370 blocks. None has a score of 6.0 or more. The
  highest score was 5.93.
- Mutation gate: 84.4%. This is above the existing 80% floor.
- `mkdocs build --strict` passed. MkDocs reported unlisted planning pages that
  existed before this review. It also showed the Material for MkDocs notice.
  Neither caused the strict build to fail.
