# ADR 0003: Split on H3 cells, not on rows

**Status:** accepted — 2026-09-13

## Context

One place often has many articles. For example, the same monument can have
twelve languages. Neighbouring places have the same surroundings and so the same
land cover. A random split of the examples can put the English article about a
church in train and the German article about the same church in test. A model
can then get a good score because it recognises the place. It does not need to
read the text.

## Decision

Give each example an **H3 resolution-5 cell** (about 252 km², about 8 km across)
from its centroid. Assign the *cell* to a split. Hash the cell id with a seed.
All rows in a cell go to the same split.

The ratios (80/10/10) are targets for **cells** and not for rows. The realised
row counts therefore differ from the ratios. The difference is larger for
geographically concentrated builds.

## Consequences

- No polygon and no document can be in two splits. `domain/validation.py`
  checks both as published guarantees. The project does not only assume them.
- The same OSM object can be in two overlapping Geofabrik extracts. It has the
  same centroid, the same cell and the same split. Regional overlap therefore
  cannot leak, also before de-duplication removes the overlap.
- The assignment depends only on the cell id and the seed. It is reproducible.
  It does not depend on the iteration order or on the number of rows in a cell.
- **Remaining weakness:** Two polygons that are one metre apart on opposite
  sides of a cell boundary can be in different splits. Resolution 5 limits this
  to a small perimeter effect. It does not remove it. See
  `docs/technical-debt.md`.
