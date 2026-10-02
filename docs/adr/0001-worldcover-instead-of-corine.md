# ADR 0001: Label with ESA WorldCover, not CORINE

**Status:** accepted — 2026-09-13

## Context

The dataset labels an OSM polygon with the land cover that covers at least 80%
of its area. The first choice was CORINE Land Cover 2018. It is the reference
product for Europe. It has a rich 44-class level-3 nomenclature.

A measurement of the source dataset decided the question. The table shows the
area of all 1,259,424 polygons:

| percentile | polygon area |
| --- | --- |
| p25 | 277 m² |
| p50 | 1,301 m² |
| p75 | 24,430 m² |

The **minimum mapping unit of CORINE is 25 ha (250,000 m²)**. Only **11.2%** of
the polygons reach it. The other 89% are fully inside one CORINE polygon. Their
dominant fraction is therefore 1.0 by construction. The 80% filter would accept
them and would test nothing.

## Decision

Label with **ESA WorldCover 2021 v200** at 10 m with 11 classes.

## Consequences

**Gained.** A 10 m pixel is 100 m². The median polygon covers about 13 pixels.
67% of the polygons cover four pixels or more. Dominance is then a real
discriminator. WorldCover is global. All 386 regions contribute and not only
Europe. For example, Andorra is fully outside the EEA-39 coverage of CORINE, but
this dataset labels it. WorldCover is published on open S3 and needs no account.
This removes a dependency on credentials.

**Lost.** WorldCover puts every settlement into one class, `Built-up`. CORINE
gives eleven artificial classes (urban fabric, industrial, ports, airports,
mines, sport facilities and others). Many Wikipedia-linked polygons *are*
buildings. This loses the distinction in which the source data is richest. The
task is now land **cover** and not land **use**.

## Alternatives considered

- **CLC+ Backbone 2021** has 10 m and 11 classes. It gives no taxonomic gain
  over WorldCover. It covers EEA-38 only and needs a Copernicus login.
- **CORINE at level 1 or 2** has coarser classes. They increase the pass rate
  but do not fix the cause. A polygon of 1,301 m² still cannot resolve a unit
  of 25 ha.
- **Both labels side by side** is attractive. The disagreement between them
  would be a useful quality signal. The project rejected it for the first
  release because it enlarges the scope. See `docs/technical-debt.md`.
