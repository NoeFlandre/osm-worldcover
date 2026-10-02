# Glossary

This page defines the project terms. Each term has one meaning in all the documents.

| Term | Meaning |
| --- | --- |
| polygon | A closed area shape from OpenStreetMap (OSM). |
| source text | The text that is linked to a polygon. |
| example | One row. It pairs one polygon with one source text. |
| recipe | A named input source: `wikidata`, `description` or `website`. |
| WorldCover class | One of the 11 land-cover types of ESA WorldCover. |
| no-data | The WorldCover value `0`. It marks ground that was not observed. It is never a label. |
| tile | One WorldCover raster file (GeoTIFF) that covers 3 by 3 degrees. |
| share | The part of a polygon that one class covers. |
| dominant class | The class with the highest share. A polygon is accepted only when this share is at or above the threshold. |
| threshold | The minimum dominant share. The default value is 0.8. |
| region | One Geofabrik extract. A build processes one region at a time. |
| shard | The output file of one region. |
| H3 cell | A hexagonal map cell (H3 resolution 5). All rows in one cell go to the same split. |
| split | One of `train`, `validation` or `test`. |
| build | One run of `owc build`. |
| release | The final dataset that `owc assemble` and `owc publish` produce. |
| manifest | The file `manifest.json`. It records the counts, settings and provenance of a release. |
| card | The generated dataset description page on Hugging Face. |
| ADR | Architecture Decision Record. It describes one design decision. |
| technical debt | A known weakness that the project accepts for now. |
