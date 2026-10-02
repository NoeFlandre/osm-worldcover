# Dataset schema

The dataset has one row for each `(polygon, source text)` pair. All recipes use
the same published schema: Wikidata-linked documents, OSM `description` tags
and fetched website text.

## Label

| column | type | meaning |
| --- | --- | --- |
| `worldcover_code` | int | ESA WorldCover class (10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100) |
| `worldcover_label` | str | Class name that a person can read |
| `dominant_fraction` | float | Share of the polygon that the dominant class covers (>= threshold) |
| `observed_fraction` | float | Share of the polygon that the raster observed; < 1 means gaps |

## Text

| column | type | meaning |
| --- | --- | --- |
| `text` | str | Full source text, whitespace-normalised |
| `lead_text` | str | Lead paragraph of the article, when available |
| `title` | str | Name of the article or of the OSM object |
| `url` | str | Source URL, when available |
| `language` | str | Source language code, when available |
| `project` | str | `wikipedia`, `wikivoyage`, `description`, `website`, or `contact_website` |
| `text_words` | int | Number of whitespace-separated tokens |
| `document_id` | str | Stable document identity |

## Place

| column | type | meaning |
| --- | --- | --- |
| `polygon_id` | str | Stable identity of the source polygon |
| `osm_type`, `osm_id` | str, int | The OSM object |
| `name` | str | OSM name tag |
| `wikidata` | str | Wikidata QID, when present; null for the other recipes |
| `lat`, `lon` | float | Centroid of the polygon |
| `centroid_wkt` | str | Centroid as WKT POINT |
| `polygon_area_m2` | float | Area of the polygon |
| `region`, `source_pbf` | str | The extract that contains the polygon |

## Split and provenance

| column | type | meaning |
| --- | --- | --- |
| `split` | str | `train`, `validation` or `test` |
| `h3_cell` | str | H3 resolution-5 cell that decided the split |
| `dataset_version` | str | Version of this build |
| `source_dataset`, `source_revision` | str | Input dataset and pinned commit |
| `worldcover_version`, `worldcover_year` | str, int | Land-cover product that the build used |

## The eleven classes

| code | label |
| --- | --- |
| 10 | Tree cover |
| 20 | Shrubland |
| 30 | Grassland |
| 40 | Cropland |
| 50 | Built-up |
| 60 | Bare / sparse vegetation |
| 70 | Snow and ice |
| 80 | Permanent water bodies |
| 90 | Herbaceous wetland |
| 95 | Mangroves |
| 100 | Moss and lichen |

`0` is the no-data value of WorldCover. It is never a label.

## Processing ledger and code provenance

A new completion receipt records `context.code_revision`. This is the full Git
commit of the clean checkout that produced the region. Processing ledger schema
2 adds `code_provenance`. This is a list of exact repository commit references.
Each reference has the region stems that the commit produced. Schema 2 also
adds `assembly_code_revision`. This is the code that assembled and finalized the
release. The audit requires that the listed regions exactly partition the
complete inventory of processed regions. Mixed pins are therefore explicit in
`manifest.json` and in the dataset card.

Schema 1 ledgers stay readable for existing releases. Some older receipts do not
have `context.code_revision`. To assemble such receipts, the operator must give
the verified legacy commit with `owc assemble --legacy-code-revision
<40-char-sha>`. This produces a schema 2 ledger with mixed provenance. The
repository URL alone is not a code pin.
