# WorldCover centroid coverage map

## Goal

Add a static world map to the Hugging Face dataset card for
`NoeFlandre/osm-wikidata-worldcover`. The map must show the geographic coverage
of the published examples. It shows one point for each distinct `polygon_id`
that has an ESA WorldCover label in the release.

The tabular layout of the dataset and its three Dataset Viewer splits do not
change.

## Chosen approach

Generate a release asset with the name `worldcover_centroids.png`. Use the exact
`train.parquet`, `validation.parquet` and `test.parquet` files. Do this
immediately before publication. The generator does these steps:

1. It reads only `polygon_id`, `lat`, `lon`, `worldcover_code` and
   `worldcover_label` from the three release Parquets.
2. It collapses the repeated article rows to one deterministic record for each
   polygon.
3. It fails if a polygon has conflicting ESA codes or labels, missing
   coordinates, or coordinates outside the longitude and latitude bounds.
4. It draws the points over a world land outline. Each of the 11 ESA WorldCover
   classes has a stable color. The map has a compact legend. It has an explicit
   caption that says the points are polygon centroids and not polygon
   boundaries.
5. It writes the PNG into the build directory. The existing folder upload then
   sends it with the card and the Parquet files.

The card renderer adds a coverage-map section after the introduction. The
section links to `worldcover_centroids.png`. It states the number of unique
polygons. It explains the centroid representation. The section also shows one
deterministic named polygon for each ESA class in the release. A compact table
shows the polygon name and the associated WorldCover class. The manifest stores
the examples. The card is then a pure function of the release metadata. The
columns `lat`, `lon` and `centroid_wkt` stay documented as the queryable
location fields.

## Dataset Viewer compatibility

The map is only a card asset. It is not a split and it is not a column. The
published repository therefore keeps the existing `default` configuration and
the `train`, `validation` and `test` Parquets. After the upload, check the
public Dataset Viewer API for these items:

- a valid dataset response with no processing error;
- exactly the three expected splits;
- non-empty first rows and row pages for each split;
- the expected schema and scalar column types; and
- available Parquet and statistics metadata.

Any Viewer failure is a release failure. This is true also if the Hub upload
succeeds.

## Testing

Unit tests cover these items: centroid aggregation and validation, deterministic
class colors and legend inputs, representative-card examples, map-card rendering
and publisher integration. The tests use small local Parquet fixtures and a
patched local boundary source. They do not need the network or the full
release.

The release verification also inspects the generated PNG. It runs the public
Dataset Viewer checks against the uploaded repository.

## Non-goals

- Do not add polygon geometries or a new Dataset Viewer split.
- Do not draw true polygon boundaries in the card image.
- Do not change the labels, the row contents, the split assignment or the
  manifest counts.
- Do not add an interactive JavaScript map. Its behavior would depend on the
  sanitization of the Hub card.
