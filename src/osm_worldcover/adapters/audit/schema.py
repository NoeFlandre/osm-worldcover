import pyarrow as pa

_SPLITS = ("train", "validation", "test")


_STRINGS = [
    "polygon_id",
    "osm_type",
    "region",
    "name",
    "wikidata",
    "document_id",
    "project",
    "language",
    "title",
    "url",
    "text",
    "lead_text",
    "worldcover_label",
    "centroid_wkt",
    "source_pbf",
    "h3_cell",
    "split",
    "dataset_version",
    "source_dataset",
    "source_revision",
    "worldcover_version",
]


_INTS = ["osm_id", "text_words", "worldcover_code", "worldcover_year"]


_FLOATS = ["dominant_fraction", "observed_fraction", "lat", "lon", "polygon_area_m2"]


_SCHEMA = {
    **dict.fromkeys(_STRINGS, "string"),
    **dict.fromkeys(_INTS, "int64"),
    **dict.fromkeys(_FLOATS, "double"),
}


_NULLABLE = {"name", "wikidata", "language", "title", "url", "lead_text"}


_PROVENANCE = (
    "dataset_version",
    "source_dataset",
    "source_revision",
    "worldcover_version",
    "worldcover_year",
)


_HASH_SCHEMA = pa.schema(
    [
        ("text_hash", pa.binary(32)),
        ("worldcover_code", pa.int64()),
        ("split", pa.string()),
        ("polygon_id", pa.string()),
    ]
)
