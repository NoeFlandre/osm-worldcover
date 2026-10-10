"""Manifest counts for the kept rows, each computed in SQL over the open connection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from osm_worldcover.domain.manifest import DatasetCounts, GeographicCoverage

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection


def _aggregate(
    connection: Any,
    rejections: dict[str, int],
    dropped: dict[str, int],
    deduplication_analysis: dict[str, Any],
) -> DatasetCounts:
    """Compute every manifest number in SQL, so no frame is ever built."""
    quantiles = connection.execute(
        "SELECT quantile_cont(dominant_fraction, [0.5, 0.9, 0.99]) FROM kept"
    ).fetchone()[0]
    bbox = connection.execute("SELECT min(lon), min(lat), max(lon), max(lat) FROM kept").fetchone()
    return DatasetCounts(
        examples=_by_split(connection, "count(*)"),
        polygons=_by_split(connection, "count(DISTINCT polygon_id)"),
        documents=_by_split(connection, "count(DISTINCT document_id)"),
        class_distribution=_tally(connection, "worldcover_code", int),
        language_distribution=_tally(connection, "language", _language_key),
        example_polygons=_example_polygons(connection),
        dominant_fraction_quantiles={
            "p50": round(float(quantiles[0]), 6),
            "p90": round(float(quantiles[1]), 6),
            "p99": round(float(quantiles[2]), 6),
        },
        coverage=GeographicCoverage(
            h3_cells=_count(connection, "SELECT count(DISTINCT h3_cell) FROM kept"),
            bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
            regions=_count(connection, "SELECT count(DISTINCT region) FROM kept"),
        ),
        rejections=dict(sorted(rejections.items())),
        deduplication=dict(sorted(dropped.items())),
        deduplication_analysis=deduplication_analysis,
    )


def _example_polygons(connection: DuckDBPyConnection) -> list[dict[str, str]]:
    """Choose one stable named polygon for every represented ESA class.

    Small compatibility fixtures and older intermediate shards may omit the
    optional OSM name column; those builds simply have no card examples.
    """
    columns = {str(row[0]) for row in connection.execute("DESCRIBE kept").fetchall()}
    if "name" not in columns:
        return []
    rows = connection.execute(
        """
        SELECT trim(name), worldcover_label
        FROM kept
        WHERE name IS NOT NULL AND length(trim(name)) > 0
        QUALIFY row_number() OVER (
            PARTITION BY worldcover_code
            ORDER BY lower(trim(name)), polygon_id, document_id
        ) = 1
        ORDER BY worldcover_code
        """
    ).fetchall()
    return [{"name": str(name), "worldcover_label": str(label)} for name, label in rows]


def _by_split(connection: DuckDBPyConnection, expression: str) -> dict[str, int]:
    """Evaluate ``expression`` per split."""
    rows = connection.execute(f"SELECT split, {expression} FROM kept GROUP BY 1").fetchall()
    return {str(split): int(value) for split, value in rows}


def _language_key(value: Any) -> str | None:
    """Keep an absent language absent.

    Most OSM ``description`` tags declare no language, and ``str(None)`` would
    publish the literal "None" as if it were a language code.
    """
    return None if value is None else str(value)


def _tally(connection: DuckDBPyConnection, column: str, cast: Any) -> dict[Any, int]:
    """Count rows per distinct value of ``column``."""
    rows = connection.execute(
        f"SELECT {column}, count(*) FROM kept GROUP BY 1 ORDER BY 1"
    ).fetchall()
    return {cast(value): int(n) for value, n in rows}


def _count(connection: DuckDBPyConnection, sql: str) -> int:
    """Run a counting query, refusing the empty result a count cannot produce."""
    row = connection.execute(sql).fetchone()
    if row is None:
        raise RuntimeError(f"count query returned no row: {sql}")
    return int(row[0])
