from typing import Final

import duckdb

from osm_worldcover.domain.nomenclature import CLASS_LABELS
from osm_worldcover.domain.text_diagnostics import group_counts_sql, retained_text_counts

from .report import AuditProblem
from .schema import _SPLITS

# DuckDB's working-memory budget for the release audit's aggregate checks.
_AUDIT_MEMORY_LIMIT: Final = "256MB"


def check_aggregates(paths, hashes, scratch, manifest, checks, strict_text_leakage):
    connection = duckdb.connect()
    try:
        connection.execute(f"SET memory_limit = '{_AUDIT_MEMORY_LIMIT}'")
        connection.execute("SET threads = 2")
        connection.execute("SET preserve_insertion_order = false")
        connection.execute("SET temp_directory = ?", [str(scratch / "spill")])
        connection.read_parquet([str(path) for path in paths]).create_view("release")
        connection.read_parquet(str(hashes)).create_view("hashes")
        _check_global(connection, checks, strict_text_leakage)
        _check_manifest(connection, manifest, checks)
    finally:
        connection.close()


def _sql_problem(connection, checks, code, query, warning=False) -> None:
    count = connection.execute(f"SELECT count(*) FROM ({query}) offenders").fetchone()[0]
    if not count:
        return
    samples = tuple(
        str(row[0]) for row in connection.execute(f"{query} ORDER BY 1 LIMIT 5").fetchall()
    )
    problem = AuditProblem(code, count, samples)
    target = checks.report.warnings if warning else checks.report.problems
    target.append(problem)


def _check_global(connection, checks, strict_text_leakage) -> None:
    for key in ("polygon_id", "document_id", "h3_cell"):
        query = f"SELECT {key} FROM release GROUP BY {key} HAVING count(DISTINCT split) > 1"
        _sql_problem(connection, checks, f"{key}_leakage", query)
    queries = {
        "object_identity_conflict": (
            "SELECT osm_type || ':' || osm_id FROM release GROUP BY osm_type, osm_id "
            "HAVING count(DISTINCT polygon_id) > 1"
        ),
        "inconsistent_polygon": (
            "SELECT polygon_id FROM release GROUP BY polygon_id "
            "HAVING count(DISTINCT (worldcover_code, lat, lon, polygon_area_m2, h3_cell)) > 1"
        ),
        "duplicate_polygon_text_label_record": (
            "SELECT polygon_id || ':' || hex(text_hash) || ':' || cast(worldcover_code AS varchar) "
            "FROM hashes GROUP BY polygon_id, text_hash, worldcover_code "
            "HAVING count(*) > 1"
        ),
    }
    for code, query in queries.items():
        _sql_problem(connection, checks, code, query)
    _sql_problem(
        connection,
        checks,
        "identical_text_cross_split",
        "SELECT hex(text_hash) FROM hashes GROUP BY text_hash HAVING count(DISTINCT split) > 1",
        warning=not strict_text_leakage,
    )
    _sql_problem(
        connection,
        checks,
        "identical_text_conflicting_labels",
        (
            "SELECT hex(text_hash) FROM hashes GROUP BY text_hash "
            "HAVING count(DISTINCT worldcover_code) > 1"
        ),
        warning=True,
    )


def _check_manifest(connection, manifest, checks) -> None:
    for key, expression in [
        ("examples", "count(*)"),
        ("polygons", "count(DISTINCT polygon_id)"),
        ("documents", "count(DISTINCT document_id)"),
    ]:
        rows = dict(
            connection.execute(f"SELECT split, {expression} FROM release GROUP BY split").fetchall()
        )
        counts = {split: rows.get(split, 0) for split in _SPLITS}
        counts["total"] = sum(counts.values())
        if manifest.get("counts", {}).get(key) != counts:
            checks.add(f"manifest_count_mismatch:{key}", counts)
    _check_distributions(connection, manifest, checks)
    _check_coverage(connection, manifest, checks)
    _check_text_diagnostics(connection, manifest, checks)
    if checks.report.rows == 0:
        checks.add("empty_dataset")


def _check_text_diagnostics(connection, manifest, checks) -> None:
    analysis = manifest.get("deduplication_analysis", {})
    if not analysis:
        return
    actual = _retained_text_diagnostics(connection)
    keys = (
        "retained_identical_text_label_groups",
        "retained_identical_text_label_rows",
        "retained_identical_text_label_cross_split_groups",
        "retained_identical_text_label_cross_split_rows",
        "identical_text_cross_split_groups",
        "identical_text_cross_split_rows",
    )
    if any(analysis.get(key) != actual[key] for key in keys):
        checks.add("deduplication_analysis_mismatch:retained_text")


def _retained_text_diagnostics(connection) -> dict[str, int]:
    label = connection.execute(
        group_counts_sql("hashes", ["text_hash", "worldcover_code"])
    ).fetchone()
    text = connection.execute(group_counts_sql("hashes", ["text_hash"])).fetchone()
    return retained_text_counts(label, text)


def _check_distributions(connection, manifest, checks) -> None:
    for section, key, column in [
        ("class_distribution", "code", "worldcover_code"),
        ("language_distribution", "language", "language"),
    ]:
        _check_distribution(connection, manifest, checks, section, key, column)
    _check_class_labels(manifest, checks)


def _check_distribution(connection, manifest, checks, section, key, column) -> None:
    expected = dict(
        connection.execute(f"SELECT {column}, count(*) FROM release GROUP BY {column}").fetchall()
    )
    entries = manifest.get(section, [])
    actual = {entry.get(key): entry.get("examples") for entry in entries}
    if actual != expected or len(entries) != len(expected):
        checks.add(f"manifest_mismatch:{section}")
    _check_shares(entries, checks)


def _check_class_labels(manifest, checks) -> None:
    for entry in manifest.get("class_distribution", []):
        if CLASS_LABELS.get(entry.get("code")) != entry.get("label"):
            checks.add("manifest_class_label_mismatch")


def _check_shares(entries, checks) -> None:
    for entry in entries:
        expected = (
            round(entry.get("examples", 0) / checks.report.rows, 6) if checks.report.rows else 0
        )
        if entry.get("share") != expected:
            checks.add("manifest_share_mismatch")


def _check_coverage(connection, manifest, checks) -> None:
    row = _coverage_aggregates(connection)
    _check_geographic_coverage(row, manifest, checks)
    _check_dominant_quantiles(row, manifest, checks)


def _coverage_aggregates(connection):
    return connection.execute(
        "SELECT count(DISTINCT h3_cell), count(DISTINCT region), "
        "min(lon), min(lat), max(lon), max(lat), "
        "quantile_cont(dominant_fraction, [0.5, 0.9, 0.99]) FROM release"
    ).fetchone()


def _check_geographic_coverage(row, manifest, checks) -> None:
    expected = {
        "h3_cells": row[0],
        "regions": row[1],
        "bbox": dict(zip(("min_lon", "min_lat", "max_lon", "max_lat"), row[2:6], strict=True)),
    }
    if manifest.get("geographic_coverage") != expected:
        checks.add("manifest_geographic_coverage_mismatch", expected)


def _check_dominant_quantiles(row, manifest, checks) -> None:
    quantiles = {
        key: round(value, 6)
        for key, value in zip(("p50", "p90", "p99"), row[6] or [], strict=False)
    }
    if manifest.get("dominant_fraction") != quantiles:
        checks.add("manifest_quantile_mismatch", quantiles)
