"""Read back release files and independently check publication guarantees.

Python keeps only one Arrow batch and bounded diagnostic samples. Global
identity checks run in DuckDB with a memory limit and disk-backed spill space.
The input files are never modified. Passing this audit does not independently
recompute raster labels, or prove that excluded source polygons were processed.
"""

import hashlib
import json
import math
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import h3
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from osm_worldcover.config import DEDUPLICATION_POLICY
from osm_worldcover.domain.nomenclature import CLASS_LABELS
from osm_worldcover.domain.splits import SplitRatios, assign_cell

__all__ = ["AuditProblem", "AuditReport", "audit_build"]

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


@dataclass(frozen=True, slots=True)
class AuditProblem:
    """A failed guarantee, with at most five diagnostic examples."""

    code: str
    count: int
    examples: tuple[str, ...] = ()


@dataclass(slots=True)
class AuditReport:
    """All detected problems and separately identified modelling risks."""

    rows: int = 0
    problems: list[AuditProblem] = field(default_factory=list)
    warnings: list[AuditProblem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether every required check passed."""
        return not self.problems

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible report."""
        return {"ok": self.ok, **asdict(self)}


class _Checks:
    def __init__(self, report: AuditReport):
        self.report = report
        self.counts: Counter[str] = Counter()
        self.samples: defaultdict[str, list[str]] = defaultdict(list)

    def add(self, code: str, example: object = "", count: int = 1) -> None:
        self.counts[code] += count
        if example != "" and len(self.samples[code]) < 5:
            self.samples[code].append(str(example))

    def finish(self) -> None:
        self.report.problems.extend(
            AuditProblem(code, n, tuple(self.samples[code]))
            for code, n in sorted(self.counts.items())
            if n
        )


def audit_build(
    build_dir: Path,
    *,
    require_complete: bool = False,
    require_card: bool = False,
    strict_text_leakage: bool = False,
) -> AuditReport:
    """Audit every release row and reconcile its exact manifest aggregates.

    Completion is a distinct processing-ledger claim, never inferred from a
    non-empty or internally consistent release. Legacy builds can be audited
    with ``require_complete=False``. Text collisions spanning different labels
    remain visible as warnings; ``strict_text_leakage`` makes cross-split text
    identity fatal even when the labels or document identities differ.
    """
    report = AuditReport()
    checks = _Checks(report)
    build_dir = Path(build_dir)
    manifest = _load_manifest(build_dir, checks)
    paths = _inspect_files(build_dir, checks)
    if manifest is None or checks.counts:
        checks.finish()
        return report
    settings = manifest.get("settings", {})
    _check_settings(settings, checks)
    _check_completion(manifest, require_complete, checks)
    if require_card:
        _check_card(build_dir, settings, checks)
    if checks.counts.get("invalid_settings"):
        checks.finish()
        return report
    with tempfile.TemporaryDirectory(prefix="owc-audit-") as scratch:
        _audit_contents(paths, Path(scratch), manifest, checks, strict_text_leakage)
    checks.finish()
    return report


def _load_manifest(build_dir: Path, checks: _Checks) -> dict[str, Any] | None:
    try:
        manifest = json.loads((build_dir / "manifest.json").read_text())
    except (OSError, ValueError) as error:
        checks.add("missing_or_invalid_manifest", error)
        return None
    if not isinstance(manifest, dict) or not _valid_manifest_shape(manifest):
        checks.add("missing_or_invalid_manifest")
        return None
    return manifest


def _valid_manifest_shape(manifest: dict) -> bool:
    """Reject malformed JSON structures before row and aggregate checks."""
    required = ("settings", "counts", "geographic_coverage", "dominant_fraction")
    optional = ("processing", "deduplication_analysis")
    distributions = ("class_distribution", "language_distribution")
    return (
        _valid_dict_fields(manifest, required)
        and _valid_dict_fields(manifest, optional, default={})
        and all(_valid_distribution(manifest.get(key)) for key in distributions)
    )


def _valid_dict_fields(manifest: dict, fields: tuple[str, ...], default=None) -> bool:
    return all(isinstance(manifest.get(key, default), dict) for key in fields)


def _valid_distribution(entries: object) -> bool:
    return isinstance(entries, list) and all(_valid_distribution_entry(entry) for entry in entries)


def _valid_distribution_entry(entry: object) -> bool:
    return isinstance(entry, dict) and isinstance(entry.get("examples"), int)


def _inspect_files(build_dir: Path, checks: _Checks) -> list[Path]:
    paths = [build_dir / f"{split}.parquet" for split in _SPLITS]
    for path in paths:
        _inspect_file(path, checks)
    return paths


def _inspect_file(path: Path, checks: _Checks) -> None:
    try:
        parquet = pq.ParquetFile(path)
        schema = {f.name: str(f.type) for f in parquet.schema_arrow}
        if schema != _SCHEMA:
            checks.add("schema_mismatch", path.name)
    except (OSError, pa.ArrowInvalid) as error:
        checks.add("missing_or_invalid_parquet", f"{path.name}: {error}")


def _check_settings(settings: dict[str, Any], checks: _Checks) -> None:
    try:
        SplitRatios(**settings["split_ratios"])
        valid = _valid_setting_values(settings)
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        checks.add("invalid_settings")
    if not re.fullmatch(r"[0-9a-f]{40}", str(settings.get("source_revision"))):
        checks.add("unpinned_source_revision")


def _valid_setting_values(settings: dict[str, Any]) -> bool:
    return _valid_processing_settings(settings) and _valid_release_settings(settings)


def _valid_processing_settings(settings: dict[str, Any]) -> bool:
    return (
        _valid_dominance_threshold(settings)
        and _valid_word_threshold(settings)
        and _valid_h3_resolution(settings)
        and isinstance(settings["split_seed"], int)
    )


def _valid_release_settings(settings: dict[str, Any]) -> bool:
    return settings.get(
        "deduplication_policy", DEDUPLICATION_POLICY
    ) == DEDUPLICATION_POLICY and _has_provenance_settings(settings)


def _has_provenance_settings(settings: dict[str, Any]) -> bool:
    return all(key in settings for key in _PROVENANCE)


def _valid_dominance_threshold(settings: dict[str, Any]) -> bool:
    return 0 <= settings["dominance_threshold"] <= 1


def _valid_word_threshold(settings: dict[str, Any]) -> bool:
    return isinstance(settings["min_words"], int) and settings["min_words"] >= 1


def _valid_h3_resolution(settings: dict[str, Any]) -> bool:
    return isinstance(settings["h3_resolution"], int) and 0 <= settings["h3_resolution"] <= 15


def _check_completion(manifest: dict, required: bool, checks: _Checks) -> None:
    if not required:
        return
    ledger = manifest.get("processing", {})
    if ledger.get("full_source_complete") is not True:
        checks.add("source_processing_incomplete")
        return
    try:
        _check_inventory(ledger)
        _check_context(ledger, manifest)
        _check_ledger_counts(ledger, manifest)
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        checks.add("invalid_processing_ledger", error)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _check_inventory(ledger: dict) -> None:
    names = ("expected", "selected", "processed")
    inventories = [ledger[f"{name}_regions"] for name in names]
    for name, regions in zip(names, inventories, strict=True):
        _check_one_inventory(name, regions, ledger)
    _require(bool(inventories[0]), "empty expected source inventory")
    _require(
        all(set(regions) == set(inventories[0]) for regions in inventories),
        "full source inventories differ",
    )
    _check_pending_inventory(ledger)


def _check_one_inventory(name: str, regions: list[Any], ledger: dict) -> None:
    _require(isinstance(regions, list), f"{name} inventory must be a list")
    _require(all(isinstance(value, str) and value for value in regions), "invalid region name")
    _require(len(regions) == len(set(regions)), f"duplicate {name} regions")
    _require(ledger["region_counts"][name] == len(regions), f"wrong {name} count")


def _check_pending_inventory(ledger: dict) -> None:
    for key in ("missing_regions", "unprocessed_selected_regions"):
        _require(ledger[key] == [], f"nonempty {key}")
    for key in ("missing", "unprocessed_selected"):
        _require(ledger["region_counts"][key] == 0, f"nonzero {key} count")
    _require(ledger["scope"] == "full", "subset scope cannot be published as complete")
    for key in ("complete", "selected_complete"):
        _require(ledger[key] is True, f"{key} is not true")


def _check_context(ledger: dict, manifest: dict) -> None:
    context = ledger["context"]
    digest = hashlib.sha256(
        json.dumps(context, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    _require(ledger["build_context_sha256"] == digest, "build context hash mismatch")
    _require(
        _processing_settings_match(context["settings"], manifest["settings"]),
        "build context settings mismatch",
    )
    _require(ledger["schema_version"] == 1, "unsupported processing ledger schema")


def _processing_settings_match(processing: dict, release: dict) -> bool:
    """Ignore fields assigned only when labelled shards become a release."""
    finalization_settings = {"dataset_version", "code_repository", "deduplication_policy"}
    keys = (set(processing) | set(release)) - finalization_settings
    return all(processing.get(key) == release.get(key) for key in keys)


_OUTCOME_COUNTS = (
    "polygons_seen",
    "polygons_invalid",
    "polygons_accepted",
    "polygons_with_examples",
    "source_links",
    "source_documents",
    "examples",
)


def _nonnegative_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validate_outcome_record(record: dict) -> None:
    for key in _OUTCOME_COUNTS:
        _require(_nonnegative_integer(record[key]), f"invalid region counter {key}")
    for key in ("rejections", "text_rejections"):
        _require(isinstance(record[key], dict), f"invalid {key} mapping")
        _require(
            all(_nonnegative_integer(value) for value in record[key].values()),
            f"invalid {key} counter",
        )
    _require(
        record["polygons_seen"]
        == record["polygons_invalid"]
        + record["polygons_accepted"]
        + sum(record["rejections"].values()),
        "spatial region accounting mismatch",
    )
    _require(
        record["polygons_accepted"]
        == record["polygons_with_examples"] + sum(record["text_rejections"].values()),
        "text region accounting mismatch",
    )
    _require(record["examples"] >= record["polygons_with_examples"], "insufficient region examples")


def _check_ledger_counts(ledger: dict, manifest: dict) -> None:
    _check_receipt_inventory(ledger)
    _check_region_totals(ledger)
    _check_ledger_rejections(ledger, manifest)
    _check_deduplication_totals(ledger, manifest)


def _check_receipt_inventory(ledger: dict) -> None:
    regions = ledger["regions"]
    stems = [record["stem"] for record in regions]
    _require(
        sorted(stems) == sorted(ledger["processed_regions"]), "region receipt inventory mismatch"
    )
    for record in regions:
        _validate_outcome_record(record)


def _check_region_totals(ledger: dict) -> None:
    regions = ledger["regions"]
    for key in _OUTCOME_COUNTS:
        _require(
            ledger["totals"][key] == sum(record[key] for record in regions),
            f"aggregate region count mismatch: {key}",
        )


def _check_deduplication_totals(ledger: dict, manifest: dict) -> None:
    drops = manifest["deduplication"]
    _require(
        all(_nonnegative_integer(value) for value in drops.values()),
        "invalid deduplication counters",
    )
    expected = manifest["counts"]["examples"]["total"] + sum(drops.values())
    _require(ledger["totals"]["examples"] == expected, "pre/post-dedup example counts mismatch")
    _check_deduplication_analysis(drops, manifest.get("deduplication_analysis", {}))


def _check_deduplication_analysis(drops: dict, analysis: dict) -> None:
    """Reconcile record-level removals and retained-text diagnostics."""
    if not analysis:
        return
    _check_deduplication_counters(analysis)
    by_length = _deduplication_length_counts(analysis)
    _require(
        sum(by_length.values()) == analysis["duplicate_records_removed"],
        "duplicate-record length counters do not reconcile",
    )
    _check_deduplication_cross_split(drops, analysis)


def _check_deduplication_counters(analysis: dict) -> None:
    keys = (
        "duplicate_polygon_text_label_groups",
        "duplicate_records_removed",
        "duplicate_record_groups_crossing_splits",
        "duplicate_records_removed_from_cross_split_groups",
        "retained_identical_text_label_groups",
        "retained_identical_text_label_rows",
        "retained_identical_text_label_cross_split_groups",
        "retained_identical_text_label_cross_split_rows",
        "identical_text_cross_split_groups",
        "identical_text_cross_split_rows",
    )
    _require(
        all(_nonnegative_integer(analysis.get(key)) for key in keys),
        "invalid duplicate-text analysis counters",
    )


def _deduplication_length_counts(analysis: dict) -> dict:
    by_length = analysis.get("duplicate_records_removed_by_text_words")
    _require(isinstance(by_length, dict), "invalid duplicate-record length counters")
    if not isinstance(by_length, dict):
        raise TypeError("invalid duplicate-record length counters")
    expected = {*(str(words) for words in range(1, 10)), "10+"}
    _require(
        set(by_length) == expected and all(_nonnegative_integer(v) for v in by_length.values()),
        "invalid duplicate-record length counters",
    )
    return by_length


def _check_deduplication_cross_split(drops: dict, analysis: dict) -> None:
    _require(
        analysis["duplicate_records_removed"] == drops.get("duplicate_polygon_text_label_records"),
        "duplicate-record analysis does not match deduplication total",
    )
    _require(
        analysis["duplicate_record_groups_crossing_splits"]
        <= analysis["duplicate_polygon_text_label_groups"]
        and analysis["duplicate_records_removed_from_cross_split_groups"]
        <= analysis["duplicate_records_removed"],
        "duplicate-record split counters are inconsistent",
    )


def _check_ledger_rejections(ledger: dict, manifest: dict) -> None:
    for key in ("rejections", "text_rejections"):
        counts: Counter[str] = Counter()
        for region in ledger["regions"]:
            counts.update(region[key])
        _require(dict(counts) == ledger["totals"][key], f"aggregate {key} mismatch")
    _require(
        manifest["rejections"] == ledger["totals"]["rejections"],
        "manifest rejection counters mismatch",
    )
    _require(ledger["reconciliation"]["valid"] is True, "ledger reconciliation not valid")


def _audit_contents(paths, scratch, manifest, checks, strict_text_leakage) -> None:
    hashes = scratch / "hashes.parquet"
    with pq.ParquetWriter(hashes, _HASH_SCHEMA, compression="zstd") as writer:
        for path in paths:
            _scan_file(path, manifest["settings"], checks, writer)
    connection = duckdb.connect()
    try:
        connection.execute("SET memory_limit = '256MB'")
        connection.execute("SET threads = 2")
        connection.execute("SET preserve_insertion_order = false")
        connection.execute("SET temp_directory = ?", [str(scratch / "spill")])
        connection.read_parquet([str(path) for path in paths]).create_view("release")
        connection.read_parquet(str(hashes)).create_view("hashes")
        _check_global(connection, checks, strict_text_leakage)
        _check_manifest(connection, manifest, checks)
    finally:
        connection.close()


def _scan_file(path, settings, checks, writer) -> None:
    columns = sorted(set(_SCHEMA) - _NULLABLE)
    for batch in pq.ParquetFile(path).iter_batches(batch_size=2048, columns=columns):
        fingerprints = []
        for row in batch.to_pylist():
            checks.report.rows += 1
            _check_row(row, path.stem, settings, checks)
            text = row["text"]
            if isinstance(text, str):
                fingerprints.append(
                    {
                        "text_hash": hashlib.sha256(" ".join(text.split()).encode()).digest(),
                        "worldcover_code": row["worldcover_code"],
                        "split": row["split"],
                        "polygon_id": row["polygon_id"],
                    }
                )
        writer.write_table(pa.Table.from_pylist(fingerprints, schema=_HASH_SCHEMA))


def _check_row(row, split, settings, checks) -> None:
    _check_required_values(row, checks)
    _check_row_identity(row, split, checks)
    _check_provenance(row, settings, checks)
    _check_numbers(row, settings, checks)
    _check_position(row, settings, checks)
    _check_text(row, settings, checks)


def _check_required_values(row, checks) -> None:
    pid = row["polygon_id"]
    for key in set(_SCHEMA) - _NULLABLE:
        if row[key] is None:
            checks.add(f"required_null:{key}", pid)


def _check_row_identity(row, split, checks) -> None:
    pid = row["polygon_id"]
    if row["split"] not in _SPLITS:
        checks.add("invalid_split", pid)
    if row["split"] != split:
        checks.add("file_split_mismatch", pid)
    if CLASS_LABELS.get(row["worldcover_code"]) != row["worldcover_label"]:
        checks.add("invalid_label", pid)


def _check_provenance(row, settings, checks) -> None:
    _check_provenance_values(row, settings, checks)
    _check_identifiers(row, checks)


def _check_provenance_values(row, settings, checks) -> None:
    for key in _PROVENANCE:
        if row[key] != settings[key]:
            checks.add(f"provenance_mismatch:{key}", row["polygon_id"])


def _check_identifiers(row, checks) -> None:
    for key in ("polygon_id", "document_id", "region", "source_pbf", "h3_cell"):
        if not isinstance(row[key], str) or not row[key].strip():
            checks.add(f"empty_identifier:{key}", row["polygon_id"])


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _check_numbers(row, settings, checks) -> None:
    _check_fractions(row, checks)
    _check_area(row, settings, checks)
    _check_dominance(row, settings, checks)


def _check_fractions(row, checks) -> None:
    pid = row["polygon_id"]
    for key in ("dominant_fraction", "observed_fraction"):
        value = row[key]
        if not _finite(value) or not 0 <= value <= 1 + 1e-6:
            checks.add(f"invalid_fraction:{key}", pid)


def _check_area(row, settings, checks) -> None:
    pid = row["polygon_id"]
    area = row["polygon_area_m2"]
    cap = settings.get("max_polygon_area_m2")
    if not _finite(area) or area <= 0:
        checks.add("invalid_area", pid)
    elif cap is not None and area > cap * (1 + 1e-6):
        checks.add("area_above_cap", pid)


def _check_dominance(row, settings, checks) -> None:
    dominant, observed = row["dominant_fraction"], row["observed_fraction"]
    _check_dominance_threshold(dominant, settings, row["polygon_id"], checks)
    _check_dominance_observation(dominant, observed, row["polygon_id"], checks)


def _check_dominance_threshold(dominant, settings, pid, checks) -> None:
    if _finite(dominant) and dominant < settings["dominance_threshold"] - 1e-12:
        checks.add("below_threshold", pid)


def _check_dominance_observation(dominant, observed, pid, checks) -> None:
    if _finite(dominant) and _finite(observed) and dominant > observed + 1e-6:
        checks.add("dominant_exceeds_observed", pid)


def _check_position(row, settings, checks) -> None:
    lat, lon = row["lat"], row["lon"]
    if not (_finite(lat) and _finite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
        checks.add("invalid_coordinates", row["polygon_id"])
        return
    _check_position_assignment(row, settings, checks, lat, lon)


def _check_position_assignment(row, settings, checks, lat, lon) -> None:
    cell = h3.latlng_to_cell(lat, lon, settings["h3_resolution"])
    if cell != row["h3_cell"]:
        checks.add("h3_cell_mismatch", row["polygon_id"])
    expected = assign_cell(cell, SplitRatios(**settings["split_ratios"]), settings["split_seed"])
    if expected.value != row["split"]:
        checks.add("h3_assignment_mismatch", row["polygon_id"])
    _check_centroid(row, checks)


def _check_centroid(row, checks) -> None:
    match = re.fullmatch(
        r"POINT\s*\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)", str(row["centroid_wkt"])
    )
    try:
        valid = match and math.isclose(float(match[1]), row["lon"], abs_tol=5.1e-8, rel_tol=0)
        valid = valid and math.isclose(float(match[2]), row["lat"], abs_tol=5.1e-8, rel_tol=0)
    except ValueError:
        valid = False
    if not valid:
        checks.add("centroid_wkt_mismatch", row["polygon_id"])


def _check_text(row, settings, checks) -> None:
    text, pid = row["text"], row["polygon_id"]
    if not isinstance(text, str):
        checks.add("invalid_text", pid)
        return
    words = text.split()
    if len(words) < settings["min_words"]:
        checks.add("unusable_text", pid)
    if text != " ".join(words):
        checks.add("unnormalized_text", pid)
    if row["text_words"] != len(words):
        checks.add("text_word_count_mismatch", pid)


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
    """Reconcile retained text-collision counts with the published rows."""
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
        """
        WITH grouped AS (
            SELECT text_hash, worldcover_code, count(*) AS row_count,
                   count(DISTINCT split) AS split_count
            FROM hashes GROUP BY text_hash, worldcover_code
        )
        SELECT count(*) FILTER (WHERE row_count > 1),
               coalesce(sum(row_count) FILTER (WHERE row_count > 1), 0),
               count(*) FILTER (WHERE row_count > 1 AND split_count > 1),
               coalesce(sum(row_count) FILTER (WHERE row_count > 1 AND split_count > 1), 0)
        FROM grouped
        """
    ).fetchone()
    text = connection.execute(
        """
        WITH grouped AS (
            SELECT text_hash, count(*) AS row_count, count(DISTINCT split) AS split_count
            FROM hashes GROUP BY text_hash
        )
        SELECT count(*) FILTER (WHERE split_count > 1),
               coalesce(sum(row_count) FILTER (WHERE split_count > 1), 0)
        FROM grouped
        """
    ).fetchone()
    return {
        "retained_identical_text_label_groups": int(label[0]),
        "retained_identical_text_label_rows": int(label[1]),
        "retained_identical_text_label_cross_split_groups": int(label[2]),
        "retained_identical_text_label_cross_split_rows": int(label[3]),
        "identical_text_cross_split_groups": int(text[0]),
        "identical_text_cross_split_rows": int(text[1]),
    }


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


def _check_card(build_dir, settings, checks) -> None:
    try:
        text = (build_dir / "README.md").read_text()
        metadata = yaml.safe_load(text.split("---", 2)[1])
        expected = [
            {
                "config_name": "default",
                "data_files": [{"split": split, "path": f"{split}.parquet"} for split in _SPLITS],
            }
        ]
        if metadata.get("configs") != expected:
            checks.add("card_data_files_mismatch")
        if metadata.get("license") != settings.get("dataset_license", "cc-by-sa-4.0"):
            checks.add("card_license_mismatch")
        _check_card_assets(build_dir, text, settings, checks)
    except (OSError, ValueError, IndexError, AttributeError, yaml.YAMLError) as error:
        checks.add("invalid_card_or_map", error)


def _check_card_assets(build_dir, text, settings, checks) -> None:
    for key in ("source_revision", "source_dataset"):
        if str(settings.get(key)) not in text:
            checks.add(f"card_missing_provenance:{key}")
    if "worldcover_centroids.png" not in text:
        checks.add("card_missing_coverage_map")
    with (build_dir / "worldcover_centroids.png").open("rb") as source:
        if source.read(8) != b"\x89PNG\r\n\x1a\n":
            checks.add("invalid_coverage_map_png")
