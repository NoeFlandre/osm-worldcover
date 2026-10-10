import hashlib
import math
import re
from pathlib import Path

import h3
import pyarrow as pa
import pyarrow.parquet as pq

from osm_worldcover.domain.nomenclature import CLASS_LABELS
from osm_worldcover.domain.splits import SplitRatios, assign_cell

from .report import _Checks
from .schema import _HASH_SCHEMA, _NULLABLE, _PROVENANCE, _SCHEMA, _SPLITS


def scan_rows(paths, settings, scratch, checks):
    hashes = scratch / "hashes.parquet"
    with pq.ParquetWriter(hashes, _HASH_SCHEMA, compression="zstd") as writer:
        for path in paths:
            _scan_file(path, settings, checks, writer)
    return hashes


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
