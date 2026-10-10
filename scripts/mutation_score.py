"""Gate mutation testing on confirmed kills and save auditable evidence.

Timeouts are inconclusive, not kills. Every generated mutant stays in the
denominator, including mutants with no tests, skipped mutants, and unresolved
outcomes. CI uploads a detailed report for the raster functions changed by PR13.
"""

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from mutmut.mutation.diff_apply import get_diff_for_mutant
from mutmut.stats import status_by_exit_code
from mutmut.utils.file_utils import walk_mutatable_files
from mutmut.utils.format_utils import (
    mangled_name_from_mutant_name,
    orig_function_and_class_names_from_key,
)

FLOOR = 0.80
MUTANTS = Path("mutants")
BASELINE = MUTANTS / "mutmut-stats.json"
STATS = MUTANTS / "mutmut-cicd-stats.json"
SCORE_REPORT = MUTANTS / "mutation-score.json"
RASTER_REPORT = MUTANTS / "worldcover-raster-mutants.json"
RASTER_SOURCE = "src/osm_worldcover/adapters/worldcover.py"
RASTER_FUNCTIONS = (
    "_add_raster",
    "_partition_feature_batch",
    "_add_ordinary_features",
    "_add_spatial_feature",
    "_extract_and_accumulate",
    "_feature_batches",
    "_feature_pixel_cells",
    "_yield_feature_batches",
    "_pixel_bbox_cells",
    "_needs_spatial_chunks",
    "_bounded_pixel_chunks",
    "_split_pixel_geometry",
    "_geometry_is_bounded",
    "_split_pixel_geometry_once",
    "_pixel_aligned_midpoint",
    "_choose_split_axis",
    "_split_is_stalled",
    "_nonempty_pixel_children",
    "_map_geometry",
    "_batch_exceeds_limits",
    "_accumulate_corrected",
    "_accumulate_boundary_cells",
    "_matched_candidate_classes",
    "_boundary_cell_chunks",
    "_accumulate_stable_cells",
)
SUMMARY_STATUS_FIELDS = {
    "killed": "killed",
    "survived": "survived",
    "timeout": "timeout",
    "no_tests": "no tests",
    "skipped": "skipped",
    "suspicious": "suspicious",
    "unreported": "not checked",
    "segfault": "segfault",
    "interrupted": "check was interrupted by user",
    "caught_by_type_check": "caught by type check",
}
STATUS_TO_SUMMARY_FIELD = {status: field for field, status in SUMMARY_STATUS_FIELDS.items()}
STATUS_TO_SUMMARY_FIELD["unreported"] = "unreported"
EXPORTED_STATUS_FIELDS = {
    "killed": "killed",
    "survived": "survived",
    "timeout": "timeout",
    "no_tests": "no_tests",
    "skipped": "skipped",
    "suspicious": "suspicious",
    "interrupted": "check_was_interrupted_by_user",
    "segfault": "segfault",
}
REQUIRED_BASELINE_FIELDS = (
    "tests_by_mangled_function_name",
    "duration_by_test",
    "stats_time",
    "function_hashes",
    "function_dependencies",
    "config_fingerprint",
    "watched_file_hashes",
    "git_commit",
)


def _json_object(path: Path) -> dict[str, Any]:
    """Read a JSON object with a path-specific error."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read valid JSON from {path}: {error}") from error
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _run_started_ns() -> int | None:
    """Read the nanosecond timestamp recorded before mutmut starts."""
    value = os.environ.get("MUTATION_RUN_STARTED_NS")
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _baseline_errors(run_status: str, started_ns: int | None) -> tuple[dict | None, list[str]]:
    """Check that mutmut completed and its baseline belongs to this run and commit."""
    errors = []
    if run_status != "success":
        errors.append(
            f"mutmut run did not complete successfully (outcome: {run_status or 'missing'})"
        )
    if started_ns is None or started_ns <= 0:
        errors.append("mutation start timestamp is missing or invalid")
    baseline = _read_baseline(started_ns, errors)
    _validate_baseline(baseline, errors)
    _check_baseline_commit(baseline, errors)
    return baseline, errors


def _read_baseline(started_ns: int | None, errors: list[str]) -> dict | None:
    """Load only a baseline written after the current mutation run began."""
    if not BASELINE.is_file():
        errors.append(f"mutation baseline is missing: {BASELINE}")
        return None
    if started_ns is not None and BASELINE.stat().st_mtime_ns < started_ns:
        errors.append(f"mutation baseline is stale: {BASELINE}")
        return None
    try:
        return _json_object(BASELINE)
    except (TypeError, ValueError) as error:
        errors.append(str(error))
        return None


def _validate_baseline(baseline: dict | None, errors: list[str]) -> None:
    """Reject a partial mutmut cache before using it as test evidence."""
    if baseline is None:
        return
    missing = _missing_baseline_fields(baseline)
    if missing:
        errors.append(f"mutation baseline is incomplete; missing: {', '.join(missing)}")
    if _has_invalid_test_mappings(baseline):
        errors.append("mutation baseline has invalid test mappings")


def _missing_baseline_fields(baseline: dict) -> list[str]:
    """List cache fields that are part of mutmut's persisted schema."""
    return [field for field in REQUIRED_BASELINE_FIELDS if field not in baseline]


def _has_invalid_test_mappings(baseline: dict) -> bool:
    """Check the test index shape without accepting a partial or malformed map."""
    tests = baseline.get("tests_by_mangled_function_name")
    if not isinstance(tests, dict):
        return True
    return any(not isinstance(value, list) for value in tests.values())


def _check_baseline_commit(baseline: dict | None, errors: list[str]) -> None:
    """Require mutmut's baseline to be pinned to this CI checkout."""
    expected = os.environ.get("MUTATION_EXPECTED_GIT_SHA")
    if baseline is not None and (not expected or baseline.get("git_commit") != expected):
        errors.append("mutation baseline commit does not match this CI checkout")


def _export_stats() -> dict[str, Any]:
    """Export mutmut's summary and reject failed or missing exports.

    The previous export is removed first, so a file present afterwards was
    written by this export. Comparing file mtimes with the clock would depend
    on the filesystem's timestamp granularity.
    """
    STATS.unlink(missing_ok=True)
    result = subprocess.run(
        [sys.executable, "-m", "mutmut", "export-cicd-stats"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip()[:500]
        suffix = f": {detail}" if detail else ""
        raise ValueError(f"mutmut stats export failed with exit code {result.returncode}{suffix}")
    if not STATS.is_file():
        raise ValueError(f"mutmut stats export is missing or stale: {STATS}")
    return _json_object(STATS)


def _mutant_records(started_ns: int | None = None) -> list[dict[str, Any]]:
    """Read raw per-mutant exit codes from every mutmut metadata file."""
    sources = list(walk_mutatable_files())
    if not sources:
        raise ValueError("mutmut has no configured source files")
    return [
        record
        for source in sources
        for record in _records_for_file(str(source), _mutant_exit_codes(source, started_ns))
    ]


def _mutant_exit_codes(source: Path, started_ns: int | None = None) -> dict:
    """Read one fresh metadata file and require its result map."""
    source_path = str(source)
    metadata_path = MUTANTS / f"{source_path}.meta"
    if not metadata_path.is_file():
        raise ValueError(f"per-mutant result file is missing: {metadata_path}")
    if started_ns is not None and metadata_path.stat().st_mtime_ns < started_ns:
        raise ValueError(f"per-mutant result file is stale: {metadata_path}")
    exit_codes = _json_object(metadata_path).get("exit_code_by_key")
    if not isinstance(exit_codes, dict):
        raise TypeError(f"mutant exit-code data is missing in {metadata_path}")
    return exit_codes


def _records_for_file(source_path: str, exit_codes: dict) -> list[dict[str, Any]]:
    """Normalize one mutmut metadata file without hiding unresolved outcomes."""
    return [
        _mutant_record(source_path, mutant, exit_code) for mutant, exit_code in exit_codes.items()
    ]


def _mutant_record(source_path: str, mutant: Any, exit_code: Any) -> dict[str, Any]:
    """Normalize one mutant key and its raw exit code."""
    if not isinstance(mutant, str) or (exit_code is not None and type(exit_code) is not int):
        raise ValueError(f"invalid mutmut result in {source_path}: {mutant!r}")
    try:
        function, class_name = orig_function_and_class_names_from_key(mutant)
        mangled = mangled_name_from_mutant_name(mutant)
    except (AssertionError, IndexError) as error:
        raise ValueError(f"invalid mutmut key in {source_path}: {mutant!r}") from error
    return {
        "name": mutant,
        "path": source_path,
        "function": function,
        "class": class_name,
        "mangled_function": mangled,
        "status": _status_for_exit_code(exit_code),
    }


def _status_for_exit_code(exit_code: int | None) -> str:
    """Classify an exit code the way mutmut's own exporter does.

    ``status_by_exit_code`` is a defaultdict: an exit code mutmut does not know
    (for example a signal other than SIGSEGV or SIGKILL) is *suspicious*. Using
    ``.get`` here classified the same mutant as unreported and made the raw
    count disagree with the export.
    """
    return status_by_exit_code[exit_code]


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Count every generated mutant; only confirmed kills are the numerator."""
    counts = Counter(
        STATUS_TO_SUMMARY_FIELD.get(record["status"], "unreported") for record in records
    )
    total = len(records)
    result = {field: counts[field] for field in SUMMARY_STATUS_FIELDS}
    result["total"] = total
    result["confirmed_kill_score"] = counts["killed"] / total if total else 0.0
    return result


def _per_file_summaries(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Score each mutated source file with the same counts as the global summary."""
    by_path: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_path.setdefault(record["path"], []).append(record)
    return {path: _summary(by_path[path]) for path in sorted(by_path)}


def _export_errors(export: dict, summary: dict[str, Any]) -> list[str]:
    """Require mutmut's export to match the detailed raw mutant metadata."""
    return [
        *_export_status_errors(export, summary),
        *_export_total_errors(export, summary),
        *_unreported_errors(summary),
    ]


def _export_status_errors(export: dict, summary: dict[str, Any]) -> list[str]:
    """Compare each status that mutmut's summary export reports directly."""
    return [
        error
        for field, export_field in EXPORTED_STATUS_FIELDS.items()
        if (error := _export_count_error(export_field, export.get(export_field), summary[field]))
    ]


def _export_count_error(field: str, actual: Any, expected: int) -> str:
    """Return a mismatch message for one exported outcome count."""
    if type(actual) is not int or actual < 0:
        return f"mutation export has an invalid {field} count"
    if actual != expected:
        return f"mutation export {field} count {actual} differs from raw count {expected}"
    return ""


def _export_total_errors(export: dict, summary: dict[str, Any]) -> list[str]:
    """Verify the export total includes all outcome classes."""
    if type(export.get("total")) is int and export["total"] == summary["total"]:
        return []
    return ["mutation export total differs from the raw mutant count"]


def _unreported_errors(summary: dict[str, Any]) -> list[str]:
    """Reject generated mutants without a completed run outcome."""
    if not summary["unreported"]:
        return []
    return [f"{summary['unreported']} mutants have no completed result"]


def _raster_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select the raster adapter functions changed in PR13."""
    return [
        record
        for record in records
        if record["path"] == RASTER_SOURCE and record["function"] in RASTER_FUNCTIONS
    ]


def _raster_evidence(
    records: list[dict[str, Any]],
    baseline: dict | None,
) -> dict[str, Any]:
    """Capture changed raster mutants and diffs for every non-killed outcome."""
    selected = _selected_raster_records(records)
    tests_by_function = _raster_test_mapping(baseline)
    mutants = [_raster_mutant_evidence(record, tests_by_function) for record in selected]
    return _raster_report(selected, mutants)


def _selected_raster_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Require mutation outcomes for PR13's configured raster functions."""
    selected = _raster_records(records)
    if not selected:
        raise ValueError("no mutation results found for PR13's changed raster functions")
    return selected


def _raster_test_mapping(baseline: dict | None) -> dict:
    """Read the baseline's test index for per-mutant evidence."""
    tests = (baseline or {}).get("tests_by_mangled_function_name", {})
    if not isinstance(tests, dict):
        raise TypeError("mutation baseline has invalid test mappings")
    return tests


def _raster_report(selected: list[dict[str, Any]], mutants: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the stable report for changed raster mutations."""
    counts = Counter(item["function"] for item in selected)
    return {
        "schema_version": 1,
        "source_path": RASTER_SOURCE,
        "changed_functions": list(RASTER_FUNCTIONS),
        "mutants_total": len(selected),
        "mutants_by_function": {name: counts[name] for name in RASTER_FUNCTIONS},
        "outcomes": dict(Counter(item["status"] for item in selected)),
        "mutants": mutants,
    }


def _raster_mutant_evidence(record: dict[str, Any], tests_by_function: dict) -> dict[str, Any]:
    """Attach associated tests and a source diff when the mutant was not killed."""
    item = {key: record[key] for key in ("name", "function", "class", "status")}
    item["tests"] = _raster_tests_for(record, tests_by_function)
    if record["status"] != "killed":
        item["diff"] = get_diff_for_mutant(record["name"], path=record["path"])
    return item


def _raster_tests_for(record: dict[str, Any], tests_by_function: dict) -> list[str]:
    """Validate test names associated with one mutated raster function."""
    tests = tests_by_function.get(record["mangled_function"], [])
    if not isinstance(tests, list):
        raise TypeError(f"invalid baseline tests for {record['mangled_function']}")
    if any(not isinstance(test, str) for test in tests):
        raise TypeError(f"invalid baseline tests for {record['mangled_function']}")
    return tests


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """Write a deterministic, reviewable JSON report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _print_summary(summary: dict[str, Any]) -> None:
    """Report every outcome category and the confirmed-kill score."""
    print(
        "confirmed-kill mutation score: "
        f"{summary['confirmed_kill_score']:.1%} "
        f"({summary['killed']}/{summary['total']} total mutants)"
    )
    for field in SUMMARY_STATUS_FIELDS:
        print(f"{field}: {summary[field]}")


def _finish(summary: dict[str, Any], errors: list[str]) -> int:
    """Fail incomplete reports before applying the unchanged 80% floor."""
    _print_summary(summary)
    for error in errors:
        print(f"FAIL: {error}", file=sys.stderr)
    if errors or summary["total"] == 0:
        return 2
    if summary["confirmed_kill_score"] < FLOOR:
        print(f"FAIL: below the {FLOOR:.0%} floor", file=sys.stderr)
        return 1
    return 0


def _read_records(errors: list[str], started_ns: int | None) -> list[dict[str, Any]]:
    """Load the per-mutant cache and preserve an error for an incomplete file set."""
    try:
        return _mutant_records(started_ns)
    except (OSError, TypeError, ValueError, KeyError) as error:
        errors.append(f"cannot read complete per-mutant results: {error}")
        return []


def _read_export(summary: dict[str, Any], errors: list[str]) -> dict[str, Any] | None:
    """Read this run's export and compare every exported outcome count."""
    try:
        export = _export_stats()
    except (OSError, TypeError, ValueError) as error:
        errors.append(str(error))
        return None
    errors.extend(_export_errors(export, summary))
    return export


def _write_raster_report(
    records: list[dict[str, Any]], baseline: dict | None, errors: list[str]
) -> None:
    """Write scoped raster mutation evidence or make the quality job fail."""
    try:
        _write_json(RASTER_REPORT, _raster_evidence(records, baseline))
    except (OSError, TypeError, ValueError, KeyError) as error:
        errors.append(f"cannot create changed-raster mutation evidence: {error}")


def _write_score_report(
    summary: dict[str, Any],
    per_file: dict[str, dict[str, Any]],
    export: dict[str, Any] | None,
    errors: list[str],
) -> None:
    """Persist the score, per-file scores, provenance, all outcomes, and validation errors."""
    report = {
        "schema_version": 1,
        "run_outcome": os.environ.get("MUTATION_RUN_OUTCOME"),
        "git_sha": os.environ.get("MUTATION_EXPECTED_GIT_SHA"),
        "floor": FLOOR,
        "summary": summary,
        "per_file": per_file,
        "export": export,
        "errors": errors,
    }
    try:
        _write_json(SCORE_REPORT, report)
    except OSError as error:
        errors.append(f"cannot write mutation score report: {error}")


def main() -> int:
    """Validate this run's mutation baseline, export, raw results, and evidence."""
    started_ns = _run_started_ns()
    baseline, errors = _baseline_errors(os.environ.get("MUTATION_RUN_OUTCOME", ""), started_ns)
    records = _read_records(errors, started_ns)
    summary = _summary(records)
    export = _read_export(summary, errors)
    if summary["total"] == 0:
        errors.append("no per-mutant outcomes were exported")
    _write_raster_report(records, baseline, errors)
    _write_score_report(summary, _per_file_summaries(records), export, errors)
    return _finish(summary, errors)


if __name__ == "__main__":
    raise SystemExit(main())
