"""Read back release files and independently check publication guarantees.

Python keeps only one Arrow batch and bounded diagnostic samples. Global
identity checks run in DuckDB with a memory limit and disk-backed spill space.
The input files are never modified. Passing this audit does not independently
recompute raster labels, or prove that excluded source polygons were processed.
"""

import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ._audit.aggregates import (
    _retained_text_diagnostics as _retained_text_diagnostics,
)
from ._audit.aggregates import (
    check_aggregates,
)
from ._audit.card import _check_card
from ._audit.manifest import _check_completion, _check_settings, _load_manifest
from ._audit.report import AuditProblem, AuditReport, _Checks
from ._audit.rows import _inspect_files, scan_rows
from ._audit.schema import _SCHEMA as _SCHEMA

__all__ = ["AuditProblem", "AuditReport", "audit_build"]

AuditProblem.__module__ = __name__
AuditReport.__module__ = __name__


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
        _check_card(build_dir, settings, manifest.get("processing", {}), checks)
    if checks.counts.get("invalid_settings"):
        checks.finish()
        return report
    with tempfile.TemporaryDirectory(prefix="owc-audit-") as scratch:
        _audit_contents(paths, Path(scratch), manifest, checks, strict_text_leakage)
    checks.finish()
    return report


def _audit_contents(
    paths: Sequence[Path],
    scratch: Path,
    manifest: Mapping[str, Any],
    checks: _Checks,
    strict_text_leakage: bool,
) -> None:
    hashes = scan_rows(paths, manifest["settings"], scratch, checks)
    check_aggregates(paths, hashes, scratch, manifest, checks, strict_text_leakage)
