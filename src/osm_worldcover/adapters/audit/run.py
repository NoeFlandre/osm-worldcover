import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from osm_worldcover.release_commit import release_read_lock

from .card import _check_card
from .manifest import _check_completion, _check_settings, _load_manifest
from .report import AuditReport, _Checks
from .rows import _inspect_files, scan_rows
from .sql import check_aggregates


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
    build_dir = Path(build_dir)
    with release_read_lock(build_dir):
        return _audit_build_locked(
            build_dir,
            require_complete=require_complete,
            require_card=require_card,
            strict_text_leakage=strict_text_leakage,
        )


def _audit_build_locked(
    build_dir: Path,
    *,
    require_complete: bool,
    require_card: bool,
    strict_text_leakage: bool,
) -> AuditReport:
    report = AuditReport()
    checks = _Checks(report)
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
