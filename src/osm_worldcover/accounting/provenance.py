"""Code revision pins recorded for each region and for the assembled ledger."""

from collections.abc import Mapping
from typing import Any

from osm_worldcover.accounting.context import BuildContext
from osm_worldcover.domain.revision import is_full_revision


def validated_assembly_code_revision(
    revision: str | None, context: BuildContext, region_revisions: Mapping[str, str] | None
) -> str | None:
    if revision is None:
        revision = context.document.get("code_revision")
    if revision is not None:
        if not is_full_revision(revision):
            raise ValueError("assembly code revision must be a full 40-character commit")
        return revision
    if region_revisions is not None:
        raise ValueError("schema 2 processing ledgers require an assembly code revision")
    return None


def validated_code_revisions(
    processed: list[str],
    context: BuildContext,
    revisions: Mapping[str, str] | None,
) -> dict[str, str] | None:
    if revisions is None:
        revisions = _context_code_revisions(processed, context)
    if revisions is None:
        return None
    _validate_code_revision_inventory(processed, revisions)
    return dict(revisions)


def _context_code_revisions(processed: list[str], context: BuildContext) -> dict[str, str] | None:
    revision = context.document.get("code_revision")
    return {stem: revision for stem in processed} if revision is not None else None


def _validate_code_revision_inventory(processed: list[str], revisions: Mapping[str, str]) -> None:
    if set(revisions) != set(processed):
        raise ValueError("code revision inventory must match processed regions exactly")
    if not all(is_full_revision(revision) for revision in revisions.values()):
        raise ValueError("region code revisions must be full 40-character commits")


def code_provenance_fields(
    revisions: Mapping[str, str] | None,
    context: BuildContext,
    assembly_revision: str | None,
) -> dict[str, Any]:
    if revisions is None:
        return {}
    return {
        "code_provenance": _code_provenance(
            revisions, context.document["settings"]["code_repository"]
        ),
        "assembly_code_revision": assembly_revision,
    }


def _code_provenance(revisions: Mapping[str, str], repository: str) -> list[dict[str, Any]]:
    by_revision: dict[str, list[str]] = {}
    for stem, revision in revisions.items():
        by_revision.setdefault(revision, []).append(stem)
    return [
        {"repository": repository, "revision": revision, "regions": sorted(stems)}
        for revision, stems in sorted(by_revision.items())
    ]
