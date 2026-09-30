"""Assemble only byte-verified, context-compatible region completions."""

import json
import re
import shutil
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from osm_worldcover.accounting import BuildContext, processing_ledger
from osm_worldcover.adapters import hub
from osm_worldcover.build import ShardStore
from osm_worldcover.config import DEDUPLICATION_POLICY, Config


@dataclass(frozen=True, slots=True)
class AssemblyInputs:
    """Verified inputs and the exact settings that produced their shards."""

    config: Config
    shards: Path
    rejections: dict[str, int]
    processing: dict[str, Any]


def verified_assembly(
    shard_dirs: list[Path],
    out: Path,
    work: Path,
    assertions: dict[str, Any],
    legacy_code_revision: str | None = None,
) -> AssemblyInputs:
    """Load all worker receipts before staging or writing any output data."""
    groups = _worker_regions(shard_dirs)
    directory, stems = next((directory, stems) for directory, stems in groups if stems)
    config, receipt_context = _receipt_config(directory / f"{stems[0]}.complete.json", out, work)
    _check_assertions(config, assertions)
    assembly_revision = BuildContext.from_config(config).document.get("code_revision")
    context_document = receipt_context.as_dict()
    context_document["code_revision"] = assembly_revision
    context = BuildContext.from_document(context_document)
    requested_version = assertions.get("dataset_version")
    config = config.with_overrides(
        dataset_version=_release_dataset_version(config, receipt_context, requested_version)
    )
    assert config.source_revision is not None  # The pinned context validates this above.
    outcomes, code_revisions = _verified_outcomes(groups, receipt_context, legacy_code_revision)
    selected = [outcome.stem for outcome in outcomes]
    expected = hub.list_region_stems(
        config.source_dataset, config.source_revision, config.source_recipe
    )
    ledger = processing_ledger(
        expected,
        selected,
        outcomes,
        context,
        region_code_revisions=code_revisions,
        assembly_code_revision=context.document.get("code_revision"),
    )
    rejections = _aggregate_rejections(outcomes)
    staged = _stage(groups, Path(work) / "verified-shards")
    return AssemblyInputs(config, staged, dict(sorted(rejections.items())), ledger)


def _release_dataset_version(
    config: Config, receipt_context: BuildContext, requested_version: str | None
) -> str:
    if requested_version:
        return requested_version
    recorded_policy = receipt_context.document["settings"].get("deduplication_policy")
    if recorded_policy == DEDUPLICATION_POLICY:
        return config.dataset_version
    return Config().dataset_version


def _verified_outcomes(
    groups: list[tuple[Path, list[str]]],
    context: BuildContext,
    legacy_code_revision: str | None,
) -> tuple[list, dict[str, str] | None]:
    outcomes = []
    revisions: dict[str, str] = {}
    missing_revisions: list[str] = []
    for directory, stem in _worker_region_files(groups):
        outcome, revision = _verified_region(directory, stem, context, legacy_code_revision)
        outcomes.append(outcome)
        _collect_region_revision(stem, revision, revisions, missing_revisions)
    return outcomes, _complete_revision_inventory(revisions, missing_revisions)


def _worker_region_files(groups: list[tuple[Path, list[str]]]) -> Iterator[tuple[Path, str]]:
    for directory, stems in groups:
        for stem in stems:
            yield directory, stem


def _collect_region_revision(
    stem: str,
    revision: str | None,
    revisions: dict[str, str],
    missing: list[str],
) -> None:
    if revision is None:
        missing.append(stem)
    else:
        revisions[stem] = revision


def _complete_revision_inventory(
    revisions: dict[str, str], missing: list[str]
) -> dict[str, str] | None:
    if revisions and missing:
        raise ValueError(
            f"code provenance is incomplete for regions without a recorded code revision: "
            f"{sorted(missing)}; provide --legacy-code-revision for verified old shards"
        )
    return revisions or None


def _verified_region(
    directory: Path,
    stem: str,
    expected_context: BuildContext,
    legacy_code_revision: str | None,
) -> tuple[Any, str | None]:
    path = directory / f"{stem}.complete.json"
    receipt_context = _load_receipt_context(path)
    _require_compatible_context(path, expected_context, receipt_context)
    outcome = ShardStore(directory, receipt_context).verified_outcomes([stem])[0]
    revision = _receipt_code_revision(stem, receipt_context, legacy_code_revision)
    return outcome, revision


def _load_receipt_context(path: Path) -> BuildContext:
    try:
        document = json.loads(path.read_text())["context"]
        return BuildContext.from_document(document)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f"unverifiable completion receipt {path}: {error}") from error


def _receipt_code_revision(
    stem: str, context: BuildContext, legacy_code_revision: str | None
) -> str | None:
    revision = context.document.get("code_revision")
    if revision is None:
        revision = legacy_code_revision
    if revision is not None and (
        not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None
    ):
        raise ValueError(f"{stem}: code revision must be a full 40-character commit")
    return revision


def _aggregate_rejections(outcomes) -> Counter[str]:
    rejections: Counter[str] = Counter()
    for outcome in outcomes:
        rejections.update(outcome.rejections)
    return rejections


def _worker_regions(shard_dirs: list[Path]) -> list[tuple[Path, list[str]]]:
    """Include orphan receipts and reject duplicate assignments before deduplication."""
    groups = []
    seen: set[str] = set()
    for raw in shard_dirs:
        directory = Path(raw)
        _require_worker_directory(directory)
        stems = _directory_stems(directory)
        _reject_duplicate_assignments(seen, stems)
        seen.update(stems)
        groups.append((directory, sorted(stems)))
    if not seen:
        raise ValueError("no region shards or completion receipts found")
    return groups


def _require_worker_directory(directory: Path) -> None:
    if not directory.is_dir():
        raise ValueError(f"shard directory does not exist: {directory}")


def _directory_stems(directory: Path) -> set[str]:
    stems = {path.stem for path in directory.glob("*.parquet")}
    stems.update(
        path.name.removesuffix(".complete.json") for path in directory.glob("*.complete.json")
    )
    return stems


def _reject_duplicate_assignments(seen: set[str], stems: set[str]) -> None:
    duplicate = sorted(seen & stems)
    if duplicate:
        raise ValueError(f"duplicate region worker assignments: {duplicate}")


def _receipt_config(path: Path, out: Path, work: Path) -> tuple[Config, BuildContext]:
    """Reconstruct every data setting, then demand an exact current-schema context."""
    try:
        document = json.loads(path.read_text())["context"]
        receipt_context = BuildContext.from_document(document)
        config = _context_config(receipt_context.document, out, work)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f"unverifiable completion receipt {path}: {error}") from error
    if not _contexts_compatible(BuildContext.from_config(config), receipt_context):
        raise ValueError(f"unverifiable completion receipt {path}: incompatible pipeline contract")
    return config, receipt_context


_FINALIZATION_ONLY_SETTINGS = {"dataset_version", "code_repository", "deduplication_policy"}


def _contexts_compatible(expected: BuildContext, recorded: BuildContext) -> bool:
    expected_document = expected.as_dict()
    recorded_document = recorded.as_dict()
    expected_document.pop("code_revision", None)
    recorded_document.pop("code_revision", None)
    expected_document.pop("code_revision", None)
    recorded_document.pop("code_revision", None)
    for document in (expected_document, recorded_document):
        settings = document.get("settings", {})
        for key in _FINALIZATION_ONLY_SETTINGS:
            settings.pop(key, None)
    return expected_document == recorded_document


def _require_compatible_context(path: Path, expected: BuildContext, recorded: BuildContext) -> None:
    if not _contexts_compatible(expected, recorded):
        raise ValueError(
            f"unverifiable region completion receipts: incompatible pipeline contract at {path}"
        )


def _context_config(document: dict[str, Any], out: Path, work: Path) -> Config:
    settings = document["settings"]
    ratios = settings["split_ratios"]
    return Config(
        out_dir=out,
        cache_dir=work,
        source=settings["source"],
        source_dataset=settings["source_dataset"],
        source_revision=settings["source_revision"],
        dataset_version=settings["dataset_version"],
        worldcover_version=settings["worldcover_version"],
        worldcover_year=settings["worldcover_year"],
        threshold=settings["dominance_threshold"],
        max_polygon_area_m2=settings["max_polygon_area_m2"],
        min_words=settings["min_words"],
        h3_resolution=settings["h3_resolution"],
        split_seed=settings["split_seed"],
        train_ratio=ratios["train"],
        validation_ratio=ratios["validation"],
        test_ratio=ratios["test"],
        extra=document["extra"],
    )


def _check_assertions(config: Config, assertions: dict[str, Any]) -> None:
    for name, value in assertions.items():
        if name == "dataset_version":
            continue
        actual = getattr(config, name)
        if value is not None and value != actual:
            option = {"source_revision": "revision"}.get(name, name).replace("_", "-")
            raise ValueError(
                f"--{option}={value!r} conflicts with verified receipt setting {actual!r}"
            )


def _stage(groups: list[tuple[Path, list[str]]], target: Path) -> Path:
    """Stage verified shards without ever replacing an input store or its ancestors."""
    _validate_stage_target(groups, target)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    _stage_groups(groups, target)
    return target


def _validate_stage_target(groups: list[tuple[Path, list[str]]], target: Path) -> None:
    resolved = target.resolve()
    for directory, _ in groups:
        source = directory.resolve()
        if resolved == source or resolved in source.parents:
            raise ValueError("assembly staging directory must not replace a shard store")


def _stage_groups(groups: list[tuple[Path, list[str]]], target: Path) -> None:
    for directory, stems in groups:
        for stem in stems:
            source = directory / f"{stem}.parquet"
            destination = target / source.name
            _link_or_copy(source, destination)


def _link_or_copy(source: Path, destination: Path) -> None:
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copy2(source, destination)
