"""Assemble only byte-verified, context-compatible region completions."""

import json
import shutil
from collections import Counter
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
) -> AssemblyInputs:
    """Load all worker receipts before staging or writing any output data."""
    groups = _worker_regions(shard_dirs)
    config, context = _verified_configuration(groups, out, work, assertions)
    return _assemble_verified_inputs(groups, config, context, work)


def _verified_configuration(
    groups: list[tuple[Path, list[str]]],
    out: Path,
    work: Path,
    assertions: dict[str, Any],
) -> tuple[Config, BuildContext]:
    """Resolve finalization-only overrides from one verified source receipt."""
    directory, stems = next((directory, stems) for directory, stems in groups if stems)
    config, context = _receipt_config(directory / f"{stems[0]}.complete.json", out, work)
    _check_assertions(config, assertions)
    requested_version = assertions.get("dataset_version")
    recorded_policy = context.document["settings"].get("deduplication_policy")
    output_version = requested_version or (
        config.dataset_version
        if recorded_policy == DEDUPLICATION_POLICY
        else Config().dataset_version
    )
    return config.with_overrides(dataset_version=output_version), context


def _assemble_verified_inputs(
    groups: list[tuple[Path, list[str]]],
    config: Config,
    context: BuildContext,
    work: Path,
) -> AssemblyInputs:
    """Reconcile every verified shard against the pinned inventory and stage it."""
    assert config.source_revision is not None  # The pinned context validates this above.
    outcomes = _verified_outcomes(groups, context)
    selected = [outcome.stem for outcome in outcomes]
    expected = hub.list_region_stems(
        config.source_dataset, config.source_revision, config.source_recipe
    )
    ledger = processing_ledger(expected, selected, outcomes, context)
    rejections = _aggregate_rejections(outcomes)
    staged = _stage(groups, Path(work) / "verified-shards")
    return AssemblyInputs(config, staged, dict(sorted(rejections.items())), ledger)


def _verified_outcomes(groups: list[tuple[Path, list[str]]], context: BuildContext) -> list:
    outcomes = []
    for directory, stems in groups:
        outcomes.extend(ShardStore(directory, context).verified_outcomes(stems))
    return outcomes


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
    """Reconstruct settings and accept only finalization-only context changes."""
    try:
        document = json.loads(path.read_text())["context"]
        config = _context_config(document, out, work)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f"unverifiable completion receipt {path}: {error}") from error
    if not _compatible_receipt_context(BuildContext.from_config(config).as_dict(), document):
        raise ValueError(f"unverifiable completion receipt {path}: incompatible pipeline contract")
    return config, BuildContext.from_document(document)


def _compatible_receipt_context(expected: dict[str, Any], recorded: dict[str, Any]) -> bool:
    """Permit version and provenance changes made only during finalization."""
    finalization_settings = {"dataset_version", "code_repository", "deduplication_policy"}
    if (set(expected) != set(recorded)) or any(
        expected[key] != recorded[key] for key in set(expected) - {"settings"}
    ):
        return False
    expected_settings = expected["settings"]
    recorded_settings = recorded["settings"]
    keys = (set(expected_settings) | set(recorded_settings)) - finalization_settings
    return all(expected_settings.get(key) == recorded_settings.get(key) for key in keys)


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
            # The region shards contain labelled examples; the release version
            # and finalization policy can change without raster recomputation.
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
