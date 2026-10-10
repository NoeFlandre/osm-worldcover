import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

from osm_worldcover.config import DEDUPLICATION_POLICY
from osm_worldcover.domain.revision import is_full_revision, is_optional_revision
from osm_worldcover.domain.splits import SplitRatios

from .report import _Checks
from .schema import _PROVENANCE


def _load_manifest(build_dir: Path, checks: _Checks) -> dict[str, Any] | None:
    try:
        text = (build_dir / "manifest.json").read_text()  # pragma: no mutate
        manifest = json.loads(text)
    except (OSError, ValueError) as error:
        checks.add("missing_or_invalid_manifest", error)
        return None
    if not isinstance(manifest, dict) or not _valid_manifest_shape(manifest):
        checks.add("missing_or_invalid_manifest")
        return None
    return manifest


def _valid_manifest_shape(manifest: dict) -> bool:
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


def _check_settings(settings: dict[str, Any], checks: _Checks) -> None:
    try:
        SplitRatios(**settings["split_ratios"])
        valid = _valid_setting_values(settings)
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        checks.add("invalid_settings")
    if not is_full_revision(settings.get("source_revision")):
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
    version = ledger["schema_version"]
    _require(version in {1, 2}, "unsupported processing ledger schema")
    _require(
        _processing_settings_match(context["settings"], manifest["settings"]),
        "build context settings mismatch",
    )
    if version == 2:
        _check_code_provenance(ledger, context)


def _check_code_provenance(ledger: dict, context: dict) -> None:
    _require(
        is_full_revision(ledger["assembly_code_revision"]),
        "invalid assembly code revision",
    )
    _require(
        is_optional_revision(context.get("code_revision")),
        "invalid context code revision",
    )
    groups = ledger["code_provenance"]
    repository = context["settings"]["code_repository"]
    regions = _validated_code_regions(groups, repository)
    _require(
        _matches_processed_regions(regions, ledger["processed_regions"]),
        "code provenance does not match processed regions",
    )
    if context.get("code_revision") is not None:
        _require(
            ledger["assembly_code_revision"] == context["code_revision"],
            "assembly code revision differs from the ledger context",
        )


def _validated_code_regions(groups: object, repository: str) -> list[str]:
    _require(isinstance(groups, list) and bool(groups), "missing region code provenance")
    groups = cast(list[object], groups)
    regions = []
    for group in groups:
        regions.extend(_validated_code_group(group, repository))
    _require(len(regions) == len(set(regions)), "duplicate code provenance region")
    return regions


def _validated_code_group(group: object, expected_repository: str) -> list[str]:
    _require(isinstance(group, dict), "invalid code provenance group")
    group = cast(dict[str, Any], group)
    repository = group.get("repository")
    revision = group.get("revision")
    raw_regions = group.get("regions")
    _require(isinstance(repository, str) and bool(repository), "invalid code repository")
    _require(repository == expected_repository, "code repository differs from the ledger context")
    _require(is_full_revision(revision), "invalid region code revision")
    _require(
        isinstance(raw_regions, list) and bool(raw_regions),
        "empty code provenance region group",
    )
    regions = cast(list[object], raw_regions)
    _require(
        all(isinstance(stem, str) and bool(stem) for stem in regions),
        "invalid code provenance region",
    )
    regions = cast(list[str], regions)
    _require(len(regions) == len(set(regions)), "duplicate code provenance region")
    return regions


def _matches_processed_regions(regions: list[str], processed: list[str]) -> bool:
    return sorted(regions) == sorted(processed)


def _processing_settings_match(processing: dict, release: dict) -> bool:
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
    by_length = cast(dict[str, Any], by_length)
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
