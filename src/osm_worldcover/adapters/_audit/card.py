import yaml

from .schema import _SPLITS


def _check_card(build_dir, settings, processing, checks) -> None:
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
        _check_card_assets(build_dir, text, settings, processing, checks)
    except (OSError, ValueError, IndexError, AttributeError, yaml.YAMLError) as error:
        checks.add("invalid_card_or_map", error)


def _check_card_assets(build_dir, text, settings, processing, checks) -> None:
    _check_card_provenance(text, settings, checks)
    _check_card_map_text(text, checks)
    _check_card_map_file(build_dir, checks)
    _check_card_code_provenance(text, processing, checks)


def _check_card_provenance(text, settings, checks) -> None:
    for key in ("source_revision", "source_dataset"):
        if str(settings.get(key)) not in text:
            checks.add(f"card_missing_provenance:{key}")


def _check_card_map_text(text, checks) -> None:
    if "worldcover_centroids.png" not in text:
        checks.add("card_missing_coverage_map")


def _check_card_map_file(build_dir, checks) -> None:
    with (build_dir / "worldcover_centroids.png").open("rb") as source:
        if source.read(8) != b"\x89PNG\r\n\x1a\n":
            checks.add("invalid_coverage_map_png")


def _check_card_code_provenance(text, processing, checks) -> None:
    _check_card_code_groups(text, processing.get("code_provenance", []), checks)
    _check_card_assembly_revision(text, processing, checks)


def _check_card_code_groups(text, groups, checks) -> None:
    for group in groups:
        revision = group["revision"]
        if not _card_code_group_present(text, group):
            checks.add("card_missing_code_provenance", revision)


def _card_code_group_present(text, group) -> bool:
    revision = group["revision"]
    repository = group["repository"]
    count = len(group["regions"])
    unit = "region" if count == 1 else "regions"
    link = f"[{revision}]({repository}/tree/{revision})"
    return f"{link} — {count:,} {unit}" in text


def _check_card_assembly_revision(text, processing, checks) -> None:
    revision = processing.get("assembly_code_revision")
    if revision and f"[{revision}]" not in text:
        checks.add("card_missing_assembly_code_revision", revision)
