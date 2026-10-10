"""Assembly accepts only byte-verified, context-compatible region completions."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from osm_worldcover import assembly
from osm_worldcover.accounting import BuildContext
from osm_worldcover.assembly import (
    _check_assertions,
    _contexts_compatible,
    _receipt_code_revision,
    _stage,
    _worker_regions,
    verified_assembly,
)
from osm_worldcover.build import ShardStore
from osm_worldcover.config import Config

HEAD = "c" * 40


@pytest.fixture(autouse=True)
def clean_checkout(monkeypatch):
    """Pretend the working tree is a clean checkout of one known commit."""
    monkeypatch.setattr("osm_worldcover.accounting.context._current_code_revision", lambda: HEAD)


@pytest.fixture
def listed_regions(monkeypatch):
    """Record the source inventory requests; the source lists exactly ``alpha``."""
    requests = []

    def list_region_stems(*arguments):
        requests.append(arguments)
        return ["alpha"]

    monkeypatch.setattr(assembly.hub, "list_region_stems", list_region_stems)
    return requests


def worker(directory, context, examples, outcome, *stems):
    """A worker shard directory holding verified receipts for ``stems``."""
    store = ShardStore(directory, context)
    for stem in stems or ("alpha",):
        store.write_outcome(stem, examples, replace(outcome, stem=stem))
    return directory


def assemble(tmp_path, *directories, assertions=None, **options):
    return verified_assembly(
        list(directories), tmp_path / "out", tmp_path / "work", assertions or {}, **options
    )


def test_verified_inputs_are_bound_to_the_requested_directories(
    tmp_path, context, examples, outcome, listed_regions
):
    shards = worker(tmp_path / "w0", context, examples, outcome)

    inputs = assemble(tmp_path, shards)

    assert (inputs.config.out_dir, inputs.config.cache_dir) == (tmp_path / "out", tmp_path / "work")
    assert inputs.shards == tmp_path / "work" / "verified-shards"
    assert [path.name for path in inputs.shards.glob("*.parquet")] == ["alpha.parquet"]


def test_verified_inputs_carry_the_receipt_accounting(
    tmp_path, context, examples, outcome, listed_regions
):
    shards = worker(tmp_path / "w0", context, examples, outcome)

    inputs = assemble(tmp_path, shards)

    assert inputs.rejections == {"below_threshold": 3}
    assert inputs.processing["assembly_code_revision"] == HEAD
    assert listed_regions == [
        (inputs.config.source_dataset, inputs.config.source_revision, inputs.config.source_recipe)
    ]


def test_every_data_setting_is_restored_from_the_receipts(
    tmp_path, examples, outcome, listed_regions
):
    config = Config(
        source="description",
        source_dataset="owner/custom",
        source_revision="a" * 40,
        threshold=0.9,
        max_polygon_area_m2=5e9,
        min_words=12,
        worldcover_version="v100",
        worldcover_year=2020,
        h3_resolution=4,
        split_seed=17,
        train_ratio=0.7,
        validation_ratio=0.15,
        test_ratio=0.15,
        dataset_version="2.1.0",
        extra={"policy": "custom"},
    )
    shards = worker(tmp_path / "w0", BuildContext.from_config(config), examples, outcome)

    inputs = assemble(tmp_path, shards)

    assert inputs.config.as_manifest_settings() == config.as_manifest_settings()
    assert inputs.config.extra == {"policy": "custom"}


def test_an_explicit_dataset_version_overrides_the_recorded_one(
    tmp_path, config, context, examples, outcome, listed_regions
):
    shards = worker(tmp_path / "w0", context, examples, outcome)
    inputs = assemble(tmp_path, shards, assertions={"dataset_version": "9.9.9"})
    assert inputs.config.dataset_version == "9.9.9"


def test_a_stale_staging_directory_is_replaced(
    tmp_path, context, examples, outcome, listed_regions
):
    shards = worker(tmp_path / "w0", context, examples, outcome)
    stale = tmp_path / "work" / "verified-shards" / "stale.parquet"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old")

    inputs = assemble(tmp_path, shards)

    assert [path.name for path in inputs.shards.iterdir()] == ["alpha.parquet"]


def test_incomplete_code_provenance_names_the_regions_without_a_revision(
    tmp_path, context, examples, outcome, listed_regions
):
    legacy = BuildContext.from_document({**context.as_dict(), "code_revision": None})
    first = worker(tmp_path / "w0", legacy, examples, outcome, "alpha")
    second = worker(tmp_path / "w1", context, examples, outcome, "beta")

    with pytest.raises(ValueError, match=r"without a recorded code revision: \['alpha'\]"):
        assemble(tmp_path, first, second)


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda tmp_path: tmp_path / "missing", "shard directory does not exist: .*missing"),
        (lambda tmp_path: tmp_path / "empty", "no region shards or completion receipts found"),
    ],
)
def test_unusable_worker_directories_are_refused(tmp_path, make, message):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match=message):
        assemble(tmp_path, make(tmp_path))


def test_worker_regions_include_orphan_receipts_in_sorted_order(tmp_path):
    directory = tmp_path / "w0"
    directory.mkdir()
    (directory / "beta.parquet").write_bytes(b"")
    (directory / "alpha.complete.json").write_text("{}")
    (directory / "alpha.parquet").write_bytes(b"")
    assert _worker_regions([directory]) == [(directory, ["alpha", "beta"])]


def test_a_region_assigned_to_two_workers_is_refused(tmp_path):
    for name in ("w0", "w1"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "alpha.parquet").write_bytes(b"")
    with pytest.raises(ValueError, match=r"duplicate region worker assignments: \['alpha'\]"):
        _worker_regions([tmp_path / "w0", tmp_path / "w1"])


def test_receipts_must_carry_a_readable_context(tmp_path, context, examples, outcome):
    shards = worker(tmp_path / "w0", context, examples, outcome)
    (shards / "alpha.complete.json").write_text(json.dumps({"no": "context"}))
    with pytest.raises(
        ValueError, match=r"unverifiable completion receipt .*alpha\.complete\.json"
    ):
        assemble(tmp_path, shards)


def test_a_receipt_with_a_foreign_contract_is_refused_with_its_path(
    tmp_path, config, examples, outcome
):
    other = BuildContext.from_config(replace(config, threshold=0.7))
    shards = worker(tmp_path / "w0", other, examples, outcome)
    receipt = shards / "alpha.complete.json"
    document = json.loads(receipt.read_text())
    document["context"]["output_columns"] = ["not", "the", "columns"]
    receipt.write_text(json.dumps(document))
    message = f"unverifiable completion receipt {receipt}: incompatible pipeline contract"
    with pytest.raises(ValueError, match=message):
        assemble(tmp_path, shards)


def test_a_worker_built_under_other_settings_is_refused_with_its_path(
    tmp_path, config, context, examples, outcome, listed_regions
):
    first = worker(tmp_path / "w0", context, examples, outcome, "alpha")
    other = BuildContext.from_config(replace(config, threshold=0.7))
    second = worker(tmp_path / "w1", other, examples, outcome, "beta")
    receipt = second / "beta.complete.json"
    message = (
        f"unverifiable region completion receipts: incompatible pipeline contract at {receipt}"
    )
    with pytest.raises(ValueError, match=message):
        assemble(tmp_path, first, second)


@pytest.mark.parametrize(
    ("name", "value", "option"),
    [
        ("threshold", 0.7, "threshold"),
        ("source_revision", "b" * 40, "revision"),
        ("h3_resolution", 3, "h3-resolution"),
    ],
)
def test_explicit_settings_that_conflict_with_receipts_are_refused(config, name, value, option):
    actual = getattr(config, name)
    with pytest.raises(ValueError) as error:
        _check_assertions(config, {name: value})
    assert (
        str(error.value)
        == f"--{option}={value!r} conflicts with verified receipt setting {actual!r}"
    )


def test_the_dataset_version_and_unset_assertions_never_conflict(config):
    _check_assertions(config, {"dataset_version": "9.9.9", "threshold": None})


def test_the_dataset_version_assertion_does_not_hide_later_conflicts(config):
    with pytest.raises(ValueError, match=r"--threshold=0\.7 conflicts"):
        _check_assertions(config, {"dataset_version": "9.9.9", "threshold": 0.7})


def document_of(context, **changes):
    return {**context.as_dict(), **changes}


@pytest.mark.parametrize(
    "change",
    [
        {"code_revision": "d" * 40},
        {"code_revision": None},
        {"settings": {"dataset_version": "0.0.1"}},
        {"settings": {"code_repository": "elsewhere"}},
        {"settings": {"deduplication_policy": "legacy"}},
    ],
)
@pytest.mark.parametrize("swap", [False, True])
def test_commit_and_finalization_only_settings_do_not_break_compatibility(context, change, swap):
    document = document_of(context)
    changed = {
        **document,
        **{k: v for k, v in change.items() if k != "settings"},
        "settings": {**document["settings"], **change.get("settings", {})},
    }
    pair = (document, changed)[:: -1 if swap else 1]
    assert _contexts_compatible(*(BuildContext.from_document(item) for item in pair))


@pytest.mark.parametrize(
    "change",
    [
        {"settings": {"dominance_threshold": 0.7}},
        {"output_columns": []},
        {"extra": {"x": 1}},
    ],
)
def test_any_other_difference_breaks_compatibility(context, change):
    changed = {**document_of(context), **change}
    changed["settings"] = {**context.as_dict()["settings"], **change.get("settings", {})}
    assert not _contexts_compatible(context, BuildContext.from_document(changed))


def test_contexts_without_settings_are_compared_on_the_rest():
    bare = {"receipt_version": 1, "pipeline_schema_version": 3}
    assert _contexts_compatible(BuildContext.from_document(bare), BuildContext.from_document(bare))
    assert not _contexts_compatible(
        BuildContext.from_document(bare), BuildContext.from_document({**bare, "extra": 1})
    )


@pytest.mark.parametrize("revision", ["short", "A" * 40, "g" * 40])
def test_a_recorded_code_revision_must_be_a_full_lowercase_commit(context, revision):
    recorded = BuildContext.from_document({**context.as_dict(), "code_revision": revision})
    with pytest.raises(ValueError, match="alpha: code revision must be a full 40-character commit"):
        _receipt_code_revision("alpha", recorded, None)


def test_the_legacy_revision_fills_in_only_when_the_receipt_has_none(context):
    unpinned = BuildContext.from_document({**context.as_dict(), "code_revision": None})
    assert _receipt_code_revision("alpha", unpinned, "e" * 40) == "e" * 40
    assert _receipt_code_revision("alpha", context, "e" * 40) == HEAD
    assert _receipt_code_revision("alpha", unpinned, None) is None
    with pytest.raises(ValueError, match="alpha: code revision"):
        _receipt_code_revision("alpha", unpinned, "nope")


def test_staging_never_replaces_a_shard_store_or_its_ancestor(tmp_path):
    store = tmp_path / "verified-shards"
    store.mkdir()
    (store / "alpha.parquet").write_bytes(b"data")
    nested = store / "deeper"
    nested.mkdir()
    (nested / "beta.parquet").write_bytes(b"data")

    for shard_dir in (store, nested):
        with pytest.raises(ValueError, match="assembly staging directory must not replace"):
            _stage([(shard_dir, ["alpha"])], store)

    assert (store / "alpha.parquet").read_bytes() == b"data"


def test_staging_copies_when_hard_links_are_unavailable(tmp_path, monkeypatch):
    source = tmp_path / "w0"
    source.mkdir()
    (source / "alpha.parquet").write_bytes(b"data")

    def refuse(self, target):
        raise OSError("cross-device link")

    monkeypatch.setattr(Path, "hardlink_to", refuse)
    staged = _stage([(source, ["alpha"])], tmp_path / "staged")

    assert (staged / "alpha.parquet").read_bytes() == b"data"
    assert not (staged / "alpha.parquet").samefile(source / "alpha.parquet")
