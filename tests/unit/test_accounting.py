"""Completion receipts bind resume safety to byte-verified inputs and counters."""

import json
from collections import Counter
from dataclasses import replace

import pandas as pd
import pytest

from osm_worldcover.accounting import (
    BuildContext,
    outcome_from_record,
    outcome_record,
    processing_ledger,
    validate_outcome,
)
from osm_worldcover.build import ShardStore
from osm_worldcover.config import Config
from osm_worldcover.pipeline import RegionOutcome


@pytest.fixture
def config():
    return Config(source_revision="a" * 40)


@pytest.fixture
def context(config):
    return BuildContext.from_config(config)


@pytest.fixture
def outcome():
    return RegionOutcome(
        "alpha",
        polygons_seen=12,
        polygons_invalid=1,
        polygons_accepted=8,
        polygons_with_examples=2,
        examples=3,
        source_links=10,
        source_documents=9,
        rejections=Counter({"below_threshold": 3}),
        text_rejections=Counter({"text_too_short": 4, "empty_text": 2}),
        tiles_missing=["N00E000"],
    )


@pytest.fixture
def examples():
    return pd.DataFrame({"polygon_id": ["p1", "p1", "p2"], "text": ["a", "b", "c"]})


def test_receipt_roundtrip_preserves_every_outcome_field(tmp_path, context, outcome, examples):
    ShardStore(tmp_path, context).write_outcome("alpha", examples, outcome)
    restored = ShardStore(tmp_path, context).outcome("alpha")
    assert restored == outcome
    assert isinstance(restored.rejections, Counter)
    assert isinstance(restored.text_rejections, Counter)


def test_raster_semantics_change_invalidates_prior_pipeline_receipts(context):
    assert context.document["pipeline_schema_version"] == 3


def test_empty_region_has_verified_completion(tmp_path, context):
    store = ShardStore(tmp_path, context)
    store.write_outcome("empty", pd.DataFrame(), RegionOutcome("empty"))
    assert store.has("empty")
    assert store.outcome("empty") == RegionOutcome("empty")


def test_legacy_shard_without_receipt_cannot_resume(tmp_path, context, examples):
    ShardStore(tmp_path).write("alpha", examples, {"below_threshold": 3})
    assert ShardStore(tmp_path, context).outcome("alpha") is None
    with pytest.raises(ValueError, match="unverifiable"):
        ShardStore(tmp_path, context).verified_outcomes(["alpha"])


@pytest.mark.parametrize(
    "change",
    [
        {"min_words": 1},
        {"source_revision": "b" * 40},
        {"source_dataset": "other/dataset"},
        {"threshold": 0.7},
        {"source": "website"},
        {"worldcover_version": "v100"},
        {"max_polygon_area_m2": None},
        {"extra": {"future_setting": "changed"}},
    ],
)
def test_data_changing_context_never_reuses_shard(
    tmp_path, config, context, outcome, examples, change
):
    ShardStore(tmp_path, context).write_outcome("alpha", examples, outcome)
    changed = BuildContext.from_config(replace(config, **change))
    assert ShardStore(tmp_path, changed).outcome("alpha") is None


def test_paths_cache_policy_and_selection_do_not_change_shard_context(config, tmp_path):
    altered = replace(
        config,
        cache_dir=tmp_path,
        out_dir=tmp_path / "other",
        cached_tiles=2,
        regions=("alpha",),
    )
    assert (
        BuildContext.from_config(config).fingerprint
        == BuildContext.from_config(altered).fingerprint
    )


@pytest.mark.parametrize("revision", [None, "main", "v1", "a" * 39])
def test_context_requires_an_immutable_commit(config, revision):
    with pytest.raises(ValueError, match="pinned source revision"):
        BuildContext.from_config(replace(config, source_revision=revision))


def test_modified_shard_bytes_invalidate_receipt(tmp_path, context, outcome, examples):
    store = ShardStore(tmp_path, context)
    store.write_outcome("alpha", examples, outcome)
    path = store.path_for("alpha")
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 1
    path.write_bytes(data)
    assert store.outcome("alpha") is None


@pytest.mark.parametrize("content", ["{", "null", "[]", '{"schema_version":999}'])
def test_invalid_receipt_is_pending_not_completed(tmp_path, context, outcome, examples, content):
    store = ShardStore(tmp_path, context)
    store.write_outcome("alpha", examples, outcome)
    (tmp_path / "alpha.complete.json").write_text(content)
    assert store.outcome("alpha") is None


def test_old_outcome_schema_is_not_silently_filled(tmp_path, context, outcome, examples):
    store = ShardStore(tmp_path, context)
    store.write_outcome("alpha", examples, outcome)
    path = tmp_path / "alpha.complete.json"
    receipt = json.loads(path.read_text())
    del receipt["outcome"]["source_links"]
    path.write_text(json.dumps(receipt))
    assert store.outcome("alpha") is None


def test_rejections_come_from_verified_receipts_not_legacy_sidecars(
    tmp_path, context, outcome, examples
):
    store = ShardStore(tmp_path, context)
    store.write_outcome("alpha", examples, outcome)
    (tmp_path / "alpha.rejections.json").write_text('{"wrong": 999}')
    assert store.rejections() == {"below_threshold": 3}


def test_generic_overwrite_invalidates_completion_even_for_identical_rows(
    tmp_path, context, outcome, examples
):
    strict = ShardStore(tmp_path, context)
    strict.write_outcome("alpha", examples, outcome)
    ShardStore(tmp_path).write("alpha", examples)
    assert strict.outcome("alpha") is None


def test_context_bound_store_rejects_partial_accounting_write(tmp_path, context, examples):
    with pytest.raises(ValueError, match="full accounting"):
        ShardStore(tmp_path, context).write("alpha", examples)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"polygons_seen": 11}, "spatial"),
        ({"polygons_with_examples": 1}, "text polygon"),
        ({"examples": 1}, "fewer examples"),
        ({"polygons_invalid": -1}, "nonnegative integer"),
        ({"source_links": 1.5}, "nonnegative integer"),
        ({"source_documents": True}, "nonnegative integer"),
        ({"text_rejections": Counter({"empty_text": -1})}, "nonnegative integer"),
    ],
)
def test_accounting_must_reconcile_and_counts_must_be_valid(outcome, change, message):
    with pytest.raises(ValueError, match=message):
        validate_outcome(replace(outcome, **change))


def test_written_shard_row_and_polygon_counts_must_match(tmp_path, context, outcome, examples):
    store = ShardStore(tmp_path, context)
    with pytest.raises(ValueError, match="row count"):
        store.write_outcome("alpha", examples.iloc[:2], outcome)
    with pytest.raises(ValueError, match="polygon count"):
        store.write_outcome("alpha", examples.assign(polygon_id="same"), outcome)


def test_stage_includes_only_verified_selected_regions(tmp_path, context, outcome, examples):
    store = ShardStore(tmp_path / "shards", context)
    store.write_outcome("alpha", examples, outcome)
    store.write_outcome("beta", examples, replace(outcome, stem="beta"))
    ShardStore(store.directory).write("legacy", examples)
    staged = store.stage(["alpha"], tmp_path / "stage")
    assert [path.name for path in staged.glob("*.parquet")] == ["alpha.parquet"]
    # A later selection must not leave the previous run's staged shard behind.
    store.stage(["beta"], staged)
    assert [path.name for path in staged.glob("*.parquet")] == ["beta.parquet"]


def test_stage_never_overwrites_the_shard_store(tmp_path, context, outcome, examples):
    store = ShardStore(tmp_path / "shards", context)
    store.write_outcome("alpha", examples, outcome)
    with pytest.raises(ValueError, match="must not replace"):
        store.stage(["alpha"], store.directory)
    assert store.has("alpha")


def test_subset_completion_is_not_full_source_completion(context, outcome):
    ledger = processing_ledger(["beta", "alpha"], ["alpha"], [outcome], context)
    assert {
        "selected_complete": ledger["selected_complete"],
        "complete": ledger["complete"],
        "full_source_complete": ledger["full_source_complete"],
        "scope": ledger["scope"],
        "missing_regions": ledger["missing_regions"],
        "region_counts": ledger["region_counts"],
    } == {
        "selected_complete": True,
        "complete": False,
        "full_source_complete": False,
        "scope": "subset",
        "missing_regions": ["beta"],
        "region_counts": {
            "expected": 2,
            "selected": 1,
            "processed": 1,
            "missing": 1,
            "unprocessed_selected": 0,
        },
    }


def test_full_ledger_accounts_for_every_region_and_all_counters(context, outcome):
    outcomes = [replace(outcome, stem="beta"), outcome]
    ledger = processing_ledger(["alpha", "beta"], ["beta", "alpha"], outcomes, context)
    assert {
        "complete": ledger["full_source_complete"],
        "scope": ledger["scope"],
        "polygons_seen": ledger["totals"]["polygons_seen"],
        "polygons_with_examples": ledger["totals"]["polygons_with_examples"],
        "rejections": ledger["totals"]["rejections"],
        "text_rejections": ledger["totals"]["text_rejections"],
        "regions": [row["stem"] for row in ledger["regions"]],
        "first_outcome": outcome_from_record(ledger["regions"][0]),
    } == {
        "complete": True,
        "scope": "full",
        "polygons_seen": 24,
        "polygons_with_examples": 4,
        "rejections": {"below_threshold": 6},
        "text_rejections": {"empty_text": 4, "text_too_short": 8},
        "regions": ["alpha", "beta"],
        "first_outcome": outcome,
    }


def test_processing_ledger_groups_exact_region_code_revisions(context, outcome):
    outcomes = [replace(outcome, stem="beta"), outcome]
    ledger = processing_ledger(
        ["alpha", "beta"],
        ["alpha", "beta"],
        outcomes,
        context,
        region_code_revisions={"alpha": "a" * 40, "beta": "b" * 40},
        assembly_code_revision="c" * 40,
    )
    assert ledger["schema_version"] == 2
    assert ledger["code_provenance"] == [
        {
            "repository": context.document["settings"]["code_repository"],
            "revision": "a" * 40,
            "regions": ["alpha"],
        },
        {
            "repository": context.document["settings"]["code_repository"],
            "revision": "b" * 40,
            "regions": ["beta"],
        },
    ]
    assert ledger["assembly_code_revision"] == "c" * 40


def test_processing_ledger_requires_code_pin_for_every_processed_region(context, outcome):
    with pytest.raises(ValueError, match="match processed regions exactly"):
        processing_ledger(
            ["alpha"],
            ["alpha"],
            [outcome],
            context,
            region_code_revisions={},
            assembly_code_revision="c" * 40,
        )


def test_legacy_context_keeps_schema_one_ledger(context, outcome):
    document = context.as_dict()
    document["code_revision"] = None
    legacy_context = BuildContext.from_document(document)
    ledger = processing_ledger(["alpha"], ["alpha"], [outcome], legacy_context)
    assert ledger["schema_version"] == 1
    assert "code_provenance" not in ledger


def test_processing_ledger_rejects_invalid_assembly_revision(context, outcome):
    with pytest.raises(ValueError, match="assembly code revision"):
        processing_ledger(
            ["alpha"],
            ["alpha"],
            [outcome],
            context,
            assembly_code_revision="not-a-commit",
        )


def test_version_two_ledger_requires_assembly_revision(context, outcome):
    document = context.as_dict()
    document["code_revision"] = None
    legacy_context = BuildContext.from_document(document)
    with pytest.raises(ValueError, match="require an assembly code revision"):
        processing_ledger(
            ["alpha"],
            ["alpha"],
            [outcome],
            legacy_context,
            region_code_revisions={"alpha": "a" * 40},
        )


def test_selected_regions_not_yet_completed_are_explicit(context, outcome):
    ledger = processing_ledger(["alpha", "beta"], ["alpha", "beta"], [outcome], context)
    assert not ledger["full_source_complete"]
    assert not ledger["selected_complete"]
    assert ledger["unprocessed_selected_regions"] == ["beta"]


@pytest.mark.parametrize(
    ("expected", "selected", "processed", "message"),
    [
        (["alpha"], ["wrong"], [], "unknown selected"),
        (["alpha", "alpha"], ["alpha"], [], "duplicate"),
        (["alpha"], ["alpha", "alpha"], [], "duplicate"),
        (["alpha", "beta"], ["beta"], ["alpha"], "outside the selected"),
        (["alpha"], ["alpha"], ["alpha", "alpha"], "duplicate"),
    ],
)
def test_ledger_refuses_ambiguous_inventory(
    context, outcome, expected, selected, processed, message
):
    with pytest.raises(ValueError, match=message):
        processing_ledger(
            expected, selected, [replace(outcome, stem=stem) for stem in processed], context
        )


def test_record_serialization_does_not_mutate_the_outcome(outcome):
    record = outcome_record(outcome)
    record["text_rejections"]["empty_text"] = 99
    assert outcome.text_rejections["empty_text"] == 2
