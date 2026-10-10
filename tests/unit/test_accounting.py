"""Completion receipts bind resume safety to byte-verified inputs and counters."""

import hashlib
import json
import subprocess
import tempfile
from collections import Counter
from dataclasses import asdict, fields, replace
from pathlib import Path

import pandas as pd
import pytest

from osm_worldcover import accounting
from osm_worldcover.accounting import (
    BuildContext,
    atomic_json,
    code_revision,
    file_sha256,
    outcome_from_record,
    outcome_record,
    processing_ledger,
    validate_outcome,
)
from osm_worldcover.build import ShardStore
from osm_worldcover.pipeline import OUTPUT_COLUMNS, RegionOutcome


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
    with pytest.raises(ValueError, match=r"^completion receipts require a pinned source revision$"):
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
    with pytest.raises(
        ValueError, match=r"^code revision inventory must match processed regions exactly$"
    ):
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
    with pytest.raises(
        ValueError, match=r"^assembly code revision must be a full 40-character commit$"
    ):
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
    with pytest.raises(
        ValueError, match=r"^schema 2 processing ledgers require an assembly code revision$"
    ):
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


def _tamper(tmp_path, mutate, stem="alpha"):
    path = tmp_path / f"{stem}.complete.json"
    receipt = json.loads(path.read_text())
    mutate(receipt)
    path.write_text(json.dumps(receipt))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r["shard"].update(filename="other.parquet"),
        lambda r: r["shard"].update(bytes=r["shard"]["bytes"] + 1),
        lambda r: r["shard"].update(sha256="0" * 64),
        lambda r: r["shard"].update(rows=r["shard"]["rows"] + 1),
        lambda r: r["shard"].update(columns=["polygon_id"]),
        lambda r: r["outcome"].update(stem="beta"),
        lambda r: r["outcome"].update(examples=r["outcome"]["examples"] + 1),
        lambda r: r.update(build_context_sha256="0" * 64),
        lambda r: r["context"].update(unrelated="change"),
        lambda r: r.update(schema_version=r["schema_version"] + 1),
    ],
    ids=[
        "filename",
        "bytes",
        "sha256",
        "rows",
        "columns",
        "outcome-stem",
        "outcome-examples",
        "fingerprint",
        "context",
        "schema",
    ],
)
def test_every_receipt_claim_must_hold_independently(tmp_path, context, outcome, examples, mutate):
    store = ShardStore(tmp_path, context)
    store.write_outcome("alpha", examples, outcome)
    assert store.outcome("alpha") == outcome

    _tamper(tmp_path, mutate)

    assert store.outcome("alpha") is None


def test_a_receipt_cannot_be_replayed_under_another_region_name(
    tmp_path, context, outcome, examples
):
    store = ShardStore(tmp_path, context)
    store.write_outcome("alpha", examples, outcome)
    (tmp_path / "beta.parquet").write_bytes((tmp_path / "alpha.parquet").read_bytes())
    (tmp_path / "beta.complete.json").write_text((tmp_path / "alpha.complete.json").read_text())
    _tamper(tmp_path, lambda r: r["shard"].update(filename="beta.parquet"), stem="beta")

    assert store.outcome("beta") is None


def test_context_bound_store_reads_back_verified_shards_and_refuses_tampered_ones(
    tmp_path, context, outcome, examples
):
    store = ShardStore(tmp_path, context)
    store.write_outcome("alpha", examples, outcome)
    assert len(store.read()) == 1

    _tamper(tmp_path, lambda r: r["shard"].update(bytes=0))

    with pytest.raises(ValueError, match=r"\['alpha'\]"):
        store.read()


def test_staging_copies_when_hardlinks_are_unavailable(
    tmp_path, context, outcome, examples, monkeypatch
):
    store = ShardStore(tmp_path / "shards", context)
    store.write_outcome("alpha", examples, outcome)

    def refuse(self, target):
        raise OSError("cross-device")

    monkeypatch.setattr(type(tmp_path), "hardlink_to", refuse)

    staged = store.stage(["alpha"], tmp_path / "stage")

    assert (staged / "alpha.parquet").read_bytes() == store.path_for("alpha").read_bytes()


def test_outcome_counters_are_written_beside_the_receipt(tmp_path, context, outcome, examples):
    ShardStore(tmp_path, context).write_outcome("alpha", examples, outcome)

    sidecar = tmp_path / "alpha.rejections.json"
    assert json.loads(sidecar.read_text()) == {"below_threshold": 3}
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "alpha.complete.json",
        "alpha.parquet",
        "alpha.rejections.json",
    ]


def test_record_serialization_does_not_mutate_the_outcome(outcome):
    record = outcome_record(outcome)
    record["text_rejections"]["empty_text"] = 99
    assert outcome.text_rejections["empty_text"] == 2


def test_record_lists_each_missing_tile_once_in_order(outcome):
    record = outcome_record(replace(outcome, tiles_missing=["N03E000", "N00E000", "N03E000"]))
    assert record["tiles_missing"] == ["N00E000", "N03E000"]


@pytest.mark.parametrize("tiles", [("N00E000",), [3]])
def test_missing_tiles_must_be_a_list_of_names(outcome, tiles):
    with pytest.raises(ValueError, match="missing-tile"):
        validate_outcome(replace(outcome, tiles_missing=tiles))


def test_context_records_the_whole_output_contract(config, context):
    document = context.document
    assert document["output_columns"] == list(OUTPUT_COLUMNS)
    assert document["outcome_fields"] == [item.name for item in fields(RegionOutcome)]
    assert document["source_recipe"] == asdict(config.source_recipe)


def test_context_accepts_an_uppercase_commit_hash(config):
    BuildContext.from_config(replace(config, source_revision="A" * 40))


@pytest.mark.parametrize("change", [{"receipt_version": 2}, {"pipeline_schema_version": 2}])
def test_a_recorded_context_must_match_both_current_versions(context, change):
    with pytest.raises(ValueError, match=r"^unsupported receipt context version$"):
        BuildContext.from_document({**context.as_dict(), **change})


def test_a_recorded_context_must_be_an_object():
    with pytest.raises(TypeError, match=r"^build context must be an object$"):
        BuildContext.from_document(["not", "an", "object"])


def fake_git(monkeypatch, returncode=0, error=None):
    """Replace ``subprocess.run`` and record how git was invoked."""
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        if error is not None:
            raise error
        return subprocess.CompletedProcess(argv, returncode, stdout=" abc\n")

    monkeypatch.setattr(subprocess, "run", run)
    return calls


def test_git_output_is_one_bounded_captured_command(tmp_path, monkeypatch):
    calls = fake_git(monkeypatch)
    assert code_revision._git_output(tmp_path, "status", "--porcelain") == "abc"
    assert calls == [
        (
            ["git", "-C", str(tmp_path), "status", "--porcelain"],
            {"check": False, "capture_output": True, "text": True, "timeout": 2},
        )
    ]


@pytest.mark.parametrize("returncode", [1, 128])
def test_a_failing_git_command_has_no_output(tmp_path, monkeypatch, returncode):
    fake_git(monkeypatch, returncode=returncode)
    assert code_revision._git_output(tmp_path, "rev-parse") is None


@pytest.mark.parametrize("error", [OSError("no git"), subprocess.TimeoutExpired("git", 2)])
def test_an_unavailable_or_stuck_git_has_no_output(tmp_path, monkeypatch, error):
    fake_git(monkeypatch, error=error)
    assert code_revision._git_output(tmp_path, "rev-parse") is None


def test_file_sha256_reads_files_larger_than_one_block(tmp_path):
    data = bytes(range(256)) * (8 * 1024 + 3)
    path = tmp_path / "large.bin"
    path.write_bytes(data)
    assert file_sha256(path) == hashlib.sha256(data).hexdigest()


def test_atomic_json_installs_canonical_bytes_and_leaves_no_temporary_file(tmp_path):
    target = tmp_path / "nested" / "deeper" / "out.json"
    atomic_json(target, {"b": 1, "a": [1, 2]})
    assert target.read_bytes() == b'{"a":[1,2],"b":1}\n'
    assert [path.name for path in target.parent.iterdir()] == ["out.json"]


def test_atomic_json_stages_a_hidden_file_beside_its_target(tmp_path, monkeypatch):
    calls = []
    mkstemp = tempfile.mkstemp

    def spy(**kwargs):
        calls.append(kwargs)
        return mkstemp(**kwargs)

    monkeypatch.setattr(tempfile, "mkstemp", spy)
    atomic_json(tmp_path / "out.json", {})
    assert calls == [{"prefix": ".out.json.", "dir": tmp_path}]


def test_atomic_json_refuses_non_finite_numbers_without_a_trace(tmp_path):
    with pytest.raises(ValueError, match="Out of range float values"):
        atomic_json(tmp_path / "out.json", {"x": float("nan")})
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("expected", "selected", "processed", "name"),
    [
        ([""], ["alpha"], [], "expected regions"),
        (["alpha"], [""], [], "selected regions"),
        (["alpha"], ["alpha"], [""], "processed regions"),
    ],
)
def test_ledger_refuses_blank_region_names(context, outcome, expected, selected, processed, name):
    outcomes = [replace(outcome, stem=stem) for stem in processed]
    with pytest.raises(ValueError, match=f"{name} must contain nonempty region names"):
        processing_ledger(expected, selected, outcomes, context)


def test_ledger_names_the_unselected_processed_regions(context, outcome):
    with pytest.raises(
        ValueError, match=r"^processed regions are outside the selected source inventory$"
    ):
        processing_ledger(["alpha", "beta"], ["beta"], [outcome], context)


def test_ledger_totals_list_every_missing_tile_once(context, outcome):
    beta = replace(outcome, stem="beta", tiles_missing=["N03E000", "N00E000"])
    ledger = processing_ledger(["alpha", "beta"], ["alpha", "beta"], [outcome, beta], context)
    assert ledger["totals"]["tiles_missing"] == ["N00E000", "N03E000"]


def test_ledger_documents_its_reconciliation_rules(context, outcome):
    assert processing_ledger(["alpha"], ["alpha"], [outcome], context)["reconciliation"] == {
        "spatial": "polygons_seen = polygons_invalid + polygons_accepted + sum(rejections)",
        "text": "polygons_accepted = polygons_with_examples + sum(text_rejections)",
        "counts_are_pre_deduplication": True,
        "valid": True,
    }


@pytest.fixture
def pinned_context(context):
    """A context recorded by a clean checkout at one exact commit."""
    return BuildContext.from_document({**context.as_dict(), "code_revision": "d" * 40})


def test_regions_default_to_the_commit_recorded_in_the_context(pinned_context, outcome):
    outcomes = [outcome, replace(outcome, stem="beta")]
    ledger = processing_ledger(["alpha", "beta"], ["alpha", "beta"], outcomes, pinned_context)
    assert ledger["schema_version"] == 2
    assert [(item["revision"], item["regions"]) for item in ledger["code_provenance"]] == [
        ("d" * 40, ["alpha", "beta"])
    ]
    assert ledger["assembly_code_revision"] == "d" * 40


def test_the_assembly_commit_defaults_to_the_one_recorded_in_the_context(pinned_context, outcome):
    ledger = processing_ledger(
        ["alpha"], ["alpha"], [outcome], pinned_context, region_code_revisions={"alpha": "a" * 40}
    )
    assert ledger["assembly_code_revision"] == "d" * 40


def test_region_revisions_must_be_full_commits(context, outcome):
    with pytest.raises(
        ValueError, match=r"^region code revisions must be full 40-character commits$"
    ):
        processing_ledger(
            ["alpha"],
            ["alpha"],
            [outcome],
            context,
            region_code_revisions={"alpha": "nope"},
            assembly_code_revision="c" * 40,
        )


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BASE_LEDGER_FIELDS = {
    "schema_version",
    "build_context_sha256",
    "context",
    "scope",
    "complete",
    "full_source_complete",
    "selected_complete",
    "expected_regions",
    "selected_regions",
    "processed_regions",
    "missing_regions",
    "unprocessed_selected_regions",
    "region_counts",
    "totals",
    "regions",
    "reconciliation",
}


@pytest.fixture
def fresh_code_revision():
    """Forget the process-wide code revision so each test observes its own git."""
    code_revision._current_code_revision.cache_clear()
    yield
    code_revision._current_code_revision.cache_clear()


def fake_repository(monkeypatch, *, status="", head="b" * 40):
    """Answer the two git queries that decide the code revision and record them."""
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        answers = {"status": status, "rev-parse": head}
        return subprocess.CompletedProcess(argv, 0, stdout=answers[argv[3]] + "\n")

    monkeypatch.setattr(subprocess, "run", run)
    return calls


def test_public_names_are_stable():
    assert sorted(accounting.__all__) == [
        "BuildContext",
        "atomic_json",
        "file_sha256",
        "outcome_from_record",
        "outcome_record",
        "processing_ledger",
        "validate_outcome",
    ]


def test_schema_versions_are_stable():
    assert accounting.RECEIPT_VERSION == 1
    assert accounting.PROCESSING_LEDGER_VERSION == 2
    assert accounting.PIPELINE_SCHEMA_VERSION == 3


def test_count_fields_are_stable():
    assert accounting.COUNT_FIELDS == (
        "polygons_seen",
        "polygons_invalid",
        "polygons_accepted",
        "polygons_with_examples",
        "source_links",
        "source_documents",
        "examples",
    )


def test_clean_checkout_records_head_queried_at_the_repository_root(
    config, monkeypatch, fresh_code_revision
):
    calls = fake_repository(monkeypatch)
    context = BuildContext.from_config(config)
    assert context.document["code_revision"] == "b" * 40
    assert calls == [
        ["git", "-C", str(REPOSITORY_ROOT), "status", "--porcelain", "--untracked-files=no"],
        ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "--verify", "HEAD"],
    ]


@pytest.mark.parametrize("status", ["M  src/osm_worldcover/build.py", " M README.md"])
def test_modified_tracked_files_record_no_code_revision(
    config, monkeypatch, fresh_code_revision, status
):
    fake_repository(monkeypatch, status=status)
    assert BuildContext.from_config(config).document["code_revision"] is None


@pytest.mark.parametrize("head", ["abc123", "g" * 40])
def test_head_that_is_not_a_full_commit_records_no_code_revision(
    config, monkeypatch, fresh_code_revision, head
):
    fake_repository(monkeypatch, head=head)
    assert BuildContext.from_config(config).document["code_revision"] is None


def test_unavailable_git_records_no_code_revision(config, monkeypatch, fresh_code_revision):
    def run(argv, **kwargs):
        raise OSError("no git")

    monkeypatch.setattr(subprocess, "run", run)
    assert BuildContext.from_config(config).document["code_revision"] is None


def test_code_revision_is_looked_up_once_per_process(config, monkeypatch, fresh_code_revision):
    calls = fake_repository(monkeypatch)
    BuildContext.from_config(config)
    BuildContext.from_config(config)
    assert len(calls) == 2


def test_ledger_top_level_fields_are_pinned(pinned_context, outcome):
    ledger = processing_ledger(
        ["alpha", "beta"],
        ["alpha", "beta"],
        [outcome, replace(outcome, stem="beta")],
        pinned_context,
    )
    assert set(ledger) == BASE_LEDGER_FIELDS | {"code_provenance", "assembly_code_revision"}
    assert ledger["schema_version"] == 2


def test_outcome_record_missing_a_field_is_refused_with_its_reason(outcome):
    record = outcome_record(outcome)
    del record["source_links"]
    with pytest.raises(
        ValueError, match=r"^region receipt does not have the current outcome schema$"
    ):
        outcome_from_record(record)


class _EndOfFileStream:
    """Return the data once, then end of file; fail loudly if read again."""

    def __init__(self, data: bytes) -> None:
        self._chunks = [data, b""]

    def __enter__(self) -> "_EndOfFileStream":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        if not self._chunks:
            raise AssertionError("read past end of file")
        return self._chunks.pop(0)


class _SingleStreamSource:
    """Stand in for a path whose only open() returns one prepared stream."""

    def __init__(self, stream: _EndOfFileStream) -> None:
        self._stream = stream

    def open(self, mode: str) -> _EndOfFileStream:
        return self._stream


def test_file_sha256_stops_at_the_first_end_of_file():
    data = b"shard bytes"
    source = _SingleStreamSource(_EndOfFileStream(data))
    assert file_sha256(source) == hashlib.sha256(data).hexdigest()
