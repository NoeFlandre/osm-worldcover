"""Shard persistence and resume behaviour of a whole build."""

import json

import pandas as pd
import pytest

from osm_worldcover.build import ShardStore


def frame(n: int = 2) -> pd.DataFrame:
    return pd.DataFrame({"polygon_id": [f"p{i}" for i in range(n)], "text": ["t"] * n})


def test_a_written_shard_is_reported_as_done(tmp_path) -> None:
    store = ShardStore(tmp_path)
    assert not store.has("alpha")
    store.write("alpha", frame())
    assert store.has("alpha")


def test_an_empty_region_is_recorded_so_it_is_not_redone(tmp_path) -> None:
    """A region with no examples is a real result, not an unfinished one."""
    store = ShardStore(tmp_path)
    store.write("alpha", pd.DataFrame())
    assert store.has("alpha")
    assert store.read() == []


def test_shards_are_read_back_in_a_deterministic_order(tmp_path) -> None:
    store = ShardStore(tmp_path)
    store.write("beta", frame(1))
    store.write("alpha", frame(2))
    assert [len(f) for f in store.read()] == [2, 1]


def test_round_trip_preserves_rows(tmp_path) -> None:
    store = ShardStore(tmp_path)
    store.write("alpha", frame(3))
    assert store.read()[0]["polygon_id"].tolist() == ["p0", "p1", "p2"]


def test_a_region_name_with_a_separator_is_still_addressable(tmp_path) -> None:
    store = ShardStore(tmp_path)
    store.write("great-britain-latest", frame())
    assert store.has("great-britain-latest")


def test_partial_writes_are_not_mistaken_for_finished_ones(tmp_path) -> None:
    store = ShardStore(tmp_path)
    (tmp_path / "alpha.parquet.part").write_bytes(b"junk")
    assert not store.has("alpha")


class TestRunBuild:
    """Whole-build orchestration, with the network and rasters stubbed out."""

    def _patch(self, monkeypatch, tmp_path, stems, examples_per_region=1):
        from osm_worldcover import build as build_module
        from osm_worldcover.adapters.source import RegionTables
        from osm_worldcover.pipeline import RegionOutcome

        monkeypatch.setattr(build_module.hub, "resolve_revision", lambda *a, **k: "a" * 40)
        monkeypatch.setattr(build_module.hub, "list_region_stems", lambda *a, **k: stems)
        monkeypatch.setattr(build_module.hub, "snapshot_region", lambda *a, **k: [])
        monkeypatch.setattr(build_module.hub, "region_files", lambda stem: [])
        monkeypatch.setattr(
            RegionTables, "load", classmethod(lambda cls, root, stem: cls(stem, None, None, None))
        )

        def fake_run_region(config, tables, tiles, keep_tiles=False):
            # Distinct objects need distinct osm ids: sharing an id across
            # regions means the same object, which assembly collapses.
            # Make fake OSM identities deterministic and unique by region.
            # Hash-modulo fixtures can collide and make unrelated regions
            # look like duplicate records to the production finalizer.
            base = (stems.index(tables.stem) + 1) * 100_000
            rows = pd.DataFrame(
                {
                    "polygon_id": [f"{tables.stem}:{i}" for i in range(examples_per_region)],
                    "osm_type": ["way"] * examples_per_region,
                    "osm_id": [base + i for i in range(examples_per_region)],
                    "region": [tables.stem] * examples_per_region,
                    "document_id": [f"{tables.stem}-d{i}" for i in range(examples_per_region)],
                    "language": ["en"] * examples_per_region,
                    "text": [
                        " ".join(["w"] * 20) + f" {tables.stem}{i}"
                        for i in range(examples_per_region)
                    ],
                    "worldcover_code": [10] * examples_per_region,
                    "worldcover_label": ["Tree cover"] * examples_per_region,
                    "dominant_fraction": [0.95] * examples_per_region,
                    "lat": [49.6] * examples_per_region,
                    "lon": [6.1] * examples_per_region,
                }
            )
            return rows, RegionOutcome(
                tables.stem,
                polygons_seen=examples_per_region,
                polygons_accepted=examples_per_region,
                polygons_with_examples=examples_per_region,
                source_links=examples_per_region,
                source_documents=examples_per_region,
                examples=examples_per_region,
            )

        monkeypatch.setattr(build_module, "run_region", fake_run_region)
        return build_module

    def test_every_region_contributes(self, tmp_path, monkeypatch) -> None:
        from osm_worldcover.config import Config

        module = self._patch(monkeypatch, tmp_path, ["alpha", "beta"])
        report = module.run_build(Config(cache_dir=tmp_path, out_dir=tmp_path / "out"))
        assert report.result.rows == 2
        assert len(report.regions) == 2

    def test_a_finished_region_is_skipped_on_a_rerun(self, tmp_path, monkeypatch) -> None:
        from osm_worldcover.config import Config

        module = self._patch(monkeypatch, tmp_path, ["alpha", "beta"])
        config = Config(cache_dir=tmp_path, out_dir=tmp_path / "out")
        module.run_build(config)

        def unexpected(*args, **kwargs):
            raise AssertionError("finished region was processed again")

        monkeypatch.setattr(module, "run_region", unexpected)
        second = module.run_build(config)
        assert [r.stem for r in second.regions] == ["alpha", "beta"]
        assert second.result.rows == 2  # but the data is still there

    def test_rejections_are_summed_across_regions(self, tmp_path, monkeypatch) -> None:
        from osm_worldcover.config import Config

        module = self._patch(monkeypatch, tmp_path, ["alpha", "beta"])
        report = module.run_build(Config(cache_dir=tmp_path, out_dir=tmp_path / "out"))
        for region in report.regions:
            region.rejections["below_threshold"] = 2
        assert report.rejections == {"below_threshold": 4}

    def test_an_explicit_region_list_overrides_discovery(self, tmp_path, monkeypatch) -> None:
        from osm_worldcover.config import Config

        module = self._patch(monkeypatch, tmp_path, ["alpha", "beta"])
        report = module.run_build(
            Config(cache_dir=tmp_path, out_dir=tmp_path / "out"), regions=["alpha"]
        )
        assert [r.stem for r in report.regions] == ["alpha"]

    def test_a_pinned_revision_is_not_resolved_again(self, tmp_path, monkeypatch) -> None:
        from osm_worldcover import build as build_module
        from osm_worldcover.config import Config

        self._patch(monkeypatch, tmp_path, ["alpha"])

        def explode(*_a, **_k):
            raise AssertionError("revision was already pinned")

        monkeypatch.setattr(build_module.hub, "resolve_revision", explode)
        report = build_module.run_build(
            Config(cache_dir=tmp_path, out_dir=tmp_path / "out", source_revision="a" * 40)
        )
        splits = [p for p in report.result.paths if p.suffix == ".parquet"]
        rows = pd.concat([pd.read_parquet(p) for p in splits], ignore_index=True)
        assert rows["source_revision"].unique().tolist() == ["a" * 40]


def test_progress_is_reported_as_each_region_starts(tmp_path, monkeypatch) -> None:
    """Regression: announcing every region up front hides where a long run is.

    A list comprehension over a pre-computed "pending" list printed all 386
    region names before any work began.
    """
    from osm_worldcover import build as build_module
    from osm_worldcover.config import Config

    seen: list[str] = []
    helper = TestRunBuild()
    module = helper._patch(monkeypatch, tmp_path, ["alpha", "beta"])
    original = module._process_region

    def spy(config, revision, stem, raw, tiles, shards, keep_tiles, progress):
        seen.append(f"processing {stem}")
        return original(config, revision, stem, raw, tiles, shards, keep_tiles, progress)

    monkeypatch.setattr(build_module, "_process_region", spy)
    module.run_build(Config(cache_dir=tmp_path, out_dir=tmp_path / "out"), progress=seen.append)

    # beta must not be announced before alpha has been processed.
    assert seen.index("processing alpha") < next(
        i for i, line in enumerate(seen) if line.startswith("[2/2] beta")
    )


class TestRejectionsSurviveTheProcess:
    """Why polygons were refused must outlive the process that refused them.

    Regression: rejection counters lived only in the build's memory, so a run
    split across worker processes -- or one whose assembly crashed -- published
    a manifest claiming no polygon was ever refused.
    """

    def test_a_region_records_why_its_polygons_were_refused(self, tmp_path) -> None:
        from osm_worldcover.build import ShardStore

        store = ShardStore(tmp_path)
        store.write("alpha", frame(2), rejections={"below_threshold": 7})
        assert store.rejections() == {"below_threshold": 7}

    def test_counters_sum_across_regions(self, tmp_path) -> None:
        from osm_worldcover.build import ShardStore

        store = ShardStore(tmp_path)
        store.write("alpha", frame(1), rejections={"below_threshold": 3, "too_large": 1})
        store.write("beta", frame(1), rejections={"below_threshold": 4})
        assert store.rejections() == {"below_threshold": 7, "too_large": 1}

    def test_a_region_that_refused_nothing_contributes_nothing(self, tmp_path) -> None:
        from osm_worldcover.build import ShardStore

        store = ShardStore(tmp_path)
        store.write("alpha", frame(1), rejections={})
        assert store.rejections() == {}

    def test_counters_are_readable_from_a_fresh_store(self, tmp_path) -> None:
        """A later process must see what an earlier one recorded."""
        from osm_worldcover.build import ShardStore

        ShardStore(tmp_path).write("alpha", frame(1), rejections={"no_valid_class": 2})
        assert ShardStore(tmp_path).rejections() == {"no_valid_class": 2}

    def test_shards_without_counters_are_tolerated(self, tmp_path) -> None:
        """Shards built before counters were recorded must still assemble."""
        from osm_worldcover.build import ShardStore

        store = ShardStore(tmp_path)
        store.write("alpha", frame(1))
        assert store.rejections() == {}


def test_unknown_selected_region_is_rejected_before_processing(tmp_path, monkeypatch):
    from osm_worldcover.config import Config

    module = TestRunBuild()._patch(monkeypatch, tmp_path, ["alpha"])
    with pytest.raises(ValueError, match="unknown selected"):
        module.run_build(Config(cache_dir=tmp_path), regions=["missing"])


def test_empty_selection_is_not_mistaken_for_full_source(tmp_path, monkeypatch):
    from osm_worldcover.config import Config

    module = TestRunBuild()._patch(monkeypatch, tmp_path, ["alpha"])
    with pytest.raises(ValueError, match="nonempty"):
        module.run_build(Config(cache_dir=tmp_path), regions=[])


def test_subset_does_not_assemble_previously_cached_unselected_regions(tmp_path, monkeypatch):
    from osm_worldcover.config import Config

    module = TestRunBuild()._patch(monkeypatch, tmp_path, ["alpha", "beta"])
    config = Config(cache_dir=tmp_path, out_dir=tmp_path / "out")
    module.run_build(config)
    subset = module.run_build(config, regions=["alpha"])
    assert subset.result.rows == 1
    assert subset.processing["expected_regions"] == ["alpha", "beta"]
    assert subset.processing["missing_regions"] == ["beta"]
    assert not subset.result.manifest["processing"]["full_source_complete"]


def test_changed_minimum_words_forces_region_reprocessing(tmp_path, monkeypatch):
    from osm_worldcover.config import Config

    module = TestRunBuild()._patch(monkeypatch, tmp_path, ["alpha"])
    config = Config(cache_dir=tmp_path, out_dir=tmp_path / "out")
    module.run_build(config)
    visited = []
    original = module.run_region

    def recording_run(*args, **kwargs):
        visited.append(args[1].stem)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "run_region", recording_run)
    module.run_build(config.with_overrides(min_words=1))
    assert visited == ["alpha"]


def test_resumed_build_preserves_full_processing_ledger(tmp_path, monkeypatch):
    from osm_worldcover.config import Config

    module = TestRunBuild()._patch(monkeypatch, tmp_path, ["alpha", "beta"])
    config = Config(cache_dir=tmp_path, out_dir=tmp_path / "out")
    original = module.run_region

    def with_rejections(*args, **kwargs):
        rows, outcome = original(*args, **kwargs)
        outcome.polygons_seen += 3
        outcome.rejections["below_threshold"] = 3
        outcome.polygons_seen += 2
        outcome.polygons_accepted += 2
        outcome.text_rejections["text_too_short"] = 2
        return rows, outcome

    monkeypatch.setattr(module, "run_region", with_rejections)
    first = module.run_build(config)
    second = module.run_build(config)
    assert first.processing == second.processing
    assert second.rejections == {"below_threshold": 6}
    assert second.processing["totals"]["text_rejections"] == {"text_too_short": 4}
    assert second.result.manifest["rejections"] == {"below_threshold": 6}


def test_named_source_revision_is_resolved_to_a_commit(tmp_path, monkeypatch):
    from osm_worldcover.config import Config

    module = TestRunBuild()._patch(monkeypatch, tmp_path, ["alpha"])
    requested = []

    def resolve(repository, revision):
        requested.append(revision)
        return "b" * 40

    monkeypatch.setattr(module.hub, "resolve_revision", resolve)
    report = module.run_build(
        Config(cache_dir=tmp_path, out_dir=tmp_path / "out", source_revision="main")
    )
    assert requested == ["main"]
    assert report.processing["context"]["settings"]["source_revision"] == "b" * 40


class _Spy:
    """Records every call as ``(name, args, kwargs)`` and returns canned values."""

    def __init__(self, **returns) -> None:
        self.calls: list[tuple] = []
        self.returns = returns

    def __call__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return self.returns.get(name)

        return record


def _recipe(name="wikidata"):
    from osm_worldcover.sources import recipe_for

    return recipe_for(name)


@pytest.mark.parametrize("source", ["wikidata", "website"])
def test_release_source_deletes_only_that_regions_tables(tmp_path, monkeypatch, source) -> None:
    from osm_worldcover import build as module

    recipe = _recipe(source)
    doomed = [tmp_path / path for path in recipe.region_paths("alpha")]
    bystander = tmp_path / "keep.txt"
    for path in [*doomed[1:], bystander]:  # the first table is already gone
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
    asked = []
    real = module.hub.region_files
    monkeypatch.setattr(module.hub, "region_files", lambda *a: asked.append(a) or real(*a))

    module._release_source(tmp_path, "alpha", recipe)

    assert not any(path.exists() for path in doomed)
    assert bystander.exists()
    # The default source keeps the one-argument call; others pass their recipe.
    assert asked == [("alpha",) if source == "wikidata" else ("alpha", recipe)]


class _Store:
    """Minimal ShardStore double: only what the region pass touches."""

    def __init__(self, spy: _Spy, cached=None) -> None:
        self.write_outcome = spy("write")
        self._cached = cached or {}

    def outcome(self, stem):
        return self._cached.get(stem)


def test_process_region_fetches_labels_records_and_releases(tmp_path, monkeypatch) -> None:
    from osm_worldcover import build as module
    from osm_worldcover.adapters.source import RegionTables
    from osm_worldcover.config import Config
    from osm_worldcover.pipeline import RegionOutcome

    config = Config(cache_dir=tmp_path, source_dataset="owner/data")
    outcome = RegionOutcome("alpha", polygons_seen=5, polygons_accepted=3, examples=2)
    tables, examples, tiles = object(), object(), object()
    spy = _Spy(run_region=(examples, outcome), load=tables)
    monkeypatch.setattr(module.hub, "snapshot_region", spy("snapshot"))
    monkeypatch.setattr(RegionTables, "load", classmethod(lambda cls, *a: spy("load")(*a)))
    monkeypatch.setattr(module, "run_region", spy("run_region"))
    monkeypatch.setattr(module, "_release_source", spy("release"))
    messages = []

    result = module._process_region(
        config, "r" * 40, "alpha", tmp_path / "raw", tiles, _Store(spy), True, messages.append
    )

    assert result is outcome
    recipe = config.source_recipe
    assert spy.calls == [
        ("snapshot", ("owner/data", "r" * 40, "alpha", tmp_path / "raw", recipe), {}),
        ("load", (tmp_path / "raw", "alpha"), {}),
        ("run_region", (config, tables, tiles), {"keep_tiles": True}),
        ("write", ("alpha", examples, outcome), {}),
        ("release", (tmp_path / "raw", "alpha", recipe), {}),
    ]
    assert messages == ["    5 polygons -> 3 labelled -> 2 examples"]


def test_process_region_hands_a_named_source_recipe_to_the_loader(tmp_path, monkeypatch) -> None:
    from osm_worldcover import build as module
    from osm_worldcover.adapters.source import RegionTables
    from osm_worldcover.config import Config
    from osm_worldcover.pipeline import RegionOutcome

    config = Config(cache_dir=tmp_path, source="website")
    spy = _Spy()
    loaded = []
    monkeypatch.setattr(module.hub, "snapshot_region", lambda *a, **k: None)
    monkeypatch.setattr(
        RegionTables, "load", classmethod(lambda cls, *a: loaded.append(a) or object())
    )
    monkeypatch.setattr(
        module, "run_region", lambda *a, **k: (None, RegionOutcome("alpha", examples=0))
    )
    monkeypatch.setattr(module, "_release_source", lambda *a: None)

    module._process_region(
        config, "r" * 40, "alpha", tmp_path, None, _Store(spy), False, lambda _: None
    )

    assert loaded == [(tmp_path, "alpha", config.source_recipe)]


def test_run_regions_numbers_regions_and_reuses_finished_ones(tmp_path, monkeypatch) -> None:
    from osm_worldcover import build as module
    from osm_worldcover.config import Config
    from osm_worldcover.pipeline import RegionOutcome

    cached = RegionOutcome("alpha", examples=1)
    fresh = RegionOutcome("beta", examples=2)
    shards = _Store(_Spy(), {"alpha": cached})
    processed = []

    def process(*args):
        processed.append(args)
        return fresh

    monkeypatch.setattr(module, "_process_region", process)
    seen = []
    config, tiles = Config(cache_dir=tmp_path), object()

    outcomes = module._run_regions(
        config, "r" * 40, ["alpha", "beta"], tmp_path, tiles, shards, True, seen.append
    )

    assert outcomes == [cached, fresh]
    assert seen == ["[1/2] alpha (already done)", "[2/2] beta"]
    assert processed == [(config, "r" * 40, "beta", tmp_path, tiles, shards, True, seen.append)]


def test_region_labels_mark_only_finished_regions() -> None:
    from osm_worldcover.build import _label

    assert _label(3, 9, "gamma", done=False) == "[3/9] gamma"
    assert _label(3, 9, "gamma", done=True) == "[3/9] gamma (already done)"


@pytest.mark.parametrize(
    ("revision", "resolved"),
    [
        ("a" * 40, False),
        ("A" * 40, False),
        ("a" * 39, True),
        ("a" * 41, True),
        ("g" * 40, True),
        (" " + "a" * 40, True),
        ("a" * 40 + "\n", True),
        (None, True),
    ],
)
def test_only_full_commit_hashes_skip_revision_resolution(monkeypatch, revision, resolved) -> None:
    from osm_worldcover import build as module
    from osm_worldcover.config import Config

    asked = []
    monkeypatch.setattr(
        module.hub, "resolve_revision", lambda dataset, rev: asked.append((dataset, rev)) or "pin"
    )
    config = Config(source_revision=revision, source_dataset="owner/data")

    assert module._resolve_revision(config) == ("pin" if resolved else revision)
    assert asked == ([("owner/data", revision)] if resolved else [])


def test_run_build_wires_every_stage_with_the_pinned_config(tmp_path, monkeypatch) -> None:
    from osm_worldcover import build as module
    from osm_worldcover.accounting import BuildContext
    from osm_worldcover.config import Config
    from osm_worldcover.pipeline import RegionOutcome

    config = Config(
        cache_dir=tmp_path / "cache",
        out_dir=tmp_path / "out",
        worldcover_version="v100",
        worldcover_year=2020,
        cached_tiles=7,
        source_dataset="owner/data",
    )
    pinned = config.with_overrides(source_revision="c" * 40)
    outcome = RegionOutcome("alpha", examples=1)
    staged = tmp_path / "staged"
    spy = _Spy(list=["alpha", "beta"], ledger={"ledger": True}, run=[outcome], finalize="built")
    monkeypatch.setattr(module.hub, "list_region_stems", spy("list"))
    monkeypatch.setattr(module.hub, "resolve_revision", lambda *a: "c" * 40)
    monkeypatch.setattr(module, "processing_ledger", spy("ledger"))
    tiles = object()
    spy.returns["tiles"] = tiles
    monkeypatch.setattr(module, "WorldCoverTiles", spy("tiles"))
    monkeypatch.setattr(module, "_run_regions", spy("run"))
    monkeypatch.setattr(module.ShardStore, "stage", lambda self, *a: spy("stage")(*a) or staged)
    monkeypatch.setattr(module, "finalize_shards", spy("finalize"))

    def progress(_message: str) -> None:
        return None

    report = module.run_build(config, regions=["alpha"], keep_tiles=True, progress=progress)

    context = BuildContext.from_config(pinned)
    names = [name for name, _, _ in spy.calls]
    assert names == ["list", "ledger", "tiles", "run", "ledger", "stage", "finalize"]
    calls = {name: (args, kwargs) for name, args, kwargs in spy.calls}
    assert calls["list"] == (("owner/data", "c" * 40, pinned.source_recipe), {})
    assert spy.calls[1][1] == (["alpha", "beta"], ["alpha"], [], context)
    assert calls["tiles"] == (
        (tmp_path / "cache" / "worldcover",),
        {"version": "v100", "year": 2020, "max_cached_tiles": 7},
    )
    run_args = calls["run"][0]
    assert run_args[:4] == (pinned, "c" * 40, ["alpha"], tmp_path / "cache" / "source")
    assert run_args[4] is tiles
    assert run_args[5].directory == tmp_path / "cache" / "shards"
    assert run_args[5].context == context
    assert run_args[6:] == (True, progress)
    assert spy.calls[4][1] == (["alpha", "beta"], ["alpha"], [outcome], context)
    assert calls["stage"] == ((["alpha"], tmp_path / "cache" / "assembly" / "selected-shards"), {})
    assert calls["finalize"] == (
        (staged, pinned, tmp_path / "cache" / "assembly", tmp_path / "out", {}),
        {"processing": {"ledger": True}},
    )
    ledger_file = tmp_path / "cache" / "processing-ledger.json"
    assert json.loads(ledger_file.read_text()) == {"ledger": True}
    # Compare listed names too: macOS resolves paths case-insensitively.
    assert "processing-ledger.json" in [path.name for path in ledger_file.parent.iterdir()]
    assert report.result == "built"
    assert report.regions == [outcome]
    assert report.processing == {"ledger": True}


@pytest.mark.parametrize("stem", ["", ".", "..", "a/b", "../x", "/abs"])
def test_unsafe_region_names_are_refused(tmp_path, stem) -> None:
    with pytest.raises(ValueError, match="unsafe region name"):
        ShardStore(tmp_path).path_for(stem)


def test_a_zero_byte_shard_is_not_a_finished_one(tmp_path) -> None:
    store = ShardStore(tmp_path)
    store.path_for("empty").write_bytes(b"")
    store.path_for("tiny").write_bytes(b"x")

    assert not store.has("empty")
    assert store.has("tiny")
    assert not store.has("absent")


def test_shards_are_written_atomically_with_a_plain_index(tmp_path, monkeypatch) -> None:
    from osm_worldcover import build as module

    seen = []
    real = module.tempfile.mkstemp
    monkeypatch.setattr(module.tempfile, "mkstemp", lambda **k: seen.append(k) or real(**k))
    custom = frame(2)
    custom.index = [7, 9]

    store = ShardStore(tmp_path)
    store.write("alpha", custom)

    assert seen[0] == {"prefix": ".alpha.parquet.", "dir": tmp_path}
    stored = pd.read_parquet(store.path_for("alpha"))
    assert list(stored.columns) == list(custom.columns)
    assert stored.index.tolist() == [0, 1]  # the caller's index is not persisted
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "alpha.parquet",
        "alpha.rejections.json",
    ]


def test_configured_regions_apply_unless_a_selection_is_passed() -> None:
    from osm_worldcover.build import _selected_regions
    from osm_worldcover.config import Config

    config = Config(regions=["alpha"])

    assert _selected_regions(["alpha", "beta"], None, config) == ["alpha"]
    assert _selected_regions(["alpha", "beta"], ["beta"], config) == ["beta"]
    assert _selected_regions(["alpha", "beta"], None, Config()) == ["alpha", "beta"]


def test_outcome_stems_and_outcomes_must_line_up() -> None:
    from osm_worldcover.build import _missing_outcome_stems

    assert _missing_outcome_stems(["a", "b"], [None, object()]) == ["a"]
    with pytest.raises(ValueError):
        _missing_outcome_stems(["a", "b"], [None])
    with pytest.raises(ValueError):
        _missing_outcome_stems(["a"], [None, None])


def test_run_build_defaults_to_quiet_and_discards_tiles(tmp_path, monkeypatch) -> None:
    from osm_worldcover.config import Config

    module = TestRunBuild()._patch(monkeypatch, tmp_path, ["alpha"])
    keep = []
    original = module.run_region
    monkeypatch.setattr(
        module, "run_region", lambda *a, **k: keep.append(k["keep_tiles"]) or original(*a, **k)
    )

    module.run_build(Config(cache_dir=tmp_path, out_dir=tmp_path / "out"))

    assert keep == [False]
