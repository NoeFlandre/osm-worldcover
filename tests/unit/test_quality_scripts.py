"""The quality gates measure every code root and enforce their documented floors."""

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import crap, mutation_score

NESTED_CALLABLE_SOURCE = """\
class Outer:
    class Inner:
        def compute(self, value):
            if value:
                return 1
            return 2

    def outer_method(self, value):
        def inner(item):
            if item:
                return 3
            return 4
        return inner(value)

def factory(value):
    class Local:
        async def compute(self, item):
            if item:
                return 5
            return 6

        class Deep:
            def measure(self, item):
                if item:
                    return 7
                return 8

    async def nested_async(item):
        if item:
            return 9
        return 10
    return value
"""


def _nested_blocks(tmp_path):
    source = tmp_path / "nested.py"
    source.write_text(NESTED_CALLABLE_SOURCE)
    return crap.blocks([str(source)])


def test_crap_formula() -> None:
    assert crap.crap(3, 0.5) == pytest.approx(4.125)


def test_blocks_collects_every_requested_root(tmp_path) -> None:
    src = tmp_path / "src"
    scripts = tmp_path / "scripts"
    src.mkdir()
    scripts.mkdir()
    (src / "module.py").write_text("def run():\n    return 1\n")
    (scripts / "tool.py").write_text("async def check():\n    return 2\n")
    (tmp_path / "notes.txt").write_text("not Python source")

    blocks = crap.blocks([str(tmp_path), str(src), str(scripts), str(tmp_path / "notes.txt")])

    assert [(block["name"], block["type"]) for block in blocks] == [
        ("check", "function"),
        ("run", "function"),
    ]


def test_blocks_include_nested_classes_local_classes_and_async_functions(tmp_path) -> None:
    blocks = _nested_blocks(tmp_path)
    qualified_names = [block["qualified_name"] for block in blocks]

    assert qualified_names == [
        "Outer.Inner.compute",
        "Outer.outer_method",
        "Outer.outer_method.inner",
        "factory",
        "factory.Local.compute",
        "factory.Local.Deep.measure",
        "factory.nested_async",
    ]
    assert len({(block["path"], block["lineno"]) for block in blocks}) == len(blocks)


def test_callable_complexity_and_type_exclude_nested_bodies(tmp_path) -> None:
    blocks = _nested_blocks(tmp_path)
    actual = {block["qualified_name"]: (block["complexity"], block["type"]) for block in blocks}

    assert actual == {
        "Outer.Inner.compute": (2, "method"),
        "Outer.outer_method": (1, "method"),
        "Outer.outer_method.inner": (2, "function"),
        "factory": (1, "function"),
        "factory.Local.compute": (2, "method"),
        "factory.Local.Deep.measure": (2, "method"),
        "factory.nested_async": (2, "function"),
    }


def test_crap_gate_rejects_an_empty_measured_inventory(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "coverage.json").write_text(json.dumps({"files": {}}))
    monkeypatch.setattr(sys, "argv", ["crap.py", "src", "scripts", "tests"])
    monkeypatch.setattr(crap, "blocks", lambda targets: [])

    assert crap.main() == 2
    assert "no code blocks were measured" in capsys.readouterr().err


def test_source_syntax_errors_are_not_silently_ignored(tmp_path) -> None:
    source = tmp_path / "broken.py"
    source.write_text("def broken(:\n    pass\n")

    with pytest.raises(ValueError, match=r"broken\.py: invalid syntax"):
        crap.blocks([str(source)])


def test_missing_source_root_is_reported(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        crap.blocks([str(tmp_path / "missing")])


def test_coverage_excludes_nested_callable_and_local_class_bodies(tmp_path) -> None:
    source = tmp_path / "owned.py"
    source.write_text(
        "def factory():\n"
        "    def nested():\n"
        "        if False:\n"
        "            return 1\n"
        "    class Local:\n"
        "        def method(self):\n"
        "            if False:\n"
        "                return 2\n"
        "    return nested\n"
    )
    factory = crap.blocks([str(source)])[0]
    coverage_files = {
        str(source): {
            "executed_lines": [1, 2, 5, 9],
            "missing_lines": [3, 4, 6, 7, 8],
        }
    }

    assert factory["owned_lines"] == [1, 2, 5, 9]
    assert crap._coverage(factory, coverage_files) == 1.0


def test_crap_gate_counts_unmeasured_functions_as_uncovered(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "coverage.json").write_text(json.dumps({"files": {}}))
    monkeypatch.setattr(sys, "argv", ["crap.py", "src", "scripts"])
    monkeypatch.setattr(
        crap,
        "blocks",
        lambda targets: [
            {
                "path": "src/uncovered.py",
                "name": "complex",
                "complexity": 3,
                "lineno": 1,
                "endline": 3,
            }
        ],
    )

    assert crap.main() == 1
    report = capsys.readouterr().out
    assert "highest CRAP: 12.00" in report
    assert "scores >= 6: 1" in report
    assert "allowlisted exceptions: 0" in report


def test_crap_gate_reports_a_clean_single_score(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "coverage.json").write_text(json.dumps({"files": {}}))
    monkeypatch.setattr(sys, "argv", ["crap.py", "scripts"])
    monkeypatch.setattr(
        crap,
        "blocks",
        lambda targets: [
            {
                "path": "scripts/simple.py",
                "name": "small",
                "complexity": 1,
                "lineno": 1,
                "endline": 1,
            }
        ],
    )

    assert crap.main() == 0
    assert "highest CRAP: 2.00" in capsys.readouterr().out


def _mutation_records(statuses):
    records = []
    for index, status in enumerate(statuses, start=1):
        function = "_add_raster"
        mangled = f"osm_worldcover.adapters.worldcover.x_{function}"
        records.append(
            {
                "name": f"{mangled}__mutmut_{index}",
                "path": mutation_score.RASTER_SOURCE,
                "function": function,
                "class": None,
                "mangled_function": mangled,
                "status": status,
            }
        )
    return records


def _mutmut_export(records):
    counts = {field: 0 for field in mutation_score.EXPORTED_STATUS_FIELDS.values()}
    for record in records:
        summary_field = mutation_score.STATUS_TO_SUMMARY_FIELD.get(record["status"])
        field = mutation_score.EXPORTED_STATUS_FIELDS.get(summary_field)
        if field is not None:
            counts[field] += 1
    return {**counts, "total": len(records)}


def _valid_mutation_baseline(git_sha):
    return {
        "tests_by_mangled_function_name": {
            "osm_worldcover.adapters.worldcover.x__add_raster": ["test_raster_chunking"]
        },
        "duration_by_test": {},
        "stats_time": 1.0,
        "function_hashes": {},
        "function_dependencies": {},
        "config_fingerprint": {},
        "watched_file_hashes": {},
        "git_commit": git_sha,
    }


def _prepare_mutation_gate(
    tmp_path,
    monkeypatch,
    statuses,
    *,
    run_outcome="success",
    baseline_updates=None,
    export_updates=None,
    export_returncode=0,
    write_export=True,
):
    mutants = tmp_path / "mutants"
    mutants.mkdir()
    baseline_path = mutants / "mutmut-stats.json"
    stats_path = mutants / "mutmut-cicd-stats.json"
    started_ns = time.time_ns() - 1_000_000
    git_sha = "expected-commit"
    baseline = _valid_mutation_baseline(git_sha)
    baseline.update(baseline_updates or {})
    for field in baseline.pop("__remove_fields__", []):
        baseline.pop(field, None)
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    records = _mutation_records(statuses)
    export = _mutmut_export(records)
    export.update(export_updates or {})

    monkeypatch.setattr(mutation_score, "MUTANTS", mutants)
    monkeypatch.setattr(mutation_score, "BASELINE", baseline_path)
    monkeypatch.setattr(mutation_score, "STATS", stats_path)
    monkeypatch.setattr(mutation_score, "SCORE_REPORT", mutants / "mutation-score.json")
    monkeypatch.setattr(mutation_score, "RASTER_REPORT", mutants / "worldcover-raster-mutants.json")
    monkeypatch.setattr(mutation_score, "_mutant_records", lambda started_ns=None: records)
    monkeypatch.setattr(
        mutation_score, "get_diff_for_mutant", lambda name, path: f"diff for {name}"
    )
    monkeypatch.setenv("MUTATION_RUN_OUTCOME", run_outcome)
    monkeypatch.setenv("MUTATION_RUN_STARTED_NS", str(started_ns))
    monkeypatch.setenv("MUTATION_EXPECTED_GIT_SHA", git_sha)

    def fake_export(*args, **kwargs):
        if write_export:
            stats_path.write_text(json.dumps(export), encoding="utf-8")
        return SimpleNamespace(returncode=export_returncode, stderr="export problem")

    monkeypatch.setattr(mutation_score.subprocess, "run", fake_export)
    return records, baseline_path, stats_path


def test_mutation_records_keep_all_mutmut_outcomes_distinct() -> None:
    exit_codes = {
        f"package.module.x_function{index}__mutmut_1": code
        for index, code in enumerate((1, 0, 36, 5, 34, 35, None, -11, 2, 37, 99))
    }

    records = mutation_score._records_for_file("src/package/module.py", exit_codes)
    summary = mutation_score._summary(records)

    assert {field: summary[field] for field in mutation_score.SUMMARY_STATUS_FIELDS} == {
        "killed": 1,
        "survived": 1,
        "timeout": 1,
        "no_tests": 1,
        "skipped": 1,
        "suspicious": 2,
        "unreported": 1,
        "segfault": 1,
        "interrupted": 1,
        "caught_by_type_check": 1,
    }
    assert summary["total"] == 11
    assert summary["confirmed_kill_score"] == pytest.approx(1 / 11)


def test_per_file_scores_partition_the_global_summary() -> None:
    records = [
        {"path": "src/b.py", "status": "killed"},
        {"path": "src/a.py", "status": "killed"},
        {"path": "src/a.py", "status": "survived"},
        {"path": "src/b.py", "status": "timeout"},
        {"path": "src/b.py", "status": "killed"},
    ]

    per_file = mutation_score._per_file_summaries(records)

    assert list(per_file) == ["src/a.py", "src/b.py"]
    assert per_file["src/a.py"]["confirmed_kill_score"] == pytest.approx(0.5)
    assert per_file["src/b.py"]["killed"] == 2
    assert per_file["src/b.py"]["timeout"] == 1
    assert per_file["src/b.py"]["confirmed_kill_score"] == pytest.approx(2 / 3)
    assert sum(score["total"] for score in per_file.values()) == len(records)


def test_mutation_report_writes_per_file_scores(tmp_path, monkeypatch) -> None:
    _prepare_mutation_gate(tmp_path, monkeypatch, ["killed"] * 8 + ["timeout"] * 2)

    assert mutation_score.main() == 0
    report = json.loads((tmp_path / "mutants" / "mutation-score.json").read_text(encoding="utf-8"))
    raster = report["per_file"][mutation_score.RASTER_SOURCE]

    assert raster["total"] == report["summary"]["total"] == 10
    assert raster["confirmed_kill_score"] == pytest.approx(0.8)


def test_mutation_gate_uses_all_mutants_and_never_counts_timeouts_as_kills(
    tmp_path, monkeypatch, capsys
) -> None:
    _prepare_mutation_gate(
        tmp_path, monkeypatch, ["killed"] * 7 + ["timeout", "survived", "survived"]
    )

    assert mutation_score.main() == 1
    output = capsys.readouterr()
    assert all(
        line in output.out
        for line in (
            "confirmed-kill mutation score: 70.0% (7/10 total mutants)",
            "timeout: 1",
            "survived: 2",
        )
    )
    assert "below the 80% floor" in output.err


def test_mutation_gate_accepts_exactly_the_existing_eighty_percent_floor(
    tmp_path, monkeypatch, capsys
) -> None:
    _prepare_mutation_gate(tmp_path, monkeypatch, ["killed"] * 8 + ["timeout"] * 2)

    assert mutation_score.main() == 0
    assert "confirmed-kill mutation score: 80.0% (8/10 total mutants)" in capsys.readouterr().out


def test_mutation_gate_rejects_a_failed_mutmut_run(tmp_path, monkeypatch, capsys) -> None:
    _prepare_mutation_gate(tmp_path, monkeypatch, ["killed"] * 10, run_outcome="failure")

    assert mutation_score.main() == 2
    assert "did not complete successfully" in capsys.readouterr().err


def test_mutation_gate_rejects_export_command_failure(tmp_path, monkeypatch, capsys) -> None:
    _prepare_mutation_gate(
        tmp_path,
        monkeypatch,
        ["killed"] * 10,
        export_returncode=1,
        write_export=False,
    )

    assert mutation_score.main() == 2
    errors = capsys.readouterr().err
    assert "export failed with exit code 1" in errors
    assert "export problem" in errors


def test_mutation_gate_rejects_an_incomplete_baseline(tmp_path, monkeypatch, capsys) -> None:
    _prepare_mutation_gate(
        tmp_path,
        monkeypatch,
        ["killed"] * 10,
        baseline_updates={"__remove_fields__": ["tests_by_mangled_function_name"]},
    )

    assert mutation_score.main() == 2
    assert "baseline is incomplete" in capsys.readouterr().err


def test_mutation_gate_rejects_a_baseline_for_a_different_commit(
    tmp_path, monkeypatch, capsys
) -> None:
    _prepare_mutation_gate(
        tmp_path,
        monkeypatch,
        ["killed"] * 10,
        baseline_updates={"git_commit": "old-commit"},
    )

    assert mutation_score.main() == 2
    assert "does not match this CI checkout" in capsys.readouterr().err


def test_read_baseline_rejects_a_missing_file(tmp_path, monkeypatch) -> None:
    baseline = tmp_path / "missing.json"
    monkeypatch.setattr(mutation_score, "BASELINE", baseline)
    errors = []

    assert mutation_score._read_baseline(None, errors) is None
    assert errors == [f"mutation baseline is missing: {baseline}"]


@pytest.mark.parametrize(
    ("content", "message"),
    [("{", "cannot read valid JSON"), ("[]", "expected a JSON object")],
)
def test_read_baseline_rejects_malformed_json(tmp_path, monkeypatch, content, message) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text(content, encoding="utf-8")
    monkeypatch.setattr(mutation_score, "BASELINE", baseline)
    errors = []

    assert mutation_score._read_baseline(None, errors) is None
    assert message in errors[0]


def test_mutation_gate_rejects_a_stale_baseline(tmp_path, monkeypatch, capsys) -> None:
    _, baseline_path, _ = _prepare_mutation_gate(tmp_path, monkeypatch, ["killed"] * 10)
    started_ns = int(mutation_score.os.environ["MUTATION_RUN_STARTED_NS"])
    baseline_path.write_text(
        json.dumps(_valid_mutation_baseline("expected-commit")), encoding="utf-8"
    )
    os.utime(baseline_path, ns=(started_ns - 2, started_ns - 2))

    assert mutation_score.main() == 2
    assert "baseline is stale" in capsys.readouterr().err


def test_mutation_gate_rejects_a_stale_export(tmp_path, monkeypatch, capsys) -> None:
    _, _, stats_path = _prepare_mutation_gate(
        tmp_path,
        monkeypatch,
        ["killed"] * 10,
        write_export=False,
    )
    stats_path.write_text(
        json.dumps(_mutmut_export(_mutation_records(["killed"] * 10))), encoding="utf-8"
    )
    stale_ns = int(mutation_score.os.environ["MUTATION_RUN_STARTED_NS"]) - 2
    os.utime(stats_path, ns=(stale_ns, stale_ns))

    assert mutation_score.main() == 2
    assert "export is missing or stale" in capsys.readouterr().err


def test_mutation_gate_rejects_unreported_mutants_even_at_the_floor(
    tmp_path, monkeypatch, capsys
) -> None:
    _prepare_mutation_gate(
        tmp_path,
        monkeypatch,
        ["killed"] * 8 + ["timeout", "not checked"],
    )

    assert mutation_score.main() == 2
    output = capsys.readouterr()
    assert "confirmed-kill mutation score: 80.0% (8/10 total mutants)" in output.out
    assert "unreported: 1" in output.out
    assert "no completed result" in output.err


def test_mutation_gate_rejects_export_counts_that_disagree_with_raw_results(
    tmp_path, monkeypatch, capsys
) -> None:
    _prepare_mutation_gate(
        tmp_path,
        monkeypatch,
        ["killed"] * 10,
        export_updates={"killed": 9},
    )

    assert mutation_score.main() == 2
    assert "differs from raw count 10" in capsys.readouterr().err


def test_raster_report_records_tests_and_diff_for_nonkilled_mutants(tmp_path, monkeypatch) -> None:
    records = _mutation_records(["killed", "survived", "timeout"])
    monkeypatch.setattr(
        mutation_score, "get_diff_for_mutant", lambda name, path: f"diff for {name}"
    )
    baseline = _valid_mutation_baseline("expected-commit")

    report = mutation_score._raster_evidence(records, baseline)

    assert (
        report["mutants_total"],
        report["outcomes"],
        [item.get("tests") for item in report["mutants"]],
        [item.get("diff") for item in report["mutants"]],
    ) == (
        3,
        {"killed": 1, "survived": 1, "timeout": 1},
        [["test_raster_chunking"]] * 3,
        [None, f"diff for {records[1]['name']}", f"diff for {records[2]['name']}"],
    )


def test_mutant_records_reject_a_missing_metadata_file(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.py"
    source.write_text("def process():\n    return True\n", encoding="utf-8")
    monkeypatch.setattr(mutation_score, "MUTANTS", tmp_path / "mutants")
    monkeypatch.setattr(mutation_score, "walk_mutatable_files", lambda: [source])

    with pytest.raises(ValueError, match="per-mutant result file is missing"):
        mutation_score._mutant_records()


def test_mutant_records_reject_stale_metadata(tmp_path, monkeypatch) -> None:
    source = Path("src/source.py")
    mutants = tmp_path / "mutants"
    metadata_path = mutants / f"{source}.meta"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(json.dumps({"exit_code_by_key": {}}), encoding="utf-8")
    started_ns = metadata_path.stat().st_mtime_ns + 10
    monkeypatch.setattr(mutation_score, "MUTANTS", mutants)
    monkeypatch.setattr(mutation_score, "walk_mutatable_files", lambda: [source])

    with pytest.raises(ValueError, match="per-mutant result file is stale"):
        mutation_score._mutant_records(started_ns)


def test_mutant_metadata_reader_accepts_fresh_results(tmp_path, monkeypatch) -> None:
    source = Path("src/source.py")
    mutants = tmp_path / "mutants"
    metadata_path = mutants / f"{source}.meta"
    metadata_path.parent.mkdir(parents=True)
    expected = {"pkg.source.x_process__mutmut_1": 1}
    metadata_path.write_text(json.dumps({"exit_code_by_key": expected}), encoding="utf-8")
    started_ns = metadata_path.stat().st_mtime_ns - 10
    monkeypatch.setattr(mutation_score, "MUTANTS", mutants)

    assert mutation_score._mutant_exit_codes(source, started_ns) == expected


def test_mutant_metadata_reader_rejects_a_missing_result_map(tmp_path, monkeypatch) -> None:
    source = Path("src/source.py")
    mutants = tmp_path / "mutants"
    metadata_path = mutants / f"{source}.meta"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(mutation_score, "MUTANTS", mutants)

    with pytest.raises(TypeError, match="mutant exit-code data is missing"):
        mutation_score._mutant_exit_codes(source)


def test_mutant_record_rejects_invalid_keys_and_exit_codes() -> None:
    with pytest.raises(ValueError, match="invalid mutmut result"):
        mutation_score._mutant_record("src/module.py", None, 1)
    with pytest.raises(ValueError, match="invalid mutmut key"):
        mutation_score._mutant_record("src/module.py", "broken", 1)
    with pytest.raises(ValueError, match="invalid mutmut result"):
        mutation_score._mutant_record("src/module.py", "pkg.module.x_process__mutmut_1", "1")


def test_mutant_records_reject_an_empty_mutation_scope(monkeypatch) -> None:
    monkeypatch.setattr(mutation_score, "walk_mutatable_files", lambda: [])

    with pytest.raises(ValueError, match="no configured source files"):
        mutation_score._mutant_records()


def test_raster_evidence_rejects_an_empty_scope() -> None:
    with pytest.raises(ValueError, match="no mutation results found"):
        mutation_score._raster_evidence([], _valid_mutation_baseline("expected-commit"))


def test_raster_evidence_rejects_malformed_test_names() -> None:
    record = _mutation_records(["survived"])[0]
    baseline = _valid_mutation_baseline("expected-commit")
    baseline["tests_by_mangled_function_name"][record["mangled_function"]] = [None]

    with pytest.raises(TypeError, match="invalid baseline tests"):
        mutation_score._raster_evidence([record], baseline)


@pytest.mark.parametrize("exit_code", [-6, -15, 99, 7])
def test_unknown_exit_codes_are_suspicious_exactly_as_mutmut_exports_them(exit_code) -> None:
    record = mutation_score._mutant_record(
        "src/osm_worldcover/x.py", "osm_worldcover.x.x_f__mutmut_1", exit_code
    )

    assert record["status"] == "suspicious"
    assert mutation_score.status_by_exit_code[exit_code] == "suspicious"


def test_a_missing_exit_code_stays_unreported_and_known_codes_keep_their_status() -> None:
    def status(code):
        return mutation_score._mutant_record("p.py", "m.x_f__mutmut_1", code)["status"]

    assert status(None) == "not checked"
    assert status(1) == "killed"
    assert status(0) == "survived"
    assert status(-11) == "segfault"
