"""The quality gates measure every code root and enforce their documented floors."""

import json
import sys

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


def test_callable_complexity_does_not_include_nested_bodies(tmp_path) -> None:
    blocks = _nested_blocks(tmp_path)
    complexities = {block["qualified_name"]: block["complexity"] for block in blocks}

    assert complexities == {
        "Outer.Inner.compute": 2,
        "Outer.outer_method": 1,
        "Outer.outer_method.inner": 2,
        "factory": 1,
        "factory.Local.compute": 2,
        "factory.Local.Deep.measure": 2,
        "factory.nested_async": 2,
    }


def test_blocks_distinguish_methods_from_nested_functions(tmp_path) -> None:
    blocks = _nested_blocks(tmp_path)
    callable_types = {block["qualified_name"]: block["type"] for block in blocks}

    assert callable_types == {
        "Outer.Inner.compute": "method",
        "Outer.outer_method": "method",
        "Outer.outer_method.inner": "function",
        "factory": "function",
        "factory.Local.compute": "method",
        "factory.Local.Deep.measure": "method",
        "factory.nested_async": "function",
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


@pytest.mark.parametrize(
    ("stats", "expected", "stdout", "stderr"),
    [
        ({"killed": 0, "timeout": 0, "survived": 0}, 2, "", "no mutants were run"),
        ({"killed": 7, "timeout": 1, "survived": 2}, 0, "80.0%", ""),
        ({"killed": 6, "timeout": 0, "survived": 2}, 1, "", "below the 80% floor"),
    ],
)
def test_mutation_score_gate_reports_the_floor(
    tmp_path, monkeypatch, capsys, stats, expected, stdout, stderr
) -> None:
    stats_path = tmp_path / "mutmut-cicd-stats.json"
    monkeypatch.setattr(mutation_score, "STATS", stats_path)
    monkeypatch.setattr(mutation_score.subprocess, "run", lambda *args, **kwargs: None)
    stats_path.write_text(json.dumps(stats))

    assert mutation_score.main() == expected
    output = capsys.readouterr()
    assert stdout in output.out
    assert stderr in output.err


def test_mutation_score_gate_requires_a_report(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(mutation_score, "STATS", tmp_path / "missing.json")
    monkeypatch.setattr(mutation_score.subprocess, "run", lambda *args, **kwargs: None)

    assert mutation_score.main() == 2
    assert "no mutation stats" in capsys.readouterr().err
