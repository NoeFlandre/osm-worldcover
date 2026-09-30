"""The quality gates measure every code root and enforce their documented floors."""

import json
import sys
from types import SimpleNamespace

import pytest
from scripts import crap, mutation_score


def test_crap_formula() -> None:
    assert crap.crap(3, 0.5) == pytest.approx(4.125)


def test_blocks_collects_every_requested_root(monkeypatch) -> None:
    def fake_run(command, **kwargs):
        assert command[-2:] == ["src", "scripts"]
        assert kwargs["check"] is True
        return SimpleNamespace(
            stdout=json.dumps({"src/module.py": [{"name": "run"}], "scripts/tool.py": []})
        )

    monkeypatch.setattr(crap.subprocess, "run", fake_run)

    assert crap.blocks(["src", "scripts"]) == [{"name": "run", "path": "src/module.py"}]


def test_radon_errors_are_not_silently_ignored(monkeypatch) -> None:
    monkeypatch.setattr(
        crap.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout='{"broken.py": {"error": "syntax"}}'),
    )

    with pytest.raises(ValueError, match=r"broken\.py: syntax"):
        crap.blocks(["src", "scripts"])


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
    ("stats", "expected"),
    [
        (None, 2),
        ({"killed": 0, "timeout": 0, "survived": 0}, 2),
        ({"killed": 7, "timeout": 1, "survived": 2}, 0),
        ({"killed": 6, "timeout": 0, "survived": 2}, 1),
    ],
)
def test_mutation_score_gate(tmp_path, monkeypatch, capsys, stats, expected) -> None:
    stats_path = tmp_path / "mutmut-cicd-stats.json"
    monkeypatch.setattr(mutation_score, "STATS", stats_path)
    monkeypatch.setattr(mutation_score.subprocess, "run", lambda *args, **kwargs: None)
    if stats is not None:
        stats_path.write_text(json.dumps(stats))

    assert mutation_score.main() == expected
    output = capsys.readouterr()
    if stats == {"killed": 7, "timeout": 1, "survived": 2}:
        assert "mutation score: 80.0%" in output.out
    if stats == {"killed": 6, "timeout": 0, "survived": 2}:
        assert "below the 80% floor" in output.err
