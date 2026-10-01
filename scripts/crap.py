"""Report CRAP scores for every source and utility function.

CRAP = complexity^2 * (1 - coverage)^3 + complexity

Per-function coverage intersects each Radon function span with the executable
lines recorded by coverage.py. Functions missing from the coverage report are
treated as uncovered so utility code cannot disappear from the quality gate.
"""

import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

THRESHOLD = 6.0


def crap(complexity: int, coverage: float) -> float:
    """Return the CRAP score for one function."""
    return complexity**2 * (1.0 - coverage) ** 3 + complexity


def blocks(targets: Sequence[str]) -> list[dict[str, Any]]:
    """Return every function, method, and closure in the requested roots."""
    raw = subprocess.run(
        [sys.executable, "-m", "radon", "cc", "-j", *targets],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    report = json.loads(raw)
    _reject_radon_errors(report)
    blocks = []
    seen = set()
    for path, items in report.items():
        for item in items:
            for function in _function_blocks(item):
                identity = _block_identity(path, function)
                if identity not in seen:
                    seen.add(identity)
                    blocks.append({**function, "path": path})
    return blocks


def _function_blocks(item: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    """Walk Radon's nested class methods and function closures recursively."""
    blocks = []
    if item.get("type") in {"function", "method"}:
        blocks.append(item)
    for child_key in ("methods", "closures"):
        for child in item.get(child_key, ()):
            blocks.extend(_function_blocks(child))
    return blocks


def _block_identity(path: str, item: Mapping[str, Any]) -> tuple:
    """Identify duplicate Radon representations of a function or method."""
    return (
        path,
        item.get("type"),
        item.get("classname"),
        item.get("name"),
        item.get("lineno"),
        item.get("endline"),
    )


def _reject_radon_errors(report: Mapping[str, Any]) -> None:
    errors = [f"{path}: {item['error']}" for path, item in report.items() if "error" in item]
    if errors:
        raise ValueError("Radon could not measure: " + "; ".join(errors))


def _coverage(block: Mapping[str, Any], files: Mapping[str, Any]) -> float:
    """Return measured line coverage, counting an absent report as zero."""
    file_data = files.get(block["path"])
    span = set(range(block["lineno"], block["endline"] + 1))
    if file_data is None:
        return 0.0
    executed = set(file_data["executed_lines"]) & span
    measured = (set(file_data["executed_lines"]) | set(file_data["missing_lines"])) & span
    return len(executed) / len(measured) if measured else 0.0


def _score_rows(measured: Sequence[dict[str, Any]], files: Mapping[str, Any]) -> list[tuple]:
    """Score every function Radon found, including functions with no test data."""
    rows = []
    for block in measured:
        coverage = _coverage(block, files)
        score = crap(block["complexity"], coverage)
        rows.append((score, block["path"], block["name"], block["complexity"], coverage))
    return sorted(rows, reverse=True)


def _print_report(rows: Sequence[tuple]) -> int:
    """Print the highest scores and every strict-gate violation."""
    if not rows:
        print("CRAP gate failed: no code blocks were measured", file=sys.stderr)
        return 2
    violations = [row for row in rows if row[0] >= THRESHOLD]
    _print_rows(rows[:15])
    _print_violations(violations)
    _print_summary(rows, violations)
    return 1 if violations else 0


def _print_rows(rows: Sequence[tuple]) -> None:
    """Print the ranked report head."""
    print(f"{'CRAP':>7}  {'cplx':>4}  {'cov':>6}  location")
    for score, path, name, complexity, coverage in rows[:15]:
        print(f"{score:7.2f}  {complexity:4d}  {coverage:6.1%}  {path}:{name}")


def _print_violations(violations: Sequence[tuple]) -> None:
    """List every score that fails the strict limit."""
    if violations:
        print("\nCRAP scores at or above the strict limit:")
        for score, path, name, complexity, coverage in violations:
            print(f"{score:7.2f}  {complexity:4d}  {coverage:6.1%}  {path}:{name}")


def _print_summary(rows: Sequence[tuple], violations: Sequence[tuple]) -> None:
    """Report the maximum score, violation count, and the absence of exceptions."""
    highest = rows[0][0] if rows else 0.0
    print(
        f"\nMeasured blocks: {len(rows)}; highest CRAP: {highest:.2f}; "
        f"scores >= {THRESHOLD:.0f}: {len(violations)}; allowlisted exceptions: 0"
    )


def main() -> int:
    """Measure all requested code roots using the complete coverage report."""
    targets = sys.argv[1:] or ["src"]
    coverage_path = Path("coverage.json")
    if not coverage_path.exists():
        print("run: pytest --cov --cov-report=json", file=sys.stderr)
        return 2
    files = json.loads(coverage_path.read_text())["files"]
    return _print_report(_score_rows(blocks(targets), files))


if __name__ == "__main__":
    raise SystemExit(main())
