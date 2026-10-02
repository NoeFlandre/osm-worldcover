"""Report CRAP scores for every source and utility callable.

CRAP = complexity^2 * (1 - coverage)^3 + complexity

Radon's JSON report omits methods in nested and function-local classes. This
inventory walks Python syntax trees and asks Radon to measure each callable
individually. Coverage for an enclosing callable excludes nested callable
bodies, so child code cannot inflate its coverage score.
"""

import ast
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from radon.complexity import cc_visit_ast

THRESHOLD = 6.0


def crap(complexity: int, coverage: float) -> float:
    """Return the CRAP score for one function."""
    return complexity**2 * (1.0 - coverage) ** 3 + complexity


def blocks(targets: Sequence[str]) -> list[dict[str, Any]]:
    """Return every function, method, and nested callable in the requested roots."""
    measured = []
    for path in _source_paths(targets):
        visitor = _CallableInventory(str(path))
        visitor.visit(_read_module(path))
        measured.extend(visitor.blocks)
    return measured


def _source_paths(targets: Sequence[str]) -> list[Path]:
    """Expand the requested roots into a stable, duplicate-free Python file list."""
    paths = {}
    for target in targets:
        _collect_target(paths, Path(target))
    return sorted(paths.values(), key=str)


def _collect_target(paths: dict[Path, Path], root: Path) -> None:
    """Collect Python files from one explicit file or directory target."""
    if not root.exists():
        raise FileNotFoundError(root)
    for path in _target_files(root):
        _add_source_path(paths, path)


def _target_files(root: Path) -> Sequence[Path]:
    """Return Python files from a file target or recursively from a directory."""
    if root.is_file():
        return [root] if root.suffix == ".py" else []
    return list(root.rglob("*.py"))


def _add_source_path(paths: dict[Path, Path], path: Path) -> None:
    """Index a source file by its resolved location and retain a coverage path."""
    resolved = path.resolve()
    paths[resolved] = _display_path(resolved)


def _display_path(path: Path) -> Path:
    """Use coverage.py's working-directory-relative form when a file is local."""
    try:
        return path.relative_to(Path.cwd().resolve())
    except ValueError:
        return path


def _read_module(path: Path) -> ast.Module:
    """Parse one source file and report syntax failures with their path."""
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as error:
        raise ValueError(f"{path}: {error}") from error


class _CallableInventory(ast.NodeVisitor):
    """Collect callables once, retaining their lexical scope and owned lines."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.scopes: list[tuple[str, str]] = []
        self.blocks: list[dict[str, Any]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record(node)
        self._visit_scope(node, "function")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._record(node)
        self._visit_scope(node, "function")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_scope(node, "class")

    def _visit_scope(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
        kind: str,
    ) -> None:
        self.scopes.append((kind, node.name))
        for statement in node.body:
            self.visit(statement)
        self.scopes.pop()

    def _record(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        is_method = bool(self.scopes and self.scopes[-1][0] == "class")
        qualified_name = ".".join([*(name for _, name in self.scopes), node.name])
        self.blocks.append(
            {
                "path": self.path,
                "type": "method" if is_method else "function",
                "name": node.name,
                "qualified_name": qualified_name,
                "complexity": _complexity(node),
                "lineno": node.lineno,
                "endline": node.end_lineno or node.lineno,
                "owned_lines": _owned_lines(node),
            }
        )


def _complexity(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Measure one callable with Radon's cyclomatic-complexity rules."""
    module = ast.Module(body=[node], type_ignores=[])
    measured = cc_visit_ast(module)
    block = next(
        (item for item in measured if item.name == node.name and item.lineno == node.lineno),
        None,
    )
    if block is None:
        raise ValueError(f"Radon could not measure {node.name} at line {node.lineno}")
    return block.complexity


def _owned_lines(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[int]:
    """Return callable lines excluding nested function and class bodies."""
    endline = node.end_lineno or node.lineno
    excluded = _excluded_lines(node)
    return [line for line in range(node.lineno, endline + 1) if line not in excluded]


def _excluded_lines(node: ast.AST) -> set[int]:
    """Combine the executable body lines owned by child scopes."""
    return {line for child in _nested_scopes(node) for line in _scope_body_lines(child)}


def _scope_body_lines(
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
) -> range:
    """Return child-scope body lines while leaving its parent-owned header."""
    first_body_line = min(statement.lineno for statement in node.body)
    endline = node.end_lineno or node.lineno
    return range(max(node.lineno + 1, first_body_line), endline + 1)


def _nested_scopes(
    node: ast.AST,
) -> Sequence[ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef]:
    """Yield nearest nested scopes without descending into their owned bodies."""
    nested = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            nested.append(child)
        else:
            nested.extend(_nested_scopes(child))
    return nested


def _coverage(block: Mapping[str, Any], files: Mapping[str, Any]) -> float:
    """Return owned executable-line coverage, treating absent data as uncovered."""
    file_data = files.get(block["path"])
    owned = set(block.get("owned_lines", range(block["lineno"], block["endline"] + 1)))
    if file_data is None:
        return 0.0
    executed = set(file_data["executed_lines"]) & owned
    measured = (set(file_data["executed_lines"]) | set(file_data["missing_lines"])) & owned
    return len(executed) / len(measured) if measured else 0.0


def _score_rows(measured: Sequence[dict[str, Any]], files: Mapping[str, Any]) -> list[tuple]:
    """Score every discovered callable, including callables without test data."""
    rows = []
    for block in measured:
        coverage = _coverage(block, files)
        score = crap(block["complexity"], coverage)
        name = block.get("qualified_name", block["name"])
        rows.append((score, block["path"], name, block["complexity"], coverage))
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
