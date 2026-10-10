"""Executable contracts for the documented reproducible build workflow."""

import re
from pathlib import Path

import pytest


def _repository_root(test_path: Path) -> Path:
    """Resolve the checkout root for regular and mutmut-copied test paths."""
    test_root = test_path.resolve().parents[2]
    return test_root.parent if test_root.name == "mutants" else test_root


ROOT = _repository_root(Path(__file__))
WEBSITE_REVISION = "5c8e56a50b5679118a28aef057af002209f80a5e"
CANONICAL_WORKFLOW_DOC = ROOT / "README.md"
LINKING_DOCS = (ROOT / "docs/index.md", ROOT / "docs/technical-debt.md")
README_USE_LINK = "https://github.com/NoeFlandre/osm-worldcover#use"
REGION_COMMAND = 'uv run owc regions --source website --revision "$SOURCE_REVISION" > regions.txt'
PARTITION_COMMAND = (
    'awk \'NF { output = "regions-" ((count++ % 2) ? "b" : "a") '
    '".txt"; print > output }\' regions.txt'
)


def _workflow(path: Path) -> str:
    """Return the bash block that drives the split global workflow."""
    blocks = re.findall(r"```bash\n(.*?)```", path.read_text(), flags=re.DOTALL)
    return next(block for block in blocks if "regions-a.txt" in block)


def _assert_split_commands_are_in_order(workflow: str) -> None:
    partition_position = workflow.index(PARTITION_COMMAND)
    assert f"SOURCE_REVISION={WEBSITE_REVISION}" in workflow
    assert workflow.index(REGION_COMMAND) < partition_position
    assert all(
        workflow.index(region_file) > partition_position
        for region_file in ("regions-a.txt", "regions-b.txt")
    )


def _pinned_build_commands(workflow: str) -> list[str]:
    return [
        line.strip()
        for line in workflow.splitlines()
        if line.strip().startswith("uv run owc build")
        or line.strip().startswith("uv run owc assemble")
    ]


def _assert_build_commands_are_pinned(commands: list[str]) -> None:
    assert commands
    assert all("--source website" in command for command in commands)
    assert all('--revision "$SOURCE_REVISION"' in command for command in commands)


def test_documentation_root_escapes_mutmut_staging_directory() -> None:
    """Copied tests must read documentation from the real checkout."""
    staged_test = Path("/checkout/mutants/tests/unit/test_documentation_contract.py")
    assert _repository_root(staged_test) == Path("/checkout")


def test_split_workflow_creates_region_files_before_consuming_them() -> None:
    """The canonical split run must show its deterministic partition step."""
    _assert_split_commands_are_in_order(_workflow(CANONICAL_WORKFLOW_DOC))


def test_canonical_split_workflow_pins_every_build_and_assemble_command() -> None:
    """The canonical split run must remain reproducible at the pinned website head."""
    workflow = _workflow(CANONICAL_WORKFLOW_DOC)
    _assert_build_commands_are_pinned(_pinned_build_commands(workflow))


@pytest.mark.parametrize("path", LINKING_DOCS)
def test_other_docs_link_to_the_canonical_workflow_instead_of_copying_it(path: Path) -> None:
    """Only the README holds the split run; other pages link to its Use section."""
    text = path.read_text()
    assert README_USE_LINK in text
    assert "regions-a.txt" not in text
