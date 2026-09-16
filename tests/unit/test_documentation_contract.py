"""Executable contracts for the documented reproducible build workflow."""

import re
from pathlib import Path

ROOT = Path(__file__).parents[2]
WEBSITE_REVISION = "5c8e56a50b5679118a28aef057af002209f80a5e"
WORKFLOW_DOCS = (
    ROOT / "README.md",
    ROOT / "docs/index.md",
    ROOT / "docs/technical-debt.md",
)
REGION_COMMAND = 'uv run owc regions --source website --revision "$SOURCE_REVISION" > regions.txt'
PARTITION_COMMAND = (
    'awk \'NF { output = "regions-" ((count++ % 2) ? "b" : "a") '
    '".txt"; print > output }\' regions.txt'
)


def _workflow(path: Path) -> str:
    """Return the bash block that drives the split global workflow."""
    blocks = re.findall(r"```bash\n(.*?)```", path.read_text(), flags=re.DOTALL)
    return next(block for block in blocks if "regions-a.txt" in block)


def test_split_workflow_creates_region_files_before_consuming_them() -> None:
    """Every documented split run must show its deterministic partition step."""
    for path in WORKFLOW_DOCS:
        workflow = _workflow(path)
        assert f"SOURCE_REVISION={WEBSITE_REVISION}" in workflow
        regions_position = workflow.index(REGION_COMMAND)
        partition_position = workflow.index(PARTITION_COMMAND)
        assert regions_position < partition_position
        for region_file in ("regions-a.txt", "regions-b.txt"):
            assert workflow.index(region_file) > partition_position


def test_technical_debt_workflow_pins_every_build_and_assemble_command() -> None:
    """Technical-debt examples must remain reproducible at the pinned website head."""
    workflow = _workflow(ROOT / "docs/technical-debt.md")
    commands = [
        line.strip()
        for line in workflow.splitlines()
        if line.strip().startswith("uv run owc build")
        or line.strip().startswith("uv run owc assemble")
    ]
    assert commands
    assert all("--source website" in command for command in commands)
    assert all('--revision "$SOURCE_REVISION"' in command for command in commands)
