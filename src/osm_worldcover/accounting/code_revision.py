"""The exact source commit of this checkout, when it is clean."""

import subprocess
from functools import lru_cache
from pathlib import Path

from osm_worldcover.domain.revision import is_full_revision


@lru_cache(maxsize=1)
def _current_code_revision() -> str | None:
    """Return the exact commit only for a clean source checkout."""
    # This file sits at src/osm_worldcover/accounting/, so the repository root
    # is four levels up from it.
    repository = Path(__file__).resolve().parents[3]
    status = _git_output(repository, "status", "--porcelain", "--untracked-files=no")
    if status is None or status:
        return None
    revision = _git_output(repository, "rev-parse", "--verify", "HEAD")
    return revision if is_full_revision(revision) else None


def _git_output(repository: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None
