"""The Python version named by the project, the local pin and the Docker image agree."""

import re
import tomllib
from pathlib import Path


def _repository_root(test_path: Path) -> Path:
    """Resolve the checkout root for regular and mutmut-copied test paths."""
    test_root = test_path.resolve().parents[2]
    return test_root.parent if test_root.name == "mutants" else test_root


ROOT = _repository_root(Path(__file__))


def _requires_python_minimum() -> str:
    """Return the ``X.Y`` lower bound of pyproject's ``requires-python``."""
    spec = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["requires-python"]
    match = re.fullmatch(r">=(\d+\.\d+)", spec)
    assert match, f"expected a '>=X.Y' lower bound, got {spec!r}"
    return match[1]


def _dockerfile_python() -> str:
    """Return the ``X.Y`` Python named by the Dockerfile's base image tag."""
    dockerfile = (ROOT / "Dockerfile").read_text()
    match = re.search(r"^FROM \S+:\S*-python(\d+\.\d+)-", dockerfile, flags=re.MULTILINE)
    assert match, "the Dockerfile FROM line does not name a python<X.Y> base image"
    return match[1]


def _local_python_pin() -> str:
    """Return the version in ``.python-version``, the pin uv and editors read."""
    return (ROOT / ".python-version").read_text().strip()


def test_dockerfile_base_python_matches_requires_python() -> None:
    """The image must run the oldest interpreter the project declares it supports."""
    assert _dockerfile_python() == _requires_python_minimum()


def test_local_python_pin_matches_dockerfile_base() -> None:
    """Local development and the release image must use the same interpreter line."""
    assert _local_python_pin() == _dockerfile_python()
