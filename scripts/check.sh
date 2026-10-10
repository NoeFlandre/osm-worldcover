#!/usr/bin/env bash
# Quality gates, defined once. CI (.github/workflows/ci.yml) and the
# Development section of README.md both run them through this script.
#
#   scripts/check.sh          every gate of the quality job, in CI order
#   scripts/check.sh <gate>   one gate
#
# Gates: lint format types architecture tests crap docs mutation.
# mutation is slow and CI runs it in its own job, so the default run skips it.
set -euo pipefail

if [ "$#" -gt 1 ]; then
  echo "usage: scripts/check.sh [gate]" >&2
  exit 2
fi

cd "$(dirname "$0")/.."

QUALITY_GATES="lint format types architecture tests crap docs"

run_gate() {
  case "$1" in
    lint) uv run ruff check . ;;
    format) uv run ruff format --check . ;;
    types) uv run ty check src/ ;;
    architecture) uv run lint-imports ;;
    tests) uv run pytest --cov --cov-report=json -q ;;
    crap) uv run python scripts/crap.py src scripts tests ;;
    docs) uv run mkdocs build --strict ;;
    mutation) uv run mutmut run ;;
    *)
      echo "unknown gate: $1" >&2
      echo "gates: all $QUALITY_GATES mutation" >&2
      exit 2
      ;;
  esac
}

target="${1:-all}"
if [ "$target" = all ]; then
  for gate in $QUALITY_GATES; do
    run_gate "$gate"
  done
else
  run_gate "$target"
fi
