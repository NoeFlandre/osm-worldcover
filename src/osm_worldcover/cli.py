"""Command line interface."""

import functools
import logging
import math
import sys
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from osm_worldcover import cli_assemble as _cli_assemble
from osm_worldcover.adapters import hub
from osm_worldcover.adapters.writer import read_manifest
from osm_worldcover.build import run_build
from osm_worldcover.config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CACHED_TILES,
    DEFAULT_DATASET_VERSION,
    DEFAULT_MAX_POLYGON_AREA_KM2,
    DEFAULT_OUT_DIR,
    DEFAULT_THRESHOLD,
    Config,
)
from osm_worldcover.domain.text import DEFAULT_MIN_WORDS
from osm_worldcover.finalize import StreamedBuild
from osm_worldcover.release import (
    migrate_release_lock as _migrate_release_lock,
)
from osm_worldcover.release import (
    release_read_lock,
)
from osm_worldcover.sources import DEFAULT_SOURCE, recipe_for, source_names

if TYPE_CHECKING:
    from osm_worldcover.adapters.audit import AuditReport

app = typer.Typer(add_completion=False, help=__doc__)
logger = logging.getLogger(__name__)


def _enable_verbose_logging(verbose: bool) -> None:
    """With --verbose, send the package's INFO logs to stderr; default output is unchanged."""
    if not verbose:
        return
    # Only this package is raised to INFO, so third-party libraries stay quiet.
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    logging.getLogger("osm_worldcover").setLevel(logging.INFO)


def _fail[**P, R](command: Callable[P, R]) -> Callable[P, R]:
    """Turn expected failures into a clean message and exit code 1, not a traceback."""

    @functools.wraps(command)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return command(*args, **kwargs)
        except KeyError as error:
            typer.echo(f"error: missing key {error}", err=True)
            raise typer.Exit(1) from error
        except (OSError, ValueError) as error:
            typer.echo(f"error: {error}", err=True)
            raise typer.Exit(1) from error

    return wrapper


SourceOption = Annotated[
    str, typer.Option(help=f"Named input source: {', '.join(source_names())}.")
]
BuildDirArgument = Annotated[Path, typer.Argument(help="A versioned build directory.")]
OutOption = Annotated[Path, typer.Option(help="Directory to write the dataset into.")]
VerboseOption = Annotated[
    bool, typer.Option("--verbose", "-v", help="Log run settings and steps to stderr.")
]


@app.command()
@_fail
def build(
    source: SourceOption = DEFAULT_SOURCE,
    out: OutOption = DEFAULT_OUT_DIR,
    cache: Annotated[
        Path, typer.Option(help="Scratch directory for source and tiles.")
    ] = DEFAULT_CACHE_DIR,
    region: Annotated[
        list[str] | None, typer.Option(help="Region stem to build; repeatable.")
    ] = None,
    regions_file: Annotated[
        Path | None,
        typer.Option(help="File of region stems, one per line; # starts a comment."),
    ] = None,
    threshold: Annotated[
        float, typer.Option(help="Minimum dominant-class share.")
    ] = DEFAULT_THRESHOLD,
    max_area_km2: Annotated[
        float, typer.Option(help="Refuse polygons larger than this, in km2.")
    ] = DEFAULT_MAX_POLYGON_AREA_KM2,
    revision: Annotated[
        str | None, typer.Option(help="Pin the source dataset to this commit.")
    ] = None,
    cached_tiles: Annotated[
        int,
        typer.Option(help="Released tiles kept on disk (~94 MB each) to avoid re-downloading."),
    ] = DEFAULT_CACHED_TILES,
    keep_tiles: Annotated[
        bool, typer.Option(help="Keep downloaded tiles instead of discarding them.")
    ] = False,
    dataset_version: Annotated[
        str, typer.Option(help="Version of the output.")
    ] = DEFAULT_DATASET_VERSION,
    verbose: VerboseOption = False,
) -> None:
    """Build the dataset and write it to disk."""
    _enable_verbose_logging(verbose)
    config = Config.from_cli(
        source=source,
        out_dir=out,
        cache_dir=cache,
        threshold=threshold,
        max_area_km2=max_area_km2,
        cached_tiles=cached_tiles,
        source_revision=revision,
        dataset_version=dataset_version,
    )
    regions = list(region or []) + _read_regions(regions_file)
    logger.info(
        "build source=%s threshold=%s revision=%s regions=%s out=%s",
        source,
        threshold,
        revision or "head",
        len(regions) or "all",
        out,
    )
    report = run_build(config, regions=regions or None, keep_tiles=keep_tiles, progress=typer.echo)
    _report(report.result)


def _read_regions(path: Path | None) -> list[str]:
    """Read region stems from ``path``, ignoring blanks and ``#`` comments."""
    if path is None:
        return []
    lines = (line.split("#", 1)[0].strip() for line in path.read_text().splitlines())
    return [line for line in lines if line]


@app.command()
@_fail
def regions(
    source: SourceOption = DEFAULT_SOURCE,
    revision: Annotated[
        str | None, typer.Option(help="Source commit to list; resolve the current head if omitted.")
    ] = None,
) -> None:
    """List the source's region stems, one per line."""
    try:
        recipe = recipe_for(source)
    except ValueError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error
    resolved_revision = revision or hub.resolve_revision(recipe.source_dataset)
    typer.echo(f"revision: {resolved_revision}", err=True)
    for stem in hub.list_region_stems(recipe.source_dataset, resolved_revision, recipe):
        typer.echo(stem)


def _report(result: StreamedBuild) -> None:
    """Report a finished build, failing if a guarantee broke."""
    typer.echo(f"\nexamples: {result.rows:,}")
    for name, count in result.manifest.get("counts", {}).get("examples", {}).items():
        typer.echo(f"  {name}: {count:,}")
    if result.paths:
        typer.echo(f"written: {result.paths[-1].parent}")
    if result.report.ok:
        return
    for violation in result.report.violations:
        typer.echo(f"  FAILED {violation.check.value}: {violation.count}", err=True)
    raise typer.Exit(1)


@app.command("migrate-release-lock")
@_fail
def migrate_release_lock(build_dir: BuildDirArgument) -> None:
    """Initialize the reader lock for an existing release directory."""
    lock_path = _migrate_release_lock(build_dir)
    typer.echo(f"release lock initialized: {lock_path}")


@app.command()
@_fail
def verify(
    build_dir: BuildDirArgument,
    threshold: Annotated[
        float | None,
        typer.Option(
            help="Minimum dominant-class fraction every row must meet. "
            "Defaults to the manifest's dominance_threshold."
        ),
    ] = None,
) -> None:
    """Re-check a build on disk against every dataset guarantee."""
    with release_read_lock(build_dir):
        _verify_locked(build_dir, threshold)


def _verify_locked(build_dir: Path, threshold: float | None) -> None:
    from osm_worldcover.domain.validation import validate

    settings = _verification_settings(build_dir)
    threshold = _verification_threshold(threshold, settings)
    rows = _load_splits(build_dir)
    if rows is None:
        typer.echo(f"no splits found in {build_dir}", err=True)
        raise typer.Exit(1)
    report = validate(
        rows, threshold=threshold, min_words=settings.get("min_words", DEFAULT_MIN_WORDS)
    )
    typer.echo(f"rows: {report.rows:,}")
    if report.ok:
        typer.echo("OK: every guarantee holds")
        return
    for violation in report.violations:
        typer.echo(f"FAILED {violation.check.value}: {violation.count} {violation.examples}")
    raise typer.Exit(1)


def _verification_threshold(explicit: float | None, settings: Mapping[str, Any]) -> float:
    """Pick the dominance threshold to verify against, rejecting values the policy cannot use.

    An explicit ``--threshold`` wins over the manifest, which wins over the default.
    The same range as the dominance decision applies to both: a finite number in (0, 1].
    Anything else would make the below-threshold check silently pass.
    """
    value = explicit
    if value is None:
        value = settings.get("dominance_threshold", DEFAULT_THRESHOLD)
    if not _is_dominance_threshold(value):
        raise ValueError(f"dominance threshold must be a finite number in (0, 1], got {value!r}")
    return float(value)


def _is_dominance_threshold(value: Any) -> bool:
    """Whether ``value`` is a finite real number the dominance decision accepts."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return math.isfinite(value) and 0.0 < value <= 1.0


def _verification_settings(build_dir: Path) -> dict[str, Any]:
    """Honor the release's source-specific text policy during readback."""
    if not (build_dir / "manifest.json").exists():
        return {}
    return read_manifest(build_dir).get("settings", {})


def _load_splits(build_dir: Path) -> Iterator[dict[str, Any]] | None:
    """Stream only validation columns, never materializing the global text set."""
    from osm_worldcover.domain.validation import REQUIRED_COLUMNS

    paths = [
        build_dir / f"{split}.parquet"
        for split in ("train", "validation", "test")
        if (build_dir / f"{split}.parquet").exists()
    ]
    if not paths:
        return None
    return _split_rows(paths, REQUIRED_COLUMNS)


def _split_rows(paths: list[Path], columns: tuple[str, ...]) -> Iterator[dict[str, Any]]:
    """Yield bounded Parquet batches for verification."""
    import pyarrow.parquet as pq

    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8192, columns=list(columns)):
            yield from batch.to_pylist()


@app.command()
@_fail
def publish(
    build_dir: BuildDirArgument,
    repo_id: Annotated[str, typer.Argument(help="Target dataset repo, e.g. user/name.")],
    private: Annotated[bool, typer.Option(help="Create the dataset private.")] = False,
) -> None:
    """Upload a build to the Hugging Face Hub with a generated dataset card."""
    from osm_worldcover.adapters.publish import publish_dataset

    url = publish_dataset(build_dir, repo_id, private=private)
    typer.echo(f"published: {url}")


@app.command()
@_fail
def audit(
    build_dir: Annotated[Path, typer.Argument(help="A versioned release directory.")],
    require_complete: Annotated[
        bool, typer.Option(help="Require receipts for every pinned source region.")
    ] = False,
    require_card: Annotated[bool, typer.Option(help="Check the generated dataset card.")] = False,
    strict_text_leakage: Annotated[
        bool, typer.Option(help="Reject identical text across splits even with different labels.")
    ] = False,
    report_file: Annotated[Path | None, typer.Option(help="Write a JSON audit report.")] = None,
) -> None:
    """Independently audit schema, every row, global split integrity and provenance."""
    import json

    from osm_worldcover.adapters.audit import audit_build

    report = audit_build(
        build_dir,
        require_complete=require_complete,
        require_card=require_card,
        strict_text_leakage=strict_text_leakage,
    )
    if report_file is not None:
        report_file.write_text(json.dumps(report.as_dict(), indent=2) + "\n")
    _print_audit_report(report)


def _print_audit_report(report: "AuditReport") -> None:
    typer.echo(f"rows: {report.rows:,}")
    for warning in report.warnings:
        typer.echo(f"WARNING {warning.code}: {warning.count}")
    for problem in report.problems:
        typer.echo(f"FAILED {problem.code}: {problem.count} {problem.examples}", err=True)
    if not report.ok:
        raise typer.Exit(1)
    typer.echo("OK: independent release audit passed")


@app.command()
@_fail
def info(build_dir: BuildDirArgument) -> None:
    """Summarise a build's manifest."""
    with release_read_lock(build_dir):
        manifest = read_manifest(build_dir)
    counts = manifest["counts"]["examples"]
    typer.echo(
        f"examples: {counts['total']:,}  "
        f"(train {counts['train']:,} / val {counts['validation']:,} / test {counts['test']:,})"
    )
    typer.echo("\nclasses:")
    for entry in manifest["class_distribution"]:
        typer.echo(
            f"  {entry['code']:>3} {entry['label']:<26} {entry['examples']:>9,}  "
            f"{entry['share'] * 100:5.1f}%"
        )
    typer.echo("\ntop languages:")
    for entry in manifest["language_distribution"][:10]:
        typer.echo(
            f"  {entry['language']:<6} {entry['examples']:>9,}  {entry['share'] * 100:5.1f}%"
        )


AssembleOptions = _cli_assemble.AssembleOptions
_assemble = _cli_assemble._assemble
_assemble_unverified = _cli_assemble._assemble_unverified
_assemble_verified = _cli_assemble._assemble_verified
_gather = _cli_assemble._gather
_link_into = _cli_assemble._link_into
assemble = _cli_assemble.register(app, _fail, _enable_verbose_logging, _report, logger)


if __name__ == "__main__":
    app()
