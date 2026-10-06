"""Command line interface."""

import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any

import typer

from osm_worldcover.adapters import hub
from osm_worldcover.adapters.writer import read_manifest
from osm_worldcover.assembly import verified_assembly
from osm_worldcover.build import ShardStore, run_build
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
from osm_worldcover.finalize import StreamedBuild, finalize_shards
from osm_worldcover.sources import DEFAULT_SOURCE, recipe_for

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def build(
    source: Annotated[
        str, typer.Option(help="Named input source: wikidata, description, or website.")
    ] = DEFAULT_SOURCE,
    out: Annotated[
        Path, typer.Option(help="Directory to write the dataset into.")
    ] = DEFAULT_OUT_DIR,
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
) -> None:
    """Build the dataset and write it to disk."""
    config = _build_config(
        source, out, cache, threshold, max_area_km2, cached_tiles, revision, dataset_version
    )
    regions = list(region or []) + _read_regions(regions_file)
    report = run_build(config, regions=regions or None, keep_tiles=keep_tiles, progress=typer.echo)
    _report(report.result)


def _read_regions(path: Path | None) -> list[str]:
    """Read region stems from ``path``, ignoring blanks and ``#`` comments."""
    if path is None:
        return []
    lines = (line.split("#", 1)[0].strip() for line in path.read_text().splitlines())
    return [line for line in lines if line]


@app.command()
def regions(
    source: Annotated[
        str, typer.Option(help="Named input source: wikidata, description, or website.")
    ] = DEFAULT_SOURCE,
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


@app.command()
def assemble(
    shard_dirs: Annotated[
        list[Path], typer.Argument(help="Directories of region shards to combine.")
    ],
    out: Annotated[Path, typer.Option(help="Directory to write the dataset into.")] = (
        DEFAULT_OUT_DIR
    ),
    source: Annotated[
        str | None, typer.Option(help="Require this source; otherwise use verified receipts.")
    ] = None,
    work: Annotated[Path, typer.Option(help="Scratch directory for assembly.")] = (
        DEFAULT_CACHE_DIR / "assembly"
    ),
    threshold: Annotated[
        float | None, typer.Option(help="Require this dominance threshold in the receipts.")
    ] = None,
    revision: Annotated[
        str | None, typer.Option(help="Require this pinned source commit in the receipts.")
    ] = None,
    dataset_version: Annotated[
        str | None, typer.Option(help="Require this output version in the receipts.")
    ] = None,
    legacy_code_revision: Annotated[
        str | None,
        typer.Option(
            help="Exact commit for verified legacy receipts that predate code revision fields."
        ),
    ] = None,
    allow_unverified_shards: Annotated[
        bool,
        typer.Option(help="Recover legacy shards without receipts; output is not publishable."),
    ] = False,
) -> None:
    """Combine region shards into the published dataset.

    Separate from `build` so a run split across processes -- each producing its
    own shards -- can be assembled once, in one place. By default all settings
    come from matching completion receipts; explicit options must agree. The
    pinned source inventory is checked, and incomplete subsets are marked as
    such. Legacy recovery uses wikidata/0.8/1.1.0 defaults and never proves
    whole-source completeness.
    """
    try:
        result = _assemble(
            shard_dirs,
            out,
            work,
            source,
            threshold,
            revision,
            dataset_version,
            legacy_code_revision,
            allow_unverified_shards,
        )
    except (OSError, ValueError) as error:
        typer.echo(f"assembly refused: {error}", err=True)
        raise typer.Exit(1) from error
    if result.rows == 0:
        typer.echo(f"no rows found in {[str(d) for d in shard_dirs]}", err=True)
        raise typer.Exit(1)
    _report(result)


def _assemble(
    shard_dirs: list[Path],
    out: Path,
    work: Path,
    source: str | None,
    threshold: float | None,
    revision: str | None,
    dataset_version: str | None,
    legacy_code_revision: str | None,
    allow_unverified: bool,
) -> StreamedBuild:
    """Keep the legacy recovery route visibly separate from verified assembly."""
    if allow_unverified:
        return _assemble_unverified(
            shard_dirs, out, work, source, threshold, revision, dataset_version
        )
    return _assemble_verified(
        shard_dirs,
        out,
        work,
        source,
        threshold,
        revision,
        dataset_version,
        legacy_code_revision,
    )


def _assemble_unverified(
    shard_dirs, out, work, source, threshold, revision, dataset_version
) -> StreamedBuild:
    typer.echo(
        "WARNING: recovering UNVERIFIED shards; provenance and full-source completion "
        "are unproven. This output is not publishable.",
        err=True,
    )
    config = Config(
        source=source if source is not None else DEFAULT_SOURCE,
        out_dir=out,
        threshold=threshold if threshold is not None else DEFAULT_THRESHOLD,
        source_revision=revision,
        dataset_version=dataset_version if dataset_version is not None else DEFAULT_DATASET_VERSION,
    )
    combined = _gather(shard_dirs, work)
    return finalize_shards(
        combined,
        config,
        work,
        out,
        ShardStore(combined).rejections(),
        processing={
            "schema_version": 1,
            "scope": "unverified",
            "complete": False,
            "full_source_complete": False,
            "selected_complete": False,
            "verified": False,
            "warning": "Legacy recovery: completion receipts and inventory were not verified.",
        },
    )


def _assemble_verified(
    shard_dirs, out, work, source, threshold, revision, dataset_version, legacy_code_revision
) -> StreamedBuild:
    inputs = verified_assembly(
        shard_dirs,
        out,
        work,
        {
            "source": source,
            "threshold": threshold,
            "source_revision": revision,
            "dataset_version": dataset_version,
        },
        legacy_code_revision=legacy_code_revision,
    )
    if not inputs.processing["full_source_complete"]:
        typer.echo(
            f"WARNING: incomplete source inventory: {len(inputs.processing['missing_regions'])} "
            "regions missing; this subset is not publishable as a complete source.",
            err=True,
        )
    return finalize_shards(
        inputs.shards, inputs.config, work, out, inputs.rejections, processing=inputs.processing
    )


def _build_config(
    source: str,
    out: Path,
    cache: Path,
    threshold: float,
    max_area_km2: float,
    cached_tiles: int,
    revision: str | None,
    dataset_version: str,
) -> Config:
    """Gather the CLI's options into one settings object."""
    return Config.from_cli(
        source=source,
        out_dir=out,
        cache_dir=cache,
        threshold=threshold,
        max_area_km2=max_area_km2,
        cached_tiles=cached_tiles,
        source_revision=revision,
        dataset_version=dataset_version,
    )


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


def _gather(shard_dirs: list[Path], work: Path) -> Path:
    """Link every shard into one directory, so assembly sees a single source.

    Names are prefixed with the directory's position, because two workers can
    each produce a file of the same name -- their directories are all called
    "shards", so the leaf name cannot tell them apart. The directory is emptied
    first so a previous assembly cannot leak into this one. Each shard's
    rejection counters travel with it, so a split build still reports why its
    polygons were refused.
    """
    if len(shard_dirs) == 1:
        return shard_dirs[0]
    combined = Path(work) / "shards"
    if combined.exists():
        shutil.rmtree(combined)
    combined.mkdir(parents=True)
    # The directory was just emptied, so no name can already be taken.
    for index, directory in enumerate(shard_dirs):
        _link_into(combined, Path(directory), f"{index:03d}")
    return combined


def _link_into(combined: Path, directory: Path, prefix: str) -> None:
    """Link one worker's shards and counters into the combined view."""
    for pattern in ("*.parquet", "*.rejections.json"):
        for path in sorted(directory.glob(pattern)):
            (combined / f"{prefix}__{path.name}").symlink_to(path.resolve())


@app.command()
def verify(
    build_dir: Annotated[Path, typer.Argument(help="A versioned build directory.")],
    threshold: Annotated[
        float, typer.Option(help="Minimum dominant-class fraction every row must meet.")
    ] = DEFAULT_THRESHOLD,
) -> None:
    """Re-check a build on disk against every dataset guarantee."""
    from osm_worldcover.domain.validation import validate

    rows = _load_splits(build_dir)
    if rows is None:
        typer.echo(f"no splits found in {build_dir}", err=True)
        raise typer.Exit(1)
    settings = _verification_settings(build_dir)
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
def publish(
    build_dir: Annotated[Path, typer.Argument(help="A versioned build directory.")],
    repo_id: Annotated[str, typer.Argument(help="Target dataset repo, e.g. user/name.")],
    private: Annotated[bool, typer.Option(help="Create the dataset private.")] = False,
) -> None:
    """Upload a build to the Hugging Face Hub with a generated dataset card."""
    from osm_worldcover.adapters.publish import publish_dataset

    url = publish_dataset(build_dir, repo_id, private=private)
    typer.echo(f"published: {url}")


@app.command()
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


def _print_audit_report(report) -> None:
    typer.echo(f"rows: {report.rows:,}")
    for warning in report.warnings:
        typer.echo(f"WARNING {warning.code}: {warning.count}")
    for problem in report.problems:
        typer.echo(f"FAILED {problem.code}: {problem.count} {problem.examples}", err=True)
    if not report.ok:
        raise typer.Exit(1)
    typer.echo("OK: independent release audit passed")


@app.command()
def info(build_dir: Annotated[Path, typer.Argument(help="A versioned build directory.")]) -> None:
    """Summarise a build's manifest."""
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


if __name__ == "__main__":
    app()
