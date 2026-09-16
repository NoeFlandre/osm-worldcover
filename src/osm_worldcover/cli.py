"""Command line interface."""

import shutil
from pathlib import Path
from typing import Annotated

import typer

from osm_worldcover.adapters import hub
from osm_worldcover.adapters.writer import read_manifest
from osm_worldcover.build import ShardStore, run_build
from osm_worldcover.config import Config
from osm_worldcover.finalize import StreamedBuild, finalize_shards
from osm_worldcover.sources import DEFAULT_SOURCE, recipe_for

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def build(
    source: Annotated[
        str, typer.Option(help="Named input source: wikidata, description, or website.")
    ] = DEFAULT_SOURCE,
    out: Annotated[Path, typer.Option(help="Directory to write the dataset into.")] = Path(
        "data/out"
    ),
    cache: Annotated[Path, typer.Option(help="Scratch directory for source and tiles.")] = Path(
        "data/cache"
    ),
    region: Annotated[
        list[str] | None, typer.Option(help="Region stem to build; repeatable.")
    ] = None,
    regions_file: Annotated[
        Path | None,
        typer.Option(help="File of region stems, one per line; # starts a comment."),
    ] = None,
    threshold: Annotated[float, typer.Option(help="Minimum dominant-class share.")] = 0.8,
    max_area_km2: Annotated[
        float, typer.Option(help="Refuse polygons larger than this, in km2.")
    ] = 10_000.0,
    revision: Annotated[
        str | None, typer.Option(help="Pin the source dataset to this commit.")
    ] = None,
    cached_tiles: Annotated[
        int,
        typer.Option(help="Released tiles kept on disk (~94 MB each) to avoid re-downloading."),
    ] = 8,
    keep_tiles: Annotated[
        bool, typer.Option(help="Keep downloaded tiles instead of discarding them.")
    ] = False,
    dataset_version: Annotated[str, typer.Option(help="Version of the output.")] = "1.0.0",
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
    out: Annotated[Path, typer.Option(help="Directory to write the dataset into.")] = Path(
        "data/out"
    ),
    source: Annotated[
        str, typer.Option(help="Named input source: wikidata, description, or website.")
    ] = DEFAULT_SOURCE,
    work: Annotated[Path, typer.Option(help="Scratch directory for assembly.")] = Path(
        "data/cache/assembly"
    ),
    threshold: Annotated[float, typer.Option(help="Minimum dominant-class share.")] = 0.8,
    revision: Annotated[str | None, typer.Option(help="Source commit to record.")] = None,
    dataset_version: Annotated[str, typer.Option(help="Version of the output.")] = "1.0.0",
) -> None:
    """Combine region shards into the published dataset.

    Separate from `build` so a run split across processes -- each producing its
    own shards -- can be assembled once, in one place.
    """
    config = Config(
        source=source,
        out_dir=out,
        threshold=threshold,
        source_revision=revision,
        dataset_version=dataset_version,
    )
    combined = _gather(shard_dirs, work)
    result = finalize_shards(combined, config, work, out, ShardStore(combined).rejections())
    if result.rows == 0:
        typer.echo(f"no rows found in {[str(d) for d in shard_dirs]}", err=True)
        raise typer.Exit(1)
    _report(result)


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
    return Config(
        source=source,
        out_dir=out,
        cache_dir=cache,
        threshold=threshold,
        max_polygon_area_m2=max_area_km2 * 1e6,
        cached_tiles=cached_tiles,
        source_revision=revision,
        dataset_version=dataset_version,
    )


def _report(result: StreamedBuild) -> None:
    """Report a finished build, failing if a guarantee broke."""
    typer.echo(f"\nexamples: {result.rows:,}")
    for name, count in result.manifest.get("counts", {}).get("examples", {}).items():
        typer.echo(f"  {name}: {count:,}")
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
    threshold: Annotated[float, typer.Option()] = 0.8,
) -> None:
    """Re-check a build on disk against every dataset guarantee."""
    from osm_worldcover.domain.validation import validate

    rows = _load_splits(build_dir)
    if rows is None:
        typer.echo(f"no splits found in {build_dir}", err=True)
        raise typer.Exit(1)
    report = validate(rows, threshold=threshold)
    typer.echo(f"rows: {report.rows:,}")
    if report.ok:
        typer.echo("OK: every guarantee holds")
        return
    for violation in report.violations:
        typer.echo(f"FAILED {violation.check.value}: {violation.count} {violation.examples}")
    raise typer.Exit(1)


def _load_splits(build_dir: Path) -> list[dict] | None:
    """Read every split written under ``build_dir``, or ``None`` if there are none."""
    import pandas as pd

    frames = [
        pd.read_parquet(build_dir / f"{split}.parquet")
        for split in ("train", "validation", "test")
        if (build_dir / f"{split}.parquet").exists()
    ]
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True).to_dict("records")


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
