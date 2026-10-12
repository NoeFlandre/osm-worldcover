"""Command-line assembly of region shards."""

import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer

from osm_worldcover.assembly import verified_assembly
from osm_worldcover.build import ShardStore
from osm_worldcover.config import DEFAULT_CACHE_DIR, DEFAULT_OUT_DIR, Config
from osm_worldcover.finalize import StreamedBuild, finalize_shards
from osm_worldcover.release import scratch_lock

OutOption = Annotated[Path, typer.Option(help="Directory to write the dataset into.")]
VerboseOption = Annotated[
    bool, typer.Option("--verbose", "-v", help="Log run settings and steps to stderr.")
]


@dataclass(frozen=True, slots=True)
class AssembleOptions:
    """Settings for one assembly, named so a revision and a dataset version cannot be swapped."""

    out: Path
    work: Path
    source: str | None
    threshold: float | None
    revision: str | None
    dataset_version: str | None
    legacy_code_revision: str | None
    allow_unverified: bool


def register(
    app: typer.Typer,
    fail: Callable[[Callable[..., Any]], Callable[..., Any]],
    enable_verbose_logging: Callable[[bool], None],
    report: Callable[[StreamedBuild], None],
    logger: logging.Logger,
) -> Callable[..., Any]:
    @fail
    def assemble(
        shard_dirs: Annotated[
            list[Path], typer.Argument(help="Directories of region shards to combine.")
        ],
        out: OutOption = DEFAULT_OUT_DIR,
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
        verbose: VerboseOption = False,
    ) -> None:
        """Combine region shards into the published dataset.

        Separate from `build` so a run split across processes -- each producing its
        own shards -- can be assembled once, in one place. By default all settings
        come from matching completion receipts; explicit options must agree. The
        pinned source inventory is checked, and incomplete subsets are marked as
        such. Legacy recovery uses wikidata/0.8/1.1.0 defaults and never proves
        whole-source completeness.
        """
        enable_verbose_logging(verbose)
        logger.info(
            "assemble shard_dirs=%d mode=%s out=%s",
            len(shard_dirs),
            "unverified" if allow_unverified_shards else "verified",
            out,
        )
        try:
            result = _assemble(
                shard_dirs,
                AssembleOptions(
                    out=out,
                    work=work,
                    source=source,
                    threshold=threshold,
                    revision=revision,
                    dataset_version=dataset_version,
                    legacy_code_revision=legacy_code_revision,
                    allow_unverified=allow_unverified_shards,
                ),
            )
        except (OSError, ValueError) as error:
            typer.echo(f"assembly refused: {error}", err=True)
            raise typer.Exit(1) from error
        if result.rows == 0:
            typer.echo(f"no rows found in {[str(d) for d in shard_dirs]}", err=True)
            raise typer.Exit(1)
        report(result)

    app.command()(assemble)
    return assemble


def _assemble(shard_dirs: list[Path], options: AssembleOptions) -> StreamedBuild:
    """Keep the legacy recovery route visibly separate from verified assembly."""
    with scratch_lock(options.work):
        if options.allow_unverified:
            return _assemble_unverified(shard_dirs, options)
        return _assemble_verified(shard_dirs, options)


def _assemble_unverified(shard_dirs: list[Path], options: AssembleOptions) -> StreamedBuild:
    typer.echo(
        "WARNING: recovering UNVERIFIED shards; provenance and full-source completion "
        "are unproven. This output is not publishable.",
        err=True,
    )
    config = Config().with_overrides(
        source=options.source,
        out_dir=options.out,
        threshold=options.threshold,
        source_revision=options.revision,
        dataset_version=options.dataset_version,
    )
    combined = _gather(shard_dirs, options.work)
    return finalize_shards(
        combined,
        config,
        options.work,
        options.out,
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


def _assemble_verified(shard_dirs: list[Path], options: AssembleOptions) -> StreamedBuild:
    inputs = verified_assembly(
        shard_dirs,
        options.out,
        options.work,
        {
            "source": options.source,
            "threshold": options.threshold,
            "source_revision": options.revision,
            "dataset_version": options.dataset_version,
        },
        legacy_code_revision=options.legacy_code_revision,
    )
    if not inputs.processing["full_source_complete"]:
        typer.echo(
            f"WARNING: incomplete source inventory: {len(inputs.processing['missing_regions'])} "
            "regions missing; this subset is not publishable as a complete source.",
            err=True,
        )
    return finalize_shards(
        inputs.shards,
        inputs.config,
        options.work,
        options.out,
        inputs.rejections,
        processing=inputs.processing,
    )


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
