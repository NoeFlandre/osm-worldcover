"""Fetch the source dataset from the Hugging Face Hub.

Only the files a run actually reads are downloaded, region by region, so a
partial build never needs the full ~21 GB snapshot. Every download is pinned to
one commit so a rebuild sees byte-identical inputs.
"""

from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from osm_worldcover.sources import DEFAULT_SOURCE, SourceRecipe, recipe_for

__all__ = [
    "list_region_stems",
    "region_files",
    "resolve_revision",
    "snapshot_region",
    "split_repo_path",
]


def region_files(stem: str, source: str | SourceRecipe = DEFAULT_SOURCE) -> list[str]:
    """Return every repository path holding data for region ``stem``."""
    recipe = source if isinstance(source, SourceRecipe) else recipe_for(source)
    return list(recipe.region_paths(stem))


def split_repo_path(path: str) -> tuple[str, str]:
    """Split a repository path into its table directory and region stem."""
    head, _, name = path.rpartition("/")
    return head, name.removesuffix(".parquet")


def resolve_revision(repo_id: str, revision: str | None = None) -> str:
    """Return the commit sha that ``revision`` names, pinning the build to it."""
    info = HfApi().dataset_info(repo_id, revision=revision)
    if info.sha is None:
        raise ValueError(f"{repo_id} has no commit for revision {revision!r}")
    return info.sha


def list_region_stems(
    repo_id: str, revision: str, source: str | SourceRecipe = DEFAULT_SOURCE
) -> list[str]:
    """Return every region present in the repository, sorted."""
    recipe = source if isinstance(source, SourceRecipe) else recipe_for(source)
    files = HfApi().list_repo_files(repo_id, repo_type="dataset", revision=revision)
    return sorted(
        split_repo_path(f)[1]
        for f in files
        if f.startswith(recipe.region_prefix) and f.endswith(".parquet")
    )


def snapshot_region(
    repo_id: str,
    revision: str,
    stem: str,
    dest: Path,
    source: str | SourceRecipe = DEFAULT_SOURCE,
) -> list[Path]:
    """Download region ``stem`` into ``dest``, mirroring the repository layout.

    Files the repository does not publish for this region -- Wikivoyage sidecars
    most often -- are skipped rather than treated as failures.
    """
    downloaded: list[Path] = []
    paths = region_files(stem, source)
    for path in paths:
        try:
            local = hf_hub_download(
                repo_id,
                path,
                repo_type="dataset",
                revision=revision,
                local_dir=dest,
            )
        except EntryNotFoundError:
            if path == paths[0]:
                raise
            continue
        downloaded.append(Path(local))
    return downloaded
