"""Publish only audited release bytes, then verify the immutable Hub commit.

A returned dataset URL means the exact uploaded files were read back or matched
against authoritative metadata at the upload's commit, not merely that an
upload request finished. The local receipt is never part of the upload.
"""

import hashlib
import json
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, RepoFile, hf_hub_download

from osm_worldcover.adapters.audit import audit_build
from osm_worldcover.adapters.coverage_map import MAP_FILENAME, write_coverage_map
from osm_worldcover.domain.card import render
from osm_worldcover.domain.manifest import SPLIT_ORDER
from osm_worldcover.domain.revision import is_full_revision
from osm_worldcover.release_commit import release_read_lock, release_write_lock

__all__ = ["PublicationError", "files_to_publish", "publish_dataset"]

MANIFEST_NAME = "manifest.json"
CHECKSUMS_NAME = "release_checksums.json"
RECEIPT_NAME = "publication_receipt.json"
_READBACK_NAMES = ("README.md", MANIFEST_NAME, CHECKSUMS_NAME)


class PublicationError(RuntimeError):
    """Publication did not establish an exact, verified release commit."""


@dataclass(frozen=True)
class _Fingerprint:
    size: int
    sha256: str
    git_blob_sha1: str
    stat: tuple[int, ...]

    def checksum(self) -> dict[str, int | str]:
        return {"size": self.size, "sha256": self.sha256}


def files_to_publish(build_dir: Path) -> list[Path]:
    """Return mandatory build inputs, before generating the card and map."""
    build_dir = Path(build_dir)
    with release_read_lock(build_dir):
        return _files_to_publish_locked(build_dir)


def _files_to_publish_locked(build_dir: Path) -> list[Path]:
    expected = [build_dir / f"{name}.parquet" for name in SPLIT_ORDER]
    expected.append(build_dir / MANIFEST_NAME)
    _require(build_dir, expected)
    return expected


def _require(build_dir: Path, expected: list[Path]) -> None:
    """Refuse missing files, directories, and indirect file references."""
    _require_regular_files(build_dir, expected)
    _reject_release_symlinks(expected)


def _require_regular_files(build_dir: Path, expected: list[Path]) -> None:
    missing = [path.name for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{build_dir} is missing regular files {missing}")


def _reject_release_symlinks(expected: list[Path]) -> None:
    if any(path.is_symlink() for path in expected):
        raise PublicationError("Release files must not be symbolic links")


def publish_dataset(
    build_dir: Path,
    repo_id: str,
    private: bool = False,
    token: str | None = None,
) -> str:
    """Audit, upload, and verify a release; return its dataset URL on success.

    Failures after upload may leave a commit on the Hub. They raise instead of
    reporting success, and never leave a receipt claiming this attempt passed.
    """
    build_dir = Path(build_dir)
    with release_write_lock(build_dir):
        return _publish_dataset_locked(build_dir, repo_id, private, token)


def _publish_dataset_locked(
    build_dir: Path,
    repo_id: str,
    private: bool,
    token: str | None,
) -> str:
    (build_dir / RECEIPT_NAME).unlink(missing_ok=True)
    files = _prepare_release(build_dir, repo_id)
    fingerprints = {path.name: _fingerprint(path) for path in files}
    report = audit_build(
        build_dir,
        require_complete=True,
        require_card=True,
        strict_text_leakage=False,
    )
    if not report.ok:
        failures = ", ".join(f"{p.code} ({p.count})" for p in report.problems)
        raise PublicationError(f"Publication audit failed: {failures}")
    _check_unchanged(build_dir, fingerprints)
    fingerprints[CHECKSUMS_NAME] = _write_checksums(build_dir, fingerprints)

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    _check_unchanged(build_dir, fingerprints)
    commit = api.upload_folder(
        folder_path=str(build_dir),
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=list(fingerprints),
        commit_message=f"Publish {repo_id} WorldCover build",
    )
    oid = _commit_oid(commit)
    verification = _verify_release(api, repo_id, oid, fingerprints, token)
    _check_unchanged(build_dir, fingerprints)
    url = f"https://huggingface.co/datasets/{repo_id}"
    _write_receipt(build_dir, repo_id, oid, url, verification, report.as_dict())
    return url


def _prepare_release(build_dir: Path, repo_id: str) -> list[Path]:
    files = files_to_publish(build_dir)
    manifest = json.loads((build_dir / MANIFEST_NAME).read_text())
    if manifest.get("settings", {}).get("output_dataset") != repo_id:
        raise PublicationError(
            "Manifest settings.output_dataset must match the publication repo_id"
        )
    card, coverage_map = build_dir / "README.md", build_dir / MAP_FILENAME
    # Refuse existing output symlinks before regeneration can follow them.
    if any(path.is_symlink() for path in (card, coverage_map, build_dir / CHECKSUMS_NAME)):
        raise PublicationError("Release files must not be symbolic links")
    write_coverage_map(build_dir, coverage_map)
    card.write_text(render(manifest))
    files.extend((card, coverage_map))
    _require(build_dir, files)
    return files


def _stat(path: Path) -> tuple[int, ...]:
    value = path.stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _fingerprint(path: Path) -> _Fingerprint:
    """Hash large Parquets in bounded memory, detecting concurrent writers."""
    before = _stat(path)
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1(f"blob {before[2]}\0".encode(), usedforsecurity=False)
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            sha256.update(block)
            git_blob.update(block)
    if _stat(path) != before:
        raise PublicationError(f"Release file changed while hashing: {path.name}")
    return _Fingerprint(before[2], sha256.hexdigest(), git_blob.hexdigest(), before)


def _check_unchanged(build_dir: Path, fingerprints: dict[str, _Fingerprint]) -> None:
    for name, expected in fingerprints.items():
        path = build_dir / name
        if path.is_symlink() or _stat(path) != expected.stat:
            raise PublicationError(f"Release file changed during publication: {name}")


def _write_checksums(build_dir: Path, fingerprints: dict[str, _Fingerprint]) -> _Fingerprint:
    payload = {
        "format_version": 1,
        "algorithm": "sha256",
        "files": {name: value.checksum() for name, value in sorted(fingerprints.items())},
    }
    path = build_dir / CHECKSUMS_NAME
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return _fingerprint(path)


def _commit_oid(commit: Any) -> str:
    oid = getattr(commit, "oid", None)
    if not is_full_revision(oid):
        raise PublicationError("Upload returned no immutable commit SHA; publication is unverified")
    return oid


def _verify_release(api, repo_id, oid, fingerprints, token) -> dict[str, Any]:
    _verify_remote_commit(api, repo_id, oid)
    entries = _release_entries(api, repo_id, oid)
    verified = _verify_release_files(oid, entries, fingerprints)
    _verify_readback_files(repo_id, oid, fingerprints, token)
    return {"ok": True, "files": verified, "pinned_readbacks": list(_READBACK_NAMES)}


def _verify_remote_commit(api, repo_id: str, oid: str) -> None:
    info = api.repo_info(repo_id, repo_type="dataset", revision=oid)
    if info.sha != oid:
        raise PublicationError(f"Remote commit mismatch: expected {oid}, got {info.sha}")


def _release_entries(api, repo_id: str, oid: str) -> dict[str, RepoFile]:
    tree = api.list_repo_tree(repo_id, repo_type="dataset", revision=oid, recursive=True)
    return {entry.path: entry for entry in tree if isinstance(entry, RepoFile)}


def _verify_release_files(oid, entries, fingerprints) -> dict[str, dict[str, int | str]]:
    verified = {}
    for name, expected in fingerprints.items():
        if name not in entries:
            raise PublicationError(f"Remote release file missing at {oid}: {name}")
        verified[name] = _verify_metadata(entries[name], expected)

    return verified


def _verify_readback_files(repo_id, oid, fingerprints, token) -> None:
    for name in _READBACK_NAMES:
        downloaded = Path(
            hf_hub_download(repo_id, name, repo_type="dataset", revision=oid, token=token)
        )
        actual = _fingerprint(downloaded)
        if actual.checksum() != fingerprints[name].checksum():
            raise PublicationError(f"Pinned remote readback mismatch: {name} at {oid}")


def _verify_metadata(remote: RepoFile, expected: _Fingerprint) -> dict[str, int | str]:
    if remote.size != expected.size:
        raise PublicationError(f"Remote size mismatch: {remote.path}")
    if remote.lfs is not None:
        algorithm, actual, wanted = "sha256", remote.lfs.sha256, expected.sha256
        if remote.lfs.size != expected.size:
            raise PublicationError(f"Remote LFS size mismatch: {remote.path}")
    else:
        algorithm, actual, wanted = "git_blob_sha1", remote.blob_id, expected.git_blob_sha1
    if actual != wanted:
        raise PublicationError(f"Remote {algorithm} mismatch: {remote.path}")
    return {**expected.checksum(), "verified_by": algorithm, "remote_hash": actual}


def _write_receipt(build_dir, repo_id, oid, url, verification, audit) -> None:
    receipt = {
        "format_version": 1,
        "repo_id": repo_id,
        "repo_type": "dataset",
        "commit_oid": oid,
        "commit_url": f"{url}/commit/{oid}",
        "dataset_url": url,
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "verification": verification,
        "audit": audit,
    }
    with tempfile.TemporaryDirectory(prefix=".publication-", dir=build_dir) as scratch:
        temporary = Path(scratch) / RECEIPT_NAME
        temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        temporary.replace(build_dir / RECEIPT_NAME)
