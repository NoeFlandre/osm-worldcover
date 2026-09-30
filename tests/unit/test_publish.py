"""A release is successful only after exact-commit, exact-byte Hub verification."""

import base64
import fnmatch
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import h3
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from huggingface_hub import RepoFile, RepoFolder

from osm_worldcover.accounting import BuildContext, processing_ledger
from osm_worldcover.adapters import publish
from osm_worldcover.adapters.audit import _SCHEMA, audit_build
from osm_worldcover.adapters.coverage_map import MAP_FILENAME
from osm_worldcover.adapters.publish import (
    CHECKSUMS_NAME,
    RECEIPT_NAME,
    PublicationError,
    files_to_publish,
    publish_dataset,
)
from osm_worldcover.config import Config
from osm_worldcover.domain.splits import assign_cell
from osm_worldcover.pipeline import RegionOutcome

CONFIG = Config(source="description", source_revision="a" * 40, min_words=10)
SETTINGS = CONFIG.as_manifest_settings()
REPO_ID = SETTINGS["output_dataset"]
COMMIT = "b" * 40
SPLITS = ("train", "validation", "test")
RELEASE_NAMES = {
    *(f"{split}.parquet" for split in SPLITS),
    "manifest.json",
    "README.md",
    MAP_FILENAME,
}
TYPES = {"string": pa.string(), "int64": pa.int64(), "double": pa.float64()}
SCHEMA = pa.schema([(key, TYPES[value]) for key, value in _SCHEMA.items()])
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aNmcAAAAASUVORK5CYII="
)


def _rows():
    rows = {}
    for lon in range(-175, 176):
        cell = h3.latlng_to_cell(40, lon, 5)
        split = assign_cell(cell).value
        if split in rows:
            continue
        row = dict.fromkeys(_SCHEMA)
        row.update(
            polygon_id=f"place:{split}",
            osm_type="way",
            osm_id=lon + 180,
            region="place",
            name="Test polygon",
            document_id=f"doc:{split}",
            project="description",
            language="en",
            text=f"{split} " + " ".join(["word"] * 12),
            text_words=13,
            worldcover_code=10,
            worldcover_label="Tree cover",
            dominant_fraction=1.0,
            observed_fraction=1.0,
            lat=40.0,
            lon=float(lon),
            centroid_wkt=f"POINT ({lon} 40)",
            polygon_area_m2=100.0,
            source_pbf="place.osm.pbf",
            h3_cell=cell,
            split=split,
            **{
                key: SETTINGS[key]
                for key in (
                    "dataset_version",
                    "source_dataset",
                    "source_revision",
                    "worldcover_version",
                    "worldcover_year",
                )
            },
        )
        rows[split] = row
        if len(rows) == 3:
            break
    return rows


@pytest.fixture
def build(tmp_path):
    target = tmp_path / "release"
    target.mkdir()
    rows = _rows()
    for split, row in rows.items():
        pq.write_table(pa.Table.from_pylist([row], schema=SCHEMA), target / f"{split}.parquet")
    counts = {**dict.fromkeys(SPLITS, 1), "total": 3}
    lons = [row["lon"] for row in rows.values()]
    outcome = RegionOutcome(
        stem="place",
        polygons_seen=3,
        polygons_accepted=3,
        polygons_with_examples=3,
        examples=3,
        source_links=3,
        source_documents=3,
    )
    manifest = {
        "settings": SETTINGS,
        "processing": processing_ledger(
            ["place"], ["place"], [outcome], BuildContext.from_config(CONFIG)
        ),
        "counts": {key: counts for key in ("examples", "polygons", "documents")},
        "class_distribution": [{"code": 10, "label": "Tree cover", "examples": 3, "share": 1.0}],
        "language_distribution": [{"language": "en", "examples": 3, "share": 1.0}],
        "example_polygons": [{"name": "Test polygon", "worldcover_label": "Tree cover"}],
        "dominant_fraction": {"p50": 1.0, "p90": 1.0, "p99": 1.0},
        "geographic_coverage": {
            "h3_cells": 3,
            "regions": 1,
            "bbox": {"min_lon": min(lons), "min_lat": 40.0, "max_lon": max(lons), "max_lat": 40.0},
        },
        "rejections": {},
        "deduplication": {
            "duplicate_objects_across_regions": 0,
            "duplicate_polygon_text_label_records": 0,
            "documents_split_across_splits": 0,
        },
    }
    (target / "manifest.json").write_text(json.dumps(manifest))
    return target


def _git_blob(data):
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data, usedforsecurity=False).hexdigest()


class FakeHub:
    """A separate immutable uploaded snapshot, not a read of local source files."""

    def __init__(self, directory):
        self.directory = directory
        self.calls = []
        self.uploaded = {}
        self.commit = SimpleNamespace(oid=COMMIT)
        self.remote_sha = COMMIT
        self.metadata = {}
        self.readbacks = {}
        self.missing = set()
        self.token = None
        self.upload_error = None

    def create_repo(self, repo_id, **kwargs):
        self.calls.append(("create", repo_id, kwargs))

    def upload_folder(self, **kwargs):
        self.calls.append(("upload", kwargs))
        if self.upload_error:
            raise self.upload_error
        root = Path(kwargs["folder_path"])
        for path in root.rglob("*"):
            name = path.relative_to(root).as_posix()
            if path.is_file() and any(fnmatch.fnmatch(name, p) for p in kwargs["allow_patterns"]):
                self.uploaded[name] = path.read_bytes()
        return self.commit

    def repo_info(self, repo_id, **kwargs):
        self.calls.append(("info", repo_id, kwargs))
        return SimpleNamespace(sha=self.remote_sha)

    def list_repo_tree(self, repo_id, **kwargs):
        self.calls.append(("tree", repo_id, kwargs))
        entries = []
        for name, data in self.uploaded.items():
            if name in self.missing:
                continue
            fields = {"path": name, "size": len(data), "oid": _git_blob(data)}
            if name.endswith((".parquet", ".png")):
                fields["lfs"] = {
                    "size": len(data),
                    "oid": hashlib.sha256(data).hexdigest(),
                    "pointerSize": 130,
                }
            fields.update(self.metadata.get(name, {}))
            entries.append(RepoFile(**fields))
        entries.append(RepoFolder(path="unrelated", oid="c" * 40))
        return entries

    def download(self, repo_id, filename, **kwargs):
        self.calls.append(("download", repo_id, filename, kwargs))
        path = self.directory / filename
        path.write_bytes(self.readbacks.get(filename, self.uploaded[filename]))
        return str(path)


@pytest.fixture
def hub(tmp_path, monkeypatch):
    directory = tmp_path / "remote"
    directory.mkdir()
    fake = FakeHub(directory)

    def api(token=None):
        fake.token = token
        fake.calls.append(("init",))
        return fake

    def coverage_map(build_dir, output_path):
        fake.calls.append(("map", build_dir, output_path))
        output_path.write_bytes(PNG)
        return 3

    monkeypatch.setattr(publish, "HfApi", api)
    monkeypatch.setattr(publish, "hf_hub_download", fake.download)
    monkeypatch.setattr(publish, "write_coverage_map", coverage_map)
    return fake


def _edit_manifest(build, edit):
    path = build / "manifest.json"
    manifest = json.loads(path.read_text())
    edit(manifest)
    path.write_text(json.dumps(manifest))


def _mutate_row(build, split="train", **changes):
    path = build / f"{split}.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[0].update(changes)
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path)


def test_required_inputs_are_three_splits_and_manifest(build):
    assert {path.name for path in files_to_publish(build)} == {
        "train.parquet",
        "validation.parquet",
        "test.parquet",
        "manifest.json",
    }


def test_valid_fixture_passes_full_source_audit(build):
    assert audit_build(build, require_complete=True).ok


def test_build_without_inputs_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        files_to_publish(tmp_path)


def test_missing_manifest_is_refused(build):
    (build / "manifest.json").unlink()
    with pytest.raises(FileNotFoundError, match=r"manifest\.json"):
        files_to_publish(build)


def test_directory_cannot_masquerade_as_release_file(build):
    (build / "test.parquet").unlink()
    (build / "test.parquet").mkdir()
    with pytest.raises(FileNotFoundError, match=r"test\.parquet"):
        files_to_publish(build)


def test_publish_uploads_only_intended_files_and_verifies_exact_commit(build, hub):
    (build / "private.log").write_text("never upload")
    (build / RECEIPT_NAME).write_text("stale receipt")
    url = publish_dataset(build, REPO_ID, token="fake-token")
    assert url == f"https://huggingface.co/datasets/{REPO_ID}"
    assert set(hub.uploaded) == RELEASE_NAMES | {CHECKSUMS_NAME}
    assert hub.token == "fake-token"
    assert "Tree cover" in hub.uploaded["README.md"].decode()
    assert "Test polygon" in hub.uploaded["README.md"].decode()
    assert MAP_FILENAME in hub.uploaded["README.md"].decode()
    assert hub.uploaded[MAP_FILENAME] == PNG
    assert hub.calls[0] == ("map", build, build / MAP_FILENAME)
    assert next(call for call in hub.calls if call[0] == "create")[2]["private"] is False
    for call in hub.calls:
        if call[0] in {"info", "tree", "download"}:
            assert call[-1]["revision"] == COMMIT
            assert call[-1]["repo_type"] == "dataset"
    downloads = [call[2] for call in hub.calls if call[0] == "download"]
    assert set(downloads) == {"README.md", "manifest.json", CHECKSUMS_NAME}
    receipt = json.loads((build / RECEIPT_NAME).read_text())
    assert receipt["commit_oid"] == COMMIT
    assert receipt["commit_url"] == f"{url}/commit/{COMMIT}"
    assert receipt["verification"]["ok"]
    assert set(receipt["verification"]["files"]) == set(hub.uploaded)
    assert receipt["verification"]["files"]["train.parquet"]["verified_by"] == "sha256"
    assert receipt["verification"]["files"]["README.md"]["verified_by"] == "git_blob_sha1"
    assert receipt["audit"]["ok"] and receipt["audit"]["rows"] == 3
    assert "fake-token" not in (build / RECEIPT_NAME).read_text()


def test_checksums_cover_exact_release_bytes_without_self_reference(build, hub):
    publish_dataset(build, REPO_ID, private=True)
    checksums = json.loads(hub.uploaded[CHECKSUMS_NAME])
    assert checksums["algorithm"] == "sha256"
    assert set(checksums["files"]) == RELEASE_NAMES
    for name, expected in checksums["files"].items():
        assert expected == {
            "size": len(hub.uploaded[name]),
            "sha256": hashlib.sha256(hub.uploaded[name]).hexdigest(),
        }
    assert next(call for call in hub.calls if call[0] == "create")[2]["private"] is True


def test_audit_runs_after_regeneration_and_before_hub_creation(build, hub, monkeypatch):
    def inspected_audit(directory, **kwargs):
        assert (directory / "README.md").exists()
        assert (directory / MAP_FILENAME).read_bytes() == PNG
        assert not any(call[0] == "init" for call in hub.calls)
        assert kwargs == {
            "require_complete": True,
            "require_card": True,
            "strict_text_leakage": False,
        }
        return audit_build(directory, **kwargs)

    monkeypatch.setattr(publish, "audit_build", inspected_audit)
    publish_dataset(build, REPO_ID)


@pytest.mark.parametrize(
    ("edit", "problem"),
    [
        (lambda m: m.pop("processing"), "source_processing_incomplete"),
        (lambda m: m["counts"]["examples"].update(train=999), "manifest_count_mismatch"),
        (lambda m: m["settings"].update(source_revision="main"), "unpinned_source_revision"),
    ],
)
def test_invalid_release_never_contacts_hub(build, hub, edit, problem):
    _edit_manifest(build, edit)
    (build / RECEIPT_NAME).write_text("stale success")
    with pytest.raises(PublicationError, match=problem):
        publish_dataset(build, REPO_ID)
    assert [call[0] for call in hub.calls] == ["map"]
    assert not (build / RECEIPT_NAME).exists()


def test_duplicate_polygon_text_label_record_is_rejected_before_hub_mutation(build, hub):
    train = pq.read_table(build / "train.parquet").to_pylist()[0]
    _mutate_row(build, "test", text=train["text"], polygon_id=train["polygon_id"])
    with pytest.raises(PublicationError, match="duplicate_polygon_text_label_record"):
        publish_dataset(build, REPO_ID)
    assert not any(call[0] == "init" for call in hub.calls)


def test_same_text_and_label_on_distinct_polygons_is_published_with_warning(build, hub):
    train = pq.read_table(build / "train.parquet").to_pylist()[0]
    _mutate_row(build, "test", text=train["text"])

    publish_dataset(build, REPO_ID)

    receipt = json.loads((build / RECEIPT_NAME).read_text())
    assert any(
        warning["code"] == "identical_text_cross_split" for warning in receipt["audit"]["warnings"]
    )


def test_cross_label_text_collisions_remain_warnings_in_receipt(build, hub):
    train = pq.read_table(build / "train.parquet").to_pylist()[0]
    _mutate_row(build, "test", text=train["text"], worldcover_code=20, worldcover_label="Shrubland")

    def new_classes(manifest):
        manifest["class_distribution"] = [
            {"code": 10, "label": "Tree cover", "examples": 2, "share": 0.666667},
            {"code": 20, "label": "Shrubland", "examples": 1, "share": 0.333333},
        ]
        manifest["example_polygons"].append(
            {"name": "Test polygon", "worldcover_label": "Shrubland"}
        )

    _edit_manifest(build, new_classes)
    publish_dataset(build, REPO_ID)
    receipt = json.loads((build / RECEIPT_NAME).read_text())
    assert {warning["code"] for warning in receipt["audit"]["warnings"]} == {
        "identical_text_cross_split",
        "identical_text_conflicting_labels",
    }


def test_schema_rejected_before_hub_mutation(build, hub):
    pq.write_table(pa.table({"text": ["invalid"]}), build / "test.parquet")
    with pytest.raises(PublicationError, match="schema_mismatch"):
        publish_dataset(build, REPO_ID)
    assert not any(call[0] == "init" for call in hub.calls)


def test_corrupt_regenerated_card_rejected(build, hub, monkeypatch):
    monkeypatch.setattr(publish, "render", lambda manifest: "no card metadata")
    with pytest.raises(PublicationError, match="invalid_card_or_map"):
        publish_dataset(build, REPO_ID)
    assert not any(call[0] == "init" for call in hub.calls)


def test_mismatched_target_is_not_silently_rewritten(build, hub):
    with pytest.raises(PublicationError, match="output_dataset"):
        publish_dataset(build, "someone/other-dataset")
    assert not hub.calls


@pytest.mark.parametrize("name", ["train.parquet", "README.md", MAP_FILENAME, CHECKSUMS_NAME])
def test_symlink_release_files_rejected_without_following_them(build, hub, tmp_path, name):
    outside = tmp_path / "outside"
    outside.write_text("untouched")
    (build / name).unlink(missing_ok=True)
    (build / name).symlink_to(outside)
    with pytest.raises(PublicationError, match="symbolic links"):
        publish_dataset(build, REPO_ID)
    assert outside.read_text() == "untouched"
    assert not hub.calls


@pytest.mark.parametrize("commit", [None, "https://hub/commit/url", SimpleNamespace(oid="main")])
def test_missing_immutable_upload_commit_cannot_report_success(build, hub, commit):
    hub.commit = commit
    with pytest.raises(PublicationError, match="no immutable commit SHA"):
        publish_dataset(build, REPO_ID)
    assert not (build / RECEIPT_NAME).exists()
    assert not any(call[0] == "info" for call in hub.calls)


def test_remote_commit_mismatch_cannot_report_success(build, hub):
    hub.remote_sha = "d" * 40
    with pytest.raises(PublicationError, match="Remote commit mismatch"):
        publish_dataset(build, REPO_ID)
    assert not (build / RECEIPT_NAME).exists()


@pytest.mark.parametrize("name", sorted(RELEASE_NAMES | {CHECKSUMS_NAME}))
def test_missing_remote_file_fails_publication(build, hub, name):
    hub.missing.add(name)
    with pytest.raises(PublicationError, match="Remote release file missing"):
        publish_dataset(build, REPO_ID)
    assert not (build / RECEIPT_NAME).exists()


@pytest.mark.parametrize(
    ("name", "metadata", "problem"),
    [
        ("train.parquet", {"lfs": {"oid": "0" * 64, "size": 0, "pointerSize": 130}}, "LFS size"),
        ("train.parquet", {"lfs": None, "oid": "0" * 40}, "git_blob_sha1"),
        ("README.md", {"oid": "0" * 40}, "git_blob_sha1"),
        ("README.md", {"size": 1}, "size mismatch"),
    ],
)
def test_remote_metadata_mismatch_fails_publication(build, hub, name, metadata, problem):
    hub.metadata[name] = metadata
    with pytest.raises(PublicationError, match=problem):
        publish_dataset(build, REPO_ID)
    assert not (build / RECEIPT_NAME).exists()


def test_remote_lfs_sha256_mismatch_fails_publication(build, hub):
    size = (build / "train.parquet").stat().st_size
    hub.metadata["train.parquet"] = {"lfs": {"oid": "0" * 64, "size": size, "pointerSize": 130}}
    with pytest.raises(PublicationError, match="Remote sha256 mismatch"):
        publish_dataset(build, REPO_ID)
    assert not (build / RECEIPT_NAME).exists()


@pytest.mark.parametrize("name", ["README.md", "manifest.json", CHECKSUMS_NAME])
def test_pinned_readback_mismatch_fails_even_when_metadata_matches(build, hub, name):
    hub.readbacks[name] = b"wrong remote bytes"
    with pytest.raises(PublicationError, match="Pinned remote readback mismatch"):
        publish_dataset(build, REPO_ID)
    assert not (build / RECEIPT_NAME).exists()


def test_upload_failure_cannot_leave_a_success_receipt(build, hub):
    hub.upload_error = ConnectionError("simulated failure")
    with pytest.raises(ConnectionError, match="simulated failure"):
        publish_dataset(build, REPO_ID)
    assert not (build / RECEIPT_NAME).exists()


def test_concurrent_change_during_audit_prevents_upload(build, hub, monkeypatch):
    def changing_audit(directory, **kwargs):
        report = audit_build(directory, **kwargs)
        (directory / "README.md").write_text("changed after audit")
        return report

    monkeypatch.setattr(publish, "audit_build", changing_audit)
    with pytest.raises(PublicationError, match="Release file changed"):
        publish_dataset(build, REPO_ID)
    assert not any(call[0] == "init" for call in hub.calls)
