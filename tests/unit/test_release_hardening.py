import fcntl
import os

import pytest

import osm_worldcover.release_commit as release
from osm_worldcover.config import Config


@pytest.mark.parametrize("version", ["", "..", ".hidden", "../escape", "a/b", "v 1", "x\\y"])
def test_dataset_version_must_be_one_plain_path_component(version):
    with pytest.raises(ValueError, match="invalid dataset version"):
        Config(dataset_version=version)


def test_plain_dataset_versions_are_accepted():
    assert Config(dataset_version="1.1.0").dataset_version == "1.1.0"
    assert Config(dataset_version="2026-10-09_rc1").dataset_version == "2026-10-09_rc1"


def test_flush_uses_full_sync_when_the_platform_has_it(tmp_path, monkeypatch):
    path = tmp_path / "file"
    path.write_text("data")
    calls = []
    monkeypatch.setattr(fcntl, "F_FULLFSYNC", 51, raising=False)
    monkeypatch.setattr(fcntl, "fcntl", lambda fd, cmd: calls.append(cmd))
    monkeypatch.setattr(os, "fsync", lambda fd: pytest.fail("os.fsync should not be used"))

    descriptor = os.open(path, os.O_RDONLY)
    try:
        release._flush_to_media(descriptor)
    finally:
        os.close(descriptor)

    assert calls == [51]


def test_flush_falls_back_to_fsync_when_full_sync_is_refused(tmp_path, monkeypatch):
    path = tmp_path / "file"
    path.write_text("data")
    synced = []

    def refuse(fd, cmd):
        raise OSError("not supported")

    monkeypatch.setattr(fcntl, "F_FULLFSYNC", 51, raising=False)
    monkeypatch.setattr(fcntl, "fcntl", refuse)
    monkeypatch.setattr(os, "fsync", lambda fd: synced.append(fd))

    descriptor = os.open(path, os.O_RDONLY)
    try:
        release._flush_to_media(descriptor)
    finally:
        os.close(descriptor)

    assert synced == [descriptor]


def test_scratch_lock_excludes_a_second_run_while_held(tmp_path):
    scratch = tmp_path / "assembly"
    with release.scratch_lock(scratch):
        assert (scratch / ".scratch.lock").is_file()
        rival = os.open(scratch / ".scratch.lock", os.O_RDWR)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(rival, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(rival)


def test_receipt_write_leaves_no_directory_behind(tmp_path):
    from osm_worldcover.adapters import publish

    publish._write_receipt(tmp_path, "owner/repo", "a" * 40, "https://x", {}, {})

    assert sorted(p.name for p in tmp_path.iterdir()) == [publish.RECEIPT_NAME]
