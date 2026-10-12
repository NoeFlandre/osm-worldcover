"""Crash-safe release commits and the locks that guard them.

A release directory is replaced only through a journaled transaction, and readers and
writers coordinate through the version's lock file. Callers import the public names
from ``osm_worldcover.release``.
"""

from osm_worldcover.release.commit import commit_release, create_stage, discard_stage, sync_stage
from osm_worldcover.release.errors import ReleaseCommitError
from osm_worldcover.release.inventory import (
    CORE_FILES,
    FileFingerprint,
    ReleaseInventory,
    core_inventory,
    inventory_release,
    stage_inventory,
)
from osm_worldcover.release.journal import transaction_pending
from osm_worldcover.release.locks import (
    migrate_release_lock,
    release_lock,
    release_read_lock,
    scratch_lock,
)
from osm_worldcover.release.recovery import (
    recover_release,
    recover_release_locked,
    release_write_lock,
)

__all__ = [
    "CORE_FILES",
    "FileFingerprint",
    "ReleaseCommitError",
    "ReleaseInventory",
    "commit_release",
    "core_inventory",
    "create_stage",
    "discard_stage",
    "inventory_release",
    "migrate_release_lock",
    "recover_release",
    "recover_release_locked",
    "release_lock",
    "release_read_lock",
    "release_write_lock",
    "scratch_lock",
    "stage_inventory",
    "sync_stage",
    "transaction_pending",
]
