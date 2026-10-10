"""Verifiable region completion records and whole-source processing accounting.

A Parquet filename does not prove which source revision or filters produced
it. Completion receipts bind the entire region outcome and the shard bytes to
an immutable build context. The final ledger distinguishes a completed subset
from complete coverage of the pinned source inventory.

The package is split by concern: ``storage`` (hashing and atomic JSON),
``code_revision`` (the clean checkout's commit), ``context`` (the receipt
binding), ``outcome`` (per-region records and their rules), ``provenance``
(revision pins) and ``ledger`` (the whole-source ledger). Import the public
names from this package.
"""

from osm_worldcover.accounting.context import PIPELINE_SCHEMA_VERSION as PIPELINE_SCHEMA_VERSION
from osm_worldcover.accounting.context import RECEIPT_VERSION as RECEIPT_VERSION
from osm_worldcover.accounting.context import BuildContext
from osm_worldcover.accounting.ledger import PROCESSING_LEDGER_VERSION as PROCESSING_LEDGER_VERSION
from osm_worldcover.accounting.ledger import processing_ledger
from osm_worldcover.accounting.outcome import COUNT_FIELDS as COUNT_FIELDS
from osm_worldcover.accounting.outcome import (
    outcome_from_record,
    outcome_record,
    validate_outcome,
)
from osm_worldcover.accounting.storage import atomic_json, file_sha256

__all__ = [
    "BuildContext",
    "atomic_json",
    "file_sha256",
    "outcome_from_record",
    "outcome_record",
    "processing_ledger",
    "validate_outcome",
]
