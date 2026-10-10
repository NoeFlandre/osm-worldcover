"""The immutable build context that completion receipts are bound to."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any

from osm_worldcover.accounting.code_revision import _current_code_revision
from osm_worldcover.accounting.storage import canonical_bytes
from osm_worldcover.config import Config
from osm_worldcover.domain.revision import is_full_revision
from osm_worldcover.pipeline import OUTPUT_COLUMNS, RegionOutcome

RECEIPT_VERSION = 1
PIPELINE_SCHEMA_VERSION = 3


@dataclass(frozen=True, slots=True)
class BuildContext:
    """The pinned input, all data-changing settings, and output contract."""

    document: dict[str, Any]

    @classmethod
    def from_config(cls, config: Config) -> "BuildContext":
        """Exclude only scratch paths, region selection and cache controls."""
        revision = config.source_revision
        if not is_full_revision(revision, allow_uppercase=True):
            raise ValueError("completion receipts require a pinned source revision")
        return cls(
            {
                "receipt_version": RECEIPT_VERSION,
                "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
                "code_revision": _current_code_revision(),
                "settings": config.as_manifest_settings(),
                "extra": config.extra,
                "source_recipe": asdict(config.source_recipe),
                "output_columns": list(OUTPUT_COLUMNS),
                "outcome_fields": [item.name for item in fields(RegionOutcome)],
            }
        )

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "BuildContext":
        """Restore an immutable context exactly as it was recorded in a receipt."""
        if not isinstance(document, Mapping):
            raise TypeError("build context must be an object")
        normalized = json.loads(canonical_bytes(document))
        if (
            normalized.get("receipt_version") != RECEIPT_VERSION
            or normalized.get("pipeline_schema_version") != PIPELINE_SCHEMA_VERSION
        ):
            raise ValueError("unsupported receipt context version")
        return cls(normalized)

    @property
    def fingerprint(self) -> str:
        """Stable identity independent of mapping order and scratch paths."""
        return hashlib.sha256(canonical_bytes(self.document)).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-normalized copy, including tuple-valued recipe data."""
        return json.loads(canonical_bytes(self.document))
