"""The build and the audit must agree on the retained-text counters."""

import duckdb
import pytest

from osm_worldcover import finalize
from osm_worldcover.adapters import audit

ROWS = [
    # (text, code, split)
    ("a  b", 10, "train"),
    ("a b", 10, "test"),
    ("a b", 20, "train"),
    ("c", 10, "train"),
    ("c", 10, "train"),
    ("d", 30, "val"),
]


@pytest.mark.parametrize("rows", [ROWS, ROWS[-1:], []])
def test_finalize_and_audit_diagnostics_agree(rows):
    import hashlib

    def key(text, code):
        return hashlib.sha256(f"{' '.join(text.split())}|{code}".encode()).hexdigest()

    finalize_con = duckdb.connect()
    finalize_con.execute(
        "CREATE TABLE kept (text VARCHAR, worldcover_code INT, split VARCHAR, _dedup_key VARCHAR)"
    )
    audit_con = duckdb.connect()
    audit_con.execute("CREATE TABLE hashes (text_hash BLOB, worldcover_code INT, split VARCHAR)")
    for text, code, split in rows:
        finalize_con.execute(
            "INSERT INTO kept VALUES (?, ?, ?, ?)", [text, code, split, key(text, code)]
        )
        digest = hashlib.sha256(" ".join(text.split()).encode()).digest()
        audit_con.execute("INSERT INTO hashes VALUES (?, ?, ?)", [digest, code, split])
    assert finalize._retained_text_diagnostics(finalize_con) == audit._retained_text_diagnostics(
        audit_con
    )
