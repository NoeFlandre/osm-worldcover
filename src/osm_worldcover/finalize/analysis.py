"""Measurements of the duplicates removed and of the repeated text the kept rows hold."""

from typing import Any

from osm_worldcover.domain.text_diagnostics import group_counts_sql, retained_text_counts


def _deduplication_analysis(connection: Any) -> dict[str, Any]:
    """Measure repeated records removed under the polygon-scoped identity key."""
    summary = connection.execute(
        """
        WITH grouped AS (
            SELECT polygon_id, _dedup_key, min(text_words) AS text_words, count(*) AS row_count,
                   count(DISTINCT split) AS split_count
            FROM objects GROUP BY polygon_id, _dedup_key
        ), duplicates AS (
            SELECT *, row_count - 1 AS removed FROM grouped WHERE row_count > 1
        )
        SELECT count(*), coalesce(sum(removed), 0),
               count(*) FILTER (WHERE split_count > 1),
               coalesce(sum(removed) FILTER (WHERE split_count > 1), 0)
        FROM duplicates
        """
    ).fetchone()
    by_length = {
        str(words): int(rows)
        for words, rows in connection.execute(
            """
            WITH grouped AS (
                SELECT polygon_id, _dedup_key, min(text_words) AS text_words,
                       count(*) AS row_count
                FROM objects GROUP BY polygon_id, _dedup_key
            )
            SELECT CASE WHEN text_words < 10 THEN cast(text_words AS varchar) ELSE '10+'
                       END AS length_bucket,
                   sum(row_count - 1) AS removed
            FROM grouped WHERE row_count > 1
            GROUP BY length_bucket
            """
        ).fetchall()
    }
    return {
        "duplicate_polygon_text_label_groups": int(summary[0]),
        "duplicate_records_removed": int(summary[1]),
        "duplicate_record_groups_crossing_splits": int(summary[2]),
        "duplicate_records_removed_from_cross_split_groups": int(summary[3]),
        "duplicate_records_removed_by_text_words": {
            **{str(words): by_length.get(str(words), 0) for words in range(1, 10)},
            "10+": by_length.get("10+", 0),
        },
    }


def _retained_text_diagnostics(connection: Any) -> dict[str, int]:
    """Count repeated text that remains, including text shared across splits."""
    label_groups = connection.execute(group_counts_sql("kept", ["_dedup_key"])).fetchone()
    text_groups = connection.execute(
        group_counts_sql(
            r"""(
                SELECT sha256(regexp_replace(trim(text), '\s+', ' ', 'g')) AS text_key,
                       split
                FROM kept
            )""",
            ["text_key"],
        )
    ).fetchone()
    return retained_text_counts(label_groups, text_groups)
