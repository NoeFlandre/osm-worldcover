"""Shared shape of the retained-text diagnostics.

The build (``finalize``) and the independent audit each derive their own key
columns; only the group/cross-split aggregation and the result dict are shared,
so the two cannot drift on what a "group" or "cross-split" means.
"""

from collections.abc import Sequence

__all__ = ["group_counts_sql", "retained_text_counts"]


def group_counts_sql(relation: str, keys: Sequence[str]) -> str:
    """SQL returning (groups, rows, cross_split_groups, cross_split_rows).

    Only keys held by more than one row are counted. ``relation`` must expose
    the ``keys`` columns and ``split``.
    """
    key_list = ", ".join(keys)
    return f"""
        WITH grouped AS (
            SELECT {key_list}, count(*) AS row_count,
                   count(DISTINCT split) AS split_count
            FROM {relation} GROUP BY {key_list}
        )
        SELECT count(*), coalesce(sum(row_count), 0),
               count(*) FILTER (WHERE split_count > 1),
               coalesce(sum(row_count) FILTER (WHERE split_count > 1), 0)
        FROM grouped WHERE row_count > 1
    """


def retained_text_counts(label: Sequence[int], text: Sequence[int]) -> dict[str, int]:
    """Name the six counters from the label-group and text-group count rows."""
    return {
        "retained_identical_text_label_groups": int(label[0]),
        "retained_identical_text_label_rows": int(label[1]),
        "retained_identical_text_label_cross_split_groups": int(label[2]),
        "retained_identical_text_label_cross_split_rows": int(label[3]),
        "identical_text_cross_split_groups": int(text[2]),
        "identical_text_cross_split_rows": int(text[3]),
    }
