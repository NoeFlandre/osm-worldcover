"""Row-local columns added to each shard before the global deduplication."""

from __future__ import annotations

import pandas as pd

from osm_worldcover.config import Config
from osm_worldcover.domain.splits import SplitRatios, assign_cell, cell_for
from osm_worldcover.pipeline import TEXT_COLUMNS


def _with_stable_text_types(frame: pd.DataFrame) -> pd.DataFrame:
    """Give every text column the same dtype in every shard.

    A region can legitimately hold no value at all for a text column -- an OSM
    ``description`` tag carries no language, for instance. Pandas then writes
    that column as NULL-typed, and a reader that takes its schema from
    whichever file it opened first will refuse the shards that do hold strings.
    Pinning the dtype makes the combined read independent of file order.
    """
    present = [column for column in TEXT_COLUMNS if column in frame.columns]
    return frame.astype({column: "string" for column in present})


def _assign_splits(frame: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Attach an H3 cell to every row and split on the cell, never on the row."""
    ratios = SplitRatios(config.train_ratio, config.validation_ratio, config.test_ratio)
    cells = [
        cell_for(lat, lon, config.h3_resolution)
        for lat, lon in zip(frame["lat"], frame["lon"], strict=True)
    ]
    frame = frame.assign(h3_cell=cells)
    # One lookup per distinct cell, so every row in a cell gets the same split.
    split_of = {cell: assign_cell(cell, ratios, config.split_seed).value for cell in set(cells)}
    return frame.assign(split=[split_of[cell] for cell in cells])


def _attach_provenance(frame: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Record which inputs and settings produced each row."""
    return frame.assign(
        dataset_version=config.dataset_version,
        source_dataset=config.source_dataset,
        source_revision=config.source_revision,
        worldcover_version=config.worldcover_version,
        worldcover_year=config.worldcover_year,
    )
