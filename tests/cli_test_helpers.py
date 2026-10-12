"""Helpers shared by CLI test modules."""

import pandas as pd


def frame(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "polygon_id": [f"p{i}" for i in range(n)],
            "document_id": [f"d{i}" for i in range(n)],
            # Distinct text per row keeps this fixture free of duplicate records.
            "text": [" ".join(["word"] * 20) + f" {i}" for i in range(n)],
            "worldcover_code": [10] * n,
            "worldcover_label": ["Tree cover"] * n,
            "dominant_fraction": [0.95] * n,
            "split": ["train", "train", "validation", "test"][:n],
        }
    )
