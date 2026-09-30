"""Recipe text eligibility preserves concise descriptions without weakening articles."""

import pandas as pd
import pytest

from osm_worldcover.adapters.source import RegionTables
from osm_worldcover.config import Config
from osm_worldcover.pipeline import to_examples


def _examples(config: Config, text: str | None) -> pd.DataFrame:
    """Apply the shared text join to one already-labelled polygon."""
    tables = RegionTables(
        stem="alpha",
        polygons=pd.DataFrame(),
        links=pd.DataFrame({"polygon_id": ["p1"], "document_id": ["d1"]}),
        documents=pd.DataFrame(
            {"document_id": ["d1"], "fetch_status": ["ok"], "full_text": [text]}
        ),
    )
    return to_examples(pd.DataFrame({"polygon_id": ["p1"]}), tables, config.effective_min_words)


@pytest.mark.parametrize("text", ["Garden", "Small public wooded garden", "公共花园"])
def test_description_keeps_short_nonempty_text(text: str) -> None:
    assert _examples(Config(source="description"), text)["text"].tolist() == [text]


@pytest.mark.parametrize("source", ["description", "wikidata", "website"])
@pytest.mark.parametrize("text", [None, "", "  \n\t\u3000 "])
def test_every_source_rejects_blank_text(source: str, text: str | None) -> None:
    assert _examples(Config(source=source), text).empty


@pytest.mark.parametrize("source", ["wikidata", "website"])
@pytest.mark.parametrize("words, kept", [(1, False), (4, False), (9, False), (10, True)])
def test_article_and_website_ten_word_boundary(source: str, words: int, kept: bool) -> None:
    rows = _examples(Config(source=source), " ".join(["word"] * words))
    assert (len(rows) == 1) == kept


def test_description_explicit_ten_word_override_still_filters_short_text() -> None:
    assert _examples(Config(source="description", min_words=10), "Small wooded garden").empty


def test_description_normalizes_whitespace_without_expanding_text() -> None:
    rows = _examples(Config(source="description"), "  Small\n wooded\t garden ")
    assert rows["text"].tolist() == ["Small wooded garden"]
