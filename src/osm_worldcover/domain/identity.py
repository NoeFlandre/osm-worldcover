"""Public identities of this project: the Hub namespace and the code repository.

Defined in ``domain`` because the pure card renderer needs them and
``domain`` may not import ``config``. ``config`` and ``sources`` import them
from here, so each name is written exactly once.
"""

from typing import Final

__all__ = ["CODE_REPOSITORY", "HUB_NAMESPACE", "WIKIDATA_OUTPUT_DATASET"]

#: Hugging Face namespace that owns the source and output datasets.
HUB_NAMESPACE: Final[str] = "NoeFlandre"
#: Public source repository of this project.
CODE_REPOSITORY: Final[str] = "https://github.com/NoeFlandre/osm-worldcover"
#: Output dataset of the default (Wikidata) source. Also the card's fallback name.
WIKIDATA_OUTPUT_DATASET: Final[str] = f"{HUB_NAMESPACE}/osm-wikidata-worldcover"
