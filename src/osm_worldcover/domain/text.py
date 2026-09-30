"""Text eligibility and normalized text-label identity.

Normalisation is deliberately conservative: whitespace only. The articles are
multilingual, so case folding, punctuation stripping or unicode normalisation
would damage some scripts while helping none of the checks made here.
"""

import hashlib
from typing import Final

__all__ = ["DEFAULT_MIN_WORDS", "dedup_key", "is_usable", "normalise", "word_count"]

#: Below this, an article carries too little signal to be a training example.
DEFAULT_MIN_WORDS: Final[int] = 10

#: Separates fields inside the hash so that ("ab", "c") and ("a", "bc") differ.
_FIELD_SEPARATOR: Final[bytes] = b"\x00"


def normalise(text: str) -> str:
    """Collapse runs of whitespace and trim, leaving all other characters intact."""
    return " ".join(text.split())


def word_count(text: str) -> int:
    """Return the number of whitespace-separated tokens in ``text``."""
    return len(text.split())


def is_usable(text: str, min_words: int = DEFAULT_MIN_WORDS) -> bool:
    """Return whether ``text`` is long enough to keep as an example."""
    return word_count(text) >= min_words


def dedup_key(text: str, label: str) -> str:
    """Return a stable normalized-text and label key.

    The release adds the stable polygon identity to this key before removing a
    record, so equal text and labels on different polygons remain distinct.
    """
    digest = hashlib.sha256()
    digest.update(normalise(text).encode("utf-8"))
    digest.update(_FIELD_SEPARATOR)
    digest.update(label.encode("utf-8"))
    return digest.hexdigest()
