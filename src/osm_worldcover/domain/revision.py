"""The single definition of a full 40-hex commit revision."""

import re
from typing import TypeGuard

_FULL_REVISION = re.compile(r"[0-9a-f]{40}")


def is_full_revision(value: object, *, allow_uppercase: bool = False) -> TypeGuard[str]:
    """Return whether ``value`` is a full 40-hex commit SHA string.

    Lowercase only by default; ``allow_uppercase`` additionally accepts
    uppercase hex digits for callers that take user-supplied pins.
    """
    if not isinstance(value, str):
        return False
    if allow_uppercase:
        value = value.lower()
    return _FULL_REVISION.fullmatch(value) is not None


def is_optional_revision(value: object) -> bool:
    """Return whether ``value`` is ``None`` or a full lowercase revision."""
    return value is None or is_full_revision(value)
