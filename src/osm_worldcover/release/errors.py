"""Exception raised when a release transaction cannot proceed or recover."""

from __future__ import annotations


class ReleaseCommitError(OSError):
    """A release transaction cannot safely proceed or recover."""
