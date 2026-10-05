import pytest

from osm_worldcover.domain.revision import is_full_revision, is_optional_revision

LOWER = "a" * 40
UPPER = "A" * 40


@pytest.mark.parametrize(
    ("value", "lower_only", "any_case"),
    [
        (LOWER, True, True),
        ("0123456789abcdef" * 2 + "01234567", True, True),
        (UPPER, False, True),
        ("aA" * 20, False, True),
        ("a" * 39, False, False),
        ("a" * 41, False, False),
        ("g" * 40, False, False),
        ("", False, False),
        (LOWER + "\n", False, False),
        (" " + LOWER, False, False),
        (None, False, False),
        (b"a" * 40, False, False),
        (123, False, False),
    ],
)
def test_is_full_revision(value, lower_only, any_case):
    assert is_full_revision(value) is lower_only
    assert is_full_revision(value, allow_uppercase=True) is any_case


def test_is_optional_revision():
    assert is_optional_revision(None)
    assert is_optional_revision(LOWER)
    assert not is_optional_revision(UPPER)
    assert not is_optional_revision("")
