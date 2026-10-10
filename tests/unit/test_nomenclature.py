"""ESA WorldCover nomenclature lookup behaviour."""

import pytest

from osm_worldcover.domain import nomenclature as nom


def test_there_are_eleven_worldcover_classes() -> None:
    assert len(nom.CLASS_LABELS) == 11


@pytest.mark.parametrize(
    ("code", "label"),
    [(10, "Tree cover"), (50, "Built-up"), (95, "Mangroves"), (100, "Moss and lichen")],
)
def test_label_for_known_code(code: int, label: str) -> None:
    assert nom.label_for(code) == label


def test_codes_are_the_official_non_sequential_values() -> None:
    assert sorted(nom.CLASS_LABELS) == [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]


def test_label_for_unknown_code_raises() -> None:
    with pytest.raises(nom.UnknownLandCoverCodeError) as raised:
        nom.label_for(0)
    assert raised.value.args == (0,)


def test_is_valid_code_discriminates() -> None:
    assert nom.is_valid_code(10)
    assert not nom.is_valid_code(0)  # WorldCover no-data
    assert not nom.is_valid_code(15)  # between real classes
    assert not nom.is_valid_code(255)


def test_nodata_value_is_declared_and_is_not_a_class() -> None:
    assert nom.NODATA == 0
    assert not nom.is_valid_code(nom.NODATA)


def test_an_unknown_code_is_named_in_the_error() -> None:
    with pytest.raises(nom.UnknownLandCoverCodeError) as raised:
        nom.label_for(999)
    assert raised.value.args == (999,)


def test_labels_are_unique() -> None:
    assert len(set(nom.CLASS_LABELS.values())) == len(nom.CLASS_LABELS)


def test_class_table_is_immutable() -> None:
    with pytest.raises(TypeError):
        # The assignment is meant to be rejected; that is the assertion.
        nom.CLASS_LABELS[10] = "something else"  # ty: ignore[invalid-assignment]
