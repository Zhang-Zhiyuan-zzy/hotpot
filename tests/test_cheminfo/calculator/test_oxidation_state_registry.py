"""PROVIDER-002 versioned oxidation-state registry tests."""

import json
from pathlib import Path

import pytest

from hotpot.cheminfo.calculator.oxidation_state import (
    element_oxidation_state_domain,
)


def test_registry_contains_ordered_rows_for_all_elements() -> None:
    path = (
        Path(__file__).parents[3]
        / "hotpot"
        / "cheminfo"
        / "calculator"
        / "oxidation_state"
        / "data"
        / "element_oxidation_states.json"
    )
    payload = json.loads(path.read_text())

    assert [row["atomic_number"] for row in payload["elements"]] == list(
        range(1, 119)
    )
    assert all(
        "source_class" in state
        for row in payload["elements"]
        for state in row["states"]
    )
    assert all(
        "sha256" in source and "url" in source
        for source in payload["sources"]
    )


@pytest.mark.parametrize(
    ("atomic_number", "expected_states"),
    (
        (1, (-1, 1)),
        (26, (-4, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7)),
        (63, (0, 2, 3)),
        (92, (-1, 1, 2, 3, 4, 5, 6)),
        (95, (2, 3, 4, 5, 6, 7)),
        (113, ()),
    ),
)
def test_registry_returns_unranked_candidate_domains(
    atomic_number: int,
    expected_states: tuple[int, ...],
) -> None:
    domain = element_oxidation_state_domain(atomic_number)

    assert domain.atomic_number == atomic_number
    assert domain.states == expected_states
    assert len(domain.content_digest) == 64


def test_missing_registry_values_are_not_changed_to_zero() -> None:
    assert element_oxidation_state_domain(113).states == ()


@pytest.mark.parametrize("atomic_number", (0, 119))
def test_registry_rejects_out_of_periodic_table_indices(atomic_number: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 118"):
        element_oxidation_state_domain(atomic_number)
