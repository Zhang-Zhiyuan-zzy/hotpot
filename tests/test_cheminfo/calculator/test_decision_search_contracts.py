"""Partial-order and immutability tests for decision-search contracts."""

from dataclasses import FrozenInstanceError

import pytest

from hotpot.cheminfo.calculator.state_composition.decision_search import (
    _AuthorityVector,
    _DecisionRelation,
    _authority_relation,
)


def test_authority_uses_pareto_dominance_not_tuple_order() -> None:
    charge_stronger = _AuthorityVector((("charge", 4), ("spin", 2)))
    spin_stronger = _AuthorityVector((("charge", 3), ("spin", 3)))

    assert _authority_relation(
        charge_stronger,
        spin_stronger,
    ) is _DecisionRelation.INCOMPARABLE


def test_authority_dominance_is_coordinate_order_independent() -> None:
    stronger = _AuthorityVector((("charge", 4), ("spin", 3)))
    weaker = _AuthorityVector((("spin", 2), ("charge", 4)))

    assert _authority_relation(
        stronger,
        weaker,
    ) is _DecisionRelation.DOMINATES
    assert _authority_relation(
        weaker,
        stronger,
    ) is _DecisionRelation.DOMINATED


def test_authority_vectors_require_the_same_unique_axes() -> None:
    with pytest.raises(ValueError, match="unique"):
        _authority_relation(
            _AuthorityVector((("charge", 4), ("charge", 3))),
            _AuthorityVector((("charge", 4),)),
        )
    with pytest.raises(ValueError, match="same quantities"):
        _authority_relation(
            _AuthorityVector((("charge", 4),)),
            _AuthorityVector((("spin", 4),)),
        )


def test_decision_search_contracts_are_immutable() -> None:
    vector = _AuthorityVector((("charge", 4),))

    with pytest.raises(FrozenInstanceError):
        vector.coordinates = ()
