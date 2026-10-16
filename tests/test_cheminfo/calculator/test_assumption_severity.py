"""Versioned assumption-severity policy tests."""

from hotpot.cheminfo.calculator.state_rules.profile_decision import (
    _assumption_severity_policy,
)
from hotpot.cheminfo.calculator.state_rules.profiles import (
    _coordination_screening_profile,
)


def test_assumption_vector_orders_more_severe_categories_first() -> None:
    policy = _assumption_severity_policy(
        _coordination_screening_profile()
    )

    assert policy.ordered_categories == (
        "execution_prior",
        "ionic_approximation",
        "domain_template",
        "representation_default",
    )
    assert policy.vector(
        (
            "domain_template:amine",
            "execution_prior:f-block-hund",
            "domain_template:pyridine",
        )
    ) == (1, 0, 2, 0)


def test_unknown_assumption_is_not_silently_ranked_as_harmless() -> None:
    policy = _assumption_severity_policy(
        _coordination_screening_profile()
    )

    assert policy.vector(("unreviewed_assumption:value",)) is None


def test_assumption_policy_identity_is_deterministic() -> None:
    profile = _coordination_screening_profile()

    assert _assumption_severity_policy(profile) == (
        _assumption_severity_policy(profile)
    )
