"""Immutable coordination-screening profile artifact tests."""

from hotpot.cheminfo.calculator.contracts import (
    DEFAULT_CANDIDATE_BUDGET,
    ScientificQuantity,
    StatePolicy,
)
from hotpot.cheminfo.calculator.state_rules.profiles import (
    _coordination_screening_profile,
)


def test_screening_profile_artifact_freezes_decision_context() -> None:
    profile = _coordination_screening_profile()

    assert profile.profile_id == StatePolicy.COORDINATION_SCREENING_V1.value
    assert profile.required_quantities == frozenset(
        {
            ScientificQuantity.NET_CHARGE,
            ScientificQuantity.OXIDATION_STATE,
        }
    )
    assert profile.candidate_budget == DEFAULT_CANDIDATE_BUDGET
    assert profile.frontier_schema_version == (
        "scientific-state-frontier-v1"
    )
    assert len(profile.required_rule_versions) == 10
    assert profile.content_digest == (
        "40693f1381ae34cc5cf3dcf7c6c20d20311aaa74e94eff46eb17e1a4598f377e"
    )


def test_screening_profile_artifact_is_stable_within_one_installation() -> None:
    first = _coordination_screening_profile()
    second = _coordination_screening_profile()

    assert first is second
    assert first.content_digest == second.content_digest
