"""Immutable coordination-screening profile artifact tests."""

from hotpot.cheminfo.rules.chemistry.scientific_state.contracts import (
    DEFAULT_CANDIDATE_BUDGET,
    ScientificQuantity,
)
from hotpot.cheminfo.operations.calculation.scientific_state.contracts import StatePolicy
from hotpot.cheminfo.operations.calculation.scientific_state.profiles import (
    _coordination_screening_profile,
)


def test_screening_profile_artifact_freezes_decision_context() -> None:
    profile = _coordination_screening_profile()

    assert profile.profile_id == StatePolicy.COORDINATION_SCREENING_V1.value
    assert profile.profile_version == "1.1.0"
    assert profile.activation_status == "active"
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
    assert len(profile.required_rule_versions) == 12
    assert profile.content_digest == (
        "dfba1eb6f19c5ccd868c462bb3276ed6cbf417e71b3f114254be166ecdc687ff"
    )


def test_screening_profile_artifact_is_stable_within_one_installation() -> None:
    first = _coordination_screening_profile()
    second = _coordination_screening_profile()

    assert first is second
    assert first.content_digest == second.content_digest
