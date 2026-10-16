"""Static scientific-profile identity and eligibility tests."""

from hotpot.cheminfo.calculator.contracts import (
    DEFAULT_CANDIDATE_BUDGET,
    CandidateBudget,
    ProviderSet,
    ProviderToggle,
    ScientificOperationRequirements,
    ScientificQuantity,
    StatePolicy,
)
from hotpot.cheminfo.calculator.profile_eligibility import (
    _static_profile_eligibility,
)
from hotpot.cheminfo.calculator.state_rules import RuleSet


def _requirements() -> ScientificOperationRequirements:
    return ScientificOperationRequirements(
        "test-operation",
        frozenset({ScientificQuantity.NET_CHARGE}),
        frozenset(),
    )


def _eligibility(
    provider_set: ProviderSet = ProviderSet(),
    candidate_budget: CandidateBudget = DEFAULT_CANDIDATE_BUDGET,
):
    return _static_profile_eligibility(
        StatePolicy.STRICT,
        _requirements(),
        provider_set,
        RuleSet(),
        candidate_budget,
    )


def test_strict_profile_is_statically_eligible() -> None:
    eligibility = _eligibility()

    assert eligibility.profile_id == StatePolicy.STRICT.value
    assert eligibility.in_domain is True
    assert eligibility.findings == ()


def test_effective_profile_identity_is_provider_order_independent() -> None:
    toggles = (
        ProviderToggle("PROVIDER-002", False),
        ProviderToggle("PROVIDER-001", True),
    )

    first = _eligibility(ProviderSet(toggles))
    second = _eligibility(ProviderSet(tuple(reversed(toggles))))

    assert first.effective_profile_identity == second.effective_profile_identity


def test_profile_override_changes_effective_identity() -> None:
    default = _eligibility()
    overridden = _eligibility(candidate_budget=CandidateBudget(8, 32, 8))

    assert (
        default.effective_profile_identity
        != overridden.effective_profile_identity
    )


def test_unactivated_screening_artifact_is_machine_readable_out_of_domain() -> None:
    eligibility = _static_profile_eligibility(
        StatePolicy.COORDINATION_SCREENING_V1,
        _requirements(),
        ProviderSet(),
        RuleSet(),
        DEFAULT_CANDIDATE_BUDGET,
    )

    assert eligibility.in_domain is False
    assert eligibility.findings[0].code == (
        "PROFILE.SCREENING_ARTIFACT_NOT_ACTIVE"
    )
