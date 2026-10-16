"""Structural validation of optional scientific-rule outcomes."""

from dataclasses import replace

import pytest

from hotpot.cheminfo.calculator.charge import NetChargeCandidateEvidence
from hotpot.cheminfo.calculator.contracts import (
    CandidateBudget,
    CandidateProvenance,
    ScientificQuantity,
    StateValueAuthority,
)
from hotpot.cheminfo.calculator.state_composition import (
    CandidateDomainScope,
    FactorKey,
    MaterializationStatus,
    ProposedCandidateFactor,
    ProposedCandidateValue,
    CandidateProposal,
)
from hotpot.cheminfo.calculator.state_rules import (
    RankAdjustment,
    ResultType,
    RuleApplicability,
    RuleKind,
    RuleOutcome,
    RulePhase,
    RuleSpecification,
)
from hotpot.cheminfo.calculator.state_rules.validation import (
    _validate_rule_outcome,
)


_BUDGET = CandidateBudget(1, 8, 8)


def _specification() -> RuleSpecification:
    return RuleSpecification(
        "OPTIONAL-TEST",
        "1.0.0",
        RuleKind.EMPIRICAL,
        RulePhase.CHARGE_EVIDENCE,
        (),
        (),
        (ResultType.CANDIDATE_PROPOSAL, ResultType.RANK_ADJUSTMENT),
        "test-v1",
    )


def _candidate(identity: str) -> NetChargeCandidateEvidence:
    return NetChargeCandidateEvidence(
        CandidateProvenance(
            identity,
            StateValueAuthority.DOMAIN_RULE,
            "OPTIONAL-TEST",
        ),
        0,
    )


def _outcome() -> RuleOutcome:
    factor = ProposedCandidateFactor(
        FactorKey(ScientificQuantity.NET_CHARGE, "molecule"),
        "test-charge-domain-v1",
        (ProposedCandidateValue(_candidate("charge:0")),),
        CandidateDomainScope.DOMAIN_BOUNDED,
        MaterializationStatus.COMPLETE,
        None,
    )
    return RuleOutcome(
        "OPTIONAL-TEST",
        "1.0.0",
        RuleApplicability(True),
        CandidateProposal("proposal:test", (factor,), ()),
        (),
        (RankAdjustment("charge:0", "test", 1.0, "test"),),
        (),
        (),
    )


def test_valid_rule_outcome_matches_specification_and_budget() -> None:
    _validate_rule_outcome(_specification(), _outcome(), _BUDGET)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("rule_id", "OTHER", "identity"),
        ("rule_version", "2.0.0", "version"),
    ),
)
def test_rule_outcome_identity_and_version_must_match(
    field: str,
    value: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _validate_rule_outcome(
            _specification(),
            replace(_outcome(), **{field: value}),
            _BUDGET,
        )


def test_rule_cannot_return_an_undeclared_output() -> None:
    specification = replace(_specification(), produces=())

    with pytest.raises(ValueError, match="undeclared candidate"):
        _validate_rule_outcome(specification, _outcome(), _BUDGET)


def test_rule_candidate_materialization_is_bounded_before_merge() -> None:
    outcome = _outcome()
    factor = outcome.candidate_proposal.factors[0]
    over_budget = replace(
        factor,
        materialized_candidates=(
            ProposedCandidateValue(_candidate("charge:0")),
            ProposedCandidateValue(_candidate("charge:1")),
        ),
    )
    proposal = replace(outcome.candidate_proposal, factors=(over_budget,))

    with pytest.raises(ValueError, match="materialization budget"):
        _validate_rule_outcome(
            _specification(),
            replace(outcome, candidate_proposal=proposal),
            _BUDGET,
        )


def test_rule_factor_requires_consistent_truncation_evidence() -> None:
    outcome = _outcome()
    factor = replace(
        outcome.candidate_proposal.factors[0],
        materialization_status=MaterializationStatus.RESOURCE_TRUNCATED,
    )
    proposal = replace(outcome.candidate_proposal, factors=(factor,))

    with pytest.raises(ValueError, match="remainder summary"):
        _validate_rule_outcome(
            _specification(),
            replace(outcome, candidate_proposal=proposal),
            _BUDGET,
        )


def test_rule_rank_adjustment_must_be_finite() -> None:
    outcome = replace(
        _outcome(),
        rank_adjustments=(
            RankAdjustment("charge:0", "test", float("nan"), "test"),
        ),
    )

    with pytest.raises(ValueError, match="must be finite"):
        _validate_rule_outcome(_specification(), outcome, _BUDGET)
