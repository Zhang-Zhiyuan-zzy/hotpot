"""Factorized candidate proposal merge tests."""

from typing import Optional

import pytest

from hotpot.cheminfo.calculator.charge import NetChargeCandidateEvidence
from hotpot.cheminfo.calculator.contracts import (
    CandidateBudget,
    CandidateProvenance,
    ScientificQuantity,
    StateValueAuthority,
)
from hotpot.cheminfo.calculator.state_composition.candidate_space import (
    _merge_candidate_proposals,
)
from hotpot.cheminfo.calculator.state_composition.contracts import (
    CandidateDomainScope,
    CandidateDomainSummary,
    CandidateProposal,
    FactorKey,
    MaterializationStatus,
    ProposedCandidateFactor,
    ProposedCandidateValue,
)


def _charge(identity: str, value: int) -> NetChargeCandidateEvidence:
    return NetChargeCandidateEvidence(
        CandidateProvenance(
            identity,
            StateValueAuthority.VERIFIED_SCIENTIFIC_SOURCE,
            "test",
        ),
        value,
    )


def _proposal(
    proposal_id: str,
    candidates: tuple[NetChargeCandidateEvidence, ...],
    *,
    status: MaterializationStatus = MaterializationStatus.COMPLETE,
    summary: Optional[CandidateDomainSummary] = None,
) -> CandidateProposal:
    return CandidateProposal(
        proposal_id,
        (
            ProposedCandidateFactor(
                FactorKey(ScientificQuantity.NET_CHARGE, "molecule"),
                "charge-domain-v1",
                tuple(ProposedCandidateValue(candidate) for candidate in candidates),
                CandidateDomainScope.DOMAIN_BOUNDED,
                status,
                summary,
            ),
        ),
        (),
    )


def test_merge_deduplicates_candidates_and_retains_sources() -> None:
    shared = _charge("q0", 0)

    factors, constraints, guarded = _merge_candidate_proposals(
        (_proposal("source:b", (shared,)), _proposal("source:a", (shared,))),
        CandidateBudget(8, 8, 8),
    )

    assert len(factors) == 1
    assert len(factors[0].values) == 1
    assert factors[0].values[0].source_proposal_ids == (
        "source:a",
        "source:b",
    )
    assert constraints == guarded == ()


def test_factor_budget_records_unmaterialized_remainder() -> None:
    factors, _, _ = _merge_candidate_proposals(
        (_proposal("source", (_charge("q0", 0), _charge("q1", 1))),),
        CandidateBudget(1, 8, 8),
    )

    factor = factors[0]
    assert factor.materialization_status is MaterializationStatus.RESOURCE_TRUNCATED
    assert factor.unexpanded_domain_summary is not None
    assert factor.unexpanded_domain_summary.remaining_count_upper_bound == 1


def test_declared_truncation_requires_remainder_summary() -> None:
    proposal = _proposal(
        "source",
        (_charge("q0", 0),),
        status=MaterializationStatus.RESOURCE_TRUNCATED,
    )

    with pytest.raises(ValueError, match="remainder summary"):
        _merge_candidate_proposals((proposal,), CandidateBudget(8, 8, 8))


def test_merge_is_independent_of_proposal_order() -> None:
    proposals = (
        _proposal("b", (_charge("q1", 1),)),
        _proposal("a", (_charge("q0", 0),)),
    )

    forward = _merge_candidate_proposals(
        proposals,
        CandidateBudget(8, 8, 8),
    )[0]
    reverse = _merge_candidate_proposals(
        tuple(reversed(proposals)),
        CandidateBudget(8, 8, 8),
    )[0]

    assert forward == reverse
