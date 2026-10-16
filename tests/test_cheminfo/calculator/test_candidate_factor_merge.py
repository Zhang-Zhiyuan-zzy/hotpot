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
    CandidateConstraint,
    CandidateProposal,
    ConstraintGuard,
    FactorKey,
    FrontierBound,
    GuardedCandidateConstraint,
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
        (),
    )

    assert len(factors) == 1
    assert len(factors[0].values) == 1
    assert factors[0].values[0].source_proposal_ids == (
        "source:a",
        "source:b",
    )
    assert constraints == guarded == ()


def test_same_candidate_identity_cannot_change_payload() -> None:
    with pytest.raises(ValueError, match="different candidate payloads"):
        _merge_candidate_proposals(
            (
                _proposal("source:a", (_charge("shared", 0),)),
                _proposal("source:b", (_charge("shared", 1),)),
            ),
            CandidateBudget(8, 8, 8),
            (),
        )


def test_factor_budget_records_unmaterialized_remainder() -> None:
    factors, _, _ = _merge_candidate_proposals(
        (_proposal("source", (_charge("q0", 0), _charge("q1", 1))),),
        CandidateBudget(1, 8, 8),
        (),
    )

    factor = factors[0]
    assert factor.materialization_status is MaterializationStatus.RESOURCE_TRUNCATED
    assert factor.unexpanded_domain_summary is not None
    assert factor.unexpanded_domain_summary.remaining_count_upper_bound == 1


def test_unknown_source_remainder_stays_unknown_after_merge() -> None:
    source_summary = CandidateDomainSummary(None, "source:next", None)

    factors, _, _ = _merge_candidate_proposals(
        (
            _proposal(
                "source",
                (_charge("q0", 0),),
                status=MaterializationStatus.RESOURCE_TRUNCATED,
                summary=source_summary,
            ),
        ),
        CandidateBudget(8, 8, 8),
        (),
    )

    assert (
        factors[0].unexpanded_domain_summary.remaining_count_upper_bound
        is None
    )


def test_single_provider_frontier_bound_survives_merge() -> None:
    bound = FrontierBound(
        "source:branch",
        "scientific-state-frontier-v1",
        ("best",),
    )
    source_summary = CandidateDomainSummary(1, "source:next", bound)

    factors, _, _ = _merge_candidate_proposals(
        (
            _proposal(
                "source",
                (_charge("q0", 0),),
                status=MaterializationStatus.RESOURCE_TRUNCATED,
                summary=source_summary,
            ),
        ),
        CandidateBudget(8, 8, 8),
        (),
    )

    assert factors[0].unexpanded_domain_summary.frontier_bound == bound


def test_local_budget_omission_cannot_reuse_provider_frontier_bound() -> None:
    bound = FrontierBound(
        "source:branch",
        "scientific-state-frontier-v1",
        ("best",),
    )
    source_summary = CandidateDomainSummary(1, "source:next", bound)

    factors, _, _ = _merge_candidate_proposals(
        (
            _proposal(
                "source",
                (_charge("q0", 0), _charge("q1", 1)),
                status=MaterializationStatus.RESOURCE_TRUNCATED,
                summary=source_summary,
            ),
        ),
        CandidateBudget(1, 8, 8),
        (),
    )

    assert factors[0].unexpanded_domain_summary.frontier_bound is None


def test_declared_truncation_requires_remainder_summary() -> None:
    proposal = _proposal(
        "source",
        (_charge("q0", 0),),
        status=MaterializationStatus.RESOURCE_TRUNCATED,
    )

    with pytest.raises(ValueError, match="remainder summary"):
        _merge_candidate_proposals(
            (proposal,),
            CandidateBudget(8, 8, 8),
            (),
        )


def test_complete_factor_cannot_carry_a_remainder_summary() -> None:
    proposal = _proposal(
        "source",
        (_charge("q0", 0),),
        summary=CandidateDomainSummary(1, "source:next", None),
    )

    with pytest.raises(ValueError, match="truncation and remainder"):
        _merge_candidate_proposals(
            (proposal,),
            CandidateBudget(8, 8, 8),
            (),
        )


@pytest.mark.parametrize("remaining", (0, -1, True, 1.0))
def test_remainder_count_must_be_a_positive_integer(remaining: object) -> None:
    proposal = _proposal(
        "source",
        (_charge("q0", 0),),
        status=MaterializationStatus.RESOURCE_TRUNCATED,
        summary=CandidateDomainSummary(remaining, "source:next", None),
    )

    with pytest.raises(ValueError, match="positive integer"):
        _merge_candidate_proposals(
            (proposal,),
            CandidateBudget(8, 8, 8),
            (),
        )


def test_remainder_requires_an_auditable_locator() -> None:
    proposal = _proposal(
        "source",
        (_charge("q0", 0),),
        status=MaterializationStatus.RESOURCE_TRUNCATED,
        summary=CandidateDomainSummary(None, None, None),
    )

    with pytest.raises(ValueError, match="continuation or frontier"):
        _merge_candidate_proposals(
            (proposal,),
            CandidateBudget(8, 8, 8),
            (),
        )


def test_merge_is_independent_of_proposal_order() -> None:
    proposals = (
        _proposal("b", (_charge("q1", 1),)),
        _proposal("a", (_charge("q0", 0),)),
    )

    forward = _merge_candidate_proposals(
        proposals,
        CandidateBudget(8, 8, 8),
        (),
    )[0]
    reverse = _merge_candidate_proposals(
        tuple(reversed(proposals)),
        CandidateBudget(8, 8, 8),
        (),
    )[0]

    assert forward == reverse


def test_unconditional_exclusion_precedes_factor_budget() -> None:
    excluded = _charge("0-excluded", 1)
    retained = _charge("z-retained", 0)
    factor_key = FactorKey(ScientificQuantity.NET_CHARGE, "molecule")
    constraint = CandidateConstraint(
        "exclude-q1",
        "excludes_all",
        "candidate-constraint-v1",
        (factor_key,),
        (("candidate_ids", (excluded.provenance.identity,)),),
    )

    factors, constraints, guarded = _merge_candidate_proposals(
        (_proposal("source", (excluded, retained)),),
        CandidateBudget(1, 8, 8),
        (
            GuardedCandidateConstraint(
                ConstraintGuard(),
                constraint,
            ),
        ),
    )

    assert tuple(
        value.candidate.provenance.identity for value in factors[0].values
    ) == ("z-retained",)
    assert factors[0].materialization_status is MaterializationStatus.COMPLETE
    assert factors[0].unexpanded_domain_summary is None
    assert constraints == (constraint,)
    assert guarded[0].constraint == constraint


def test_unconditional_requirement_precedes_factor_budget() -> None:
    skipped = _charge("a-skipped", 0)
    retained = _charge("z-required", 1)
    factor_key = FactorKey(ScientificQuantity.NET_CHARGE, "molecule")
    constraint = CandidateConstraint(
        "require-q1",
        "requires_all",
        "candidate-constraint-v1",
        (factor_key,),
        (("candidate_ids", (retained.provenance.identity,)),),
    )

    factors, _, _ = _merge_candidate_proposals(
        (_proposal("source", (skipped, retained)),),
        CandidateBudget(1, 8, 8),
        (
            GuardedCandidateConstraint(
                ConstraintGuard(),
                constraint,
            ),
        ),
    )

    assert tuple(
        value.candidate.provenance.identity for value in factors[0].values
    ) == ("z-required",)
    assert factors[0].materialization_status is MaterializationStatus.COMPLETE
