"""Structural validation for factorized scientific candidate proposals."""

from __future__ import annotations

from typing import Tuple

from ..charge.contracts import (
    LewisFormalChargeCandidate,
    NetChargeCandidateEvidence,
)
from ..contracts import CandidateBudget, ScientificQuantity
from ..electronic_state.contracts import (
    ElectronicStateCandidate,
    SpinContributionCandidate,
)
from ..oxidation_state.contracts import (
    CoordinationCountingCandidate,
    OxidationStateCandidateEvidence,
)
from ._artifacts import _FRONTIER_SCHEMA
from .contracts import (
    CandidateDomainSummary,
    CandidateProposal,
    DerivedStateValue,
    FactorKey,
    MaterializationStatus,
    ProposedCandidateFactor,
    StateCandidate,
)

__all__ = []


def _candidate_matches_factor(
    factor_key: FactorKey,
    candidate: StateCandidate,
) -> bool:
    if isinstance(candidate, DerivedStateValue):
        return candidate.quantity is factor_key.quantity
    candidate_types = {
        ScientificQuantity.NET_CHARGE: (NetChargeCandidateEvidence,),
        ScientificQuantity.LEWIS_FORMAL_CHARGE: (
            LewisFormalChargeCandidate,
        ),
        ScientificQuantity.COORDINATION_COUNTING: (
            CoordinationCountingCandidate,
        ),
        ScientificQuantity.OXIDATION_STATE: (
            OxidationStateCandidateEvidence,
        ),
        ScientificQuantity.ELECTRONIC_STATE: (
            SpinContributionCandidate,
            ElectronicStateCandidate,
        ),
    }
    return isinstance(candidate, candidate_types[factor_key.quantity])


def _validate_domain_summary(summary: CandidateDomainSummary) -> None:
    remaining = summary.remaining_count_upper_bound
    if remaining is not None and (
        type(remaining) is not int or remaining <= 0
    ):
        raise ValueError(
            "A known remainder count must be a positive integer"
        )
    continuation = summary.continuation_identity
    if continuation is not None and not continuation:
        raise ValueError("A continuation identity cannot be empty")
    bound = summary.frontier_bound
    if continuation is None and bound is None:
        raise ValueError(
            "A remainder summary requires a continuation or frontier bound"
        )
    if bound is None:
        return
    if not bound.branch_identity:
        raise ValueError("A frontier branch identity cannot be empty")
    if bound.schema_version != _FRONTIER_SCHEMA:
        raise ValueError("A frontier bound uses an unsupported schema")
    if not bound.optimistic_key or any(
        not isinstance(value, str) or not value
        for value in bound.optimistic_key
    ):
        raise ValueError("A frontier bound requires non-empty string keys")


def _validate_proposed_factor(factor: ProposedCandidateFactor) -> None:
    if not factor.domain_definition_id:
        raise ValueError("A candidate factor requires a domain definition")
    truncated = (
        factor.materialization_status
        is MaterializationStatus.RESOURCE_TRUNCATED
    )
    if truncated != (factor.unexpanded_domain_summary is not None):
        raise ValueError(
            "Factor truncation and remainder summary must agree"
        )
    if factor.unexpanded_domain_summary is not None:
        _validate_domain_summary(factor.unexpanded_domain_summary)

    candidate_ids = tuple(
        value.candidate.provenance.identity
        for value in factor.materialized_candidates
    )
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Candidate identities must be unique within a factor")
    for proposed in factor.materialized_candidates:
        if not _candidate_matches_factor(
            factor.factor_key,
            proposed.candidate,
        ):
            raise ValueError(
                "Candidate type does not match its scientific factor"
            )
        branch_groups = tuple(
            assignment.group_id for assignment in proposed.branch_assignments
        )
        if len(set(branch_groups)) != len(branch_groups):
            raise ValueError(
                "One candidate cannot repeat a branch-assignment group"
            )


def _validate_candidate_proposal_structure(
    proposal: CandidateProposal,
) -> None:
    if not proposal.proposal_id:
        raise ValueError("A candidate proposal requires an identity")
    factor_keys = tuple(factor.factor_key for factor in proposal.factors)
    if len(set(factor_keys)) != len(factor_keys):
        raise ValueError("Candidate proposal factor keys must be unique")
    for factor in proposal.factors:
        _validate_proposed_factor(factor)


def _validate_candidate_proposal_budget(
    proposal: CandidateProposal,
    candidate_budget: CandidateBudget,
) -> None:
    _validate_candidate_proposal_structure(proposal)
    if any(
        len(factor.materialized_candidates)
        > candidate_budget.max_materialized_values_per_factor
        for factor in proposal.factors
    ):
        raise ValueError(
            "Candidate proposal exceeds its materialization budget"
        )


def _validate_candidate_proposals(
    proposals: Tuple[CandidateProposal, ...],
) -> None:
    for proposal in proposals:
        _validate_candidate_proposal_structure(proposal)
