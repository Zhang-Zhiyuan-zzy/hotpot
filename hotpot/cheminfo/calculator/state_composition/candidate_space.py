"""Deterministic merging of factorized scientific candidate proposals."""

from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Tuple

from ..contracts import CandidateBudget
from .constraints import (
    _constraint_candidate_ids,
    _unconditional_constraint,
)
from .contracts import (
    BranchAssignment,
    CandidateConstraint,
    CandidateDomainScope,
    CandidateDomainSummary,
    CandidateFactor,
    CandidateProposal,
    FactorCandidateValue,
    FactorDomainContribution,
    FactorKey,
    GuardedCandidateConstraint,
    MaterializationStatus,
    ProposedCandidateFactor,
    StateCandidate,
)

__all__ = []


_SCOPE_PRECEDENCE = {
    CandidateDomainScope.EXHAUSTIVE: 0,
    CandidateDomainScope.DOMAIN_BOUNDED: 1,
    CandidateDomainScope.OPEN: 2,
}


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _candidate_identity(candidate: StateCandidate) -> str:
    return candidate.provenance.identity


def _merged_scope(
    factors: Tuple[ProposedCandidateFactor, ...],
) -> CandidateDomainScope:
    return max(
        (factor.domain_scope for factor in factors),
        key=_SCOPE_PRECEDENCE.__getitem__,
    )


def _merged_summary(
    factor_key: FactorKey,
    factors: Tuple[ProposedCandidateFactor, ...],
    omitted_count: int,
) -> CandidateDomainSummary:
    summaries = tuple(
        factor.unexpanded_domain_summary
        for factor in factors
        if factor.unexpanded_domain_summary is not None
    )
    known_remaining = tuple(
        summary.remaining_count_upper_bound
        for summary in summaries
        if summary.remaining_count_upper_bound is not None
    )
    remaining = (
        None
        if any(
            summary.remaining_count_upper_bound is None
            for summary in summaries
        )
        else omitted_count + sum(known_remaining)
    )
    frontier_bounds = tuple(
        summary.frontier_bound for summary in summaries
    )
    unique_frontier_bounds = frozenset(frontier_bounds)
    frontier_bound = (
        frontier_bounds[0]
        if omitted_count == 0
        and frontier_bounds
        and None not in unique_frontier_bounds
        and len(unique_frontier_bounds) == 1
        else None
    )
    continuation_payload = {
        "factor": (factor_key.quantity.value, factor_key.subject_identity),
        "continuations": tuple(
            summary.continuation_identity for summary in summaries
        ),
        "omitted_count": omitted_count,
    }
    return CandidateDomainSummary(
        remaining_count_upper_bound=remaining,
        continuation_identity=_digest(continuation_payload),
        frontier_bound=frontier_bound,
    )


def _merge_factor(
    factor_key: FactorKey,
    entries: Tuple[Tuple[str, ProposedCandidateFactor], ...],
    candidate_budget: CandidateBudget,
    excluded_candidate_ids: frozenset[str],
    required_candidate_ids: frozenset[str],
) -> CandidateFactor:
    factors = tuple(factor for _, factor in entries)
    values_by_identity: Dict[
        str,
        Tuple[StateCandidate, set[str], set[BranchAssignment]],
    ] = {}
    for proposal_id, factor in entries:
        if (
            factor.materialization_status
            is MaterializationStatus.RESOURCE_TRUNCATED
            and factor.unexpanded_domain_summary is None
        ):
            raise ValueError(
                "Resource-truncated candidate factors require a remainder summary"
            )
        for proposed in factor.materialized_candidates:
            identity = _candidate_identity(proposed.candidate)
            if identity in excluded_candidate_ids:
                continue
            incumbent = values_by_identity.get(identity)
            if incumbent is None:
                values_by_identity[identity] = (
                    proposed.candidate,
                    {proposal_id},
                    set(proposed.branch_assignments),
                )
            else:
                if incumbent[0] != proposed.candidate:
                    raise ValueError(
                        "One candidate identity cannot describe different "
                        "candidate payloads"
                    )
                incumbent[1].add(proposal_id)
                incumbent[2].update(proposed.branch_assignments)

    required_in_factor = required_candidate_ids.intersection(
        values_by_identity
    )
    eligible_identities = (
        required_in_factor
        if required_in_factor
        else values_by_identity.keys()
    )
    ordered_identities = tuple(sorted(eligible_identities))
    limit = candidate_budget.max_materialized_values_per_factor
    retained_identities = ordered_identities[:limit]
    omitted_count = len(ordered_identities) - len(retained_identities)
    values = tuple(
        FactorCandidateValue(
            candidate=values_by_identity[identity][0],
            source_proposal_ids=tuple(sorted(values_by_identity[identity][1])),
            branch_assignments=tuple(
                sorted(
                    values_by_identity[identity][2],
                    key=lambda assignment: (
                        assignment.group_id,
                        assignment.option_id,
                    ),
                )
            ),
        )
        for identity in retained_identities
    )
    truncated = omitted_count > 0 or any(
        factor.materialization_status
        is MaterializationStatus.RESOURCE_TRUNCATED
        for factor in factors
    )
    materialization_status = (
        MaterializationStatus.RESOURCE_TRUNCATED
        if truncated
        else MaterializationStatus.COMPLETE
    )
    summary = (
        _merged_summary(factor_key, factors, omitted_count)
        if truncated
        else None
    )
    domain_definition_ids = tuple(
        sorted({factor.domain_definition_id for factor in factors})
    )
    domain_definition_id = _digest(domain_definition_ids)
    domain_contributions = tuple(
        FactorDomainContribution(
            proposal_id=proposal_id,
            domain_scope=factor.domain_scope,
            materialization_status=factor.materialization_status,
            unexpanded_domain_summary=factor.unexpanded_domain_summary,
        )
        for proposal_id, factor in sorted(entries, key=lambda item: item[0])
    )
    identity = _digest(
        {
            "factor": (factor_key.quantity.value, factor_key.subject_identity),
            "domain_definition_id": domain_definition_id,
            "candidate_ids": retained_identities,
            "contributions": tuple(
                contribution.proposal_id
                for contribution in domain_contributions
            ),
            "materialization": materialization_status.value,
        }
    )
    return CandidateFactor(
        identity=identity,
        factor_key=factor_key,
        domain_definition_id=domain_definition_id,
        values=values,
        domain_contributions=domain_contributions,
        unexpanded_domain_summary=summary,
        domain_scope=_merged_scope(factors),
        materialization_status=materialization_status,
    )


def _merge_candidate_proposals(
    proposals: Tuple[CandidateProposal, ...],
    candidate_budget: CandidateBudget,
    direct_guarded_constraints: Tuple[GuardedCandidateConstraint, ...],
) -> Tuple[
    Tuple[CandidateFactor, ...],
    Tuple[CandidateConstraint, ...],
    Tuple[GuardedCandidateConstraint, ...],
]:
    proposal_ids = tuple(proposal.proposal_id for proposal in proposals)
    if len(set(proposal_ids)) != len(proposal_ids):
        raise ValueError("Candidate proposal IDs must be unique")

    entries_by_key: Dict[
        FactorKey,
        List[Tuple[str, ProposedCandidateFactor]],
    ] = {}
    guarded_constraints = list(direct_guarded_constraints)
    for proposal in proposals:
        for factor in proposal.factors:
            entries_by_key.setdefault(factor.factor_key, []).append(
                (proposal.proposal_id, factor)
            )
        guarded_constraints.extend(proposal.constraints)

    ordered_guarded_constraints = tuple(
        sorted(
            guarded_constraints,
            key=lambda guarded: guarded.constraint.identity,
        )
    )
    excluded_candidate_ids = frozenset(
        candidate_ids[0]
        for guarded in ordered_guarded_constraints
        if _unconditional_constraint(guarded)
        and guarded.constraint.constraint_type == "excludes_all"
        for candidate_ids in (
            _constraint_candidate_ids(guarded.constraint),
        )
        if len(candidate_ids) == 1
    )
    required_candidate_ids = frozenset(
        candidate_id
        for guarded in ordered_guarded_constraints
        if _unconditional_constraint(guarded)
        and guarded.constraint.constraint_type == "requires_all"
        for candidate_id in _constraint_candidate_ids(guarded.constraint)
    )
    factors = tuple(
        _merge_factor(
            factor_key,
            tuple(entries_by_key[factor_key]),
            candidate_budget,
            excluded_candidate_ids,
            required_candidate_ids,
        )
        for factor_key in sorted(
            entries_by_key,
            key=lambda key: (key.quantity.value, key.subject_identity),
        )
    )
    constraints_by_identity = {}
    for guarded in ordered_guarded_constraints:
        constraint = guarded.constraint
        incumbent = constraints_by_identity.get(constraint.identity)
        if incumbent is not None and incumbent != constraint:
            raise ValueError(
                "One constraint identity cannot describe different constraints"
            )
        constraints_by_identity[constraint.identity] = constraint
    return (
        factors,
        tuple(
            constraints_by_identity[identity]
            for identity in sorted(constraints_by_identity)
        ),
        ordered_guarded_constraints,
    )
