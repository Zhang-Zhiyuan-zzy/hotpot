"""Materialize molecular spin domains from local spin contributions."""

from __future__ import annotations

import hashlib
import itertools
import json
from typing import Optional, Tuple

from ..contracts import (
    CandidateBudget,
    CandidateProvenance,
    MolecularFacts,
    ScientificQuantity,
    StateValueAuthority,
)
from ..electronic_state.contracts import (
    SpinContributionCandidate,
    SpinContributionScope,
)
from .constraints import _molecular_subject_identity
from .contracts import (
    BranchAssignment,
    CandidateConstraint,
    CandidateDomainScope,
    CandidateDomainSummary,
    CandidateFactor,
    CandidateProposal,
    ConstraintGuard,
    FactorCandidateValue,
    FactorKey,
    GuardedCandidateConstraint,
    MaterializationStatus,
    ProposedCandidateFactor,
    ProposedCandidateValue,
)

__all__ = []


_COUPLING_SOURCE = "state-composition:local-spin-coupling-v1"
_CONSTRAINT_SCHEMA = "candidate-constraint-v1"
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


def _coupled_two_s_values(local_two_s: Tuple[int, ...]) -> Tuple[int, ...]:
    """Return the complete spin-only angular-momentum coupling domain."""
    totals = {0}
    for two_s in local_two_s:
        totals = {
            coupled
            for incumbent in totals
            for coupled in range(
                abs(incumbent - two_s),
                incumbent + two_s + 1,
                2,
            )
        }
    return tuple(sorted(totals))


def _local_spin_factors(
    factors: Tuple[CandidateFactor, ...],
) -> Optional[Tuple[CandidateFactor, ...]]:
    local_factors = []
    for factor in factors:
        if factor.factor_key.quantity is not ScientificQuantity.ELECTRONIC_STATE:
            continue
        local_values = tuple(
            value
            for value in factor.values
            if isinstance(value.candidate, SpinContributionCandidate)
            and value.candidate.scope is not SpinContributionScope.MOLECULAR
        )
        if not local_values:
            continue
        if len(local_values) != len(factor.values) or any(
            value.candidate.subject_identity
            != factor.factor_key.subject_identity
            for value in local_values
        ):
            return None
        local_factors.append(factor)
    return tuple(local_factors)


def _coupled_candidate(
    facts: MolecularFacts,
    local_values: Tuple[FactorCandidateValue, ...],
    two_s: int,
) -> SpinContributionCandidate:
    contributions = tuple(value.candidate for value in local_values)
    parent_ids = tuple(
        sorted(candidate.provenance.identity for candidate in contributions)
    )
    identity = _digest(
        {
            "source": _COUPLING_SOURCE,
            "facts": facts.identity.chemical_identity_digest,
            "parents": parent_ids,
            "two_s": two_s,
        }
    )
    return SpinContributionCandidate(
        provenance=CandidateProvenance(
            identity=identity,
            authority=StateValueAuthority.DERIVED_CONSTRAINT,
            source_identity=_COUPLING_SOURCE,
            evidence_ids=tuple(
                sorted(
                    {
                        *parent_ids,
                        *(
                            evidence_id
                            for candidate in contributions
                            for evidence_id in candidate.provenance.evidence_ids
                        ),
                    }
                )
            ),
            assumption_ids=tuple(
                sorted(
                    {
                        assumption
                        for candidate in contributions
                        for assumption in candidate.provenance.assumption_ids
                    }
                )
            ),
        ),
        scope=SpinContributionScope.MOLECULAR,
        subject_identity=_molecular_subject_identity(facts),
        parent_candidate_ids=parent_ids,
        two_s=two_s,
        pure_spin=True,
    )


def _spin_subject_atoms(
    facts: MolecularFacts,
    candidate: SpinContributionCandidate,
) -> Optional[frozenset[int]]:
    if candidate.scope is SpinContributionScope.ATOM:
        atom_index_by_subject = {
            f"atom:{atom_map}": atom_index
            for atom_index, atom_map in enumerate(facts.atom_map)
        }
        atom_index = atom_index_by_subject.get(candidate.subject_identity)
        return None if atom_index is None else frozenset({atom_index})
    if candidate.scope is SpinContributionScope.COMPONENT:
        return next(
            (
                frozenset(component.atom_indices)
                for component in facts.covalent_components
                if component.identity == candidate.subject_identity
            ),
            None,
        )
    return None


def _local_spin_subjects_form_partition(
    facts: MolecularFacts,
    contributions: Tuple[SpinContributionCandidate, ...],
) -> bool:
    occupied = set()
    for candidate in contributions:
        atoms = _spin_subject_atoms(facts, candidate)
        if atoms is None or occupied.intersection(atoms):
            return False
        occupied.update(atoms)
    return occupied == set(range(len(facts.atomic_numbers)))


def _merged_branch_assignments(
    local_values: Tuple[FactorCandidateValue, ...],
) -> Optional[Tuple[BranchAssignment, ...]]:
    options = {}
    for value in local_values:
        for assignment in value.branch_assignments:
            incumbent = options.get(assignment.group_id)
            if incumbent is not None and incumbent != assignment.option_id:
                return None
            options[assignment.group_id] = assignment.option_id
    return tuple(
        BranchAssignment(group_id, options[group_id])
        for group_id in sorted(options)
    )


def _coupling_constraint(
    molecular_key: FactorKey,
    local_factors: Tuple[CandidateFactor, ...],
    candidate: SpinContributionCandidate,
) -> GuardedCandidateConstraint:
    candidate_ids = (
        candidate.provenance.identity,
    ) + candidate.parent_candidate_ids
    return GuardedCandidateConstraint(
        guard=ConstraintGuard(
            all_candidate_ids=(candidate.provenance.identity,),
        ),
        constraint=CandidateConstraint(
            identity=_digest(
                {
                    "source": _COUPLING_SOURCE,
                    "candidate_ids": candidate_ids,
                }
            ),
            constraint_type="requires_all",
            schema_version=_CONSTRAINT_SCHEMA,
            factor_keys=(molecular_key,)
            + tuple(factor.factor_key for factor in local_factors),
            parameters=(("candidate_ids", candidate_ids),),
        ),
    )


def _local_spin_coupling_proposal(
    facts: MolecularFacts,
    factors: Tuple[CandidateFactor, ...],
    candidate_budget: CandidateBudget,
) -> Optional[CandidateProposal]:
    """Build one bounded molecular-spin factor from local-spin factors."""
    local_factors = _local_spin_factors(factors)
    if local_factors is None or not local_factors:
        return None
    molecular_key = FactorKey(
        ScientificQuantity.ELECTRONIC_STATE,
        _molecular_subject_identity(facts),
    )
    if any(factor.factor_key == molecular_key for factor in factors):
        return None
    if any(
        not value.candidate.pure_spin
        for factor in local_factors
        for value in factor.values
    ):
        return None
    limit = candidate_budget.max_materialized_values_per_factor
    retained = []
    constraints = []
    found_omitted = False
    for local_values in itertools.product(
        *(factor.values for factor in local_factors)
    ):
        contributions = tuple(value.candidate for value in local_values)
        branch_assignments = _merged_branch_assignments(local_values)
        if branch_assignments is None or not (
            _local_spin_subjects_form_partition(facts, contributions)
        ):
            continue
        for two_s in _coupled_two_s_values(
            tuple(candidate.two_s for candidate in contributions)
        ):
            candidate = _coupled_candidate(facts, local_values, two_s)
            if len(retained) == limit:
                found_omitted = True
                break
            retained.append(
                ProposedCandidateValue(candidate, branch_assignments)
            )
            constraints.append(
                _coupling_constraint(
                    molecular_key,
                    local_factors,
                    candidate,
                )
            )
        if found_omitted:
            break
    if not retained:
        return None
    source_truncated = any(
        factor.materialization_status
        is MaterializationStatus.RESOURCE_TRUNCATED
        for factor in local_factors
    )
    truncated = found_omitted or source_truncated
    continuation_identity = _digest(
        {
            "source": _COUPLING_SOURCE,
            "facts": facts.identity.chemical_identity_digest,
            "local_factors": tuple(
                factor.identity for factor in local_factors
            ),
            "retained": tuple(
                proposed.candidate.provenance.identity
                for proposed in retained
            ),
        }
    )
    proposal_id = _digest(
        {
            "source": _COUPLING_SOURCE,
            "facts": facts.identity.chemical_identity_digest,
            "candidates": tuple(
                proposed.candidate.provenance.identity
                for proposed in retained
            ),
            "truncated": truncated,
        }
    )
    return CandidateProposal(
        proposal_id=proposal_id,
        factors=(
            ProposedCandidateFactor(
                factor_key=molecular_key,
                domain_definition_id=_digest(
                    {
                        "source": _COUPLING_SOURCE,
                        "local_domains": tuple(
                            factor.domain_definition_id
                            for factor in local_factors
                        ),
                    }
                ),
                materialized_candidates=tuple(retained),
                domain_scope=max(
                    (
                        factor.domain_scope for factor in local_factors
                    ),
                    key=_SCOPE_PRECEDENCE.__getitem__,
                ),
                materialization_status=(
                    MaterializationStatus.RESOURCE_TRUNCATED
                    if truncated
                    else MaterializationStatus.COMPLETE
                ),
                unexpanded_domain_summary=(
                    CandidateDomainSummary(
                        remaining_count_upper_bound=None,
                        continuation_identity=continuation_identity,
                        frontier_bound=None,
                    )
                    if truncated
                    else None
                ),
            ),
        ),
        constraints=tuple(constraints),
    )
