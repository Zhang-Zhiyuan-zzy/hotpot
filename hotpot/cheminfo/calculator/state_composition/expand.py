"""Budgeted expansion and hard validation of coupled scientific paths."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from typing import Dict, Optional, Tuple

from ..charge.contracts import (
    LewisFormalChargeCandidate,
    NetChargeCandidateEvidence,
)
from ..contracts import (
    MolecularFacts,
    ScientificAdmissibility,
    ScientificQuantity,
)
from ..electronic_state.contracts import ElectronicStateCandidate
from ..oxidation_state.contracts import (
    CoordinationCountingCandidate,
    OxidationStateCandidateEvidence,
)
from .constraints import _validate_coupled_candidates
from .contracts import (
    BranchAssignment,
    CandidateConstraint,
    CandidateDomainScope,
    CandidateFactor,
    CandidateSearchStatus,
    CoupledScientificStateCandidate,
    FrontierBound,
    GuardedCandidateConstraint,
    MaterializationStatus,
    RejectedPathSummary,
    ScientificStateIdentity,
    StateValueOrigin,
    StateValueRef,
)

__all__ = []


_HARD_CONSTRAINT_ID = "hard-scientific-consistency-v1"
_FRONTIER_SCHEMA = "scientific-state-frontier-v1"


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _candidate_collections(
    factors: Tuple[CandidateFactor, ...],
) -> Tuple[
    Tuple[NetChargeCandidateEvidence, ...],
    Tuple[LewisFormalChargeCandidate, ...],
    Tuple[CoordinationCountingCandidate, ...],
    Tuple[OxidationStateCandidateEvidence, ...],
    Tuple[ElectronicStateCandidate, ...],
    Dict[str, Tuple[BranchAssignment, ...]],
]:
    charges = []
    lewis_candidates = []
    counting_candidates = []
    oxidation_candidates = []
    electronic_candidates = []
    branches = {}
    for factor in factors:
        for value in factor.values:
            candidate = value.candidate
            branches[candidate.provenance.identity] = value.branch_assignments
            if isinstance(candidate, NetChargeCandidateEvidence):
                charges.append(candidate)
            elif isinstance(candidate, LewisFormalChargeCandidate):
                lewis_candidates.append(candidate)
            elif isinstance(candidate, CoordinationCountingCandidate):
                counting_candidates.append(candidate)
            elif isinstance(candidate, OxidationStateCandidateEvidence):
                if all(state is not None for state in candidate.atom_states):
                    oxidation_candidates.append(candidate)
            elif isinstance(candidate, ElectronicStateCandidate):
                electronic_candidates.append(candidate)
    return (
        tuple(charges),
        tuple(lewis_candidates),
        tuple(counting_candidates),
        tuple(oxidation_candidates),
        tuple(electronic_candidates),
        branches,
    )


def _selected_candidate_ids(
    charge: NetChargeCandidateEvidence,
    lewis: Optional[LewisFormalChargeCandidate],
    counting: Optional[CoordinationCountingCandidate],
    oxidation: Optional[OxidationStateCandidateEvidence],
    electronic: Optional[ElectronicStateCandidate],
) -> Tuple[str, ...]:
    return tuple(
        candidate.provenance.identity
        for candidate in (charge, lewis, counting, oxidation, electronic)
        if candidate is not None
    )


def _branch_conflict(
    selected_ids: Tuple[str, ...],
    branches: Dict[str, Tuple[BranchAssignment, ...]],
) -> bool:
    selected_options: Dict[str, str] = {}
    for candidate_id in selected_ids:
        for assignment in branches.get(candidate_id, ()):
            incumbent = selected_options.get(assignment.group_id)
            if incumbent is not None and incumbent != assignment.option_id:
                return True
            selected_options[assignment.group_id] = assignment.option_id
    return False


def _guard_matches(
    guarded: GuardedCandidateConstraint,
    selected_ids: Tuple[str, ...],
    branches: Dict[str, Tuple[BranchAssignment, ...]],
) -> bool:
    selected_id_set = set(selected_ids)
    if not set(guarded.guard.all_candidate_ids).issubset(selected_id_set):
        return False
    selected_assignments = {
        assignment
        for candidate_id in selected_ids
        for assignment in branches.get(candidate_id, ())
    }
    return set(guarded.guard.branch_assignments).issubset(selected_assignments)


def _constraint_rejection(
    constraint: CandidateConstraint,
    selected_ids: Tuple[str, ...],
) -> Optional[str]:
    parameters = dict(constraint.parameters)
    candidate_ids = parameters.get("candidate_ids", ())
    selected = set(selected_ids)
    required = set(candidate_ids) if isinstance(candidate_ids, tuple) else set()
    if constraint.constraint_type == "requires_all":
        return None if required.issubset(selected) else "CONSTRAINT.REQUIRES_ALL"
    if constraint.constraint_type == "excludes_all":
        return "CONSTRAINT.EXCLUDES_ALL" if required.issubset(selected) else None
    return "CONSTRAINT.UNSUPPORTED_TYPE"


def _active_constraint_rejection(
    guarded_constraints: Tuple[GuardedCandidateConstraint, ...],
    selected_ids: Tuple[str, ...],
    branches: Dict[str, Tuple[BranchAssignment, ...]],
) -> Optional[Tuple[str, str]]:
    for guarded in guarded_constraints:
        if not _guard_matches(guarded, selected_ids, branches):
            continue
        reason = _constraint_rejection(guarded.constraint, selected_ids)
        if reason is not None:
            return guarded.constraint.identity, reason
    return None


def _path_identity(
    facts: MolecularFacts,
    selected_ids: Tuple[str, ...],
    electronic_result_id: str,
) -> ScientificStateIdentity:
    return ScientificStateIdentity(
        _digest(
            {
                "facts": facts.identity.chemical_identity_digest,
                "candidate_ids": selected_ids,
                "electronic_result_id": electronic_result_id,
            }
        )
    )


def _coupled_path(
    facts: MolecularFacts,
    charge: NetChargeCandidateEvidence,
    lewis: Optional[LewisFormalChargeCandidate],
    counting: Optional[CoordinationCountingCandidate],
    oxidation: Optional[OxidationStateCandidateEvidence],
    electronic: Optional[ElectronicStateCandidate],
    electronic_result_id: str,
) -> CoupledScientificStateCandidate:
    selected = tuple(
        candidate
        for candidate in (charge, lewis, counting, oxidation, electronic)
        if candidate is not None
    )
    selected_ids = tuple(
        candidate.provenance.identity for candidate in selected
    )
    return CoupledScientificStateCandidate(
        identity=_path_identity(facts, selected_ids, electronic_result_id),
        net_charge=StateValueRef(
            ScientificQuantity.NET_CHARGE,
            charge.charge,
            StateValueOrigin.RESULT,
            charge.provenance.identity,
        ),
        component_assignment_ids=(
            (lewis.provenance.identity,) if lewis is not None else ()
        ),
        coordination_assignment_ids=(
            (counting.provenance.identity,) if counting is not None else ()
        ),
        oxidation_state_assignment_id=(
            oxidation.provenance.identity if oxidation is not None else None
        ),
        electronic_state_result_id=electronic_result_id,
        electronic_state_candidate_identity=(
            electronic.provenance.identity if electronic is not None else None
        ),
        derived_values=(),
        evidence_ids=selected_ids,
        assumption_ids=tuple(
            sorted(
                {
                    assumption
                    for candidate in selected
                    for assumption in candidate.provenance.assumption_ids
                }
            )
        ),
    )


def _expand_scientific_paths(
    facts: MolecularFacts,
    factors: Tuple[CandidateFactor, ...],
    guarded_constraints: Tuple[GuardedCandidateConstraint, ...],
    electronic_result_by_candidate: Dict[str, str],
    unresolved_electronic_result_ids: Tuple[str, ...],
    *,
    max_expanded_paths: int,
    max_frontier_paths: int,
) -> Tuple[
    Tuple[CoupledScientificStateCandidate, ...],
    Tuple[RejectedPathSummary, ...],
    Tuple[FrontierBound, ...],
    CandidateSearchStatus,
]:
    (
        charges,
        lewis_candidates,
        counting_candidates,
        oxidation_candidates,
        electronic_candidates,
        branches,
    ) = _candidate_collections(factors)
    if not charges:
        return (), (), (), CandidateSearchStatus.NOT_EVALUATED

    lewis_domain = lewis_candidates or (None,)
    counting_domain = counting_candidates or (None,)
    oxidation_domain = oxidation_candidates or (None,)
    if electronic_candidates:
        electronic_domain = tuple(
            (
                candidate,
                electronic_result_by_candidate[candidate.provenance.identity],
            )
            for candidate in electronic_candidates
            if candidate.provenance.identity in electronic_result_by_candidate
        )
    else:
        electronic_domain = tuple(
            (None, result_id) for result_id in unresolved_electronic_result_ids
        )
    if not electronic_domain:
        return (), (), (), CandidateSearchStatus.NOT_EVALUATED

    domains = (
        charges,
        lewis_domain,
        counting_domain,
        oxidation_domain,
        electronic_domain,
    )
    total_combinations = math.prod(len(domain) for domain in domains)
    paths: Dict[str, CoupledScientificStateCandidate] = {}
    rejections = []
    attempted = 0
    for charge, lewis, counting, oxidation, electronic_entry in itertools.product(
        *domains
    ):
        if attempted >= max_expanded_paths:
            break
        attempted += 1
        electronic, electronic_result_id = electronic_entry
        selected_ids = _selected_candidate_ids(
            charge,
            lewis,
            counting,
            oxidation,
            electronic,
        )
        if _branch_conflict(selected_ids, branches):
            rejections.append(
                RejectedPathSummary(
                    selected_ids,
                    "branch-assignment-v1",
                    ("COMPOSITION.BRANCH_CONFLICT",),
                )
            )
            continue
        constraint_rejection = _active_constraint_rejection(
            guarded_constraints,
            selected_ids,
            branches,
        )
        if constraint_rejection is not None:
            constraint_identity, reason = constraint_rejection
            rejections.append(
                RejectedPathSummary(
                    selected_ids,
                    constraint_identity,
                    (reason,),
                )
            )
            continue
        assessment = _validate_coupled_candidates(
            facts,
            charge,
            lewis=lewis,
            counting=counting,
            oxidation_state=oxidation,
            electronic_state=electronic,
        )
        if assessment.admissibility is not ScientificAdmissibility.ADMISSIBLE:
            rejections.append(
                RejectedPathSummary(
                    selected_ids,
                    _HARD_CONSTRAINT_ID,
                    tuple(finding.code for finding in assessment.findings),
                )
            )
            continue
        path = _coupled_path(
            facts,
            charge,
            lewis,
            counting,
            oxidation,
            electronic,
            electronic_result_id,
        )
        paths[path.identity.digest] = path

    truncated = attempted < total_combinations or any(
        factor.materialization_status
        is MaterializationStatus.RESOURCE_TRUNCATED
        for factor in factors
    )
    frontier = (
        (
            FrontierBound(
                branch_identity=_digest(
                    {
                        "facts": facts.identity.chemical_identity_digest,
                        "attempted": attempted,
                        "total": total_combinations,
                    }
                ),
                schema_version=_FRONTIER_SCHEMA,
                optimistic_key=(),
            ),
        )
        if truncated and max_frontier_paths > 0
        else ()
    )
    ordered_paths = tuple(paths[identity] for identity in sorted(paths))
    factors_are_exhaustive = all(
        factor.domain_scope is CandidateDomainScope.EXHAUSTIVE
        and factor.materialization_status is MaterializationStatus.COMPLETE
        for factor in factors
    )
    if truncated:
        status = CandidateSearchStatus.RESOURCE_LIMIT_NO_CERTIFIED_DECISION
    elif len(ordered_paths) == 1 and factors_are_exhaustive:
        status = CandidateSearchStatus.DECISION_CERTIFIED
    else:
        status = CandidateSearchStatus.NO_CERTIFIED_DECISION
    return ordered_paths, tuple(rejections), frontier, status
