"""Public composition of attributable scientific-state candidate domains."""

from __future__ import annotations

import hashlib
import json
from typing import Dict, Tuple

from ..charge.contracts import (
    LewisFormalChargeResult,
    NetChargeResult,
)
from ..contracts import (
    CandidateBudget,
    DEFAULT_CANDIDATE_BUDGET,
    MolecularFacts,
    MolecularFactsIdentity,
    ResolutionStatus,
    ScientificAdmissibility,
    ScientificFinding,
    ScientificQuantity,
    StateValueAuthority,
)
from ..electronic_state.contracts import (
    ElectronicStateCandidate,
    ElectronicStateInferenceResult,
)
from ..oxidation_state.contracts import (
    CoordinationElectronCountingResult,
    OxidationStateInferenceResult,
)
from .candidate_space import _merge_candidate_proposals
from .contracts import (
    CandidateConstraint,
    CandidateDomainScope,
    CandidateDomainSummary,
    CandidateProposal,
    ConstraintGuard,
    FactorKey,
    GuardedCandidateConstraint,
    MaterializationStatus,
    ProposedCandidateFactor,
    ProposedCandidateValue,
    ScientificStateCandidateSpace,
    StateCandidate,
)
from .expand import _FRONTIER_SCHEMA, _expand_scientific_paths

__all__ = ["compose_scientific_state_space"]


_BUDGET_SCHEMA = "scientific-state-candidate-budget-v1"
_RESULT_EXHAUSTIVE_AUTHORITIES = frozenset(
    {
        StateValueAuthority.VERIFIED_EXPLICIT,
        StateValueAuthority.VERIFIED_SCIENTIFIC_SOURCE,
        StateValueAuthority.EXACT_IDENTITY_TRUSTED_SOURCE,
    }
)


def _digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _molecular_subject(facts: MolecularFacts) -> str:
    return f"molecule:{facts.identity.chemical_identity_digest}"


def _is_resource_truncated(
    findings: Tuple[ScientificFinding, ...],
) -> bool:
    return any("RESOURCE_TRUNCATED" in finding.code for finding in findings)


def _result_domain_scope(
    candidates: Tuple[StateCandidate, ...],
    *,
    bounded_provider: bool,
) -> CandidateDomainScope:
    if bounded_provider:
        return CandidateDomainScope.DOMAIN_BOUNDED
    if candidates and all(
        candidate.provenance.authority in _RESULT_EXHAUSTIVE_AUTHORITIES
        for candidate in candidates
    ):
        return CandidateDomainScope.EXHAUSTIVE
    return CandidateDomainScope.DOMAIN_BOUNDED


def _result_proposal(
    result_kind: str,
    result_identity: str,
    subject_identity: str,
    quantity: ScientificQuantity,
    candidates: Tuple[StateCandidate, ...],
    findings: Tuple[ScientificFinding, ...],
    *,
    bounded_provider: bool,
) -> CandidateProposal:
    truncated = _is_resource_truncated(findings)
    summary = (
        CandidateDomainSummary(
            remaining_count_upper_bound=None,
            continuation_identity=f"result:{result_kind}:{result_identity}",
            frontier_bound=None,
        )
        if truncated
        else None
    )
    return CandidateProposal(
        proposal_id=f"result:{result_kind}:{result_identity}",
        factors=(
            ProposedCandidateFactor(
                factor_key=FactorKey(quantity, subject_identity),
                domain_definition_id=(
                    f"{result_kind}-result-domain:{result_identity}"
                ),
                materialized_candidates=tuple(
                    ProposedCandidateValue(candidate)
                    for candidate in candidates
                ),
                domain_scope=_result_domain_scope(
                    candidates,
                    bounded_provider=bounded_provider,
                ),
                materialization_status=(
                    MaterializationStatus.RESOURCE_TRUNCATED
                    if truncated
                    else MaterializationStatus.COMPLETE
                ),
                unexpanded_domain_summary=summary,
            ),
        ),
        constraints=(),
    )


def _eligible_result(
    facts: MolecularFacts,
    facts_identity: MolecularFactsIdentity,
    admissibility: ScientificAdmissibility,
) -> bool:
    return (
        facts_identity == facts.identity
        and admissibility is ScientificAdmissibility.ADMISSIBLE
    )


def _append_result_mapping(
    mapping: Dict[str, set[str]],
    result_identity: str,
    candidates: Tuple[StateCandidate, ...],
) -> None:
    for candidate in candidates:
        mapping.setdefault(candidate.provenance.identity, set()).add(
            result_identity
        )


def _proposal_result_mappings(
    proposals: Tuple[CandidateProposal, ...],
    charge_mapping: Dict[str, set[str]],
    electronic_mapping: Dict[str, set[str]],
) -> None:
    for proposal in proposals:
        for factor in proposal.factors:
            for proposed in factor.materialized_candidates:
                candidate = proposed.candidate
                if factor.factor_key.quantity is ScientificQuantity.NET_CHARGE:
                    charge_mapping.setdefault(
                        candidate.provenance.identity,
                        set(),
                    ).add(proposal.proposal_id)
                elif (
                    factor.factor_key.quantity
                    is ScientificQuantity.ELECTRONIC_STATE
                    and isinstance(candidate, ElectronicStateCandidate)
                ):
                    electronic_mapping.setdefault(
                        candidate.provenance.identity,
                        set(),
                    ).add(proposal.proposal_id)


def _budget_artifact_id(candidate_budget: CandidateBudget) -> str:
    return _digest(
        {
            "schema": _BUDGET_SCHEMA,
            "max_materialized_values_per_factor": (
                candidate_budget.max_materialized_values_per_factor
            ),
            "max_expanded_paths": candidate_budget.max_expanded_paths,
            "max_frontier_paths": candidate_budget.max_frontier_paths,
        }
    )


def compose_scientific_state_space(
    facts: MolecularFacts,
    *,
    net_charge_results: Tuple[NetChargeResult, ...],
    lewis_results: Tuple[LewisFormalChargeResult, ...] = (),
    coordination_counting_results: Tuple[
        CoordinationElectronCountingResult,
        ...,
    ] = (),
    oxidation_state_results: Tuple[OxidationStateInferenceResult, ...] = (),
    electronic_state_results: Tuple[ElectronicStateInferenceResult, ...] = (),
    candidate_proposals: Tuple[CandidateProposal, ...] = (),
    constraints: Tuple[CandidateConstraint, ...] = (),
    candidate_budget: CandidateBudget = DEFAULT_CANDIDATE_BUDGET,
) -> ScientificStateCandidateSpace:
    """Compose hard-valid result envelopes into a factorized state space."""
    subject = _molecular_subject(facts)
    proposals = list(candidate_proposals)
    charge_result_ids: Dict[str, set[str]] = {}
    electronic_result_ids: Dict[str, set[str]] = {}

    for result in net_charge_results:
        if not _eligible_result(
            facts,
            result.facts_identity,
            result.admissibility,
        ) or not result.candidates:
            continue
        candidates = tuple(result.candidates)
        proposals.append(
            _result_proposal(
                "net-charge",
                result.identity,
                subject,
                ScientificQuantity.NET_CHARGE,
                candidates,
                result.findings,
                bounded_provider=False,
            )
        )
        _append_result_mapping(
            charge_result_ids,
            result.identity,
            candidates,
        )

    result_groups = (
        (
            "lewis",
            ScientificQuantity.LEWIS_FORMAL_CHARGE,
            lewis_results,
        ),
        (
            "coordination-counting",
            ScientificQuantity.COORDINATION_COUNTING,
            coordination_counting_results,
        ),
        (
            "oxidation-state",
            ScientificQuantity.OXIDATION_STATE,
            oxidation_state_results,
        ),
    )
    for result_kind, quantity, results in result_groups:
        for result in results:
            if not _eligible_result(
                facts,
                result.facts_identity,
                result.admissibility,
            ) or not result.candidates:
                continue
            proposals.append(
                _result_proposal(
                    result_kind,
                    result.identity,
                    subject,
                    quantity,
                    tuple(result.candidates),
                    result.findings,
                    bounded_provider=True,
                )
            )

    unresolved_electronic_result_ids = []
    for result in electronic_state_results:
        if result.facts_identity != facts.identity:
            continue
        if (
            result.admissibility is ScientificAdmissibility.ADMISSIBLE
            and result.candidates
        ):
            candidates = tuple(result.candidates)
            proposals.append(
                _result_proposal(
                    "electronic-state",
                    result.identity,
                    subject,
                    ScientificQuantity.ELECTRONIC_STATE,
                    candidates,
                    result.findings,
                    bounded_provider=False,
                )
            )
            _append_result_mapping(
                electronic_result_ids,
                result.identity,
                candidates,
            )
        elif (
            result.resolution is ResolutionStatus.UNRESOLVED
            and not result.candidates
        ):
            unresolved_electronic_result_ids.append(result.identity)

    _proposal_result_mappings(
        candidate_proposals,
        charge_result_ids,
        electronic_result_ids,
    )
    factors, merged_constraints, guarded_constraints = (
        _merge_candidate_proposals(tuple(proposals), candidate_budget)
        if proposals
        else ((), (), ())
    )
    direct_guarded_constraints = tuple(
        GuardedCandidateConstraint(ConstraintGuard(), constraint)
        for constraint in constraints
    )
    all_guarded_constraints = tuple(
        sorted(
            guarded_constraints + direct_guarded_constraints,
            key=lambda guarded: guarded.constraint.identity,
        )
    )
    constraints_by_identity = {}
    for constraint in merged_constraints + constraints:
        incumbent = constraints_by_identity.get(constraint.identity)
        if incumbent is not None and incumbent != constraint:
            raise ValueError(
                "One constraint identity cannot describe different constraints"
            )
        constraints_by_identity[constraint.identity] = constraint
    paths, rejections, frontier, search_status = _expand_scientific_paths(
        facts,
        factors,
        all_guarded_constraints,
        {
            identity: tuple(sorted(result_ids))
            for identity, result_ids in charge_result_ids.items()
        },
        {
            identity: tuple(sorted(result_ids))
            for identity, result_ids in electronic_result_ids.items()
        },
        tuple(sorted(set(unresolved_electronic_result_ids))),
        max_expanded_paths=candidate_budget.max_expanded_paths,
        max_frontier_paths=candidate_budget.max_frontier_paths,
    )
    return ScientificStateCandidateSpace(
        factors=factors,
        constraints=tuple(
            constraints_by_identity[identity]
            for identity in sorted(constraints_by_identity)
        ),
        candidate_budget=candidate_budget,
        candidate_budget_artifact_id=_budget_artifact_id(candidate_budget),
        frontier_bound_schema_version=_FRONTIER_SCHEMA,
        expanded_paths=paths,
        rejected_path_summaries=rejections,
        frontier=frontier,
        search_status=search_status,
    )
