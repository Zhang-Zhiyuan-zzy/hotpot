"""Named scientific-state decision policies over hard-valid paths."""

from __future__ import annotations

from typing import Optional, Tuple

from .contracts import (
    CoverageStatus,
    MolecularFacts,
    ResolutionStatus,
    ScientificAdmissibility,
    ScientificOperationRequirements,
    ScientificQuantity,
    StatePolicy,
)
from .state_composition import (
    CandidateDomainScope,
    CandidateSearchStatus,
    CoupledScientificStateCandidate,
    ScientificStateCandidateSpace,
)
from .state_composition.decision_search import (
    _DecisionSearchCompiler,
    _select_decision_record,
)
from .oxidation_state import OxidationStateCandidateEvidence
from .state_selection import (
    DecisionCertifiability,
    DecisionCertificationStatus,
    DecisionReason,
    ExecutionStateDecision,
    StateDecisionAction,
    StateDecisionSource,
    StaticProfileEligibility,
)
from .state_rules import RankAdjustment
from .state_rules.profile_decision import (
    _CoordinationScreeningDecisionCompiler,
)
from .state_rules.profiles import _coordination_screening_profile

__all__ = []


def _decision_search_compiler(
    policy: StatePolicy,
    requirements: ScientificOperationRequirements,
    eligibility: StaticProfileEligibility,
    rank_adjustments: Tuple[RankAdjustment, ...],
) -> Optional[_DecisionSearchCompiler]:
    """Return the active profile compiler without inventing inactive policy."""
    if policy is StatePolicy.STRICT:
        if rank_adjustments:
            raise ValueError("STRICT policy cannot consume rule rank adjustments")
        return None
    if not eligibility.in_domain:
        return None
    return _CoordinationScreeningDecisionCompiler(
        _coordination_screening_profile(),
        requirements,
        rank_adjustments,
    )


def _terminal_action(
    result_states: Tuple[
        Tuple[ScientificAdmissibility, ResolutionStatus],
        ...,
    ],
) -> Optional[StateDecisionAction]:
    if any(
        resolution is ResolutionStatus.CONFLICT
        for _, resolution in result_states
    ):
        return StateDecisionAction.CONFLICT
    if any(
        admissibility is ScientificAdmissibility.INADMISSIBLE
        for admissibility, _ in result_states
    ):
        return StateDecisionAction.INADMISSIBLE
    return None


def _complete_oxidation_state_ids(
    candidate_space: ScientificStateCandidateSpace,
) -> frozenset[str]:
    return frozenset(
        candidate.provenance.identity
        for factor in candidate_space.factors
        for value in factor.values
        for candidate in (value.candidate,)
        if factor.factor_key.quantity is ScientificQuantity.OXIDATION_STATE
        and isinstance(candidate, OxidationStateCandidateEvidence)
        and candidate.coverage is CoverageStatus.COMPLETE
        and all(
            state is not None
            for state in candidate.atom_states
        )
    )


def _path_resolves_requirements(
    path: CoupledScientificStateCandidate,
    requirements: ScientificOperationRequirements,
    policy: StatePolicy,
    candidate_space: ScientificStateCandidateSpace,
    facts: Optional[MolecularFacts],
) -> bool:
    resolved = {
        ScientificQuantity.NET_CHARGE,
    }
    if path.component_assignment_ids:
        resolved.add(ScientificQuantity.LEWIS_FORMAL_CHARGE)
    if path.coordination_assignment_ids:
        resolved.add(ScientificQuantity.COORDINATION_COUNTING)
    if _path_resolves_oxidation_state(
        path,
        policy,
        candidate_space,
        facts,
    ):
        resolved.add(ScientificQuantity.OXIDATION_STATE)
    if path.electronic_state_candidate_identity is not None:
        resolved.add(ScientificQuantity.ELECTRONIC_STATE)
    return requirements.required_resolved_quantities.issubset(resolved)


def _selected_oxidation_candidates(
    path: CoupledScientificStateCandidate,
    candidate_space: ScientificStateCandidateSpace,
) -> Tuple[OxidationStateCandidateEvidence, ...]:
    selected_ids = set(path.evidence_ids)
    return tuple(
        candidate
        for factor in candidate_space.factors
        for value in factor.values
        for candidate in (value.candidate,)
        if isinstance(candidate, OxidationStateCandidateEvidence)
        and candidate.provenance.identity in selected_ids
    )


def _screening_coordination_centers(
    facts: MolecularFacts,
) -> frozenset[int]:
    supported_atomic_numbers = (
        _coordination_screening_profile().coordination_center_atomic_numbers
    )
    coordinated_atoms = frozenset(
        atom_index
        for bond in facts.semantic_bonds
        if bond.kind == "coordination"
        for atom_index in bond.atom_indices
    )
    return frozenset(
        atom_index
        for atom_index in coordinated_atoms
        if facts.atomic_numbers[atom_index] in supported_atomic_numbers
    )


def _oxidation_states_cover_atoms(
    candidates: Tuple[OxidationStateCandidateEvidence, ...],
    atom_indices: frozenset[int],
) -> bool:
    return bool(atom_indices) and all(
        any(
            atom_index < len(candidate.atom_states)
            and candidate.atom_states[atom_index] is not None
            for candidate in candidates
        )
        for atom_index in atom_indices
    )


def _path_resolves_oxidation_state(
    path: CoupledScientificStateCandidate,
    policy: StatePolicy,
    candidate_space: ScientificStateCandidateSpace,
    facts: Optional[MolecularFacts],
) -> bool:
    selected = _selected_oxidation_candidates(path, candidate_space)
    if policy is not StatePolicy.COORDINATION_SCREENING_V1:
        if facts is not None and selected:
            has_implicit_hydrogen = any(
                explicit_count == 0 and implicit_count not in (None, 0)
                for explicit_count, implicit_count in zip(
                    facts.hydrogen_inventory.explicit_counts,
                    facts.hydrogen_inventory.implicit_counts,
                )
            )
            return (
                any(
                    candidate.coverage is CoverageStatus.COMPLETE
                    and all(
                        state is not None
                        for state in candidate.atom_states
                    )
                    for candidate in selected
                )
                or (
                    not has_implicit_hydrogen
                    and _oxidation_states_cover_atoms(
                        selected,
                        frozenset(range(len(facts.atomic_numbers))),
                    )
                )
            )
        complete_ids = _complete_oxidation_state_ids(candidate_space)
        return path.oxidation_state_assignment_id in complete_ids
    if facts is None:
        return False
    return _oxidation_states_cover_atoms(
        selected,
        _screening_coordination_centers(facts),
    )


def _certifiability(
    candidate_space: ScientificStateCandidateSpace,
    eligible_paths: Tuple[CoupledScientificStateCandidate, ...],
    selected_path: Optional[CoupledScientificStateCandidate],
) -> DecisionCertifiability:
    if candidate_space.search_status is CandidateSearchStatus.NOT_EVALUATED:
        return DecisionCertifiability(
            DecisionCertificationStatus.NOT_EVALUATED,
            (
                DecisionReason(
                    "DECISION.SEARCH_NOT_EVALUATED",
                    "No coupled candidate path was available for decision.",
                ),
            ),
        )
    if (
        candidate_space.search_status
        is CandidateSearchStatus.RESOURCE_LIMIT_NO_CERTIFIED_DECISION
    ):
        return DecisionCertifiability(
            DecisionCertificationStatus.RESOURCE_LIMIT_UNCERTIFIED,
            (
                DecisionReason(
                    "DECISION.RESOURCE_LIMIT",
                    "Candidate search ended without a certified decision.",
                ),
            ),
        )
    if any(
        factor.domain_scope is CandidateDomainScope.OPEN
        for factor in candidate_space.factors
    ):
        return DecisionCertifiability(
            DecisionCertificationStatus.OPEN_OR_UNBOUNDABLE_DOMAIN,
            (
                DecisionReason(
                    "DECISION.OPEN_DOMAIN",
                    "An operation-relevant candidate factor remains open.",
                ),
            ),
        )
    if (
        candidate_space.search_status
        is CandidateSearchStatus.NO_CERTIFIED_DECISION
        or selected_path is None
    ):
        return DecisionCertifiability(
            DecisionCertificationStatus.NON_EQUIVALENT_TIE,
            (
                DecisionReason(
                    "DECISION.PATH_COUNT",
                    "The policy cannot certify exactly one eligible path.",
                    tuple(path.identity.digest for path in eligible_paths),
                ),
            ),
        )
    return DecisionCertifiability(
        DecisionCertificationStatus.CERTIFIED,
        (),
    )


def _certified_candidate_path(
    candidate_space: ScientificStateCandidateSpace,
    eligible_paths: Tuple[CoupledScientificStateCandidate, ...],
    decision_search: Optional[_DecisionSearchCompiler],
) -> Optional[CoupledScientificStateCandidate]:
    if candidate_space.search_status not in (
        CandidateSearchStatus.DECISION_CERTIFIED,
        CandidateSearchStatus.TRUNCATED_BUT_DECISION_CERTIFIED,
    ):
        return None
    if decision_search is None:
        return eligible_paths[0] if len(eligible_paths) == 1 else None
    record = _select_decision_record(
        eligible_paths,
        candidate_space.factors,
        decision_search,
    )
    if record is None:
        return None
    return next(
        (
            path
            for path in eligible_paths
            if path.identity == record.state_identity
        ),
        None,
    )


def _unselected_decision(
    action: StateDecisionAction,
    eligibility: StaticProfileEligibility,
    certifiability: DecisionCertifiability,
    code: str,
    message: str,
) -> ExecutionStateDecision:
    return ExecutionStateDecision(
        selected_state_identity=None,
        action=action,
        source=StateDecisionSource.NOT_SELECTED,
        static_profile_eligibility=eligibility,
        decision_certifiability=certifiability,
        rationale=(DecisionReason(code, message),),
    )


def _decide_scientific_state(
    policy: StatePolicy,
    requirements: ScientificOperationRequirements,
    candidate_space: ScientificStateCandidateSpace,
    eligibility: StaticProfileEligibility,
    result_states: Tuple[
        Tuple[ScientificAdmissibility, ResolutionStatus],
        ...,
    ],
    explicit_candidate_ids: frozenset[str],
    decision_search: Optional[_DecisionSearchCompiler] = None,
    facts: Optional[MolecularFacts] = None,
) -> ExecutionStateDecision:
    terminal_action = _terminal_action(result_states)
    if terminal_action is not None:
        return _unselected_decision(
            terminal_action,
            eligibility,
            DecisionCertifiability(
                DecisionCertificationStatus.NOT_EVALUATED,
                (),
            ),
            f"DECISION.{terminal_action.value.upper()}",
            "Authoritative scientific input failed hard validation.",
        )
    if not requirements.required_resolved_quantities:
        return _unselected_decision(
            StateDecisionAction.NOT_REQUESTED,
            eligibility,
            DecisionCertifiability(
                DecisionCertificationStatus.NOT_EVALUATED,
                (),
            ),
            "DECISION.NOT_REQUESTED",
            "The effective operation requests no resolved execution state.",
        )
    if not eligibility.in_domain:
        return _unselected_decision(
            StateDecisionAction.NO_AUTOMATIC_STATE,
            eligibility,
            DecisionCertifiability(
                DecisionCertificationStatus.NOT_EVALUATED,
                (),
            ),
            "DECISION.PROFILE_OUT_OF_DOMAIN",
            "The effective profile is outside its static applicability domain.",
        )

    eligible_paths = tuple(
        path
        for path in candidate_space.expanded_paths
        if _path_resolves_requirements(
            path,
            requirements,
            policy,
            candidate_space,
            facts,
        )
    )
    selected = _certified_candidate_path(
        candidate_space,
        eligible_paths,
        decision_search,
    )
    certifiability = _certifiability(
        candidate_space,
        eligible_paths,
        selected,
    )
    if certifiability.status is not DecisionCertificationStatus.CERTIFIED:
        return _unselected_decision(
            StateDecisionAction.NO_AUTOMATIC_STATE,
            eligibility,
            certifiability,
            "DECISION.NO_CERTIFIED_PATH",
            "No unique operation-complete scientific path is certified.",
        )

    if selected is None:
        raise RuntimeError("Certified decision is missing its selected path")
    if policy is StatePolicy.COORDINATION_SCREENING_V1:
        source = StateDecisionSource.EXECUTION_DEFAULT
    elif explicit_candidate_ids.intersection(selected.evidence_ids):
        source = StateDecisionSource.EXPLICIT
    else:
        source = StateDecisionSource.SCIENTIFIC_SOURCE
    return ExecutionStateDecision(
        selected_state_identity=selected.identity,
        action=StateDecisionAction.SELECT_STATE,
        source=source,
        static_profile_eligibility=eligibility,
        decision_certifiability=certifiability,
        rationale=(
            DecisionReason(
                "DECISION.CERTIFIED_PATH",
                "The named policy certified one operation-complete path.",
                (selected.identity.digest,),
            ),
        ),
    )
