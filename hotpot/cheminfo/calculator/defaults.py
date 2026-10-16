"""Named scientific-state decision policies over hard-valid paths."""

from __future__ import annotations

from typing import Optional, Tuple

from .contracts import (
    CoverageStatus,
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
from .state_composition.decision_search import _DecisionSearchCompiler
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
    raise RuntimeError(
        "An active screening profile requires a versioned decision-search "
        f"compiler for operation {requirements.operation_id!r}"
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
    complete_oxidation_state_ids: frozenset[str],
) -> bool:
    resolved = {
        ScientificQuantity.NET_CHARGE,
    }
    if path.component_assignment_ids:
        resolved.add(ScientificQuantity.LEWIS_FORMAL_CHARGE)
    if path.coordination_assignment_ids:
        resolved.add(ScientificQuantity.COORDINATION_COUNTING)
    if path.oxidation_state_assignment_id in complete_oxidation_state_ids:
        resolved.add(ScientificQuantity.OXIDATION_STATE)
    if path.electronic_state_candidate_identity is not None:
        resolved.add(ScientificQuantity.ELECTRONIC_STATE)
    return requirements.required_resolved_quantities.issubset(resolved)


def _certifiability(
    candidate_space: ScientificStateCandidateSpace,
    eligible_paths: Tuple[CoupledScientificStateCandidate, ...],
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
    if len(eligible_paths) != 1:
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

    complete_os_ids = _complete_oxidation_state_ids(candidate_space)
    eligible_paths = tuple(
        path
        for path in candidate_space.expanded_paths
        if _path_resolves_requirements(path, requirements, complete_os_ids)
    )
    certifiability = _certifiability(candidate_space, eligible_paths)
    if certifiability.status is not DecisionCertificationStatus.CERTIFIED:
        return _unselected_decision(
            StateDecisionAction.NO_AUTOMATIC_STATE,
            eligibility,
            certifiability,
            "DECISION.NO_CERTIFIED_PATH",
            "No unique operation-complete scientific path is certified.",
        )

    selected = eligible_paths[0]
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
