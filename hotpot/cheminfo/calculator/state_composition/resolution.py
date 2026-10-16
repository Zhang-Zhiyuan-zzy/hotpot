"""Scientific resolution derived from an immutable candidate space."""

from __future__ import annotations

from ..contracts import ResolutionStatus
from .contracts import (
    CandidateDomainScope,
    CandidateSearchStatus,
    MaterializationStatus,
    ScientificResolutionResult,
    ScientificStateCandidateSpace,
)

__all__ = []


def _resolve_candidate_space(
    candidate_space: ScientificStateCandidateSpace,
) -> ScientificResolutionResult:
    identities = tuple(
        path.identity for path in candidate_space.expanded_paths
    )
    exhaustive = (
        bool(candidate_space.factors)
        and candidate_space.search_status
        is not CandidateSearchStatus.NOT_EVALUATED
        and candidate_space.search_status
        is not CandidateSearchStatus.RESOURCE_LIMIT_NO_CERTIFIED_DECISION
        and not candidate_space.frontier
        and all(
            factor.domain_scope is CandidateDomainScope.EXHAUSTIVE
            and factor.materialization_status is MaterializationStatus.COMPLETE
            for factor in candidate_space.factors
        )
    )
    unresolved_electronic_state = bool(identities) and all(
        path.electronic_state_candidate_identity is None
        for path in candidate_space.expanded_paths
    )
    one_resolved_path = (
        len(identities) == 1
        and candidate_space.expanded_paths[
            0
        ].electronic_state_candidate_identity
        is not None
    )
    if not identities or unresolved_electronic_state:
        status = ResolutionStatus.UNRESOLVED
    elif one_resolved_path and exhaustive:
        status = ResolutionStatus.UNIQUE
    else:
        status = ResolutionStatus.AMBIGUOUS
    return ScientificResolutionResult(status, exhaustive, identities)
