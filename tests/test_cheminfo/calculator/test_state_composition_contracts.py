"""Factorized scientific-state contract tests."""

import json
from dataclasses import FrozenInstanceError, asdict

import pytest

from hotpot.cheminfo.calculator.contracts import (
    DEFAULT_CANDIDATE_BUDGET,
    ResolutionStatus,
    ScientificQuantity,
)
from hotpot.cheminfo.calculator.state_composition import (
    CandidateDomainScope,
    CandidateDomainSummary,
    CandidateSearchStatus,
    FactorKey,
    FrontierBound,
    MaterializationStatus,
    ScientificResolutionResult,
    ScientificStateCandidateSpace,
    ScientificStateIdentity,
)


def test_empty_candidate_space_is_frozen_and_serializable() -> None:
    space = ScientificStateCandidateSpace(
        factors=(),
        constraints=(),
        candidate_budget=DEFAULT_CANDIDATE_BUDGET,
        candidate_budget_artifact_id="budget-v1",
        frontier_bound_schema_version="frontier-v1",
        expanded_paths=(),
        rejected_path_summaries=(),
        frontier=(),
        search_status=CandidateSearchStatus.NOT_EVALUATED,
    )

    with pytest.raises(FrozenInstanceError):
        space.search_status = CandidateSearchStatus.DECISION_CERTIFIED
    assert json.loads(json.dumps(asdict(space), default=str))["factors"] == []


def test_domain_summary_retains_frontier_certification_evidence() -> None:
    key = FactorKey(ScientificQuantity.NET_CHARGE, "molecule")
    bound = FrontierBound("branch:1", "frontier-v1", ("0", "source"))
    summary = CandidateDomainSummary(12, "page:2", bound)

    assert key.quantity is ScientificQuantity.NET_CHARGE
    assert summary.frontier_bound == bound
    assert CandidateDomainScope.DOMAIN_BOUNDED.value == "domain_bounded"
    assert MaterializationStatus.RESOURCE_TRUNCATED.value == "resource_truncated"


def test_resolution_is_separate_from_execution_selection() -> None:
    identity = ScientificStateIdentity("state")
    resolution = ScientificResolutionResult(
        ResolutionStatus.AMBIGUOUS,
        False,
        (identity,),
    )

    assert resolution.status is ResolutionStatus.AMBIGUOUS
    assert resolution.equivalent_state_identities == (identity,)
