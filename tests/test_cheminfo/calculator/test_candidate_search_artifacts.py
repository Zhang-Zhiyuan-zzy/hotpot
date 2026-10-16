"""Versioned candidate-search artifact identity and integrity tests."""

from dataclasses import replace

import pytest

from hotpot.cheminfo.calculator.contracts import CandidateBudget
from hotpot.cheminfo.calculator.state_composition._artifacts import (
    _FRONTIER_SCHEMA,
    _candidate_budget_artifact_id,
    _validate_candidate_space_artifacts,
)
from hotpot.cheminfo.calculator.state_composition.contracts import (
    CandidateSearchStatus,
    FrontierBound,
    ScientificStateCandidateSpace,
)


def _space(candidate_budget: CandidateBudget) -> ScientificStateCandidateSpace:
    return ScientificStateCandidateSpace(
        factors=(),
        constraints=(),
        candidate_budget=candidate_budget,
        candidate_budget_artifact_id=_candidate_budget_artifact_id(
            candidate_budget
        ),
        frontier_bound_schema_version=_FRONTIER_SCHEMA,
        expanded_paths=(),
        rejected_path_summaries=(),
        frontier=(),
        search_status=CandidateSearchStatus.NOT_EVALUATED,
    )


def test_candidate_budget_identity_covers_every_limit() -> None:
    baseline = CandidateBudget(8, 16, 4)
    identities = {
        _candidate_budget_artifact_id(baseline),
        _candidate_budget_artifact_id(CandidateBudget(9, 16, 4)),
        _candidate_budget_artifact_id(CandidateBudget(8, 17, 4)),
        _candidate_budget_artifact_id(CandidateBudget(8, 16, 5)),
    }

    assert len(identities) == 4
    assert _candidate_budget_artifact_id(baseline) == (
        _candidate_budget_artifact_id(baseline)
    )


@pytest.mark.parametrize(
    "limits",
    (
        (0, 1, 1),
        (1, -1, 1),
        (1, 1, 0),
        (True, 1, 1),
        (1.0, 1, 1),
    ),
)
def test_candidate_budget_requires_positive_integer_limits(
    limits: tuple[object, object, object],
) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        CandidateBudget(*limits)


def test_candidate_space_rejects_mismatched_budget_identity() -> None:
    space = _space(CandidateBudget(8, 16, 4))

    with pytest.raises(ValueError, match="budget artifact"):
        _validate_candidate_space_artifacts(
            replace(space, candidate_budget_artifact_id="wrong")
        )


def test_candidate_space_rejects_mismatched_frontier_schema() -> None:
    space = replace(
        _space(CandidateBudget(8, 16, 4)),
        frontier=(
            FrontierBound("branch", "unsupported-frontier-v0", ("key",)),
        ),
    )

    with pytest.raises(ValueError, match="Nested frontier schema"):
        _validate_candidate_space_artifacts(space)


def test_candidate_space_accepts_matching_artifacts() -> None:
    _validate_candidate_space_artifacts(
        _space(CandidateBudget(8, 16, 4))
    )
