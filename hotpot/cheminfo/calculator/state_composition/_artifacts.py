"""Versioned identities and integrity checks for candidate-search artifacts."""

from __future__ import annotations

import hashlib
import json
from typing import Optional

from ..contracts import CandidateBudget
from .contracts import (
    CandidateDomainSummary,
    ScientificStateCandidateSpace,
)

__all__ = []


_BUDGET_SCHEMA = "scientific-state-candidate-budget-v1"
_FRONTIER_SCHEMA = "scientific-state-frontier-v1"


def _candidate_budget_artifact_id(candidate_budget: CandidateBudget) -> str:
    payload = {
        "schema": _BUDGET_SCHEMA,
        "max_materialized_values_per_factor": (
            candidate_budget.max_materialized_values_per_factor
        ),
        "max_expanded_paths": candidate_budget.max_expanded_paths,
        "max_frontier_paths": candidate_budget.max_frontier_paths,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _summary_frontier_schema(
    summary: CandidateDomainSummary,
) -> Optional[str]:
    if summary.frontier_bound is None:
        return None
    return summary.frontier_bound.schema_version


def _validate_candidate_space_artifacts(
    candidate_space: ScientificStateCandidateSpace,
) -> None:
    expected_budget_id = _candidate_budget_artifact_id(
        candidate_space.candidate_budget
    )
    if candidate_space.candidate_budget_artifact_id != expected_budget_id:
        raise ValueError("Candidate-budget artifact identity does not match")
    if candidate_space.frontier_bound_schema_version != _FRONTIER_SCHEMA:
        raise ValueError("Unsupported candidate-frontier schema version")
    nested_schema_versions = tuple(
        bound.schema_version for bound in candidate_space.frontier
    ) + tuple(
        schema
        for factor in candidate_space.factors
        for summary in (
            factor.unexpanded_domain_summary,
            *tuple(
                contribution.unexpanded_domain_summary
                for contribution in factor.domain_contributions
            ),
        )
        if summary is not None
        for schema in (_summary_frontier_schema(summary),)
        if schema is not None
    )
    if any(
        schema != candidate_space.frontier_bound_schema_version
        for schema in nested_schema_versions
    ):
        raise ValueError("Nested frontier schema does not match candidate space")
