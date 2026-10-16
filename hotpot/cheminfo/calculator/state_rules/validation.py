"""Private structural validation of optional-rule outcomes."""

from __future__ import annotations

import math

from ..contracts import CandidateBudget
from ..state_composition.contracts import MaterializationStatus
from ..state_composition.expand import _candidate_matches_factor
from .contracts import ResultType, RuleOutcome, RuleSpecification

__all__ = []


def _validate_declared_outputs(
    specification: RuleSpecification,
    outcome: RuleOutcome,
) -> None:
    produced = frozenset(specification.produces)
    if (
        outcome.candidate_proposal is not None
        and ResultType.CANDIDATE_PROPOSAL not in produced
    ):
        raise ValueError("Rule returned an undeclared candidate proposal")
    if outcome.evidence and ResultType.EVIDENCE not in produced:
        raise ValueError("Rule returned undeclared evidence")
    if (
        outcome.rank_adjustments
        and ResultType.RANK_ADJUSTMENT not in produced
    ):
        raise ValueError("Rule returned undeclared rank adjustments")
    if not outcome.applicability.matches and (
        outcome.candidate_proposal is not None
        or outcome.evidence
        or outcome.rank_adjustments
        or outcome.assumptions
    ):
        raise ValueError("An inapplicable rule cannot produce scientific data")


def _validate_candidate_proposal(
    outcome: RuleOutcome,
    candidate_budget: CandidateBudget,
) -> None:
    proposal = outcome.candidate_proposal
    if proposal is None:
        return
    for factor in proposal.factors:
        if (
            len(factor.materialized_candidates)
            > candidate_budget.max_materialized_values_per_factor
        ):
            raise ValueError("Rule candidate factor exceeds its materialization budget")
        truncated = (
            factor.materialization_status
            is MaterializationStatus.RESOURCE_TRUNCATED
        )
        if truncated != (factor.unexpanded_domain_summary is not None):
            raise ValueError(
                "Rule factor truncation and remainder summary must agree"
            )
        for proposed in factor.materialized_candidates:
            if not _candidate_matches_factor(
                factor.factor_key,
                proposed.candidate,
            ):
                raise ValueError(
                    "Rule candidate type does not match its factor key"
                )


def _validate_rule_outcome(
    specification: RuleSpecification,
    outcome: RuleOutcome,
    candidate_budget: CandidateBudget,
) -> None:
    """Validate one outcome against its immutable specification and budget."""
    if outcome.rule_id != specification.rule_id:
        raise ValueError("Rule outcome identity does not match specification")
    if outcome.rule_version != specification.rule_version:
        raise ValueError("Rule outcome version does not match specification")
    if any(
        not math.isfinite(adjustment.value)
        for adjustment in outcome.rank_adjustments
    ):
        raise ValueError("Rule rank adjustments must be finite")
    _validate_declared_outputs(specification, outcome)
    _validate_candidate_proposal(outcome, candidate_budget)
