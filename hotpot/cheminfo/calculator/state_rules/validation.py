"""Private structural validation of optional-rule outcomes."""

from __future__ import annotations

import math

from ..contracts import CandidateBudget
from ..state_composition.validation import (
    _validate_candidate_proposal_budget,
)
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
    _validate_candidate_proposal_budget(proposal, candidate_budget)


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
