"""Private deterministic controller for dependency-layered state rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

from ..contracts import (
    CandidateBudget,
    MolecularFacts,
    ScientificOperationRequirements,
)
from ..state_composition.contracts import (
    CandidateProposal,
    ScientificStateCandidateSpace,
)
from .contracts import (
    ActivatedRule,
    RuleApplicability,
    RuleContext,
    RuleOutcome,
    RulePhase,
    RuleSet,
    RuleTrace,
)
from .validation import _validate_rule_outcome

__all__ = []


_CandidateSpaceBuilder = Callable[
    [Tuple[CandidateProposal, ...]],
    ScientificStateCandidateSpace,
]


@dataclass(frozen=True)
class _RuleExecution:
    candidate_space: ScientificStateCandidateSpace
    traces: Tuple[RuleTrace, ...]
    candidate_proposals: Tuple[CandidateProposal, ...]


def _dependency_unavailable_outcome(
    activated_rule: ActivatedRule,
) -> RuleOutcome:
    specification = activated_rule.rule.specification
    return RuleOutcome(
        rule_id=specification.rule_id,
        rule_version=specification.rule_version,
        applicability=RuleApplicability(
            False,
            ("DEPENDENCY_UNAVAILABLE",),
        ),
        candidate_proposal=None,
        evidence=(),
        rank_adjustments=(),
        assumptions=(),
        warnings=(),
    )


def _dependencies_available(
    activated_rule: ActivatedRule,
    prior_outcomes: Dict[str, RuleOutcome],
) -> bool:
    return all(
        dependency.rule_id in prior_outcomes
        and prior_outcomes[dependency.rule_id].applicability.matches
        for dependency in activated_rule.rule.specification.requires
    )


def _layer_phase(layer: Tuple[ActivatedRule, ...]) -> RulePhase:
    phases = frozenset(
        activated.rule.specification.phase for activated in layer
    )
    if len(phases) != 1:
        raise ValueError("One optional-rule layer must have exactly one phase")
    return next(iter(phases))


def _execute_rule_layers(
    rule_set: RuleSet,
    facts: MolecularFacts,
    requirements: ScientificOperationRequirements,
    candidate_budget: CandidateBudget,
    initial_candidate_space: ScientificStateCandidateSpace,
    candidate_space_builder: _CandidateSpaceBuilder,
) -> _RuleExecution:
    """Evaluate immutable rule layers and compose after each committed layer."""
    candidate_space = initial_candidate_space
    prior_outcomes: Tuple[RuleOutcome, ...] = ()
    outcome_by_id: Dict[str, RuleOutcome] = {}
    traces = []
    proposals = []

    for layer in rule_set.layers:
        if not layer:
            raise ValueError("Optional-rule layers cannot be empty")
        phase = _layer_phase(layer)
        context = RuleContext(
            facts=facts,
            phase=phase,
            requirements=requirements,
            candidate_space=candidate_space,
            prior_outcomes=prior_outcomes,
            candidate_budget=candidate_budget,
        )
        layer_outcomes = []
        for activated_rule in layer:
            outcome = (
                activated_rule.rule.evaluate(activated_rule.toggle, context)
                if _dependencies_available(activated_rule, outcome_by_id)
                else _dependency_unavailable_outcome(activated_rule)
            )
            _validate_rule_outcome(
                activated_rule.rule.specification,
                outcome,
                candidate_budget,
            )
            layer_outcomes.append(outcome)

        for outcome in layer_outcomes:
            outcome_by_id[outcome.rule_id] = outcome
            if outcome.candidate_proposal is not None:
                proposals.append(outcome.candidate_proposal)
            traces.append(RuleTrace(phase, outcome))
        prior_outcomes = prior_outcomes + tuple(layer_outcomes)
        candidate_space = candidate_space_builder(tuple(proposals))

    return _RuleExecution(
        candidate_space,
        tuple(traces),
        tuple(proposals),
    )
