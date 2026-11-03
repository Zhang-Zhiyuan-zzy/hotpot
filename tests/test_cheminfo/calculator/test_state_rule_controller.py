"""Deterministic dependency-layer execution of optional state rules."""

from typing import List, Optional, Tuple

import hotpot

from hotpot.cheminfo.rules.chemistry.scientific_state.contracts import (
    CandidateBudget,
    ScientificOperationRequirements,
)
from hotpot.cheminfo.structure.snapshots import MolecularFacts
from hotpot.cheminfo.operations.calculation.scientific_state.contracts import StatePolicy
from hotpot.cheminfo.rules.chemistry.scientific_state.composition import (
    CandidateProposal,
    ScientificStateCandidateSpace,
)
from hotpot.cheminfo.operations.calculation.scientific_state.composition.compose import (
    compose_scientific_state_space,
)
from hotpot.cheminfo.operations.calculation.scientific_state.facts import extract_molecular_facts
from hotpot.cheminfo.rules.chemistry.scientific_state.providers.base import InferenceRule
from hotpot.cheminfo.rules.chemistry.scientific_state.providers import (
    RankAdjustment,
    ResultType,
    RuleApplicability,
    RuleContext,
    RuleDependency,
    RuleKind,
    RuleOutcome,
    RulePhase,
    RuleSpecification,
    RuleToggle,
    RuleValue,
)
from hotpot.cheminfo.operations.calculation.scientific_state.rule_registry import (
    InferenceRuleRegistry,
)
from hotpot.cheminfo.operations.calculation.scientific_state.rule_controller import (
    _execute_rule_layers,
)


class _ObservedRule(InferenceRule):
    def __init__(
        self,
        rule_id: str,
        observations: List[Tuple[int, Tuple[str, ...]]],
        *,
        applicable: bool = True,
        requires: Tuple[RuleDependency, ...] = (),
        rank_value: Optional[float] = None,
    ) -> None:
        self._observations = observations
        self._applicable = applicable
        self._rank_value = rank_value
        self.specification = RuleSpecification(
            rule_id,
            "1.0.0",
            RuleKind.EMPIRICAL,
            RulePhase.CHARGE_EVIDENCE,
            requires,
            (),
            (ResultType.RANK_ADJUSTMENT,) if rank_value is not None else (),
            "test-v1",
        )

    def applicable(self, context: RuleContext) -> RuleApplicability:
        self._observations.append(
            (
                id(context),
                tuple(outcome.rule_id for outcome in context.prior_outcomes),
            )
        )
        return RuleApplicability(self._applicable, ())

    def action(
        self,
        context: RuleContext,
        configuration: Tuple[Tuple[str, RuleValue], ...],
    ) -> RuleOutcome:
        del context, configuration
        rank_adjustments = (
            ()
            if self._rank_value is None
            else (
                RankAdjustment(
                    self.specification.rule_id,
                    "test-rank",
                    self._rank_value,
                    "test",
                ),
            )
        )
        return RuleOutcome(
            self.specification.rule_id,
            self.specification.rule_version,
            RuleApplicability(True),
            None,
            (),
            rank_adjustments,
            (),
            (),
        )


def _facts() -> MolecularFacts:
    return extract_molecular_facts(hotpot.read_mol("CO", "smi"))


def _requirements() -> ScientificOperationRequirements:
    return ScientificOperationRequirements(
        "rule-controller-test",
        frozenset(),
        frozenset(),
    )


def _candidate_space(
    facts: MolecularFacts,
    budget: CandidateBudget,
) -> ScientificStateCandidateSpace:
    return compose_scientific_state_space(
        facts,
        net_charge_results=(),
        candidate_budget=budget,
    )


def test_one_layer_shares_one_immutable_context() -> None:
    observations: List[Tuple[int, Tuple[str, ...]]] = []
    registry = InferenceRuleRegistry()
    registry.register(_ObservedRule("B", observations))
    registry.register(_ObservedRule("A", observations))
    rule_set = registry.activate(
        StatePolicy.COORDINATION_SCREENING_V1,
        (RuleToggle("B", True), RuleToggle("A", True)),
    )
    facts = _facts()
    budget = CandidateBudget(8, 8, 8)
    candidate_space = _candidate_space(facts, budget)
    builder_calls: List[
        Tuple[Tuple[CandidateProposal, ...], Tuple[RankAdjustment, ...]]
    ] = []

    def build_space(
        proposals: Tuple[CandidateProposal, ...],
        rank_adjustments: Tuple[RankAdjustment, ...],
    ) -> ScientificStateCandidateSpace:
        builder_calls.append((proposals, rank_adjustments))
        return candidate_space

    execution = _execute_rule_layers(
        rule_set,
        facts,
        _requirements(),
        budget,
        candidate_space,
        build_space,
    )

    assert len({context_id for context_id, _ in observations}) == 1
    assert all(prior == () for _, prior in observations)
    assert tuple(trace.outcome.rule_id for trace in execution.traces) == (
        "A",
        "B",
    )
    assert builder_calls == [((), ())]


def test_builder_receives_cumulative_rank_adjustments_explicitly() -> None:
    observations: List[Tuple[int, Tuple[str, ...]]] = []
    registry = InferenceRuleRegistry()
    registry.register(_ObservedRule("A", observations, rank_value=1.0))
    registry.register(
        _ObservedRule(
            "B",
            observations,
            requires=(RuleDependency("A", ">=1"),),
            rank_value=2.0,
        )
    )
    rule_set = registry.activate(
        StatePolicy.COORDINATION_SCREENING_V1,
        (RuleToggle("A", True), RuleToggle("B", True)),
    )
    facts = _facts()
    budget = CandidateBudget(8, 8, 8)
    candidate_space = _candidate_space(facts, budget)
    rank_calls: List[Tuple[RankAdjustment, ...]] = []

    def build_space(
        proposals: Tuple[CandidateProposal, ...],
        rank_adjustments: Tuple[RankAdjustment, ...],
    ) -> ScientificStateCandidateSpace:
        del proposals
        rank_calls.append(rank_adjustments)
        return candidate_space

    execution = _execute_rule_layers(
        rule_set,
        facts,
        _requirements(),
        budget,
        candidate_space,
        build_space,
    )

    assert tuple(len(adjustments) for adjustments in rank_calls) == (1, 2)
    assert execution.rank_adjustments == rank_calls[-1]


def test_abstained_dependency_prevents_dependent_action() -> None:
    observations: List[Tuple[int, Tuple[str, ...]]] = []
    registry = InferenceRuleRegistry()
    registry.register(
        _ObservedRule("A", observations, applicable=False)
    )
    registry.register(
        _ObservedRule(
            "B",
            observations,
            requires=(RuleDependency("A", ">=1"),),
        )
    )
    rule_set = registry.activate(
        StatePolicy.COORDINATION_SCREENING_V1,
        (RuleToggle("A", True), RuleToggle("B", True)),
    )
    facts = _facts()
    budget = CandidateBudget(8, 8, 8)
    candidate_space = _candidate_space(facts, budget)

    execution = _execute_rule_layers(
        rule_set,
        facts,
        _requirements(),
        budget,
        candidate_space,
        lambda proposals, rank_adjustments: candidate_space,
    )

    assert tuple(trace.outcome.rule_id for trace in execution.traces) == (
        "A",
        "B",
    )
    assert execution.traces[1].outcome.applicability.reason_codes == (
        "DEPENDENCY_UNAVAILABLE",
    )
    assert len(observations) == 1


def test_explicitly_disabled_dependency_is_retained_for_controller() -> None:
    observations: List[Tuple[int, Tuple[str, ...]]] = []
    registry = InferenceRuleRegistry()
    registry.register(_ObservedRule("A", observations))
    registry.register(
        _ObservedRule(
            "B",
            observations,
            requires=(RuleDependency("A", ">=1"),),
        )
    )
    rule_set = registry.activate(
        StatePolicy.COORDINATION_SCREENING_V1,
        (RuleToggle("A", False), RuleToggle("B", True)),
    )
    facts = _facts()
    budget = CandidateBudget(8, 8, 8)
    candidate_space = _candidate_space(facts, budget)

    execution = _execute_rule_layers(
        rule_set,
        facts,
        _requirements(),
        budget,
        candidate_space,
        lambda proposals, rank_adjustments: candidate_space,
    )

    assert execution.traces[0].outcome.applicability.reason_codes == (
        "DISABLED",
    )
    assert execution.traces[1].outcome.applicability.reason_codes == (
        "DEPENDENCY_UNAVAILABLE",
    )
    assert observations == []
