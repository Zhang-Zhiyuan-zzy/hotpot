"""Deterministic optional-rule registry and DAG tests."""

import pytest

from hotpot.cheminfo.calculator.contracts import StatePolicy
from hotpot.cheminfo.calculator.state_rules import (
    InferenceRule,
    InferenceRuleRegistry,
    RuleApplicability,
    RuleDependency,
    RuleKind,
    RuleOutcome,
    RulePhase,
    RuleSpecification,
    RuleToggle,
)


class _Rule(InferenceRule):
    def __init__(
        self,
        rule_id: str,
        phase: RulePhase,
        requires: tuple[RuleDependency, ...] = (),
        version: str = "1.0.0",
    ) -> None:
        self.specification = RuleSpecification(
            rule_id,
            version,
            RuleKind.EMPIRICAL,
            phase,
            requires,
            (),
            (),
            "test-v1",
        )

    def applicable(self, context):
        return RuleApplicability(True)

    def action(self, context, configuration):
        return RuleOutcome(
            self.specification.rule_id,
            self.specification.rule_version,
            RuleApplicability(True),
            None,
            (),
            (),
            (),
            (),
        )


def _enabled(*rule_ids: str) -> tuple[RuleToggle, ...]:
    return tuple(RuleToggle(rule_id, True) for rule_id in rule_ids)


def test_available_is_independent_of_registration_order() -> None:
    registry = InferenceRuleRegistry()
    registry.register(_Rule("B", RulePhase.CHARGE_EVIDENCE))
    registry.register(_Rule("A", RulePhase.SOURCE_EVIDENCE))

    assert tuple(item.rule_id for item in registry.available()) == ("A", "B")


def test_duplicate_registration_is_rejected() -> None:
    registry = InferenceRuleRegistry()
    registry.register(_Rule("A", RulePhase.SOURCE_EVIDENCE))

    with pytest.raises(ValueError, match="already registered"):
        registry.register(_Rule("A", RulePhase.SOURCE_EVIDENCE))


def test_activation_respects_phase_and_dependency_layers() -> None:
    registry = InferenceRuleRegistry()
    registry.register(_Rule("B", RulePhase.CHARGE_EVIDENCE))
    registry.register(
        _Rule(
            "C",
            RulePhase.CHARGE_EVIDENCE,
            (RuleDependency("B", ">=1,<2"),),
        )
    )
    registry.register(_Rule("A", RulePhase.SOURCE_EVIDENCE))

    rule_set = registry.activate(
        StatePolicy.COORDINATION_SCREENING_V1,
        _enabled("C", "A", "B"),
    )

    assert tuple(
        tuple(item.rule.specification.rule_id for item in layer)
        for layer in rule_set.layers
    ) == (("A",), ("B",), ("C",))


def test_missing_or_wrong_version_dependency_is_rejected() -> None:
    registry = InferenceRuleRegistry()
    registry.register(_Rule("A", RulePhase.SOURCE_EVIDENCE, version="1.0.0"))
    registry.register(
        _Rule(
            "B",
            RulePhase.CHARGE_EVIDENCE,
            (RuleDependency("A", ">=2"),),
        )
    )

    with pytest.raises(ValueError, match="requires configured rule"):
        registry.activate(StatePolicy.STRICT, _enabled("B"))
    with pytest.raises(ValueError, match="not 1.0.0"):
        registry.activate(StatePolicy.STRICT, _enabled("A", "B"))


def test_dependency_cycle_is_rejected() -> None:
    registry = InferenceRuleRegistry()
    registry.register(
        _Rule("A", RulePhase.CHARGE_EVIDENCE, (RuleDependency("B", ">=1"),))
    )
    registry.register(
        _Rule("B", RulePhase.CHARGE_EVIDENCE, (RuleDependency("A", ">=1"),))
    )

    with pytest.raises(ValueError, match="cycle"):
        registry.activate(StatePolicy.STRICT, _enabled("A", "B"))


def test_disabled_rules_produce_empty_rule_set() -> None:
    registry = InferenceRuleRegistry()
    registry.register(_Rule("A", RulePhase.SOURCE_EVIDENCE))

    rule_set = registry.activate(
        StatePolicy.STRICT,
        (RuleToggle("A", False),),
    )

    assert rule_set.layers == ()
