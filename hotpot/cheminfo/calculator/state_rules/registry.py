"""Deterministic dependency registry for optional scientific-state rules."""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from packaging.specifiers import SpecifierSet
from packaging.version import Version

from ..contracts import StatePolicy
from .base import InferenceRule
from .contracts import (
    ActivatedRule,
    RulePhase,
    RuleSet,
    RuleSpecification,
    RuleToggle,
)

__all__ = ["InferenceRuleRegistry"]


_PHASES = tuple(RulePhase)
_PHASE_INDEX = {phase: index for index, phase in enumerate(_PHASES)}


class InferenceRuleRegistry:
    """Register immutable rules and activate deterministic topological layers."""

    def __init__(self) -> None:
        self._rules: Dict[str, InferenceRule] = {}

    def register(self, rule: InferenceRule) -> None:
        """Register one unique rule implementation."""
        rule_id = rule.specification.rule_id
        if rule_id in self._rules:
            raise ValueError(f"Rule {rule_id!r} is already registered")
        self._rules[rule_id] = rule

    def available(self) -> Tuple[RuleSpecification, ...]:
        """Return registered specifications in deterministic phase/ID order."""
        return tuple(
            self._rules[rule_id].specification
            for rule_id in sorted(
                self._rules,
                key=lambda item: (
                    _PHASE_INDEX[self._rules[item].specification.phase],
                    item,
                ),
            )
        )

    def activate(
        self,
        profile: StatePolicy,
        toggles: Tuple[RuleToggle, ...],
    ) -> RuleSet:
        """Validate enabled rules and return phase-respecting DAG layers."""
        del profile
        toggle_by_id = {toggle.rule_id: toggle for toggle in toggles}
        if len(toggle_by_id) != len(toggles):
            raise ValueError("Rule toggles contain duplicate rule IDs")
        unknown = tuple(sorted(set(toggle_by_id) - set(self._rules)))
        if unknown:
            raise ValueError(f"Unknown rule IDs: {unknown}")

        enabled_ids = {
            rule_id for rule_id, toggle in toggle_by_id.items() if toggle.enabled
        }
        included_ids = set(enabled_ids)
        for rule_id in enabled_ids:
            for dependency in self._rules[rule_id].specification.requires:
                if dependency.rule_id not in toggle_by_id:
                    raise ValueError(
                        f"Rule {rule_id!r} requires configured rule "
                        f"{dependency.rule_id!r}"
                    )
                included_ids.add(dependency.rule_id)
        activated = {
            rule_id: ActivatedRule(
                self._rules[rule_id],
                toggle_by_id[rule_id],
            )
            for rule_id in included_ids
        }
        self._validate_dependencies(activated)
        return RuleSet(self._topological_layers(activated))

    def _validate_dependencies(
        self,
        activated: Dict[str, ActivatedRule],
    ) -> None:
        for rule_id, activated_rule in activated.items():
            if not activated_rule.toggle.enabled:
                continue
            specification = activated_rule.rule.specification
            for dependency in specification.requires:
                required = activated.get(dependency.rule_id)
                if required is None:
                    raise ValueError(
                        f"Rule {rule_id!r} requires enabled rule "
                        f"{dependency.rule_id!r}"
                    )
                required_specification = required.rule.specification
                if Version(required_specification.rule_version) not in SpecifierSet(
                    dependency.version_specifier
                ):
                    raise ValueError(
                        f"Rule {rule_id!r} requires {dependency.rule_id!r} "
                        f"{dependency.version_specifier}, not "
                        f"{required_specification.rule_version}"
                    )
                if (
                    _PHASE_INDEX[required_specification.phase]
                    > _PHASE_INDEX[specification.phase]
                ):
                    raise ValueError(
                        f"Rule {rule_id!r} depends on a later execution phase"
                    )

    def _topological_layers(
        self,
        activated: Dict[str, ActivatedRule],
    ) -> Tuple[Tuple[ActivatedRule, ...], ...]:
        completed: Set[str] = set()
        layers: List[Tuple[ActivatedRule, ...]] = []
        for phase in _PHASES:
            pending = {
                rule_id
                for rule_id, activated_rule in activated.items()
                if activated_rule.rule.specification.phase is phase
            }
            while pending:
                ready = tuple(
                    sorted(
                        rule_id
                        for rule_id in pending
                        if not activated[rule_id].toggle.enabled
                        or all(
                            dependency.rule_id in completed
                            for dependency in activated[
                                rule_id
                            ].rule.specification.requires
                        )
                    )
                )
                if not ready:
                    raise ValueError(
                        "Activated rule dependencies contain a cycle"
                    )
                layers.append(tuple(activated[rule_id] for rule_id in ready))
                pending.difference_update(ready)
                completed.update(ready)
        return tuple(layers)
