"""Abstract toggle-to-action interface for optional state rules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, final

from .contracts import (
    RuleApplicability,
    RuleContext,
    RuleOutcome,
    RuleSpecification,
    RuleToggle,
    RuleValue,
)

__all__ = ["InferenceRule"]


def _disabled_rule_outcome(specification: RuleSpecification) -> RuleOutcome:
    return RuleOutcome(
        rule_id=specification.rule_id,
        rule_version=specification.rule_version,
        applicability=RuleApplicability(False, ("DISABLED",)),
        candidate_proposal=None,
        evidence=(),
        rank_adjustments=(),
        assumptions=(),
        warnings=(),
    )


def _abstained_rule_outcome(
    specification: RuleSpecification,
    applicability: RuleApplicability,
) -> RuleOutcome:
    return RuleOutcome(
        rule_id=specification.rule_id,
        rule_version=specification.rule_version,
        applicability=applicability,
        candidate_proposal=None,
        evidence=(),
        rank_adjustments=(),
        assumptions=(),
        warnings=(),
    )


class InferenceRule(ABC):
    """Stateless optional scientific rule with explicit applicability."""

    specification: RuleSpecification

    @final
    def evaluate(
        self,
        toggle: RuleToggle,
        context: RuleContext,
    ) -> RuleOutcome:
        """Evaluate applicability and action without controlling workflow."""
        if not toggle.enabled:
            return _disabled_rule_outcome(self.specification)
        applicability = self.applicable(context)
        if not applicability.matches:
            return _abstained_rule_outcome(self.specification, applicability)
        return self.action(context, toggle.configuration)

    @abstractmethod
    def applicable(self, context: RuleContext) -> RuleApplicability:
        """Return domain applicability without producing candidates."""

    @abstractmethod
    def action(
        self,
        context: RuleContext,
        configuration: Tuple[Tuple[str, RuleValue], ...],
    ) -> RuleOutcome:
        """Produce attributable candidates, evidence or ranking adjustments."""
