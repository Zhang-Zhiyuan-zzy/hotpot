"""Optional scientific-state rule abstraction tests."""

import hotpot

from hotpot.cheminfo.calculator.contracts import (
    DEFAULT_CANDIDATE_BUDGET,
    ScientificOperationRequirements,
)
from hotpot.cheminfo.calculator.state_resolver import extract_molecular_facts
from hotpot.cheminfo.calculator.state_rules import (
    InferenceRule,
    RuleApplicability,
    RuleContext,
    RuleKind,
    RuleOutcome,
    RulePhase,
    RuleSpecification,
    RuleToggle,
    RuleValue,
)


class _Rule(InferenceRule):
    specification = RuleSpecification(
        rule_id="TEST-001",
        rule_version="1",
        kind=RuleKind.EMPIRICAL,
        phase=RulePhase.CHARGE_EVIDENCE,
        requires=(),
        consumes=(),
        produces=(),
        applicability_schema="test-v1",
    )

    def applicable(self, context: RuleContext) -> RuleApplicability:
        return RuleApplicability(context.phase is self.specification.phase)

    def action(
        self,
        context: RuleContext,
        configuration: tuple[tuple[str, RuleValue], ...],
    ) -> RuleOutcome:
        return RuleOutcome(
            self.specification.rule_id,
            self.specification.rule_version,
            RuleApplicability(True),
            None,
            (),
            (),
            tuple(key for key, _ in configuration),
            (),
        )


def _context(phase: RulePhase) -> RuleContext:
    facts = extract_molecular_facts(hotpot.read_mol("C", "smi"))
    return RuleContext(
        facts=facts,
        phase=phase,
        requirements=ScientificOperationRequirements(
            "test",
            frozenset(),
            frozenset(),
        ),
        candidate_space=None,
        prior_outcomes=(),
        candidate_budget=DEFAULT_CANDIDATE_BUDGET,
    )


def test_disabled_toggle_never_calls_action() -> None:
    outcome = _Rule().evaluate(
        RuleToggle("TEST-001", False),
        _context(RulePhase.CHARGE_EVIDENCE),
    )

    assert not outcome.applicability.matches
    assert outcome.applicability.reason_codes == ("DISABLED",)
    assert outcome.candidate_proposal is None


def test_out_of_domain_rule_abstains_without_fallback() -> None:
    outcome = _Rule().evaluate(
        RuleToggle("TEST-001", True),
        _context(RulePhase.ELECTRONIC_STATE_CANDIDATE),
    )

    assert not outcome.applicability.matches
    assert outcome.evidence == ()


def test_enabled_rule_receives_only_immutable_configuration() -> None:
    outcome = _Rule().evaluate(
        RuleToggle("TEST-001", True, (("mode", "test"),)),
        _context(RulePhase.CHARGE_EVIDENCE),
    )

    assert outcome.applicability.matches
    assert outcome.assumptions == ("mode",)
