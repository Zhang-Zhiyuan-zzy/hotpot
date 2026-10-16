"""Scientific-state decision contract tests."""

from dataclasses import FrozenInstanceError

import pytest

from hotpot.cheminfo.calculator.state_composition import ScientificStateIdentity
from hotpot.cheminfo.calculator.state_selection import (
    DecisionCertifiability,
    DecisionCertificationStatus,
    DecisionReason,
    ExecutionStateDecision,
    StateDecisionAction,
    StateDecisionSource,
    StaticProfileEligibility,
)


def _eligibility() -> StaticProfileEligibility:
    return StaticProfileEligibility("strict", "strict:empty", True, ())


def test_no_automatic_state_carries_no_selected_identity() -> None:
    decision = ExecutionStateDecision(
        selected_state_identity=None,
        action=StateDecisionAction.NO_AUTOMATIC_STATE,
        source=StateDecisionSource.NOT_SELECTED,
        static_profile_eligibility=_eligibility(),
        decision_certifiability=DecisionCertifiability(
            DecisionCertificationStatus.NON_EQUIVALENT_TIE,
            (DecisionReason("STRICT.AMBIGUOUS", "State remains ambiguous."),),
        ),
        rationale=(),
    )

    assert decision.selected_state_identity is None
    with pytest.raises(FrozenInstanceError):
        decision.action = StateDecisionAction.SELECT_STATE


def test_selection_identity_is_separate_from_decision_source() -> None:
    identity = ScientificStateIdentity("state:explicit")
    decision = ExecutionStateDecision(
        selected_state_identity=identity,
        action=StateDecisionAction.SELECT_STATE,
        source=StateDecisionSource.EXPLICIT,
        static_profile_eligibility=_eligibility(),
        decision_certifiability=DecisionCertifiability(
            DecisionCertificationStatus.CERTIFIED,
            (),
        ),
        rationale=(),
    )

    assert decision.selected_state_identity == identity
    assert decision.source is StateDecisionSource.EXPLICIT
