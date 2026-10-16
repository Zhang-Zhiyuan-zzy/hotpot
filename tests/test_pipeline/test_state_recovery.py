"""Contracts for pure, opt-in backend-state recovery decisions."""

from dataclasses import FrozenInstanceError
import inspect

import pytest

from hotpot.pipeline.state_recovery import (
    DEFAULT_RECOVERY_BUDGET,
    RecoveryAction,
    RecoveryBudget,
    RecoveryDecision,
    RecoveryPolicySet,
    decide_state_recovery,
)


def test_state_recovery_exports_and_signature_are_stable() -> None:
    import hotpot.pipeline.state_recovery as recovery

    assert recovery.__all__ == [
        "decide_state_recovery",
        "RecoveryAction",
        "RecoveryPolicySet",
        "RecoveryBudget",
        "DEFAULT_RECOVERY_BUDGET",
        "RecoveryDecision",
    ]
    assert tuple(inspect.signature(decide_state_recovery).parameters) == (
        "state",
        "backend",
        "request",
        "execution_policies",
        "execution_history",
        "recovery_policies",
        "recovery_budget",
    )


@pytest.mark.parametrize("value", (True, False, 0, -1, 1.0, "1"))
def test_recovery_budget_rejects_nonpositive_or_noninteger_values(value) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        RecoveryBudget(value)


def test_recovery_contracts_are_immutable() -> None:
    policies = RecoveryPolicySet()

    with pytest.raises(FrozenInstanceError):
        policies.policy_ids = ("RECOVERY-001",)


def test_recovery_decision_requires_projection_exactly_for_retry() -> None:
    with pytest.raises(ValueError, match="RETRY requires"):
        RecoveryDecision(RecoveryAction.RETRY, None, ())


def test_default_budget_permits_only_original_attempt() -> None:
    assert DEFAULT_RECOVERY_BUDGET.max_backend_attempts == 1


def test_disabled_recovery_stops_without_reading_scientific_inputs() -> None:
    decision = decide_state_recovery(
        None,
        None,
        execution_history=(),
    )

    assert decision.action is RecoveryAction.STOP
    assert decision.projection is None
    assert decision.rationale[0].code == "RECOVERY.DISABLED"


def test_recovery_policy_ids_are_validated_at_decision_time() -> None:
    with pytest.raises(ValueError, match="unique"):
        decide_state_recovery(
            None,
            None,
            execution_history=(),
            recovery_policies=RecoveryPolicySet(
                ("RECOVERY-001", "RECOVERY-001")
            ),
        )
    with pytest.raises(ValueError, match="Unknown"):
        decide_state_recovery(
            None,
            None,
            execution_history=(),
            recovery_policies=RecoveryPolicySet(("UNKNOWN",)),
        )


def test_default_budget_stops_an_enabled_recovery_policy() -> None:
    decision = decide_state_recovery(
        None,
        None,
        execution_history=(),
        recovery_policies=RecoveryPolicySet(("RECOVERY-001",)),
    )

    assert decision.action is RecoveryAction.STOP
    assert decision.rationale[0].code == "RECOVERY.BUDGET_EXHAUSTED"


def test_recovery_does_not_fabricate_an_unavailable_alternative_state() -> None:
    decision = decide_state_recovery(
        None,
        None,
        execution_history=(),
        recovery_policies=RecoveryPolicySet(("RECOVERY-001",)),
        recovery_budget=RecoveryBudget(2),
    )

    assert decision.action is RecoveryAction.STOP
    assert decision.projection is None
    assert decision.rationale[0].code == "RECOVERY-001.NO_CONTINUATION"
