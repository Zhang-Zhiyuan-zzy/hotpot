"""Pure pipeline decisions for opt-in backend-state recovery."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from hotpot.cheminfo.calculator.contracts import BackendContext
from hotpot.cheminfo.calculator.execution_selection import (
    BackendExecutionEvidence,
    BackendStateProjection,
    BackendStateRequest,
    ExecutionPolicySet,
)
from hotpot.cheminfo.calculator.state_resolver import ScientificStateBundle
from hotpot.cheminfo.calculator.state_selection import DecisionReason


__all__ = [
    "decide_state_recovery",
    "RecoveryAction",
    "RecoveryPolicySet",
    "RecoveryBudget",
    "DEFAULT_RECOVERY_BUDGET",
    "RecoveryDecision",
]


_KNOWN_RECOVERY_POLICY_IDS = frozenset({"RECOVERY-001"})


@dataclass(frozen=True)
class RecoveryBudget:
    """Maximum backend attempts permitted for one controlled route."""

    max_backend_attempts: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_backend_attempts, bool)
            or not isinstance(self.max_backend_attempts, int)
            or self.max_backend_attempts < 1
        ):
            raise ValueError("max_backend_attempts must be a positive integer")


DEFAULT_RECOVERY_BUDGET = RecoveryBudget(max_backend_attempts=1)


class RecoveryAction(Enum):
    """Controller action selected after one backend attempt."""

    STOP = "stop"
    RETRY = "retry"


@dataclass(frozen=True)
class RecoveryPolicySet:
    """Named opt-in policies admitted to one recovery decision."""

    policy_ids: Tuple[str, ...] = ()


@dataclass(frozen=True)
class RecoveryDecision:
    """A pure recovery action and an optional next backend projection."""

    action: RecoveryAction
    projection: Optional[BackendStateProjection]
    rationale: Tuple[DecisionReason, ...]

    def __post_init__(self) -> None:
        retry = self.action is RecoveryAction.RETRY
        if retry != (self.projection is not None):
            raise ValueError("RETRY requires a projection and STOP forbids one")


def _validated_policy_ids(
    policies: RecoveryPolicySet,
) -> frozenset[str]:
    if len(set(policies.policy_ids)) != len(policies.policy_ids):
        raise ValueError("Recovery policy IDs must be unique")
    unknown = set(policies.policy_ids) - _KNOWN_RECOVERY_POLICY_IDS
    if unknown:
        raise ValueError(f"Unknown recovery policy IDs: {sorted(unknown)}")
    return frozenset(policies.policy_ids)


def _stop(code: str, message: str) -> RecoveryDecision:
    return RecoveryDecision(
        action=RecoveryAction.STOP,
        projection=None,
        rationale=(DecisionReason(code, message),),
    )


def decide_state_recovery(
    state: ScientificStateBundle,
    backend: BackendContext,
    *,
    request: BackendStateRequest = BackendStateRequest(),
    execution_policies: ExecutionPolicySet = ExecutionPolicySet(),
    execution_history: Tuple[BackendExecutionEvidence, ...],
    recovery_policies: RecoveryPolicySet = RecoveryPolicySet(),
    recovery_budget: RecoveryBudget = DEFAULT_RECOVERY_BUDGET,
) -> RecoveryDecision:
    """Select a next state without launching a backend or mutating science."""

    policy_ids = _validated_policy_ids(recovery_policies)
    if not policy_ids:
        return _stop(
            "RECOVERY.DISABLED",
            "No backend-state recovery policy is enabled.",
        )
    if recovery_budget.max_backend_attempts == 1:
        return _stop(
            "RECOVERY.BUDGET_EXHAUSTED",
            "The recovery budget permits only the original backend attempt.",
        )
    return _stop(
        "RECOVERY-001.NO_CONTINUATION",
        "No attributable alternative-state continuation is available.",
    )
