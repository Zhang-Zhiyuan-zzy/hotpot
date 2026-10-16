"""Backend-context state projection contracts and operations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from .contracts import BackendIdentity, MolecularFactsIdentity
from .state_composition import ScientificStateIdentity
from .state_resolver import ScientificStateBundle
from .state_selection import DecisionReason

__all__ = [
    "BackendStateValues",
    "BackendStateRequest",
    "BackendDecisionSource",
    "BackendExecutability",
    "BackendProjectionFinding",
    "BackendExecutionDecision",
    "BackendExecutionEvidence",
    "ExecutionPolicySet",
    "BackendStateProjection",
    "BackendProjectionStatus",
    "BackendProjectionResult",
    "ScientificStateWorkflowResult",
]


# Backend execution-selection contracts.


@dataclass(frozen=True)
class BackendStateValues:
    """Backend-facing charge, UHF and multiplicity values."""

    charge: int
    uhf: Optional[int]
    multiplicity: Optional[int]


@dataclass(frozen=True)
class BackendStateRequest:
    """Explicit backend-only state settings supplied by a caller."""

    uhf: Optional[int] = None


class BackendDecisionSource(Enum):
    """Authority responsible for one backend execution-state choice."""

    EXPLICIT = "explicit"
    EXACT_CONTEXT_HISTORY = "exact_context_history"
    EXECUTION_DEFAULT = "execution_default"


class BackendExecutability(Enum):
    """Static executability of one state in an exact backend context."""

    NOT_EVALUATED = "not_evaluated"
    SUPPORTED = "supported"
    UNSUPPORTED_METHOD = "unsupported_method"
    UNSUPPORTED_ELEMENT = "unsupported_element"
    INVALID_EFFECTIVE_ELECTRON_COUNT = "invalid_effective_electron_count"


@dataclass(frozen=True)
class BackendProjectionFinding:
    """Typed diagnostic emitted while projecting a scientific state."""

    code: str
    message: str
    related_identities: Tuple[str, ...] = ()


@dataclass(frozen=True)
class BackendExecutionDecision:
    """Auditable backend-state decision independent of scientific evidence."""

    backend_identity: BackendIdentity
    requested_state: BackendStateValues
    selected_state: BackendStateValues
    source: BackendDecisionSource
    rationale: Tuple[DecisionReason, ...]


@dataclass(frozen=True)
class BackendExecutionEvidence:
    """Verified execution evidence tied to an exact state and backend."""

    scientific_state_identity: ScientificStateIdentity
    backend_identity: BackendIdentity
    requested_state: BackendStateValues
    executed_state: BackendStateValues
    artifact_digests: Tuple[str, ...]


@dataclass(frozen=True)
class ExecutionPolicySet:
    """Named backend execution-selection policies enabled for projection."""

    policy_ids: Tuple[str, ...] = ("EXECUTION-001", "DEFAULT-004")


@dataclass(frozen=True)
class BackendStateProjection:
    """Validated mapping from one scientific state into a backend context."""

    scientific_state_identity: ScientificStateIdentity
    facts_identity: MolecularFactsIdentity
    backend_identity: BackendIdentity
    effective_electron_count: int
    charge: int
    uhf: Optional[int]
    multiplicity: Optional[int]
    projection_convention: str
    execution_decision: BackendExecutionDecision
    executability: BackendExecutability


class BackendProjectionStatus(Enum):
    """Terminal status of one backend projection attempt."""

    NOT_PROJECTED = "not_projected"
    PROJECTED = "projected"
    PROJECTION_CONFLICT = "projection_conflict"
    BACKEND_UNSUPPORTED = "backend_unsupported"


@dataclass(frozen=True)
class BackendProjectionResult:
    """Tagged backend projection result with no fabricated failure values."""

    status: BackendProjectionStatus
    backend_identity: BackendIdentity
    request: BackendStateRequest
    projection: Optional[BackendStateProjection]
    findings: Tuple[BackendProjectionFinding, ...]

    def __post_init__(self) -> None:
        projected = self.status is BackendProjectionStatus.PROJECTED
        if projected != (self.projection is not None):
            raise ValueError("Only PROJECTED results may carry a projection")
        if (
            self.projection is not None
            and self.projection.backend_identity != self.backend_identity
        ):
            raise ValueError(
                "Projection and result must identify the same backend context"
            )


@dataclass(frozen=True)
class ScientificStateWorkflowResult:
    """Scientific-state resolution paired with its backend projection."""

    scientific_state: ScientificStateBundle
    backend_projection: BackendProjectionResult
