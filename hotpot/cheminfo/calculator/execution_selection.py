"""Backend-context state projection contracts and operations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

from .contracts import (
    BackendContext,
    BackendIdentity,
    MolecularFacts,
    MolecularFactsIdentity,
)
from .electronic_state import (
    ElectronicStateCandidate,
    SpinContributionCandidate,
    SpinContributionScope,
)
from .state_composition import (
    CoupledScientificStateCandidate,
    ScientificStateIdentity,
    StateCandidate,
)
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


# Backend electron-count and selected-state helpers.


_HIGHEST_MS_PROJECTION = "pure_spin_highest_ms"
_CHARGE_ONLY_PROJECTION = "charge_only"
_KNOWN_EXECUTION_POLICY_IDS = frozenset(
    {"EXECUTION-001", "DEFAULT-004"}
)


def _effective_electron_count(
    facts: MolecularFacts,
    backend: BackendContext,
    charge: int,
) -> int:
    explicit_electrons = sum(
        backend.effective_nuclear_charge(atomic_number)
        for atomic_number in facts.atomic_numbers
    )
    implicit_hydrogens = sum(
        implicit_count
        for atomic_number, explicit_count, implicit_count in zip(
            facts.atomic_numbers,
            facts.hydrogen_inventory.explicit_counts,
            facts.hydrogen_inventory.implicit_counts,
        )
        if atomic_number != 1
        and explicit_count == 0
        and implicit_count is not None
    )
    return (
        explicit_electrons
        + implicit_hydrogens * backend.effective_nuclear_charge(1)
        - charge
    )


def _uhf_legality_finding(
    effective_electron_count: int,
    uhf: int,
) -> Optional[BackendProjectionFinding]:
    if effective_electron_count < 0:
        return BackendProjectionFinding(
            "HARD-013.NEGATIVE_EFFECTIVE_ELECTRON_COUNT",
            "The backend effective-electron count is negative.",
        )
    if uhf < 0 or uhf > effective_electron_count:
        return BackendProjectionFinding(
            "HARD-013.UHF_OUT_OF_RANGE",
            "UHF must lie between zero and the effective-electron count.",
        )
    if (effective_electron_count - uhf) % 2:
        return BackendProjectionFinding(
            "HARD-013.UHF_PARITY",
            "UHF and the effective-electron count have incompatible parity.",
        )
    return None


def _lowest_legal_uhf(effective_electron_count: int) -> int:
    return effective_electron_count % 2


def _validated_execution_policy_ids(
    policies: ExecutionPolicySet,
) -> frozenset[str]:
    if len(set(policies.policy_ids)) != len(policies.policy_ids):
        raise ValueError("Execution policy IDs must be unique")
    unknown = set(policies.policy_ids) - _KNOWN_EXECUTION_POLICY_IDS
    if unknown:
        raise ValueError(f"Unknown execution policy IDs: {tuple(sorted(unknown))}")
    return frozenset(policies.policy_ids)


def _selected_path(
    state: ScientificStateBundle,
) -> Optional[CoupledScientificStateCandidate]:
    selected_identity = state.execution_decision.selected_state_identity
    if selected_identity is None:
        return None
    return next(
        (
            path
            for path in state.candidate_space.expanded_paths
            if path.identity == selected_identity
        ),
        None,
    )


def _candidate_index(
    state: ScientificStateBundle,
) -> Dict[str, StateCandidate]:
    return {
        value.candidate.provenance.identity: value.candidate
        for factor in state.candidate_space.factors
        for value in factor.values
    }


def _selected_pure_spin(
    state: ScientificStateBundle,
    path: CoupledScientificStateCandidate,
) -> Optional[Tuple[int, int]]:
    candidates = _candidate_index(state)
    electronic_identity = path.electronic_state_candidate_identity
    electronic = candidates.get(electronic_identity or "")
    if isinstance(electronic, ElectronicStateCandidate):
        if not electronic.pure_spin or electronic.multiplicity is None:
            return None
        return electronic.two_s, electronic.multiplicity

    contributions = tuple(
        candidate
        for evidence_id in path.evidence_ids
        for candidate in (candidates.get(evidence_id),)
        if isinstance(candidate, SpinContributionCandidate)
        and candidate.scope is SpinContributionScope.MOLECULAR
    )
    if len(contributions) != 1 or not contributions[0].pure_spin:
        return None
    return contributions[0].two_s, contributions[0].two_s + 1
