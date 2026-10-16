"""Contracts for scientific-state workflow decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from .state_composition import ScientificStateIdentity

__all__ = [
    "StateDecisionAction",
    "StateDecisionSource",
    "DecisionCertificationStatus",
    "StaticEligibilityFinding",
    "DecisionReason",
    "StaticProfileEligibility",
    "DecisionCertifiability",
    "ExecutionStateDecision",
]


class StateDecisionAction(Enum):
    """Terminal scientific-path action selected by a named policy."""

    NOT_REQUESTED = "not_requested"
    SELECT_STATE = "select_state"
    NO_AUTOMATIC_STATE = "no_automatic_state"
    CONFLICT = "conflict"
    INADMISSIBLE = "inadmissible"


class StateDecisionSource(Enum):
    """Authority class responsible for a workflow selection."""

    NOT_SELECTED = "not_selected"
    EXPLICIT = "explicit"
    SCIENTIFIC_SOURCE = "scientific_source"
    EXECUTION_DEFAULT = "execution_default"
    EXECUTION_PRIOR = "execution_prior"
    CANONICAL_EQUIVALENT_TIE = "canonical_equivalent_tie"


class DecisionCertificationStatus(Enum):
    """Whether the candidate search certifies one deterministic decision."""

    NOT_EVALUATED = "not_evaluated"
    CERTIFIED = "certified"
    NON_EQUIVALENT_TIE = "non_equivalent_tie"
    OPEN_OR_UNBOUNDABLE_DOMAIN = "open_or_unboundable_domain"
    RESOURCE_LIMIT_UNCERTIFIED = "resource_limit_uncertified"


@dataclass(frozen=True)
class StaticEligibilityFinding:
    """Static profile-domain finding independent of candidate ranking."""

    code: str
    message: str


@dataclass(frozen=True)
class DecisionReason:
    """Typed explanation attached to a workflow decision."""

    code: str
    message: str
    related_identities: Tuple[str, ...] = ()


@dataclass(frozen=True)
class StaticProfileEligibility:
    """Static applicability of one exact effective profile configuration."""

    profile_id: str
    effective_profile_identity: str
    in_domain: bool
    findings: Tuple[StaticEligibilityFinding, ...]


@dataclass(frozen=True)
class DecisionCertifiability:
    """Dynamic proof status for choosing among hard-valid paths."""

    status: DecisionCertificationStatus
    findings: Tuple[DecisionReason, ...]


@dataclass(frozen=True)
class ExecutionStateDecision:
    """Named policy action without changing scientific resolution."""

    selected_state_identity: Optional[ScientificStateIdentity]
    action: StateDecisionAction
    source: StateDecisionSource
    static_profile_eligibility: StaticProfileEligibility
    decision_certifiability: DecisionCertifiability
    rationale: Tuple[DecisionReason, ...]
