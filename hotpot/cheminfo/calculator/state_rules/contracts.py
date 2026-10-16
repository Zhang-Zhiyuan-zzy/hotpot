"""Immutable contracts for optional scientific-state rules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from ..contracts import (
    CandidateBudget,
    MolecularFacts,
    ProviderValue,
    ScientificOperationRequirements,
    StateValueAuthority,
)
from ..state_composition.contracts import (
    CandidateProposal,
    ScientificStateCandidateSpace,
)

__all__ = [
    "RuleKind",
    "RulePhase",
    "ResultType",
    "RuleValue",
    "RuleDependency",
    "RuleSpecification",
    "RuleToggle",
    "EvidenceItem",
    "RankAdjustment",
    "RuleApplicability",
    "RuleContext",
    "RuleOutcome",
    "RuleTrace",
    "ActivatedRule",
    "RuleSet",
]


class RuleKind(Enum):
    """Implementation family of one optional scientific rule."""

    EMPIRICAL = "empirical"
    AI = "ai"


class RulePhase(Enum):
    """Closed optional-rule execution phases."""

    SOURCE_EVIDENCE = "source_evidence"
    CHARGE_EVIDENCE = "charge_evidence"
    COORDINATION_COUNTING = "coordination_counting"
    REDOX_CANDIDATE = "redox_candidate"
    OXIDATION_STATE_CANDIDATE = "oxidation_state_candidate"
    OXIDATION_STATE_RANKING = "oxidation_state_ranking"
    ELECTRONIC_STATE_CANDIDATE = "electronic_state_candidate"
    ELECTRONIC_STATE_RANKING = "electronic_state_ranking"


class ResultType(Enum):
    """Typed inputs and outputs declared by one rule specification."""

    NET_CHARGE_RESULT = "net_charge_result"
    LEWIS_FORMAL_CHARGE_RESULT = "lewis_formal_charge_result"
    COORDINATION_COUNTING_RESULT = "coordination_counting_result"
    OXIDATION_STATE_RESULT = "oxidation_state_result"
    ELECTRONIC_STATE_RESULT = "electronic_state_result"
    SPIN_CONTRIBUTION = "spin_contribution"
    CANDIDATE_PROPOSAL = "candidate_proposal"
    EVIDENCE = "evidence"
    RANK_ADJUSTMENT = "rank_adjustment"


RuleValue = ProviderValue


@dataclass(frozen=True)
class RuleDependency:
    """Required rule and accepted version specifier."""

    rule_id: str
    version_specifier: str


@dataclass(frozen=True)
class RuleSpecification:
    """Immutable discovery and dependency metadata for one rule."""

    rule_id: str
    rule_version: str
    kind: RuleKind
    phase: RulePhase
    requires: Tuple[RuleDependency, ...]
    consumes: Tuple[ResultType, ...]
    produces: Tuple[ResultType, ...]
    applicability_schema: str


@dataclass(frozen=True)
class RuleToggle:
    """Per-run activation and immutable configuration of one rule."""

    rule_id: str
    enabled: bool
    configuration: Tuple[Tuple[str, RuleValue], ...] = ()


@dataclass(frozen=True)
class EvidenceItem:
    """Attributable scientific statement produced or consumed by a rule."""

    identity: str
    authority: StateValueAuthority
    source_identity: str
    claim: str
    locator: Optional[str] = None


@dataclass(frozen=True)
class RankAdjustment:
    """Attributable preference contribution without direct selection."""

    target_identity: str
    rank_dimension: str
    value: float
    reason: str
    assumption_ids: Tuple[str, ...] = ()


@dataclass(frozen=True)
class RuleApplicability:
    """Whether one rule's reviewed domain matches the immutable context."""

    matches: bool
    reason_codes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class RuleContext:
    """Immutable input boundary supplied to one optional rule layer."""

    facts: MolecularFacts
    phase: RulePhase
    requirements: ScientificOperationRequirements
    candidate_space: Optional[ScientificStateCandidateSpace]
    prior_outcomes: Tuple["RuleOutcome", ...]
    candidate_budget: CandidateBudget


@dataclass(frozen=True)
class RuleOutcome:
    """Immutable proposal, evidence and preference output of one rule."""

    rule_id: str
    rule_version: str
    applicability: RuleApplicability
    candidate_proposal: Optional[CandidateProposal]
    evidence: Tuple[EvidenceItem, ...]
    rank_adjustments: Tuple[RankAdjustment, ...]
    assumptions: Tuple[str, ...]
    warnings: Tuple[str, ...]


@dataclass(frozen=True)
class RuleTrace:
    """One phase-labelled rule outcome retained for audit."""

    phase: RulePhase
    outcome: RuleOutcome


@dataclass(frozen=True)
class ActivatedRule:
    """One registered rule paired with its per-run toggle."""

    rule: "InferenceRule"
    toggle: RuleToggle


@dataclass(frozen=True)
class RuleSet:
    """Dependency-validated topological layers of activated rules."""

    layers: Tuple[Tuple[ActivatedRule, ...], ...] = ()
