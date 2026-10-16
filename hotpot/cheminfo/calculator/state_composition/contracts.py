"""Immutable contracts for factorized scientific-state composition."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Union

from ..charge.contracts import (
    LewisFormalChargeCandidate,
    NetChargeCandidateEvidence,
)
from ..contracts import (
    CandidateBudget,
    CandidateProvenance,
    ResolutionStatus,
    ScientificQuantity,
)
from ..electronic_state.contracts import (
    ElectronicStateCandidate,
    SpinContributionCandidate,
)
from ..oxidation_state.contracts import (
    CoordinationCountingCandidate,
    CountingConvention,
    OxidationStateCandidateEvidence,
)

__all__ = [
    "CandidateDomainScope",
    "MaterializationStatus",
    "CandidateSearchStatus",
    "StateValueOrigin",
    "ScientificStateIdentity",
    "FactorKey",
    "ConstraintValue",
    "CandidateConstraint",
    "FrontierBound",
    "CandidateDomainSummary",
    "RejectedPathSummary",
    "BranchAssignment",
    "StateValueRef",
    "DerivedStateValue",
    "StateCandidate",
    "FactorCandidateValue",
    "FactorDomainContribution",
    "CandidateFactor",
    "ProposedCandidateValue",
    "ProposedCandidateFactor",
    "ConstraintGuard",
    "GuardedCandidateConstraint",
    "CandidateProposal",
    "CoupledScientificStateCandidate",
    "ScientificStateCandidateSpace",
    "ScientificResolutionResult",
]


class CandidateDomainScope(Enum):
    """Scientific completeness of a candidate factor's declared domain."""

    EXHAUSTIVE = "exhaustive"
    DOMAIN_BOUNDED = "domain_bounded"
    OPEN = "open"


class MaterializationStatus(Enum):
    """Whether all values in a declared factor domain were materialized."""

    COMPLETE = "complete"
    RESOURCE_TRUNCATED = "resource_truncated"


class CandidateSearchStatus(Enum):
    """Certification status of budgeted joint-path expansion."""

    NOT_EVALUATED = "not_evaluated"
    DECISION_CERTIFIED = "decision_certified"
    TRUNCATED_BUT_DECISION_CERTIFIED = "truncated_but_decision_certified"
    NO_CERTIFIED_DECISION = "no_certified_decision"
    RESOURCE_LIMIT_NO_CERTIFIED_DECISION = (
        "resource_limit_no_certified_decision"
    )


class StateValueOrigin(Enum):
    """Origin class for one scalar value in a coupled path."""

    RESULT = "result"
    DERIVED_CONSTRAINT = "derived_constraint"


@dataclass(frozen=True)
class ScientificStateIdentity:
    """Stable identity of one coupled scientific-state path."""

    digest: str


@dataclass(frozen=True)
class FactorKey:
    """Scientific quantity and subject defining one candidate factor."""

    quantity: ScientificQuantity
    subject_identity: str


ConstraintValue = Union[
    None,
    bool,
    int,
    float,
    str,
    Tuple["ConstraintValue", ...],
]


@dataclass(frozen=True)
class CandidateConstraint:
    """Serializable compatibility constraint between candidate factors."""

    identity: str
    constraint_type: str
    schema_version: str
    factor_keys: Tuple[FactorKey, ...]
    parameters: Tuple[Tuple[str, ConstraintValue], ...]


@dataclass(frozen=True)
class FrontierBound:
    """Optimistic decision-key bound for one unexpanded branch."""

    branch_identity: str
    schema_version: str
    optimistic_key: Tuple[str, ...]


@dataclass(frozen=True)
class CandidateDomainSummary:
    """Compact description of an unmaterialized candidate-domain remainder."""

    remaining_count_upper_bound: Optional[int]
    continuation_identity: Optional[str]
    frontier_bound: Optional[FrontierBound]


@dataclass(frozen=True)
class RejectedPathSummary:
    """Auditable compact record of one rejected candidate combination."""

    candidate_identities: Tuple[str, ...]
    constraint_identity: str
    reason_codes: Tuple[str, ...]


@dataclass(frozen=True)
class StateValueRef:
    """Reference to a scalar scientific value used by one path."""

    quantity: ScientificQuantity
    value: int
    origin: StateValueOrigin
    origin_identity: str


@dataclass(frozen=True)
class DerivedStateValue:
    """Value derived exactly from named constraints and parent evidence."""

    provenance: CandidateProvenance
    quantity: ScientificQuantity
    value: int
    equation_id: str
    parent_candidate_ids: Tuple[str, ...]
    counting_convention: CountingConvention


StateCandidate = Union[
    NetChargeCandidateEvidence,
    LewisFormalChargeCandidate,
    CoordinationCountingCandidate,
    OxidationStateCandidateEvidence,
    SpinContributionCandidate,
    ElectronicStateCandidate,
    DerivedStateValue,
]


@dataclass(frozen=True)
class BranchAssignment:
    """Selection of one correlated option within a proposal branch group."""

    group_id: str
    option_id: str


@dataclass(frozen=True)
class FactorCandidateValue:
    """One deduplicated value retained in a composed candidate factor."""

    candidate: StateCandidate
    source_proposal_ids: Tuple[str, ...]
    branch_assignments: Tuple[BranchAssignment, ...]


@dataclass(frozen=True)
class FactorDomainContribution:
    """Domain and materialization claim made by one proposal."""

    proposal_id: str
    domain_scope: CandidateDomainScope
    materialization_status: MaterializationStatus
    unexpanded_domain_summary: Optional[CandidateDomainSummary]


@dataclass(frozen=True)
class CandidateFactor:
    """Merged candidate values and domain evidence for one factor key."""

    identity: str
    factor_key: FactorKey
    domain_definition_id: str
    values: Tuple[FactorCandidateValue, ...]
    domain_contributions: Tuple[FactorDomainContribution, ...]
    unexpanded_domain_summary: Optional[CandidateDomainSummary]
    domain_scope: CandidateDomainScope
    materialization_status: MaterializationStatus


@dataclass(frozen=True)
class ProposedCandidateValue:
    """Candidate value supplied by one provider or rule proposal."""

    candidate: StateCandidate
    branch_assignments: Tuple[BranchAssignment, ...] = ()


@dataclass(frozen=True)
class ProposedCandidateFactor:
    """One proposal's contribution to a factorized candidate domain."""

    factor_key: FactorKey
    domain_definition_id: str
    materialized_candidates: Tuple[ProposedCandidateValue, ...]
    domain_scope: CandidateDomainScope
    materialization_status: MaterializationStatus
    unexpanded_domain_summary: Optional[CandidateDomainSummary]


@dataclass(frozen=True)
class ConstraintGuard:
    """Conditions under which a proposed compatibility constraint applies."""

    all_candidate_ids: Tuple[str, ...] = ()
    branch_assignments: Tuple[BranchAssignment, ...] = ()


@dataclass(frozen=True)
class GuardedCandidateConstraint:
    """Constraint paired with its correlated-branch activation guard."""

    guard: ConstraintGuard
    constraint: CandidateConstraint


@dataclass(frozen=True)
class CandidateProposal:
    """Correlated factor values and constraints emitted by one action."""

    proposal_id: str
    factors: Tuple[ProposedCandidateFactor, ...]
    constraints: Tuple[GuardedCandidateConstraint, ...]


@dataclass(frozen=True)
class CoupledScientificStateCandidate:
    """One expanded and hard-valid coupled scientific-state path."""

    identity: ScientificStateIdentity
    net_charge: StateValueRef
    component_assignment_ids: Tuple[str, ...]
    coordination_assignment_ids: Tuple[str, ...]
    oxidation_state_assignment_id: Optional[str]
    electronic_state_result_id: str
    electronic_state_candidate_identity: Optional[str]
    derived_values: Tuple[DerivedStateValue, ...]
    evidence_ids: Tuple[str, ...]
    assumption_ids: Tuple[str, ...]


@dataclass(frozen=True)
class ScientificStateCandidateSpace:
    """Factorized domains, expanded paths and retained search evidence."""

    factors: Tuple[CandidateFactor, ...]
    constraints: Tuple[CandidateConstraint, ...]
    candidate_budget: CandidateBudget
    candidate_budget_artifact_id: str
    frontier_bound_schema_version: str
    expanded_paths: Tuple[CoupledScientificStateCandidate, ...]
    rejected_path_summaries: Tuple[RejectedPathSummary, ...]
    frontier: Tuple[FrontierBound, ...]
    search_status: CandidateSearchStatus


@dataclass(frozen=True)
class ScientificResolutionResult:
    """Scientific resolution independent of workflow path selection."""

    status: ResolutionStatus
    exhaustive: bool
    equivalent_state_identities: Tuple[ScientificStateIdentity, ...]
