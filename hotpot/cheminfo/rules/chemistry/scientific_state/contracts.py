"""Shared immutable contracts for scientific-state resolution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Union

__all__ = [
    'ScientificQuantity',
    'ScientificAdmissibility',
    'ResolutionStatus',
    'ChemicalConfidence',
    'CoverageStatus',
    'StateValueAuthority',
    'ScientificFinding',
    'CandidateProvenance',
    'ProviderValue',
    'CandidateBudget',
    'DEFAULT_CANDIDATE_BUDGET',
    'ScientificOperationRequirements',
]


# Scientific and workflow state sets.


class ScientificQuantity(Enum):
    """Independent scientific quantities carried by a state path."""

    NET_CHARGE = "net_charge"
    LEWIS_FORMAL_CHARGE = "lewis_formal_charge"
    COORDINATION_COUNTING = "coordination_counting"
    OXIDATION_STATE = "oxidation_state"
    ELECTRONIC_STATE = "electronic_state"


class ScientificAdmissibility(Enum):
    """Hard scientific legality of an evidence or result envelope."""

    ADMISSIBLE = "admissible"
    INADMISSIBLE = "inadmissible"
    NOT_EVALUATED = "not_evaluated"


class ResolutionStatus(Enum):
    """Degree to which attributable evidence resolves a quantity."""

    UNIQUE = "unique"
    AMBIGUOUS = "ambiguous"
    UNRESOLVED = "unresolved"
    OUT_OF_DOMAIN = "out_of_domain"
    CONFLICT = "conflict"


class ChemicalConfidence(Enum):
    """Scientific confidence independent of hard legality."""

    AUTHORITATIVE = "authoritative"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNRESOLVED = "unresolved"


class CoverageStatus(Enum):
    """Declared coverage of structured atom or component evidence."""

    COMPLETE = "complete"
    PARTIAL = "partial"


class StateValueAuthority(Enum):
    """Authority class attached to one candidate provenance record."""

    VERIFIED_EXPLICIT = "verified_explicit"
    VERIFIED_SCIENTIFIC_SOURCE = "verified_scientific_source"
    EXACT_IDENTITY_TRUSTED_SOURCE = "exact_identity_trusted_source"
    DOMAIN_RULE = "domain_rule"
    DERIVED_CONSTRAINT = "derived_constraint"
    UNRESOLVED = "unresolved"


# Scientific findings and candidate provenance.


@dataclass(frozen=True)
class ScientificFinding:
    """Typed human-readable scientific diagnostic."""

    code: str
    message: str
    related_identities: Tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateProvenance:
    """Attributable source and assumption lineage for one candidate."""

    identity: str
    authority: StateValueAuthority
    source_identity: str
    evidence_ids: Tuple[str, ...] = ()
    assumption_ids: Tuple[str, ...] = ()


# Provider values, search budget and operation requirements.


ProviderValue = Union[
    None,
    bool,
    int,
    float,
    str,
    Tuple["ProviderValue", ...],
]


@dataclass(frozen=True)
class CandidateBudget:
    """Independent limits for candidate materialization and path search."""

    max_materialized_values_per_factor: int
    max_expanded_paths: int
    max_frontier_paths: int

    def __post_init__(self) -> None:
        values = (
            self.max_materialized_values_per_factor,
            self.max_expanded_paths,
            self.max_frontier_paths,
        )
        if any(type(value) is not int or value <= 0 for value in values):
            raise ValueError("Candidate-budget limits must be positive integers")


DEFAULT_CANDIDATE_BUDGET = CandidateBudget(
    max_materialized_values_per_factor=64,
    max_expanded_paths=4096,
    max_frontier_paths=1024,
)


@dataclass(frozen=True)
class ScientificOperationRequirements:
    """Scientific quantities required or allowed unresolved by one operation."""

    operation_id: str
    required_resolved_quantities: frozenset[ScientificQuantity]
    allowed_unresolved_quantities: frozenset[ScientificQuantity]
