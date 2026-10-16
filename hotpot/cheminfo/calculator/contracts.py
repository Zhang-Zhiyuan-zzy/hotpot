"""Shared immutable contracts for scientific-state resolution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Protocol, Tuple, Union

__all__ = [
    "ScientificQuantity",
    "ScientificAdmissibility",
    "ResolutionStatus",
    "ChemicalConfidence",
    "CoverageStatus",
    "StateValueAuthority",
    "StatePolicy",
    "MolecularFactsIdentity",
    "BackendIdentity",
    "ComponentIdentity",
    "HydrogenInventory",
    "SemanticBondKind",
    "SemanticBond",
    "ScientificFinding",
    "CandidateProvenance",
    "MolecularFacts",
    "ProviderValue",
    "ProviderToggle",
    "ProviderSet",
    "CandidateBudget",
    "DEFAULT_CANDIDATE_BUDGET",
    "ScientificOperationRequirements",
    "BackendContext",
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


class StatePolicy(Enum):
    """Named scientific-state decision profiles."""

    STRICT = "strict"
    COORDINATION_SCREENING_V1 = "coordination_screening_v1"


# Identity and molecular-fact contracts.


@dataclass(frozen=True)
class MolecularFactsIdentity:
    """Coordinate-independent identity of one chemical boundary."""

    chemical_identity_digest: str


@dataclass(frozen=True)
class BackendIdentity:
    """Identity of one exact backend and projection context."""

    digest: str


@dataclass(frozen=True)
class ComponentIdentity:
    """Stable component identity with indices in source atom order."""

    identity: str
    atom_indices: Tuple[int, ...]


@dataclass(frozen=True)
class HydrogenInventory:
    """Explicit and stored implicit hydrogen counts for every atom."""

    explicit_counts: Tuple[Union[int, None], ...]
    implicit_counts: Tuple[Union[int, None], ...]


SemanticBondKind = Literal["covalent", "coordination", "unknown"]


@dataclass(frozen=True)
class SemanticBond:
    """One immutable bond fact without backend-owned objects."""

    atom_indices: Tuple[int, int]
    order: Union[float, None]
    aromatic: bool
    kind: SemanticBondKind


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


@dataclass(frozen=True)
class MolecularFacts:
    """Immutable chemical and optional geometric facts extracted from a molecule."""

    identity: MolecularFactsIdentity
    geometry_digest: Union[str, None]
    atomic_numbers: Tuple[int, ...]
    mass_numbers: Tuple[Union[int, None], ...]
    atom_map: Tuple[int, ...]
    source_components: Tuple[ComponentIdentity, ...]
    covalent_components: Tuple[ComponentIdentity, ...]
    hydrogen_inventory: HydrogenInventory
    semantic_bonds: Tuple[SemanticBond, ...]
    coordinates: Union[Tuple[Tuple[float, float, float], ...], None]
    nuclear_charge_sum: int


# Provider, search-budget and backend boundaries.


ProviderValue = Union[
    None,
    bool,
    int,
    float,
    str,
    Tuple["ProviderValue", ...],
]


@dataclass(frozen=True)
class ProviderToggle:
    """Immutable activation and configuration of one deterministic provider."""

    provider_id: str
    enabled: bool
    configuration: Tuple[Tuple[str, ProviderValue], ...] = ()


@dataclass(frozen=True)
class ProviderSet:
    """Provider settings supplied to one resolution run."""

    toggles: Tuple[ProviderToggle, ...] = ()


@dataclass(frozen=True)
class CandidateBudget:
    """Independent limits for candidate materialization and path search."""

    max_materialized_values_per_factor: int
    max_expanded_paths: int
    max_frontier_paths: int


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


class BackendContext(Protocol):
    """Minimal backend facts consumed by state projection."""

    @property
    def identity(self) -> BackendIdentity:
        """Return the exact backend-context identity."""

    @property
    def scientific_operation_requirements(self) -> ScientificOperationRequirements:
        """Return scientific quantities required by the requested operation."""

    @property
    def projection_convention(self) -> str:
        """Return the named scientific-to-backend projection convention."""

    def effective_nuclear_charge(self, atomic_number: int) -> int:
        """Return the electron-count contribution used by this backend context."""
