"""Immutable contracts for charge and spin inference."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol, TYPE_CHECKING, Tuple

from ..contracts import (
    CandidateProvenance,
    ChemicalConfidence,
    MolecularFactsIdentity,
    ResolutionStatus,
    ScientificAdmissibility,
    ScientificFinding,
)

if TYPE_CHECKING:
    from ...core import Molecule

__all__ = [
    "AmbiguousHydrogenRepresentationError",
    "ChargeEstimator",
    "ChargeInferenceError",
    "ChargeInferenceResult",
    "ChargeInferenceSource",
    "ElectronicState",
    "ElectronicStateError",
    "FragmentCharge",
    "IncompleteExplicitAtomError",
    "SpinEstimator",
    "SpinInferenceResult",
    "SpinInferenceSource",
    "ElectronicStateCandidateEvidence",
    "SpinContributionScope",
    "SpinContributionCandidate",
    "ElectronicStateCandidate",
    "ElectronicStateInferenceResult",
    "NeutralAtomReference",
]


# Exceptions and finite state sets.


class ElectronicStateError(ValueError):
    """Base error for an electronic state that cannot be resolved."""


class ChargeInferenceError(ElectronicStateError):
    """Raised when classical integer charge inference has no valid result."""


class AmbiguousHydrogenRepresentationError(ChargeInferenceError):
    """Raised when explicit and stored implicit hydrogen facts conflict."""


class IncompleteExplicitAtomError(ElectronicStateError):
    """Raised when an execution state omits atoms represented implicitly."""


class ChargeInferenceSource(str, Enum):
    """Provenance of an inferred or authoritative integer charge."""

    VALENCE = "valence"
    VALENCE_CONSTRAINED = "valence-constrained"
    PRESERVED = "preserve"
    METAL_DEFAULT = "metal-default"
    METAL_RESOLVER = "metal-resolver"
    EXPLICIT = "explicit"


class SpinInferenceSource(str, Enum):
    """Provenance of an unpaired-electron count."""

    LOWEST_SPIN_PARITY = "lowest-spin-parity"
    EXPLICIT = "explicit"


class SpinContributionScope(Enum):
    """Boundary to which a local spin contribution applies."""

    MOLECULAR = "molecular"
    COMPONENT = "component"
    ATOM = "atom"


# Immutable data contracts.


@dataclass(frozen=True)
class FragmentCharge:
    """Integer charge assigned to one deterministic molecular fragment."""

    atom_indices: tuple[int, ...]
    charge: int
    source: ChargeInferenceSource


@dataclass(frozen=True)
class ChargeInferenceResult:
    """Non-mutating atom, fragment and molecular formal-charge evidence."""

    atom_formal_charges: tuple[int, ...]
    fragments: tuple[FragmentCharge, ...]
    total_charge: int
    source: ChargeInferenceSource
    assumptions: tuple[str, ...]


@dataclass(frozen=True)
class SpinInferenceResult:
    """Electron-count evidence for one spin policy."""

    unpaired_electrons: int
    multiplicity: int
    electron_count: int
    source: SpinInferenceSource
    assumptions: tuple[str, ...]


@dataclass(frozen=True)
class ElectronicState:
    """Resolved charge and spin values supplied to a numerical backend."""

    charge: int
    unpaired_electrons: int
    multiplicity: int
    fragment_charges: tuple[int, ...]
    charge_source: ChargeInferenceSource
    spin_source: SpinInferenceSource
    assumptions: tuple[str, ...]


@dataclass(frozen=True)
class ElectronicStateCandidateEvidence:
    """Direct attributable evidence for a molecular scientific spin state."""

    provenance: CandidateProvenance
    two_s: int
    multiplicity: Optional[int]
    pure_spin: bool


@dataclass(frozen=True)
class SpinContributionCandidate:
    """Local spin contribution awaiting molecular coupling."""

    provenance: CandidateProvenance
    scope: SpinContributionScope
    subject_identity: str
    parent_candidate_ids: Tuple[str, ...]
    two_s: int
    pure_spin: bool


@dataclass(frozen=True)
class ElectronicStateCandidate:
    """One complete molecular electronic-state candidate."""

    provenance: CandidateProvenance
    net_charge_candidate_identity: str
    oxidation_state_candidate_identity: Optional[str]
    electron_count: int
    two_s: int
    multiplicity: Optional[int]
    pure_spin: bool


@dataclass(frozen=True)
class ElectronicStateInferenceResult:
    """Result envelope for molecular electronic-state candidates."""

    identity: str
    facts_identity: MolecularFactsIdentity
    candidates: Tuple[ElectronicStateCandidate, ...]
    admissibility: ScientificAdmissibility
    resolution: ResolutionStatus
    confidence: ChemicalConfidence
    findings: Tuple[ScientificFinding, ...]


@dataclass(frozen=True)
class NeutralAtomReference:
    """Versioned isolated-neutral-atom electron and spin reference."""

    atomic_number: int
    electron_count: int
    ground_configuration: str
    two_s: int
    source_identity: str
    registry_version: str
    content_digest: str


# Replaceable calculator interfaces.


class ChargeEstimator(Protocol):
    """Interface for a non-mutating molecular charge estimator."""

    def infer(self, mol: "Molecule") -> ChargeInferenceResult:
        """Infer atom, fragment and total integer charges."""


class SpinEstimator(Protocol):
    """Interface for an unpaired-electron estimator."""

    def infer(self, mol: "Molecule", charge: int) -> SpinInferenceResult:
        """Infer a spin state for a fixed total charge."""

