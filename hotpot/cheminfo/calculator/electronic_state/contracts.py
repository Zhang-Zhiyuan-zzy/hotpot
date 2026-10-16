"""Immutable contracts for scientific electronic-state inference."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from ..contracts import (
    CandidateProvenance,
    ChemicalConfidence,
    MolecularFactsIdentity,
    ResolutionStatus,
    ScientificAdmissibility,
    ScientificFinding,
)

__all__ = [
    "ElectronicStateCandidateEvidence",
    "SpinContributionScope",
    "SpinContributionCandidate",
    "ElectronicStateCandidate",
    "ElectronicStateInferenceResult",
    "NeutralAtomReference",
]


class SpinContributionScope(Enum):
    """Boundary to which a local spin contribution applies."""

    MOLECULAR = "molecular"
    COMPONENT = "component"
    ATOM = "atom"


# Immutable data contracts.


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

