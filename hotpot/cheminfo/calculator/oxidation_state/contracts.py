"""Immutable coordination-counting and oxidation-state contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from ..contracts import (
    CandidateProvenance,
    ChemicalConfidence,
    CoverageStatus,
    MolecularFactsIdentity,
    ResolutionStatus,
    ScientificAdmissibility,
    ScientificFinding,
)

__all__ = [
    "CountingConvention",
    "OxidationStateOverride",
    "CoordinationFragmentAssignment",
    "CoordinationCountingCandidate",
    "CoordinationElectronCountingResult",
    "OxidationStateCandidateEvidence",
    "OxidationStateInferenceResult",
    "ElementOxidationStateDomain",
]


class CountingConvention(Enum):
    """Mutually exclusive metal-ligand electron-counting conventions."""

    IONIC = "ionic"
    DONOR_RETURN = "donor_return"
    CBC = "cbc"


@dataclass(frozen=True)
class OxidationStateOverride:
    """Explicit oxidation-state constraint for one atom index."""

    atom_index: int
    oxidation_state: int


@dataclass(frozen=True)
class CoordinationFragmentAssignment:
    """Convention-labelled local accounting for one molecular component."""

    component_identity: str
    ionic_bookkeeping_charge: Optional[int]
    lxz_class: Optional[str]
    donated_electrons: Optional[int]


@dataclass(frozen=True)
class CoordinationCountingCandidate:
    """One attributable coordination-counting assignment."""

    provenance: CandidateProvenance
    convention: CountingConvention
    fragments: Tuple[CoordinationFragmentAssignment, ...]


@dataclass(frozen=True)
class CoordinationElectronCountingResult:
    """Result envelope for convention-labelled coordination counting."""

    identity: str
    facts_identity: MolecularFactsIdentity
    candidates: Tuple[CoordinationCountingCandidate, ...]
    admissibility: ScientificAdmissibility
    resolution: ResolutionStatus
    confidence: ChemicalConfidence
    findings: Tuple[ScientificFinding, ...]


@dataclass(frozen=True)
class OxidationStateCandidateEvidence:
    """Complete or partial atom oxidation-state evidence."""

    provenance: CandidateProvenance
    convention: CountingConvention
    atom_states: Tuple[Optional[int], ...]
    coverage: CoverageStatus
    coordination_counting_candidate_identity: Optional[str] = None


@dataclass(frozen=True)
class OxidationStateInferenceResult:
    """Result envelope for attributable oxidation-state assignments."""

    identity: str
    facts_identity: MolecularFactsIdentity
    candidates: Tuple[OxidationStateCandidateEvidence, ...]
    admissibility: ScientificAdmissibility
    resolution: ResolutionStatus
    confidence: ChemicalConfidence
    findings: Tuple[ScientificFinding, ...]


@dataclass(frozen=True)
class ElementOxidationStateDomain:
    """Versioned admissible oxidation-state domain for one element."""

    atomic_number: int
    states: Tuple[int, ...]
    source_identity: str
    registry_version: str
    content_digest: str
