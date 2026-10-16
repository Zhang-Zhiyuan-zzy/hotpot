"""Immutable contracts for net-charge and Lewis-formal-charge evidence."""

from __future__ import annotations

from dataclasses import dataclass
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
    "NetChargeCandidateEvidence",
    "AtomFormalChargeEvidence",
    "ComponentChargeEvidence",
    "NetChargeResult",
    "LewisFormalChargeCandidate",
    "LewisFormalChargeResult",
]


@dataclass(frozen=True)
class NetChargeCandidateEvidence:
    """One attributable integer molecular-charge candidate."""

    provenance: CandidateProvenance
    charge: int


@dataclass(frozen=True)
class AtomFormalChargeEvidence:
    """Complete or partial per-atom formal-charge evidence."""

    provenance: CandidateProvenance
    charges: Tuple[Optional[int], ...]
    coverage: CoverageStatus


@dataclass(frozen=True)
class ComponentChargeEvidence:
    """Complete or partial component-level charge evidence."""

    provenance: CandidateProvenance
    component_charges: Tuple[Tuple[str, int], ...]
    coverage: CoverageStatus


@dataclass(frozen=True)
class NetChargeResult:
    """Result envelope for molecular net-charge inference."""

    identity: str
    facts_identity: MolecularFactsIdentity
    candidates: Tuple[NetChargeCandidateEvidence, ...]
    admissibility: ScientificAdmissibility
    resolution: ResolutionStatus
    confidence: ChemicalConfidence
    findings: Tuple[ScientificFinding, ...]


@dataclass(frozen=True)
class LewisFormalChargeCandidate:
    """One complete Lewis allocation for a fixed net-charge candidate."""

    provenance: CandidateProvenance
    net_charge_candidate_identity: str
    atom_formal_charges: Tuple[int, ...]
    bond_orders: Tuple[float, ...]
    nonbonding_electrons: Tuple[int, ...]
    radical_electrons: Tuple[int, ...]
    component_charges: Tuple[Tuple[str, int], ...]


@dataclass(frozen=True)
class LewisFormalChargeResult:
    """Result envelope for zero or more attributable Lewis allocations."""

    identity: str
    facts_identity: MolecularFactsIdentity
    candidates: Tuple[LewisFormalChargeCandidate, ...]
    admissibility: ScientificAdmissibility
    resolution: ResolutionStatus
    confidence: ChemicalConfidence
    findings: Tuple[ScientificFinding, ...]
