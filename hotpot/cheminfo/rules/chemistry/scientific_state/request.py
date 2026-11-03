"""Explicit immutable scientific-state request inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .charge.contracts import (
    AtomFormalChargeEvidence,
    ComponentChargeEvidence,
    NetChargeCandidateEvidence,
)
from .oxidation_state.contracts import (
    OxidationStateOverride,
    OxidationStateCandidateEvidence,
)
from .electronic_state.contracts import ElectronicStateCandidateEvidence

__all__ = [
    'StateRequest',
]


@dataclass(frozen=True)
class StateRequest:
    """Explicit constraints and attributable allowed candidate domains."""

    net_charge: Optional[int] = None
    atom_formal_charges: Optional[AtomFormalChargeEvidence] = None
    component_charges: Optional[ComponentChargeEvidence] = None
    oxidation_overrides: Tuple[OxidationStateOverride, ...] = ()
    two_s: Optional[int] = None
    multiplicity: Optional[int] = None
    charge_candidates: Tuple[NetChargeCandidateEvidence, ...] = ()
    oxidation_candidates: Tuple[OxidationStateCandidateEvidence, ...] = ()
    electronic_candidates: Tuple[ElectronicStateCandidateEvidence, ...] = ()
