"""Validated numerical policy contracts for force-field optimizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence, Tuple, TypedDict, Union

from ...mathematics import geometry as geo
from ....structure.forcefields import AtomTopologySignature, BondTopology

import numpy as np


__all__ = (
    "OptimizationStoppingCriteria",
    "AcceptanceCheck",
    "StructureAcceptanceThresholds",
    "ForceFieldAcceptanceEvidence",
    "ForceFieldDiagnosticValue",
)


@dataclass(frozen=True)
class OptimizationStoppingCriteria:
    """Opt-in numerical thresholds for ending one optimizer segment."""

    window: int = 5
    maximum_energy_change_kj_mol: float = 1.0e-4
    maximum_atom_displacement_angstrom: float = 1.0e-4
    maximum_rms_gradient_kj_mol_angstrom: float = 1.0
    maximum_gradient_kj_mol_angstrom: float = 5.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.window, int)
            or isinstance(self.window, bool)
            or self.window < 1
        ):
            raise ValueError("window must be a positive integer")
        thresholds = (
            self.maximum_energy_change_kj_mol,
            self.maximum_atom_displacement_angstrom,
            self.maximum_rms_gradient_kj_mol_angstrom,
            self.maximum_gradient_kj_mol_angstrom,
        )
        if not all(
            np.isfinite(threshold) and threshold >= 0.0
            for threshold in thresholds
        ):
            raise ValueError(
                "optimization stopping thresholds must be finite and non-negative"
            )


AcceptanceLevel = Literal["off", "basic", "standard", "strict"]
ForceFieldStage = Literal["candidate", "final"]
ForceFieldDiagnosticValue = Union[
    None,
    bool,
    int,
    float,
    str,
    AtomTopologySignature,
    BondTopology,
    Tuple["ForceFieldDiagnosticValue", ...],
    Mapping[str, "ForceFieldDiagnosticValue"],
]


@dataclass(frozen=True)
class AcceptanceCheck:
    """One serializable decision made by force-field acceptance policy."""

    name: str
    passed: bool
    severity: Literal["info", "warning", "error"] = "error"
    measured: ForceFieldDiagnosticValue = None
    threshold: ForceFieldDiagnosticValue = None
    atom_indices: Tuple[int, ...] = ()
    bond_indices: Tuple[int, ...] = ()
    message: str = ""


@dataclass(frozen=True)
class StructureAcceptanceThresholds:
    """Chemical and force-field thresholds for structure acceptance."""

    overlap_tolerance: float = 1.0e-3
    basic_minimum_distance: float = 0.40
    standard_minimum_distance: float = 0.50
    standard_covalent_radius_scale: float = 0.55
    maximum_bond_distance: float = 30.0
    covalent_bond_ratio: Tuple[float, float] = (0.65, 1.45)
    metal_ligand_bond_ratio: Tuple[float, float] = (0.65, 1.60)
    strict_rms_gradient: float = 1.0
    strict_max_gradient: float = 5.0
    strict_energy_change: float = 1.0e-4
    strict_max_displacement: float = 1.0e-4
    strict_stability_window: int = 5


class ForceFieldAcceptanceEvidence(TypedDict, total=False):
    """Force-field observations consumed by structure-acceptance policy."""

    setup_succeeded: bool
    converged: bool
    epochs_completed: int
    segment_epochs_completed: int
    final_energy: float
    energy_unit: str
    rms_gradient: float
    max_gradient: float
    exploded: bool
    energy_changes: Sequence[float]
    max_displacements: Sequence[float]


@dataclass(frozen=True)
class _BondRingAcceptanceFinding:
    """Source-independent bond--ring evidence consumed by acceptance policy."""

    ring_key: Tuple[int, ...]
    bond_key: Tuple[int, int]
    state: geo.PiercingState
    indeterminacy_causes: Tuple[geo.SegmentCycleIndeterminacy, ...]


@dataclass(frozen=True)
class _BondRingAcceptanceFacts:
    """Normalized facts from one completed bond--ring checkpoint."""

    actionable_findings: Tuple[_BondRingAcceptanceFinding, ...]
    ring_scope: Literal["full_graph", "ligand_skeleton"]
    max_ring_size: int
    selected_ring_count: int
    excluded_ring_count: int
    piercing_pair_count: int
    undetermined_pair_count: int
    scan_complete: bool

    @property
    def piercings(self) -> Tuple[_BondRingAcceptanceFinding, ...]:
        return tuple(
            finding
            for finding in self.actionable_findings
            if finding.state is geo.PiercingState.PIERCES
        )

    @property
    def undetermined(self) -> Tuple[_BondRingAcceptanceFinding, ...]:
        return tuple(
            finding
            for finding in self.actionable_findings
            if finding.state is geo.PiercingState.UNDETERMINED
        )


@dataclass(frozen=True)
class _AtomPairAcceptanceIssue:
    kind: Literal["overlap", "too_close"]
    atom_indices: Tuple[int, int]
    distance: float
    threshold: float
