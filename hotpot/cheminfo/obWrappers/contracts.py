"""Public value contracts for Open Babel wrapper rule execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


__all__ = (
    "BuildReport",
    "BondKindCode",
    "CoordinateChange",
    "HybridizationChange",
    "OptimizationFrame",
    "OptimizationReport",
    "RuleApplication",
    "RuleDescriptor",
    "RuleExecutionReport",
    "RuleStage",
    "SingleOptimizationReport",
)


class BondKindCode(IntEnum):
    """Stable native-boundary codes for Hotpot bond semantics."""

    SINGLE = 1
    DOUBLE = 2
    TRIPLE = 3
    AROMATIC = 4
    ZERO = 5
    DATIVE = 6
    UNKNOWN = 7


class RuleStage(Enum):
    """Lifecycle stage at which an Open Babel wrapper rule executes."""

    PRE_BUILD = "pre_build"
    PRE_FORCEFIELD_SETUP = "pre_forcefield_setup"


@dataclass(frozen=True)
class RuleDescriptor:
    """Stable identity and ordering metadata for one registered rule."""

    rule_id: str
    version: str
    stage: RuleStage
    priority: int


@dataclass(frozen=True)
class HybridizationChange:
    """Temporary hybridization mutation requested by a native rule."""

    atom_index: int
    before: int
    after: int


@dataclass(frozen=True)
class CoordinateChange:
    """One coordinate mutation requested by a native rule."""

    atom_index: int
    before: Tuple[float, float, float]
    after: Tuple[float, float, float]


@dataclass(frozen=True)
class RuleApplication:
    """Auditable evidence and mutations produced by one rule application."""

    descriptor: RuleDescriptor
    atom_indices: Tuple[int, ...]
    metric_before: Optional[float]
    hybridization_changes: Tuple[HybridizationChange, ...]
    coordinate_changes: Tuple[CoordinateChange, ...]


@dataclass(frozen=True)
class RuleExecutionReport:
    """Ordered applications produced for one wrapper stage."""

    stage: RuleStage
    applications: Tuple[RuleApplication, ...] = ()

    @property
    def applied(self) -> bool:
        return bool(self.applications)


@dataclass(frozen=True)
class BuildReport:
    """Result of one delegated Open Babel coordinate build."""

    succeeded: bool
    rules: RuleExecutionReport


@dataclass(frozen=True)
class SingleOptimizationReport:
    """Result of one native steepest-descent optimization."""

    coordinates: NDArray[np.float64]
    energy: float
    energy_unit: str
    backend_energy_unit: str
    exploded: bool
    rules: RuleExecutionReport


@dataclass(frozen=True)
class OptimizationFrame:
    """Numerical facts recorded after one native optimization epoch."""

    coordinates: NDArray[np.float64]
    energy: float
    rms_gradient: float
    max_gradient: float
    exploded: bool
    converged: bool
    segment_epochs_completed: int
    segment_index: int
    energy_change: Optional[float]
    max_displacement: Optional[float]


@dataclass(frozen=True)
class OptimizationReport:
    """Complete result returned by the native Open Babel optimizer."""

    coordinates: NDArray[np.float64]
    terminal_coordinates: NDArray[np.float64]
    frames: Tuple[OptimizationFrame, ...]
    selected_frame_index: int
    best_epoch: int
    final_energy: float
    best_energy: float
    rms_gradient: float
    max_gradient: float
    exploded: bool
    converged: bool
    epochs_completed: int
    steps_submitted: int
    initialization_steps: int
    selected_segment_epochs_completed: int
    energy_unit: str
    backend_energy_unit: str
    termination_reason: str
    terminal_converged: bool
    energy_changes: Tuple[float, ...]
    max_displacements: Tuple[float, ...]
    epoch_energies: Tuple[float, ...]
    rules: RuleExecutionReport
