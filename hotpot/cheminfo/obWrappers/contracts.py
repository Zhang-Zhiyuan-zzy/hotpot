"""Public value contracts for Open Babel wrapper rule execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


__all__ = (
    "BuildReport",
    "CoordinateChange",
    "ForceFieldStateReport",
    "HybridizationChange",
    "OptimizationPreparationReport",
    "RuleApplication",
    "RuleDescriptor",
    "RuleExecutionReport",
    "RuleStage",
)


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
class OptimizationPreparationReport:
    """Coordinate preparation performed before force-field setup."""

    forcefield: str
    rules: RuleExecutionReport

    @property
    def applied(self) -> bool:
        return self.rules.applied


@dataclass(frozen=True)
class ForceFieldStateReport:
    """Finite-energy and finite-gradient status after force-field setup."""

    energy: float
    finite_energy: bool
    finite_gradients: bool
    nonfinite_gradient_atom_indices: Tuple[int, ...]

    @property
    def passed(self) -> bool:
        return self.finite_energy and self.finite_gradients
