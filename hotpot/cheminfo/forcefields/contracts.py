"""Public data contracts and exceptions for force-field workflows."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Literal, Mapping, Optional, Sequence, Tuple, TypedDict, Union

import numpy as np

from .topology import (
    AtomTopologySignature,
    BondTopologySignature,
)
from .trajectory import ForceFieldTrajectory, ForceFieldTrajectoryArchive


__all__ = (
    "TrajectoryPath",
    "OptimizationAlgorithm",
    "TerminationReason",
    "ForceFieldDiagnosticValue",
    "ForceFieldRunReport",
    "Build3DReport",
    "CandidateRejection",
    "RingUntanglingReport",
    "CoordinationBondRestorationReport",
    "ComplexBuildDiagnostics",
    "BuildWorkerResult",
    "ForceFieldWorkflowReport",
    "BuildAndOptimizeReport",
    "ComplexBuildReport",
    "ForceFieldSetupReport",
    "AcceptanceCheck",
    "StructureAcceptanceThresholds",
    "ForceFieldAcceptanceEvidence",
    "ForceFieldValidationReport",
    "CoordinationEnvironment",
    "CoordinationGeometryCandidate",
    "CoordinationGeometryResult",
    "ForceFieldError",
    "ForceFieldSetupError",
    "BuildWorkerError",
    "BuildTimeoutError",
    "ComplexBuildError",
    "ComplexBuildWarning",
    "ComplexBuildWorkerError",
    "ComplexBuildTimeoutError",
    "GeometryQualityError",
    "GeometryQualityWarning",
)


OptimizationAlgorithm = Literal["steepest", "conjugate"]
TrajectoryPath = Union[str, os.PathLike[str]]
TerminationReason = Literal[
    "converged",
    "budget_exhausted",
]
AcceptanceLevel = Literal["off", "basic", "standard", "strict"]
ForceFieldStage = Literal["candidate", "final"]
ForceFieldDiagnosticValue = Union[
    None,
    bool,
    int,
    float,
    str,
    AtomTopologySignature,
    BondTopologySignature,
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
class ForceFieldValidationReport:
    """Acceptance result for coordinates, topology, and force-field evidence."""

    level: AcceptanceLevel
    passed: bool
    checks: Tuple[AcceptanceCheck, ...]
    metrics: Mapping[str, ForceFieldDiagnosticValue] = field(default_factory=dict)

    @property
    def failures(self) -> Tuple[AcceptanceCheck, ...]:
        return tuple(
            check
            for check in self.checks
            if not check.passed and check.severity == "error"
        )

    @property
    def warnings(self) -> Tuple[AcceptanceCheck, ...]:
        return tuple(
            check
            for check in self.checks
            if not check.passed and check.severity == "warning"
        )

    def to_dict(self) -> dict[str, ForceFieldDiagnosticValue]:
        """Return a JSON-serializable representation of the report."""
        return asdict(self)


@dataclass(frozen=True)
class ForceFieldRunReport:
    """Summary of an optimization run.

    ``converged`` and the gradient/quality fields describe the selected frame;
    ``final_energy`` and ``termination_reason`` describe the terminal frame.
    Open Babel does not expose its exact internal step counter (and conjugate
    gradient initialization itself takes a step).  Therefore
    ``steps_submitted`` records the number passed to ``TakeNSteps`` and
    ``initialization_steps`` records the first steps performed by conjugate
    gradient initialization. ``steps_completed`` remains ``None`` rather than
    claiming how many submitted steps Open Babel completed before stopping.
    ``best_epoch`` is the zero-based ordinal among frames actually observed by
    the optimizer, not the outer scheduling-loop index.  A value of ``-1``
    means that the selected coordinates are the segment's unoptimized initial
    frame.  ``selected_segment_epochs_completed`` counts observations in the
    numerical segment that produced the selected frame.
    """

    requested_forcefield: Optional[str]
    effective_forcefield: str
    setup_succeeded: bool
    converged: bool
    epochs_completed: int
    steps_submitted: int
    initialization_steps: int
    steps_completed: Optional[int]
    final_energy: float
    best_energy: float
    energy_unit: str
    rms_gradient: float
    max_gradient: float
    exploded: bool
    quality_report: Optional[ForceFieldValidationReport] = None
    backend_energy_unit: Optional[str] = None
    gradient_unit: str = "kJ/(mol*angstrom)"
    energy_changes: Tuple[float, ...] = ()
    max_displacements: Tuple[float, ...] = ()
    best_epoch: int = 0
    selected_segment_epochs_completed: int = 0
    epoch_energies: Tuple[float, ...] = ()
    termination_reason: TerminationReason = "budget_exhausted"
    terminal_converged: bool = False
    untangling: Optional["RingUntanglingReport"] = None
    trajectory: Optional[ForceFieldTrajectoryArchive] = None


@dataclass(frozen=True)
class Build3DReport:
    atom_count: int
    added_hydrogen_count: int
    quality_report: ForceFieldValidationReport


@dataclass(frozen=True)
class CandidateRejection:
    component_index: int
    attempt: int
    reason: str
    quality_failures: Tuple[AcceptanceCheck, ...] = ()


@dataclass(frozen=True)
class RingUntanglingReport:
    """Outcome of one bounded covalent-ring untangling stage."""

    attempt_limit: int
    attempts_completed: int
    initial_piercing_count: int
    final_piercing_count: int
    minimum_piercing_count: int
    resolved: bool
    warning_messages: Tuple[str, ...] = ()


@dataclass(frozen=True)
class CoordinationBondRestorationReport:
    """Outcome of incremental restoration of original coordination bonds.

    ``attempts_completed`` counts stalled relax-and-retry rounds.  Successful
    one-bond restoration rounds do not consume that failure budget.
    """

    attempt_limit: int
    attempts_completed: int
    bond_count: int
    metal_relocation_attempt_count: int
    relocated_metal_indices: Tuple[int, ...]
    infeasible_metal_indices: Tuple[int, ...]
    forced_bond_keys: Tuple[Tuple[int, int], ...]
    rejected_piercing_trial_count: int
    undetermined_trial_count: int
    excluded_ring_observation_count: int
    warning_messages: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ComplexBuildDiagnostics:
    attempt_count: int
    accepted_candidates: int
    rejected_candidates: Tuple[CandidateRejection, ...]
    elapsed_seconds: float
    warning_messages: Tuple[str, ...] = ()
    ligand_untangling: Tuple[RingUntanglingReport, ...] = ()
    coordination_restoration: Optional[CoordinationBondRestorationReport] = None


@dataclass(frozen=True)
class BuildWorkerResult:
    status: Literal["ok", "error"]
    coordinates: Optional[np.ndarray] = None
    diagnostics: Optional[ComplexBuildDiagnostics] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    traceback: Optional[str] = None
    ligand_build_attempts: Tuple[ForceFieldTrajectory, ...] = ()


@dataclass(frozen=True)
class ForceFieldWorkflowReport:
    requested_forcefield: Optional[str]
    effective_forcefield: str
    build: Union[Build3DReport, ComplexBuildDiagnostics]
    optimization: Optional[ForceFieldRunReport]
    quality_report: ForceFieldValidationReport
    trajectory: Optional[ForceFieldTrajectoryArchive] = None


@dataclass(frozen=True)
class BuildAndOptimizeReport(ForceFieldWorkflowReport):
    build: Build3DReport
    optimization: ForceFieldRunReport


@dataclass(frozen=True)
class ComplexBuildReport(ForceFieldWorkflowReport):
    build: ComplexBuildDiagnostics


@dataclass(frozen=True)
class ForceFieldSetupReport:
    requested_forcefield: Optional[str]
    effective_forcefield: str
    stage: Literal["lookup", "setup"]
    setup_succeeded: bool = False


@dataclass(frozen=True)
class CoordinationEnvironment:
    metal_idx: int
    donor_indices: Tuple[int, ...]
    coordination_number: int
    metal_atomic_number: int
    metal_formal_charge: int
    donor_atomic_numbers: Tuple[int, ...]
    chelate_groups: Tuple[Tuple[int, ...], ...]


@dataclass(frozen=True)
class CoordinationGeometryCandidate:
    coordinates: np.ndarray
    assigned_geometries: Tuple[str, ...]
    score: Optional[float]


@dataclass(frozen=True)
class CoordinationGeometryResult:
    environments: Tuple[CoordinationEnvironment, ...]
    candidates: Tuple[CoordinationGeometryCandidate, ...]
    diagnostics: Mapping[str, object]


class ForceFieldError(RuntimeError):
    """Base class for force-field workflow failures."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.trajectory: Optional[ForceFieldTrajectoryArchive] = None
        self.ligand_build_attempts: Tuple[ForceFieldTrajectory, ...] = ()


class ForceFieldSetupError(ForceFieldError):
    """Raised when Open Babel cannot initialize a requested force field."""

    def __init__(
        self,
        message: str,
        report: Optional[ForceFieldSetupReport] = None,
    ) -> None:
        super().__init__(message)
        self.report = report


class BuildWorkerError(ForceFieldError):
    """Raised when a generic coordinate-build worker fails."""

    def __init__(
        self,
        error_type: str,
        error_message: str,
        worker_traceback: Optional[str],
        diagnostics: Optional[ComplexBuildDiagnostics] = None,
    ) -> None:
        super().__init__(f"{error_type}: {error_message}")
        self.error_type = error_type
        self.error_message = error_message
        self.worker_traceback = worker_traceback
        self.diagnostics = diagnostics


class BuildTimeoutError(ForceFieldError, TimeoutError):
    """Raised after a generic coordinate-build worker times out."""


class ComplexBuildError(ForceFieldError):
    """Raised when bounded ligand-proxy construction cannot produce a result."""

    def __init__(
        self, message: str, diagnostics: Optional[ComplexBuildDiagnostics] = None
    ) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


class ComplexBuildWorkerError(ComplexBuildError):
    """Raised in the parent process when the proxy-build worker fails."""

    def __init__(
        self,
        error_type: str,
        error_message: str,
        worker_traceback: Optional[str],
        diagnostics: Optional[ComplexBuildDiagnostics] = None,
    ) -> None:
        super().__init__(f"{error_type}: {error_message}", diagnostics)
        self.error_type = error_type
        self.error_message = error_message
        self.worker_traceback = worker_traceback


class ComplexBuildTimeoutError(ComplexBuildError, TimeoutError):
    """Raised after a proxy-build worker exceeds its allotted wall time."""


class ComplexBuildWarning(UserWarning):
    """Warn that complex construction continued with a partial search result."""


class GeometryQualityError(ForceFieldError):
    """Raised when optimization yields no usable finite-topology frame."""

    def __init__(self, report: Optional[ForceFieldValidationReport]) -> None:
        super().__init__(
            "The generated geometry did not pass the requested quality gate"
        )
        self.report = report


class GeometryQualityWarning(UserWarning):
    """Warn that a returned finite-topology frame failed its quality gate."""
