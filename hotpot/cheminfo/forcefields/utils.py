"""Transactional force-field construction and optimization workflows."""

from __future__ import annotations

import multiprocessing as mp
import os
import threading
import time
import traceback as traceback_module
import warnings
from collections import deque
from copy import copy, deepcopy
from dataclasses import asdict, dataclass, field, replace
from functools import wraps
from itertools import combinations
from multiprocessing.connection import Connection, wait as wait_for_connections
from typing import (
    Callable,
    TYPE_CHECKING,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
    cast,
)

import networkx as nx
import numpy as np
from openbabel import openbabel as ob

from .. import geometry as geo
from ..obconvert import extract_obmol_coordinates, mol2obmol, set_obmol_coordinates
from .trajectory import (
    AtomIdentity,
    BondTopology,
    BondTopologyRevision,
    CoordinationFrameEvidence,
    ForceFieldFrame,
    ForceFieldTrajectory,
    ForceFieldTrajectoryArchive,
    FrameEvidence,
    OptimizationFrameEvidence,
    RingFrameEvidence,
    TrajectoryEvent,
    TrajectoryStage,
    TrajectoryStart,
)


if TYPE_CHECKING:
    from ..core import Angle, Atom, AtomPair, Bond, BondKind, Molecule, Ring, Torsion


__all__ = (
    "TrajectoryPath",
    "TrajectoryStart",
    "TrajectoryStage",
    "TrajectoryEvent",
    "AtomIdentity",
    "BondTopology",
    "BondTopologyRevision",
    "RingFrameEvidence",
    "CoordinationFrameEvidence",
    "OptimizationFrameEvidence",
    "FrameEvidence",
    "ForceFieldFrame",
    "ForceFieldTrajectory",
    "ForceFieldTrajectoryArchive",
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
    "AtomTopologySignature",
    "BondTopologySignature",
    "TopologyReference",
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
    "capture_topology",
    "evaluate_structure_acceptance",
    "is_structure_accepted",
    "perturb",
    "collect_coordination_environments",
    "prepare_coordination_geometry",
    "build3d",
    "optimize",
    "build_complex3d",
    "optimize_complex",
    "complexes_build",
    "build_and_optimize",
    "auto_optimize",
)


OptimizationAlgorithm = Literal["steepest", "conjugate"]
TrajectoryPath = Union[str, os.PathLike[str]]
TerminationReason = Literal[
    "converged",
    "budget_exhausted",
    "ring_piercing",
    "quality_gate_failed",
]
AcceptanceLevel = Literal["off", "basic", "standard", "strict"]
ForceFieldStage = Literal["candidate", "final"]
CallableT = TypeVar("CallableT", bound=Callable[..., object])
ForceFieldDiagnosticValue = Union[
    None,
    bool,
    int,
    float,
    str,
    "AtomTopologySignature",
    "BondTopologySignature",
    Tuple["ForceFieldDiagnosticValue", ...],
    Mapping[str, "ForceFieldDiagnosticValue"],
]


class _SeededBuildWorker(Protocol):
    def __call__(
        self,
        mol: "Molecule",
        connection: Connection,
        seed: int,
    ) -> None:
        ...


class _ComplexBuildWorker(Protocol):
    def __call__(
        self,
        mol: "Molecule",
        connection: Connection,
        max_attempts: int,
        candidate_warmup_steps: int,
        candidate_score_steps: int,
        best_candidate_refine_steps: int,
        effective_forcefield: str,
        seed: Optional[int],
        ligand_untangling_attempts: int,
        perturb_sigma: float,
        record_ligand_trajectories: bool,
    ) -> None:
        ...


class _SeedInitializer(Protocol):
    def __call__(self, seed: int) -> None:
        ...

_SUPPORTED_FORCEFIELDS = frozenset({"UFF", "MMFF94", "MMFF94s", "GAFF", "Ghemical"})
_NEUTRAL_DONOR_ATOMIC_NUMBERS = frozenset({7, 8, 15, 16, 33, 34})
_BOND_RING_MAX_SIZE = 16


# Public report and coordination data contracts.


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


class _CoordinationMetrics(TypedDict):
    metal_index: int
    coordination_number: int
    donor_indices: Tuple[int, ...]
    distances: Tuple[float, ...]
    angles: Tuple[float, ...]


@dataclass(frozen=True)
class AtomTopologySignature:
    """Stable identity and chemistry for an atom present before optimization."""

    index: int
    atom_id: int
    atomic_number: int
    formal_charge: int


@dataclass(frozen=True)
class BondTopologySignature:
    """Stable topology for a bond present before optimization."""

    atom_indices: Tuple[int, int]
    bond_order: float
    bond_kind: str


@dataclass(frozen=True)
class TopologyReference:
    """Immutable topology snapshot used by force-field transactions."""

    atoms: Tuple[AtomTopologySignature, ...]
    bonds: Tuple[BondTopologySignature, ...]
    allow_added_hydrogens: bool = True


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
    epoch_energies: Tuple[float, ...] = ()
    epoch_quality_reports: Tuple[ForceFieldValidationReport, ...] = ()
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
    restored_without_forcing: int
    forced_bond_keys: Tuple[Tuple[int, int], ...]
    final_piercing_count: int
    final_undetermined_count: int
    excluded_ring_count: int
    resolved: bool
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


# Public exception hierarchy.


class ForceFieldError(RuntimeError):
    """Base class for force-field workflow failures."""


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


# Internal workflow data contracts.


@dataclass(frozen=True)
class _CandidateOptimizationResult:
    energy: float
    energy_unit: str
    exploded: bool


@dataclass(frozen=True)
class _ObservedFrame:
    coordinates: np.ndarray
    energy: float
    rms_gradient: float
    max_gradient: float
    exploded: bool
    converged: bool
    quality_report: ForceFieldValidationReport
    energy_changes: Tuple[float, ...]
    max_displacements: Tuple[float, ...]


@dataclass(frozen=True)
class _RingUntanglingResult:
    report: RingUntanglingReport
    energy: float


@dataclass(frozen=True)
class _LigandCandidate:
    coordinates: np.ndarray
    energy: float
    attempt: int
    untangling: RingUntanglingReport
    trajectory: Optional[ForceFieldTrajectory] = None


@dataclass(frozen=True)
class _CoordinationRestorationResult:
    report: CoordinationBondRestorationReport


@dataclass(frozen=True)
class _PreparedComplex:
    mol: "Molecule"
    diagnostics: ComplexBuildDiagnostics
    trajectory: ForceFieldTrajectory
    ligand_build_attempts: Tuple[ForceFieldTrajectory, ...] = ()


@dataclass(frozen=True)
class _CoordinationRelationCounts:
    piercing: int
    undetermined: int
    excluded_rings: int


class _BondAttributePayload(TypedDict):
    bond_order: float
    constraint: bool
    id: int
    bond_kind: "BondKind"
    bond_direction: Optional[str]
    bond_source: Optional[str]
    bond_source_metadata: Mapping[str, object]


@dataclass(frozen=True)
class _WorkingCopyCommit:
    original_atom_attrs: Tuple[np.ndarray, ...]
    added_atom_attrs: Tuple[np.ndarray, ...]
    added_bonds: Tuple[Tuple[int, int, _BondAttributePayload], ...]
    conformer_state: Mapping[str, object]
    conformer_index: int


@dataclass(frozen=True)
class _MoleculeCommitSnapshot:
    atoms: Tuple["Atom", ...]
    bonds: Tuple["Bond", ...]
    atom_state: Tuple[
        Tuple["Atom", np.ndarray, list["Atom"], list["Bond"]], ...
    ]
    graph: nx.Graph
    row_to_index: Optional[dict[int, int]]
    angles: list["Angle"]
    torsions: list["Torsion"]
    rings: list["Ring"]
    cycle_basis_rings: list["Ring"]
    ring_indices_cache: dict[
        Tuple[
            bool,
            Optional[int],
            Optional[int],
            Optional[Tuple[Tuple[int, ...], Tuple[Tuple[int, int], ...]]],
        ],
        Tuple[Tuple[int, ...], ...],
    ]
    ligand_rings: Optional[list["Ring"]]
    ligand_cycle_basis_rings: Optional[list["Ring"]]
    ligand_rings_signature: Optional[
        Tuple[Tuple[int, ...], Tuple[Tuple[int, int], ...]]
    ]
    obmol: Optional[ob.OBMol]
    atom_pair_items: Tuple[Tuple[frozenset["Atom"], "AtomPair"], ...]
    conformer_state: Mapping[str, object]
    conformer_index: int


# Diagnostic formatting helpers.


_UNRETURNABLE_FRAME_FAILURES = frozenset({
    "coordinate_shape",
    "finite_coordinates",
})


def _has_unreturnable_frame_failure(
    report: ForceFieldValidationReport,
) -> bool:
    """Return whether a frame cannot safely cross a workflow boundary."""
    return any(
        check.name in _UNRETURNABLE_FRAME_FAILURES
        or check.name == "topology"
        or check.name.startswith("topology_")
        for check in report.failures
    )


def _format_geometry_checks(
    prefix: str,
    checks: Tuple[AcceptanceCheck, ...],
) -> str:
    """Render failed geometry checks without discarding measured evidence."""
    details = "; ".join(
        f"{check.name}(measured={check.measured!r}, "
        f"threshold={check.threshold!r}, "
        f"atom_indices={check.atom_indices!r}, "
        f"bond_indices={check.bond_indices!r})"
        for check in checks
    )
    return f"{prefix}: {details}"


# Shared molecular value and identity helpers.


def _copy_coordinates(coordinates: np.ndarray) -> np.ndarray:
    """Return an independent floating-point Cartesian-coordinate array."""
    return np.asarray(coordinates, dtype=float).copy()


def _atom_index_map(atoms: Sequence["Atom"]) -> dict[int, int]:
    """Map each atom object's identity to its molecular atom-table position."""
    return {id(atom): index for index, atom in enumerate(atoms)}


def _bond_endpoint_indices(
    bond: "Bond",
    atom_indices: Mapping[int, int],
) -> Tuple[int, int]:
    """Return the atom-table positions of a bond's ordered atoms."""
    return atom_indices[id(bond.atom1)], atom_indices[id(bond.atom2)]


def _atom_identity(atom: "Atom") -> Tuple[int, int, int]:
    """Return the stable chemical identity used by topology transactions."""
    return int(atom.id), int(atom.atomic_number), int(atom.formal_charge)


def _bond_identity(
    bond: "Bond",
    atom_indices: Mapping[int, int],
) -> Tuple[Tuple[int, int], float, str]:
    """Return orientation-independent endpoints, order, and kind for a bond."""
    first, second = sorted(_bond_endpoint_indices(bond, atom_indices))
    endpoints = first, second
    return endpoints, float(bond.bond_order), bond.bond_kind.value


def _piercing_count(
    report: Optional["geo.BondRingScanReport[Ring, Bond]"],
) -> int:
    """Return the confirmed piercing count of an optional geometry scan."""
    return 0 if report is None else len(report.piercings)


def _unique_messages(messages: Sequence[str]) -> Tuple[str, ...]:
    """Deduplicate messages while preserving their first-seen order."""
    return tuple(dict.fromkeys(messages))


def _iter_metal_donor_pairs(
    mol: "Molecule",
) -> Iterator[Tuple["Atom", "Atom"]]:
    """Yield the metal and donor atoms of each explicit coordination bond."""
    for bond in mol.bonds:
        if not bond.is_metal_ligand_bond:
            continue
        if bond.atom1.is_metal:
            yield bond.atom1, bond.atom2
        else:
            yield bond.atom2, bond.atom1


# Structure-acceptance policy helpers.  Geometry supplies measurements and
# relation states; this module owns every chemical threshold and pass/fail
# decision made from those facts.


@dataclass(frozen=True)
class _AtomPairAcceptanceIssue:
    kind: Literal["overlap", "too_close"]
    atom_indices: Tuple[int, int]
    distance: float
    threshold: float


def _bond_key(bond: "Bond") -> Tuple[int, int]:
    first, second = sorted((int(bond.atom1.idx), int(bond.atom2.idx)))
    return first, second


def _overlap_issues(
    distances: Sequence["geo.AtomPairDistance[Atom]"],
    tolerance: float,
) -> Tuple[_AtomPairAcceptanceIssue, ...]:
    return tuple(
        _AtomPairAcceptanceIssue(
            kind="overlap",
            atom_indices=(distance.target.first.key, distance.target.second.key),
            distance=float(distance.measurement.distance),
            threshold=float(tolerance),
        )
        for distance in distances
        if distance.measurement.distance <= tolerance
    )


def _too_close_issues(
    distances: Sequence["geo.AtomPairDistance[Atom]"],
    *,
    minimum_distance: float,
    covalent_radius_scale: Optional[float],
    pair_scope: geo.PairScope,
    include_overlaps: bool,
    overlap_tolerance: float,
) -> Tuple[_AtomPairAcceptanceIssue, ...]:
    issues = []
    for distance in distances:
        if pair_scope == "bonded" and not distance.target.bonded:
            continue
        if pair_scope == "nonbonded" and distance.target.bonded:
            continue
        measured = float(distance.measurement.distance)
        if not include_overlaps and measured <= overlap_tolerance:
            continue
        threshold = minimum_distance
        if covalent_radius_scale is not None:
            threshold = max(
                threshold,
                covalent_radius_scale * (
                    float(distance.target.first.atom.covalent_radius)
                    + float(distance.target.second.atom.covalent_radius)
                ),
            )
        if measured < threshold:
            issues.append(_AtomPairAcceptanceIssue(
                kind="too_close",
                atom_indices=(
                    distance.target.first.key,
                    distance.target.second.key,
                ),
                distance=measured,
                threshold=float(threshold),
            ))
    return tuple(issues)


def _topology_bond_signature(
    bond: "Bond",
    atom_indices: Mapping[int, int],
) -> BondTopologySignature:
    endpoints, bond_order, bond_kind = _bond_identity(bond, atom_indices)
    return BondTopologySignature(
        atom_indices=endpoints,
        bond_order=bond_order,
        bond_kind=bond_kind,
    )


def _topology_checks(
    mol: "Molecule",
    reference: TopologyReference,
) -> Tuple[AcceptanceCheck, ...]:
    atoms = tuple(mol.atoms)
    checks = []
    original_count = len(reference.atoms)

    if len(atoms) < original_count:
        return (AcceptanceCheck(
            name="topology_atom_count",
            passed=False,
            measured=len(atoms),
            threshold=f">={original_count}",
            message="Original atoms were removed",
        ),)

    for signature, atom in zip(reference.atoms, atoms[:original_count]):
        measured = _atom_identity(atom)
        expected = (
            signature.atom_id,
            signature.atomic_number,
            signature.formal_charge,
        )
        if measured != expected:
            checks.append(AcceptanceCheck(
                name="topology_atom_identity",
                passed=False,
                measured=measured,
                threshold=expected,
                atom_indices=(signature.index,),
                message="An original atom identity or formal charge changed",
            ))

    added_indices = set(range(original_count, len(atoms)))
    if added_indices and not reference.allow_added_hydrogens:
        checks.append(AcceptanceCheck(
            name="topology_added_atoms",
            passed=False,
            measured=len(added_indices),
            threshold=0,
            atom_indices=tuple(sorted(added_indices)),
            message="Additional atoms are not allowed by this topology reference",
        ))
    elif added_indices:
        non_hydrogens = tuple(
            index
            for index in added_indices
            if int(atoms[index].atomic_number) != 1
        )
        if non_hydrogens:
            checks.append(AcceptanceCheck(
                name="topology_added_atoms",
                passed=False,
                measured=tuple(
                    int(atoms[index].atomic_number)
                    for index in non_hydrogens
                ),
                threshold="hydrogen only",
                atom_indices=non_hydrogens,
                message="Only hydrogen atoms may be added during preparation",
            ))

    atom_indices = _atom_index_map(atoms)
    candidate_bonds = {
        signature.atom_indices: signature
        for signature in (
            _topology_bond_signature(bond, atom_indices) for bond in mol.bonds
        )
    }
    reference_bonds = {
        signature.atom_indices: signature for signature in reference.bonds
    }

    for endpoints, expected in reference_bonds.items():
        measured = candidate_bonds.get(endpoints)
        if measured != expected:
            checks.append(AcceptanceCheck(
                name="topology_original_bond",
                passed=False,
                measured=measured,
                threshold=expected,
                atom_indices=endpoints,
                message="An original bond was removed or changed",
            ))

    added_bonds = set(candidate_bonds).difference(reference_bonds)
    invalid_added_bonds = tuple(sorted(
        endpoints
        for endpoints in added_bonds
        if not reference.allow_added_hydrogens
        or sum(endpoint in added_indices for endpoint in endpoints) != 1
    ))
    for endpoints in invalid_added_bonds:
        checks.append(AcceptanceCheck(
            name="topology_added_bond",
            passed=False,
            measured=endpoints,
            threshold="one added H endpoint",
            atom_indices=endpoints,
            message="Only new X-H bonds may be added during preparation",
        ))

    if reference.allow_added_hydrogens:
        degree = {index: 0 for index in added_indices}
        for endpoints in added_bonds:
            for endpoint in endpoints:
                if endpoint in degree:
                    degree[endpoint] += 1
        invalid_hydrogens = tuple(
            index for index, count in sorted(degree.items()) if count != 1
        )
        if invalid_hydrogens:
            checks.append(AcceptanceCheck(
                name="topology_added_hydrogen_degree",
                passed=False,
                measured=tuple(
                    degree[index] for index in invalid_hydrogens
                ),
                threshold=1,
                atom_indices=invalid_hydrogens,
                message="Each added hydrogen must have exactly one new bond",
            ))

    if not checks:
        checks.append(AcceptanceCheck(
            name="topology",
            passed=True,
            measured=(len(atoms), len(candidate_bonds)),
            threshold=(original_count, len(reference_bonds)),
            message="Original topology is preserved",
        ))
    return tuple(checks)


def _resolve_acceptance_thresholds(
    thresholds: Optional[StructureAcceptanceThresholds],
) -> StructureAcceptanceThresholds:
    return thresholds if thresholds is not None else StructureAcceptanceThresholds()


def _forcefield_acceptance_checks(
    report: Optional[ForceFieldAcceptanceEvidence],
    level: AcceptanceLevel,
    thresholds: StructureAcceptanceThresholds,
    stage: ForceFieldStage,
) -> Tuple[AcceptanceCheck, ...]:
    if report is None:
        if level == "strict":
            return (AcceptanceCheck(
                name="forcefield_report",
                passed=False,
                measured=None,
                threshold="complete force-field report",
                message="Strict structure validation requires force-field diagnostics",
            ),)
        return ()

    checks = []
    setup_succeeded = report.get("setup_succeeded")
    checks.append(AcceptanceCheck(
        name="forcefield_setup",
        passed=setup_succeeded is not None and bool(setup_succeeded),
        measured=setup_succeeded,
        threshold=True,
        message="Force-field setup must succeed",
    ))

    required_finite_fields = ["final_energy"]
    if stage == "final":
        required_finite_fields.extend(("rms_gradient", "max_gradient"))
    for field_name in required_finite_fields:
        value = report.get(field_name)
        finite = value is not None and bool(np.isfinite(value))
        checks.append(AcceptanceCheck(
            name=f"finite_{field_name}",
            passed=finite,
            measured=None if value is None else float(value),
            threshold="finite",
            message=f"{field_name.replace('_', ' ')} must be finite",
        ))

    if level in ("basic", "standard", "strict"):
        exploded = report.get("exploded")
        checks.append(AcceptanceCheck(
            name="backend_explosion",
            passed=exploded is not None and not bool(exploded),
            measured=exploded,
            threshold=False,
            message="The force-field backend must report a non-exploded structure",
        ))

    if stage == "final" and level in ("standard", "strict"):
        converged = report.get("converged")
        if converged is not None or level == "strict":
            checks.append(AcceptanceCheck(
                name="forcefield_convergence",
                passed=converged is not None and bool(converged),
                severity="error" if level == "strict" else "warning",
                measured=converged,
                threshold=True,
                message="The force-field backend did not report convergence",
            ))

    if stage == "final" and level == "strict":
        gradient_limits = (
            ("rms_gradient", thresholds.strict_rms_gradient),
            ("max_gradient", thresholds.strict_max_gradient),
        )
        for field_name, limit in gradient_limits:
            value = report.get(field_name)
            if value is not None and np.isfinite(value):
                checks.append(AcceptanceCheck(
                    name=field_name,
                    passed=float(value) <= limit,
                    measured=float(value),
                    threshold=limit,
                    message=(
                        f"{field_name.replace('_', ' ')} exceeds the strict limit"
                    ),
                ))

        segment_epochs_completed = report.get("segment_epochs_completed")
        converged = bool(report.get("converged"))
        no_history_required = (
            segment_epochs_completed is not None
            and int(segment_epochs_completed) == 1
            and converged
        )
        stability_checks = (
            ("energy_changes", thresholds.strict_energy_change),
            ("max_displacements", thresholds.strict_max_displacement),
        )
        stability_observations = []
        for field_name, limit in stability_checks:
            history = report.get(field_name)
            values = () if history is None else tuple(history)
            recent = values[-thresholds.strict_stability_window:]
            value = max(recent) if recent else None
            stability_observations.append(len(recent))
            checks.append(AcceptanceCheck(
                name=field_name.removesuffix("s"),
                passed=no_history_required or (
                    value is not None
                    and np.isfinite(value)
                    and float(value) <= limit
                ),
                measured=None if value is None else float(value),
                threshold=limit,
                message=(
                    f"{field_name.replace('_', ' ')} do not satisfy the strict limit"
                ),
            ))

        observations = min(stability_observations)
        if segment_epochs_completed is None:
            epochs_completed = report.get("epochs_completed")
            required_observations = (
                min(thresholds.strict_stability_window, int(epochs_completed))
                if epochs_completed is not None
                else thresholds.strict_stability_window
            )
        else:
            required_observations = min(
                thresholds.strict_stability_window,
                max(int(segment_epochs_completed) - 1, 0),
            )
        checks.append(AcceptanceCheck(
            name="stability_observations",
            passed=(
                no_history_required
                or required_observations > 0
                and observations >= required_observations
            ),
            measured=observations,
            threshold=required_observations,
            message="Strict validation requires a stable multi-epoch history",
        ))
    return tuple(checks)


def _bond_position_data(
    mol: "Molecule",
    atoms: Sequence["Atom"],
) -> Iterator[Tuple[int, "Bond", int, int]]:
    atom_indices = _atom_index_map(atoms)
    for bond_index, bond in enumerate(mol.bonds):
        first, second = _bond_endpoint_indices(bond, atom_indices)
        yield (
            bond_index,
            bond,
            first,
            second,
        )


def _coordination_metrics(
    mol: "Molecule",
    atoms: Sequence["Atom"],
    coordinates: np.ndarray,
) -> Tuple[_CoordinationMetrics, ...]:
    atom_indices = _atom_index_map(atoms)
    donors = {index: [] for index, atom in enumerate(atoms) if atom.is_metal}
    for metal, donor in _iter_metal_donor_pairs(mol):
        donors[atom_indices[id(metal)]].append(atom_indices[id(donor)])

    environments = []
    for metal, donor_indices in sorted(donors.items()):
        donor_indices = sorted(donor_indices)
        vectors = [
            coordinates[index] - coordinates[metal]
            for index in donor_indices
        ]
        distances = [float(np.linalg.norm(vector)) for vector in vectors]
        angles = []
        for first, second in combinations(vectors, 2):
            denominator = np.linalg.norm(first) * np.linalg.norm(second)
            if denominator > 0.0:
                cosine = np.clip(
                    np.dot(first, second) / denominator,
                    -1.0,
                    1.0,
                )
                angles.append(float(np.degrees(np.arccos(cosine))))
        environments.append({
            "metal_index": int(atoms[metal].idx),
            "coordination_number": len(donor_indices),
            "donor_indices": tuple(
                int(atoms[index].idx) for index in donor_indices
            ),
            "distances": tuple(distances),
            "angles": tuple(angles),
        })
    return tuple(environments)


def _bond_ring_acceptance_checks(
    mol: "Molecule",
    report: "geo.BondRingScanReport[Ring, Bond]",
) -> Tuple[AcceptanceCheck, ...]:
    bond_positions = {
        _bond_key(candidate): index
        for index, candidate in enumerate(mol.bonds)
    }
    checks = []
    for finding in report.piercings:
        bond_key = finding.target.bond.key
        checks.append(AcceptanceCheck(
            name="bond_ring_piercing",
            passed=False,
            measured=finding.target.ring.key,
            threshold=geo.PiercingState.DOES_NOT_PIERCE.value,
            atom_indices=bond_key,
            bond_indices=(bond_positions[bond_key],),
            message="A finite bond segment pierces a selected ring surface",
        ))
    for finding in report.undetermined:
        bond_key = finding.target.bond.key
        checks.append(AcceptanceCheck(
            name="bond_ring_piercing",
            passed=False,
            severity="warning",
            measured=tuple(
                sorted(cause.value for cause in finding.relation.indeterminacy_causes)
            ),
            threshold=geo.PiercingState.DOES_NOT_PIERCE.value,
            atom_indices=bond_key,
            bond_indices=(bond_positions[bond_key],),
            message="The bond-ring spatial relation is mathematically undetermined",
        ))
    if report.excluded_ring_count:
        checks.append(AcceptanceCheck(
            name="bond_ring_scope_coverage",
            passed=False,
            severity="warning",
            measured=report.excluded_ring_count,
            threshold=0,
            message=(
                "Some rings exceed the configured maximum size and were not "
                "evaluated for bond-ring piercing"
            ),
        ))
    if not checks:
        checks.append(AcceptanceCheck(
            name="bond_ring_piercing",
            passed=True,
            measured=geo.PiercingState.DOES_NOT_PIERCE.value,
            threshold=geo.PiercingState.DOES_NOT_PIERCE.value,
        ))
    return tuple(checks)


def _coordinate_acceptance_section(
    mol: "Molecule",
    atoms: Sequence["Atom"],
    coordinates: np.ndarray,
) -> Tuple[
    Tuple[AcceptanceCheck, ...],
    bool,
    dict[str, ForceFieldDiagnosticValue],
]:
    """Return coordinate checks, finiteness, and base structure metrics."""
    expected_shape = (len(atoms), 3)
    shape_ok = coordinates.shape == expected_shape
    checks = [AcceptanceCheck(
        name="coordinate_shape",
        passed=shape_ok,
        measured=tuple(coordinates.shape),
        threshold=expected_shape,
        message="Coordinates must contain one Cartesian row per atom",
    )]

    finite_ok = shape_ok and bool(np.all(np.isfinite(coordinates)))
    nonfinite_indices = ()
    if shape_ok and not finite_ok:
        nonfinite_indices = tuple(
            int(atoms[index].idx)
            for index in np.flatnonzero(
                ~np.all(np.isfinite(coordinates), axis=1)
            )
        )
    checks.append(AcceptanceCheck(
        name="finite_coordinates",
        passed=finite_ok,
        measured=finite_ok,
        threshold=True,
        atom_indices=nonfinite_indices,
        message="All Cartesian coordinates must be finite",
    ))
    metrics: dict[str, ForceFieldDiagnosticValue] = {
        "atom_count": len(atoms),
        "bond_count": len(mol.bonds),
    }
    return tuple(checks), finite_ok, metrics


def _atom_pair_distance_acceptance_section(
    mol: "Molecule",
    level: AcceptanceLevel,
    limits: StructureAcceptanceThresholds,
) -> Tuple[
    Tuple[AcceptanceCheck, ...],
    dict[str, ForceFieldDiagnosticValue],
]:
    """Return atom-pair distance checks and distance metrics."""
    distances = geo.measure_atom_pair_distances(mol, "all")
    metrics: dict[str, ForceFieldDiagnosticValue] = {}
    if distances:
        metrics["minimum_pair_distance"] = min(
            float(distance.measurement.distance) for distance in distances
        )

    if level == "off":
        return (), metrics

    checks = []
    overlaps = _overlap_issues(distances, limits.overlap_tolerance)
    if overlaps:
        checks.extend(AcceptanceCheck(
            name="atom_overlap",
            passed=False,
            measured=issue.distance,
            threshold=issue.threshold,
            atom_indices=issue.atom_indices,
            message="Two atoms occupy indistinguishable coordinates",
        ) for issue in overlaps)
    else:
        checks.append(AcceptanceCheck(
            name="atom_overlap",
            passed=True,
            measured=0,
            threshold=limits.overlap_tolerance,
        ))

    close_pairs_by_atoms = {
        issue.atom_indices: issue
        for issue in _too_close_issues(
            distances,
            minimum_distance=limits.basic_minimum_distance,
            covalent_radius_scale=None,
            pair_scope="all",
            include_overlaps=False,
            overlap_tolerance=limits.overlap_tolerance,
        )
    }
    if level in ("standard", "strict"):
        close_pairs_by_atoms.update(
            (issue.atom_indices, issue)
            for issue in _too_close_issues(
                distances,
                minimum_distance=limits.standard_minimum_distance,
                covalent_radius_scale=limits.standard_covalent_radius_scale,
                pair_scope="nonbonded",
                include_overlaps=False,
                overlap_tolerance=limits.overlap_tolerance,
            )
        )
    close_pairs = tuple(
        close_pairs_by_atoms[key] for key in sorted(close_pairs_by_atoms)
    )
    if close_pairs:
        checks.extend(AcceptanceCheck(
            name="atom_too_close",
            passed=False,
            measured=issue.distance,
            threshold=issue.threshold,
            atom_indices=issue.atom_indices,
            message="An atom pair is closer than the allowed separation",
        ) for issue in close_pairs)
    else:
        checks.append(AcceptanceCheck(
            name="atom_too_close",
            passed=True,
            measured=0,
            threshold=(
                limits.basic_minimum_distance
                if level == "basic"
                else (
                    limits.basic_minimum_distance,
                    limits.standard_minimum_distance,
                    limits.standard_covalent_radius_scale,
                )
            ),
        ))
    return tuple(checks), metrics


def _bond_geometry_acceptance_section(
    mol: "Molecule",
    atoms: Sequence["Atom"],
    coordinates: np.ndarray,
    level: AcceptanceLevel,
    limits: StructureAcceptanceThresholds,
) -> Tuple[
    Tuple[AcceptanceCheck, ...],
    dict[str, ForceFieldDiagnosticValue],
]:
    """Return explicit-bond geometry checks and bond-length metrics."""
    checks = []
    maximum_bond_length = 0.0
    short_bond_count = 0
    for bond_index, bond, first, second in _bond_position_data(mol, atoms):
        distance = float(np.linalg.norm(
            coordinates[first] - coordinates[second]
        ))
        maximum_bond_length = max(maximum_bond_length, distance)
        atom_indices = (
            int(atoms[first].idx),
            int(atoms[second].idx),
        )
        valid_length = 0.0 < distance <= limits.maximum_bond_distance
        if not valid_length:
            checks.append(AcceptanceCheck(
                name="bond_distance",
                passed=False,
                measured=distance,
                threshold=(0.0, limits.maximum_bond_distance),
                atom_indices=atom_indices,
                bond_indices=(bond_index,),
                message="An explicit bond has an invalid or exploded length",
            ))

        if level in ("standard", "strict"):
            radius_sum = (
                float(atoms[first].covalent_radius)
                + float(atoms[second].covalent_radius)
            )
            if radius_sum > 0.0:
                ratio = distance / radius_sum
                ratio_limits = (
                    limits.metal_ligand_bond_ratio
                    if bond.is_metal_ligand_bond
                    else limits.covalent_bond_ratio
                )
                if ratio < ratio_limits[0]:
                    short_bond_count += 1
                    checks.append(AcceptanceCheck(
                        name="short_bond",
                        passed=False,
                        measured=distance,
                        threshold=ratio_limits[0] * radius_sum,
                        atom_indices=atom_indices,
                        bond_indices=(bond_index,),
                        message=(
                            "An explicit bond is shorter than its "
                            "radius-scaled limit"
                        ),
                    ))
                if not ratio_limits[0] <= ratio <= ratio_limits[1]:
                    checks.append(AcceptanceCheck(
                        name="bond_length_ratio",
                        passed=False,
                        measured=ratio,
                        threshold=ratio_limits,
                        atom_indices=atom_indices,
                        bond_indices=(bond_index,),
                        message=(
                            "Bond length is inconsistent with covalent radii"
                        ),
                    ))
    metrics: dict[str, ForceFieldDiagnosticValue] = {
        "maximum_bond_length": maximum_bond_length,
    }
    if not any(check.name == "bond_distance" for check in checks):
        checks.append(AcceptanceCheck(
            name="bond_distance",
            passed=True,
            measured=maximum_bond_length,
            threshold=(0.0, limits.maximum_bond_distance),
        ))
    if level in ("standard", "strict") and not any(
        check.name == "bond_length_ratio" for check in checks
    ):
        checks.append(AcceptanceCheck(
            name="bond_length_ratio",
            passed=True,
            measured=None,
            threshold=(
                limits.covalent_bond_ratio,
                limits.metal_ligand_bond_ratio,
            ),
        ))
    if level in ("standard", "strict") and short_bond_count == 0:
        checks.append(AcceptanceCheck(
            name="short_bond",
            passed=True,
            measured=0,
            threshold=(
                limits.covalent_bond_ratio[0],
                limits.metal_ligand_bond_ratio[0],
            ),
            message="No explicit bond is below its radius-scaled limit",
        ))
    return tuple(checks), metrics


def _bond_ring_coordination_acceptance_section(
    mol: "Molecule",
    atoms: Sequence["Atom"],
    coordinates: np.ndarray,
) -> Tuple[
    Tuple[AcceptanceCheck, ...],
    dict[str, ForceFieldDiagnosticValue],
]:
    """Return bond-ring checks and coordination-environment metrics."""
    bond_ring_report = geo.scan_bond_ring_relations(
        mol,
        ring_scope="ligand_skeleton",
        max_ring_size=_BOND_RING_MAX_SIZE,
    )
    metrics: dict[str, ForceFieldDiagnosticValue] = {
        "bond_ring_piercing_count": bond_ring_report.piercing_pair_count,
        "bond_ring_undetermined_count": bond_ring_report.undetermined_pair_count,
        "bond_ring_scan_complete": bond_ring_report.scan_complete,
        "bond_ring_selected_ring_count": bond_ring_report.selected_ring_count,
        "bond_ring_excluded_ring_count": bond_ring_report.excluded_ring_count,
        "bond_ring_max_ring_size": bond_ring_report.max_ring_size,
        "bond_ring_scope": bond_ring_report.ring_scope,
        "coordination_environments": _coordination_metrics(
            mol,
            atoms,
            coordinates,
        ),
    }
    return _bond_ring_acceptance_checks(mol, bond_ring_report), metrics


def _select_ring_opening_edge(
    mol: "Molecule",
    ring: "Ring",
    bond: "Bond",
    *,
    ring_scope: geo.RingScope = "ligand_skeleton",
) -> Optional["Bond"]:
    """Choose the nearest single edge not shared by another ring in scope.

    Ring detection may be size-limited, but fused-edge membership must not be:
    an edge in the detected ring can also belong to a larger ring.
    The legacy cycle-basis views are deliberately outside this workflow.
    """
    ring_memberships = {}
    for candidate_ring in mol.rings_for_scope(ring_scope):
        for edge in candidate_ring.bonds:
            key = _bond_key(edge)
            ring_memberships[key] = ring_memberships.get(key, 0) + 1

    eligible_edges = tuple(
        edge
        for edge in ring.bonds
        if float(edge.bond_order) == 1.0
        and ring_memberships.get(_bond_key(edge), 0) == 1
    )
    if not eligible_edges:
        return None

    target_segment = geo.segment_from_bond(bond)

    def edge_distance(edge: "Bond") -> Tuple[float, Tuple[int, int]]:
        return (
            geo.segment_segment_distance(
                geo.segment_from_bond(edge),
                target_segment,
            ),
            _bond_key(edge),
        )

    return min(eligible_edges, key=edge_distance)


def _scan_confirmed_ring_piercings(
    mol: "Molecule",
    *,
    ring_scope: geo.RingScope,
) -> Tuple[
    geo.PiercingState,
    Optional["geo.BondRingScanReport[Ring, Bond]"],
]:
    """Return a dense report only when a confirmed piercing needs repair."""
    state = geo.determine_bond_ring_piercing_state(
        mol,
        ring_scope=ring_scope,
        max_ring_size=_BOND_RING_MAX_SIZE,
    )
    if state is not geo.PiercingState.PIERCES:
        return state, None
    return state, geo.scan_bond_ring_relations(
        mol,
        ring_scope=ring_scope,
        max_ring_size=_BOND_RING_MAX_SIZE,
    )


def _first_openable_ring_edge(
    mol: "Molecule",
    report: "geo.BondRingScanReport[Ring, Bond]",
) -> Optional["Bond"]:
    """Choose one deterministic ring edge for the next repair attempt."""
    for finding in report.piercings:
        ring_edge = _select_ring_opening_edge(
            mol,
            finding.target.ring.ring,
            finding.target.bond.bond,
            ring_scope=report.ring_scope,
        )
        if ring_edge is not None:
            return ring_edge
    return None


def _ring_frame_evidence(
    state: geo.PiercingState,
    report: Optional["geo.BondRingScanReport[Ring, Bond]"],
    *,
    confirmed_piercing_count: Optional[int] = None,
) -> RingFrameEvidence:
    """Describe one observed ring state without triggering another scan."""
    if report is None:
        uncertain_relation_count = (
            None if state is geo.PiercingState.UNDETERMINED else 0
        )
    else:
        uncertain_relation_count = len(report.undetermined)
    return RingFrameEvidence(
        confirmed_piercing_count=(
            _piercing_count(report)
            if confirmed_piercing_count is None
            else confirmed_piercing_count
        ),
        uncertain_relation_count=uncertain_relation_count,
    )


def _untangle_ring_piercings(
    mol: "Molecule",
    effective_forcefield: str,
    *,
    attempt_limit: int,
    short_steps: int,
    settling_steps: int,
    perturb_sigma: float,
    rng: np.random.Generator,
    ring_scope: geo.RingScope = "ligand_skeleton",
    initial_energy: float = float("nan"),
    trajectory: Optional[ForceFieldTrajectory] = None,
    trajectory_stage: TrajectoryStage = TrajectoryStage.COMPLEX_UNTANGLING,
) -> _RingUntanglingResult:
    """Repair confirmed ring piercing without rebuilding the molecular graph.

    One covalent ring edge is opened per attempt.  The open structure is
    perturbed and relaxed, then the exact bond object is restored before the
    next geometric observation.  A shared trajectory records both the open
    and closed topology revisions without taking over workflow control.
    """
    records_trajectory = (
        trajectory is not None
        and trajectory.records(trajectory_stage)
    )

    def record_ring_frame(
        event: TrajectoryEvent,
        *,
        energy: Optional[float] = None,
        state: Optional[geo.PiercingState] = None,
        report: Optional["geo.BondRingScanReport[Ring, Bond]"] = None,
        confirmed_piercing_count: Optional[int] = None,
        attempt: Optional[int] = None,
    ) -> Optional[int]:
        if not records_trajectory or trajectory is None:
            return None
        frame = trajectory.record_molecule(
            mol,
            stage=trajectory_stage,
            event=event,
            energy_kj_mol=energy,
            attempt=attempt,
            evidence=(
                None
                if state is None
                else _ring_frame_evidence(
                    state,
                    report,
                    confirmed_piercing_count=confirmed_piercing_count,
                )
            ),
        )
        return frame.index

    state, report = _scan_confirmed_ring_piercings(
        mol,
        ring_scope=ring_scope,
    )
    initial_count = _piercing_count(report)
    current_count = initial_count
    minimum_count = initial_count
    best_coordinates = _copy_coordinates(mol.coordinates)
    best_energy = float(initial_energy)
    best_trace_energy = (
        best_energy if np.isfinite(best_energy) else None
    )
    warning_messages = []
    attempts_completed = 0
    settled = False
    unresolved_reason: Optional[str] = None
    record_ring_frame(
        TrajectoryEvent.INITIAL,
        energy=best_trace_energy,
        state=state,
        report=report,
        attempt=0,
    )

    while True:
        if state is not geo.PiercingState.PIERCES:
            if state is geo.PiercingState.UNDETERMINED:
                warning_messages.append(
                    "A bond-ring relation remained mathematically undetermined"
                )
            if settled or settling_steps == 0:
                break
            optimized = _single_ob_optimization(
                mol,
                effective_forcefield,
                settling_steps,
            )
            settled = True
            state, report = _scan_confirmed_ring_piercings(
                mol,
                ring_scope=ring_scope,
            )
            current_count = _piercing_count(report)
            if current_count <= minimum_count:
                minimum_count = current_count
                best_coordinates = _copy_coordinates(mol.coordinates)
                best_energy = float(optimized.energy)
                best_trace_energy = float(optimized.energy)
            record_ring_frame(
                TrajectoryEvent.SETTLED,
                energy=float(optimized.energy),
                state=state,
                report=report,
                attempt=attempts_completed,
            )
            continue

        if attempts_completed >= attempt_limit:
            unresolved_reason = (
                f"Confirmed bond-ring piercing remains after {attempt_limit} "
                "untangling attempts; retaining the closed-topology frame with "
                "the lowest piercing count"
            )
            break

        if report is None:
            raise RuntimeError("A confirmed piercing requires a dense geometry report")
        ring_edge = _first_openable_ring_edge(mol, report)
        if ring_edge is None:
            unresolved_reason = (
                "Confirmed bond-ring piercing has no eligible single ring edge; "
                "retaining the closed-topology frame with the lowest piercing count"
            )
            break

        attempts_completed += 1
        mol.hide_bonds(ring_edge, clear_conformers=False)
        record_ring_frame(
            TrajectoryEvent.RING_OPENED,
            attempt=attempts_completed,
        )
        optimized_successfully = False
        try:
            mol.coordinates = _perturbed_coordinates(
                mol.coordinates,
                sigma=perturb_sigma,
                rng=rng,
            )
            record_ring_frame(
                TrajectoryEvent.PERTURBED,
                attempt=attempts_completed,
            )
            optimized = _single_ob_optimization(
                mol,
                effective_forcefield,
                short_steps,
            )
            optimized_successfully = True
            record_ring_frame(
                TrajectoryEvent.OPTIMIZED,
                energy=float(optimized.energy),
                attempt=attempts_completed,
            )
        finally:
            mol.restore_bonds(ring_edge, clear_conformers=False)
            if not optimized_successfully:
                record_ring_frame(
                    TrajectoryEvent.RING_CLOSED,
                    attempt=attempts_completed,
                )

        state, report = _scan_confirmed_ring_piercings(
            mol,
            ring_scope=ring_scope,
        )
        current_count = _piercing_count(report)
        record_ring_frame(
            TrajectoryEvent.RING_CLOSED,
            state=state,
            report=report,
            attempt=attempts_completed,
        )
        if current_count <= minimum_count:
            minimum_count = current_count
            best_coordinates = _copy_coordinates(mol.coordinates)
            best_energy = float("nan")
            best_trace_energy = None
        settled = False

    if unresolved_reason is not None:
        mol.coordinates = best_coordinates
        current_count = minimum_count
        record_ring_frame(
            TrajectoryEvent.ROLLED_BACK,
            energy=best_trace_energy,
            state=state,
            confirmed_piercing_count=current_count,
            attempt=attempts_completed,
        )
        if settling_steps:
            retained_coordinates = best_coordinates.copy()
            retained_energy = best_energy
            retained_trace_energy = best_trace_energy
            retained_count = minimum_count
            optimized = _single_ob_optimization(
                mol,
                effective_forcefield,
                settling_steps,
            )
            settled_state, settled_report = _scan_confirmed_ring_piercings(
                mol,
                ring_scope=ring_scope,
            )
            settled_count = _piercing_count(settled_report)
            record_ring_frame(
                TrajectoryEvent.SETTLED,
                energy=float(optimized.energy),
                state=settled_state,
                report=settled_report,
                attempt=attempts_completed,
            )
            if settled_count <= retained_count:
                state = settled_state
                report = settled_report
                current_count = settled_count
                minimum_count = settled_count
                best_coordinates = _copy_coordinates(mol.coordinates)
                best_energy = float(optimized.energy)
                best_trace_energy = float(optimized.energy)
            else:
                mol.coordinates = retained_coordinates
                state = geo.PiercingState.PIERCES
                current_count = retained_count
                best_energy = retained_energy
                best_trace_energy = retained_trace_energy
                record_ring_frame(
                    TrajectoryEvent.ROLLED_BACK,
                    energy=best_trace_energy,
                    state=state,
                    confirmed_piercing_count=retained_count,
                    attempt=attempts_completed,
                )
        if state is geo.PiercingState.PIERCES:
            warning_messages.append(unresolved_reason)
        elif state is geo.PiercingState.UNDETERMINED:
            warning_messages.append(
                "A bond-ring relation remained mathematically undetermined"
            )
    resolved = current_count == 0
    terminal_index = record_ring_frame(
        TrajectoryEvent.TERMINAL,
        energy=best_trace_energy,
        state=state,
        report=report,
        confirmed_piercing_count=current_count,
        attempt=attempts_completed,
    )
    if terminal_index is not None and trajectory is not None:
        trajectory.select(terminal_index)

    return _RingUntanglingResult(
        report=RingUntanglingReport(
            attempt_limit=attempt_limit,
            attempts_completed=attempts_completed,
            initial_piercing_count=initial_count,
            final_piercing_count=current_count,
            minimum_piercing_count=minimum_count,
            resolved=resolved,
            warning_messages=_unique_messages(warning_messages),
        ),
        energy=best_energy,
    )


def _is_coordination_cycle_closure(
    finding: "geo.BondRingFinding[Ring, Bond]",
    coordination_bond: "Bond",
) -> bool:
    """Identify a geometric finding created only by closing a chelate cycle."""
    candidate_key = _bond_key(coordination_bond)
    return (
        finding.target.bond.key == candidate_key
        and set(candidate_key).issubset(finding.target.ring.key)
    )


def _scan_full_graph_bond_ring_relations(
    mol: "Molecule",
) -> "geo.BondRingScanReport[Ring, Bond]":
    return geo.scan_bond_ring_relations(
        mol,
        ring_scope="full_graph",
        max_ring_size=_BOND_RING_MAX_SIZE,
    )


def _bond_ring_finding_key(
    finding: "geo.BondRingFinding[Ring, Bond]",
) -> Tuple[Tuple[int, ...], Tuple[int, int]]:
    return finding.target.ring.key, finding.target.bond.key


def _candidate_coordination_relation_counts(
    before: "geo.BondRingScanReport[Ring, Bond]",
    after: "geo.BondRingScanReport[Ring, Bond]",
    coordination_bond: "Bond",
) -> _CoordinationRelationCounts:
    """Count new relations caused by an active candidate coordination bond."""
    before_states = {
        _bond_ring_finding_key(finding): finding.relation.state
        for finding in before.findings
    }
    candidate_key = _bond_key(coordination_bond)
    candidate_endpoints = set(candidate_key)
    piercing_count = 0
    undetermined_count = 0
    for finding in after.findings:
        involves_candidate = finding.target.bond.key == candidate_key
        candidate_closed_ring = candidate_endpoints.issubset(
            finding.target.ring.key
        )
        if not involves_candidate and not candidate_closed_ring:
            continue
        if _is_coordination_cycle_closure(finding, coordination_bond):
            continue
        previous_state = before_states.get(_bond_ring_finding_key(finding))
        if finding.relation.state is geo.PiercingState.PIERCES:
            piercing_count += previous_state is not geo.PiercingState.PIERCES
        elif finding.relation.state is geo.PiercingState.UNDETERMINED:
            undetermined_count += previous_state is not geo.PiercingState.UNDETERMINED
    return _CoordinationRelationCounts(
        piercing=piercing_count,
        undetermined=undetermined_count,
        excluded_rings=after.excluded_ring_count,
    )


def _coordination_topology_relation_counts(
    report: "geo.BondRingScanReport[Ring, Bond]",
    coordination_bonds: Sequence["Bond"],
) -> _CoordinationRelationCounts:
    """Count unique relations associated with the restored coordination graph."""
    coordination_by_key = {
        _bond_key(bond): bond for bond in coordination_bonds
    }
    endpoint_sets = tuple(
        set(candidate_key) for candidate_key in coordination_by_key
    )
    piercing_count = 0
    undetermined_count = 0
    for finding in report.findings:
        target_key = finding.target.bond.key
        associated_ring = any(
            endpoints.issubset(finding.target.ring.key)
            for endpoints in endpoint_sets
        )
        if target_key not in coordination_by_key and not associated_ring:
            continue
        target_coordination_bond = coordination_by_key.get(target_key)
        if (
            target_coordination_bond is not None
            and _is_coordination_cycle_closure(
                finding,
                target_coordination_bond,
            )
        ):
            continue
        if finding.relation.state is geo.PiercingState.PIERCES:
            piercing_count += 1
        elif finding.relation.state is geo.PiercingState.UNDETERMINED:
            undetermined_count += 1
    return _CoordinationRelationCounts(
        piercing=piercing_count,
        undetermined=undetermined_count,
        excluded_rings=report.excluded_ring_count,
    )


def _restore_next_nonpiercing_coordination_bond(
    mol: "Molecule",
    pending_bonds: list["Bond"],
    *,
    trajectory: Optional[ForceFieldTrajectory] = None,
    attempt: Optional[int] = None,
) -> Tuple[Optional["Bond"], Tuple[str, ...]]:
    """Restore the first bond whose post-addition graph has no new piercing."""
    records_trajectory = (
        trajectory is not None
        and trajectory.records(TrajectoryStage.COORDINATION_RESTORATION)
    )

    def record_candidate(
        event: TrajectoryEvent,
        bond: "Bond",
        *,
        accepted: Optional[bool],
        pending_count: int,
        relation_counts: Optional[_CoordinationRelationCounts] = None,
    ) -> None:
        if not records_trajectory or trajectory is None:
            return
        trajectory.record_molecule(
            mol,
            stage=TrajectoryStage.COORDINATION_RESTORATION,
            event=event,
            attempt=attempt,
            evidence=CoordinationFrameEvidence(
                bond_atom_indices=_bond_key(bond),
                accepted=accepted,
                pending_bond_count=pending_count,
                introduced_piercing_count=(
                    0 if relation_counts is None else relation_counts.piercing
                ),
                introduced_undetermined_count=(
                    0 if relation_counts is None else relation_counts.undetermined
                ),
                excluded_ring_count=(
                    0 if relation_counts is None else relation_counts.excluded_rings
                ),
            ),
        )

    warning_messages = []
    before = _scan_full_graph_bond_ring_relations(mol)
    for bond in tuple(pending_bonds):
        mol.restore_bonds(bond, clear_conformers=False)
        record_candidate(
            TrajectoryEvent.BOND_TRIAL,
            bond,
            accepted=None,
            pending_count=len(pending_bonds),
        )
        keep_restored = False
        try:
            relation_counts = _candidate_coordination_relation_counts(
                before,
                _scan_full_graph_bond_ring_relations(mol),
                bond,
            )
            keep_restored = relation_counts.piercing == 0
            record_candidate(
                (
                    TrajectoryEvent.BOND_ACCEPTED
                    if keep_restored
                    else TrajectoryEvent.BOND_REJECTED
                ),
                bond,
                accepted=keep_restored,
                pending_count=(
                    len(pending_bonds) - 1
                    if keep_restored
                    else len(pending_bonds)
                ),
                relation_counts=relation_counts,
            )
        finally:
            if not keep_restored:
                mol.hide_bonds(bond, clear_conformers=False)
                record_candidate(
                    TrajectoryEvent.BOND_ROLLBACK,
                    bond,
                    accepted=False,
                    pending_count=len(pending_bonds),
                )
        if relation_counts.undetermined:
            warning_messages.append(
                f"Coordination bond {_bond_key(bond)} has "
                f"{relation_counts.undetermined} newly introduced, "
                "mathematically undetermined "
                "ring relation(s)"
            )
        if relation_counts.excluded_rings:
            warning_messages.append(
                f"Coordination bond {_bond_key(bond)} was checked while "
                f"{relation_counts.excluded_rings} ring(s) larger than "
                f"{_BOND_RING_MAX_SIZE} atoms were excluded"
            )
        if relation_counts.piercing:
            continue
        pending_bonds.remove(bond)
        return bond, tuple(warning_messages)
    return None, tuple(warning_messages)


def _restore_coordination_bonds_incrementally(
    mol: "Molecule",
    effective_forcefield: str,
    *,
    attempt_limit: int,
    relaxation_steps: int,
    perturb_sigma: float,
    rng: np.random.Generator,
    trajectory: Optional[ForceFieldTrajectory] = None,
) -> _CoordinationRestorationResult:
    """Restore original metal--ligand bonds through explicit bounded rounds."""
    records_trajectory = (
        trajectory is not None
        and trajectory.records(TrajectoryStage.COORDINATION_RESTORATION)
    )

    def record_restoration_frame(
        event: TrajectoryEvent,
        *,
        energy: Optional[float] = None,
        evidence: Optional[CoordinationFrameEvidence] = None,
        attempt: Optional[int] = None,
    ) -> Optional[int]:
        if not records_trajectory or trajectory is None:
            return None
        frame = trajectory.record_molecule(
            mol,
            stage=TrajectoryStage.COORDINATION_RESTORATION,
            event=event,
            energy_kj_mol=energy,
            attempt=attempt,
            evidence=evidence,
        )
        return frame.index

    coordination_bonds = tuple(sorted(
        (bond for bond in mol.bonds if bond.is_metal_ligand_bond),
        key=_bond_key,
    ))
    if not coordination_bonds:
        record_restoration_frame(TrajectoryEvent.COORDINATION_READY)
        terminal_index = record_restoration_frame(
            TrajectoryEvent.TERMINAL,
            evidence=CoordinationFrameEvidence(
                bond_atom_indices=None,
                accepted=True,
            ),
        )
        if terminal_index is not None and trajectory is not None:
            trajectory.select(terminal_index)
        return _CoordinationRestorationResult(
            report=CoordinationBondRestorationReport(
                attempt_limit=attempt_limit,
                attempts_completed=0,
                bond_count=0,
                restored_without_forcing=0,
                forced_bond_keys=(),
                final_piercing_count=0,
                final_undetermined_count=0,
                excluded_ring_count=0,
                resolved=True,
            ),
        )

    mol.hide_bonds(*coordination_bonds, clear_conformers=False)
    pending_bonds = list(coordination_bonds)
    record_restoration_frame(
        TrajectoryEvent.COORDINATION_READY,
        evidence=CoordinationFrameEvidence(
            bond_atom_indices=None,
            accepted=True,
            pending_bond_count=len(pending_bonds),
        ),
    )
    warning_messages: list[str] = []
    stalled_attempts = 0
    last_energy: Optional[float] = None

    while pending_bonds:
        restored_bond, relation_warnings = (
            _restore_next_nonpiercing_coordination_bond(
                mol,
                pending_bonds,
                trajectory=trajectory,
                attempt=stalled_attempts,
            )
        )
        warning_messages.extend(relation_warnings)
        if restored_bond is None:
            if stalled_attempts >= attempt_limit:
                break
            if stalled_attempts:
                mol.coordinates = _perturbed_coordinates(
                    mol.coordinates,
                    sigma=perturb_sigma,
                    rng=rng,
                )
                record_restoration_frame(
                    TrajectoryEvent.PERTURBED,
                    attempt=stalled_attempts,
                )
            stalled_attempts += 1
        optimized = _single_ob_optimization(
            mol,
            effective_forcefield,
            relaxation_steps,
        )
        last_energy = float(optimized.energy)
        record_restoration_frame(
            TrajectoryEvent.OPTIMIZED,
            energy=last_energy,
            attempt=stalled_attempts,
        )

    forced_bond_keys = tuple(_bond_key(bond) for bond in pending_bonds)
    if pending_bonds:
        for forced_index, bond in enumerate(pending_bonds):
            mol.restore_bonds(bond, clear_conformers=False)
            record_restoration_frame(
                TrajectoryEvent.BOND_FORCED,
                evidence=CoordinationFrameEvidence(
                    bond_atom_indices=_bond_key(bond),
                    accepted=False,
                    pending_bond_count=(
                        len(pending_bonds) - forced_index - 1
                    ),
                    forced=True,
                ),
                attempt=stalled_attempts,
            )
        last_energy = None
        warning_messages.append(
            f"Forced restoration of {len(pending_bonds)} coordination bond(s) "
            f"after {attempt_limit} stalled attempts"
        )

    final_relation_counts = _coordination_topology_relation_counts(
        _scan_full_graph_bond_ring_relations(mol),
        coordination_bonds,
    )
    if final_relation_counts.piercing:
        warning_messages.append(
            f"The restored coordination topology has "
            f"{final_relation_counts.piercing} confirmed bond-ring piercing "
            "relation(s); Stage 2.2 will repair the complete complex"
        )
    if final_relation_counts.undetermined:
        warning_messages.append(
            f"The restored coordination topology has "
            f"{final_relation_counts.undetermined} mathematically "
            "undetermined bond-ring relation(s)"
        )
    if final_relation_counts.excluded_rings:
        warning_messages.append(
            f"The coordination topology contains "
            f"{final_relation_counts.excluded_rings} ring(s) larger than "
            f"{_BOND_RING_MAX_SIZE} atoms that were not tested for piercing"
        )

    report = CoordinationBondRestorationReport(
        attempt_limit=attempt_limit,
        attempts_completed=stalled_attempts,
        bond_count=len(coordination_bonds),
        restored_without_forcing=len(coordination_bonds) - len(forced_bond_keys),
        forced_bond_keys=forced_bond_keys,
        final_piercing_count=final_relation_counts.piercing,
        final_undetermined_count=final_relation_counts.undetermined,
        excluded_ring_count=final_relation_counts.excluded_rings,
        resolved=(
            not forced_bond_keys
            and final_relation_counts.piercing == 0
        ),
        warning_messages=_unique_messages(warning_messages),
    )
    terminal_index = record_restoration_frame(
        TrajectoryEvent.TERMINAL,
        energy=last_energy,
        evidence=CoordinationFrameEvidence(
            bond_atom_indices=None,
            accepted=report.resolved,
            pending_bond_count=0,
            forced=bool(forced_bond_keys),
            introduced_piercing_count=final_relation_counts.piercing,
            introduced_undetermined_count=final_relation_counts.undetermined,
            excluded_ring_count=final_relation_counts.excluded_rings,
        ),
        attempt=stalled_attempts,
    )
    if terminal_index is not None and trajectory is not None:
        trajectory.select(terminal_index)
    return _CoordinationRestorationResult(
        report=report,
    )


# Synchronization and force-field policy helpers.


_WORKER_LIFECYCLE_LOCK = threading.Lock()
_OPENBABEL_FORCEFIELD_LOCK = threading.RLock()
_WORKER_EXIT_GRACE_SECONDS = 30.0


def _serialized_forcefield_call(function: CallableT) -> CallableT:
    @wraps(function)
    def synchronized(*args: object, **kwargs: object) -> object:
        with _OPENBABEL_FORCEFIELD_LOCK:
            return function(*args, **kwargs)

    return cast(CallableT, synchronized)


def _serialized_builder_call(function: CallableT) -> CallableT:
    @wraps(function)
    def synchronized(*args: object, **kwargs: object) -> object:
        with _WORKER_LIFECYCLE_LOCK:
            with _OPENBABEL_FORCEFIELD_LOCK:
                return function(*args, **kwargs)

    return cast(CallableT, synchronized)


def _resolve_complex_forcefield(requested: Optional[str]) -> str:
    """Resolve every currently supported complex request to UFF."""
    if requested is not None and requested not in _SUPPORTED_FORCEFIELDS:
        raise ValueError(f"Unsupported force field: {requested!r}")
    return "UFF"


def _resolve_organic_forcefield(requested: Optional[str]) -> str:
    """Resolve an omitted organic force field without changing explicit choices."""
    effective = requested or "MMFF94s"
    if effective not in _SUPPORTED_FORCEFIELDS:
        raise ValueError(f"Unsupported force field: {effective!r}")
    return effective


def _require_explicit_complex(mol: "Molecule") -> None:
    """Require a metal center with at least one explicit metal--ligand bond."""
    if not mol.has_metal or not any(
        bond.is_metal_ligand_bond for bond in mol.bonds
    ):
        raise ValueError(
            "The complex workflow requires a molecule with at least one "
            "explicit metal-ligand bond"
        )


def _make_constraints(mol: "Molecule") -> ob.OBFFConstraints:
    """Return the intentionally empty force-field constraint adapter."""
    return ob.OBFFConstraints()


def _setup_forcefield_backend(
    backend: ob.OBForceField,
    mol: "Molecule",
    obmol: ob.OBMol,
    *,
    requested_forcefield: Optional[str],
    effective_forcefield: str,
) -> None:
    """Set up an Open Babel force field or raise structured diagnostics."""
    if backend.Setup(obmol, _make_constraints(mol)):
        return
    raise ForceFieldSetupError(
        f"Open Babel could not initialize force field {effective_forcefield!r}",
        ForceFieldSetupReport(
            requested_forcefield,
            effective_forcefield,
            "setup",
        ),
    )


def _energy_factor_to_kj(unit: str) -> float:
    normalized = unit.strip().lower().replace(" ", "")
    if normalized in {"kj/mol", "kjmol-1", "kjmol^-1"}:
        return 1.0
    if normalized in {"kcal/mol", "kcalmol-1", "kcalmol^-1"}:
        return 4.184
    raise ValueError(f"Unsupported Open Babel energy unit: {unit!r}")


def _forcefield_energy_in_kj(
    ob_forcefield: ob.OBForceField,
    calc_grad: bool = True,
) -> float:
    return float(ob_forcefield.Energy(calc_grad)) * _energy_factor_to_kj(ob_forcefield.GetUnit())


@_serialized_forcefield_call
def _get_forcefield(name: str) -> ob.OBForceField:
    """Retrieve a force-field plugin guarded by the process-local FF lock."""
    backend = _find_forcefield_prototype(name)
    if backend is None:
        raise ForceFieldSetupError(
            f"Unknown Open Babel force field: {name!r}",
            ForceFieldSetupReport(name, name, "lookup"),
        )
    return backend


def _find_forcefield_prototype(name: str) -> Optional[ob.OBForceField]:
    return ob.OBForceField.FindType(name)


def _seed_openbabel_random(seed: int) -> None:
    """Seed the current Open Babel RNG before using ``OBBuilder``."""
    os.environ["OB_RANDOM_SEED"] = str(seed)


@_serialized_forcefield_call
def _single_ob_optimization(
    mol: "Molecule", forcefield: str, steps: int
) -> _CandidateOptimizationResult:
    backend = _get_forcefield(forcefield)
    backend.EnableCutOff(False)
    obmol, _ = mol2obmol(mol)
    _setup_forcefield_backend(
        backend,
        mol,
        obmol,
        requested_forcefield=forcefield,
        effective_forcefield=forcefield,
    )
    backend.SteepestDescent(steps)
    backend.GetCoordinates(obmol)
    mol.coordinates = extract_obmol_coordinates(obmol)
    energy = _forcefield_energy_in_kj(backend)
    return _CandidateOptimizationResult(
        energy=energy,
        energy_unit="kJ/mol",
        exploded=bool(backend.DetectExplosion()),
    )


# Low-level Open Babel build and optimization primitives.


@_serialized_builder_call
def _ob_build(mol: "Molecule") -> None:
    """Run OBBuilder directly on an internal working molecule."""
    builder = ob.OBBuilder()
    obmol, _ = mol2obmol(mol)
    if not builder.Build(obmol):
        raise ForceFieldError("Open Babel could not build initial 3D coordinates")
    mol.coordinates = extract_obmol_coordinates(obmol)


# Working-copy preparation and transactional commit helpers.


def _copy_molecule_metadata(source_mol: "Molecule", target_mol: "Molecule") -> None:
    target_mol.charge = source_mol.charge
    target_mol.properties = dict(source_mol.properties)
    target_mol._model = source_mol._model
    target_mol._environ = source_mol._environ
    target_mol._crystal = source_mol._crystal


def _recalculate_neutral_donor_valence(
    mol: "Molecule",
    donor_indices: set[int],
) -> None:
    """Infer neutral donor hydrogens from the metal-free ligand skeleton."""
    for donor_index in donor_indices:
        donor = mol.atoms[donor_index]
        if (
            donor.formal_charge == 0
            and donor.atomic_number in _NEUTRAL_DONOR_ATOMIC_NUMBERS
        ):
            donor.valence = donor.get_valence()
            donor.calc_implicit_hydrogens()


def _hydrogenated_working_copy(
    mol: "Molecule",
    *,
    add_hydrogens: bool,
    seed: Optional[int] = None,
) -> "Molecule":
    """Copy ``mol`` and infer H atoms against its ligand covalent skeleton."""
    working_mol = copy(mol)
    _copy_molecule_metadata(mol, working_mol)
    original_atom_count = len(working_mol.atoms)
    if add_hydrogens:
        if working_mol.has_metal:
            donor_indices = {
                donor.idx
                for _, donor in _iter_metal_donor_pairs(working_mol)
            }
            working_mol.hide_metal_ligand_bonds(clear_conformers=False)
            _recalculate_neutral_donor_valence(working_mol, donor_indices)
            working_mol.add_hydrogens(
                rm_polar_hs=False,
                rng=np.random.default_rng(seed),
            )
            working_mol.recover_hided_metal_ligand_bonds(clear_conformers=False)
        else:
            working_mol.add_hydrogens(
                rm_polar_hs=False,
                rng=np.random.default_rng(seed),
            )
    used_ids = {
        int(atom.id) for atom in working_mol.atoms[:original_atom_count]
    }
    next_id = max(used_ids, default=-1) + 1
    for atom in working_mol.atoms[original_atom_count:]:
        while next_id in used_ids:
            next_id += 1
        atom.id = next_id
        used_ids.add(next_id)
        next_id += 1
    return working_mol


def _make_worker_mol(mol: "Molecule") -> "Molecule":
    """Return a structure-only clone with private positional IDs for a worker."""
    worker_mol = copy(mol)
    worker_mol.charge = mol.charge
    worker_mol.refresh_atom_id()
    return worker_mol


def _prepare_working_copy_commit(
    original_mol: "Molecule",
    working_mol: "Molecule",
) -> _WorkingCopyCommit:
    original_atoms = tuple(original_mol._atoms)
    working_atoms = tuple(working_mol.atoms)
    original_atom_count = len(original_atoms)
    if len(working_atoms) < original_atom_count:
        raise ValueError("The working copy removed an original atom")

    for original_atom, working_atom in zip(
        original_atoms,
        working_atoms[:original_atom_count],
    ):
        if _atom_identity(original_atom) != _atom_identity(working_atom):
            raise ValueError("The working copy changed an original atom identity")

    original_atom_indices = _atom_index_map(original_atoms)
    working_atom_indices = _atom_index_map(working_atoms)
    original_bonds = {
        _bond_identity(bond, original_atom_indices)
        for bond in original_mol.bonds
    }
    working_original_bonds = set()
    working_bond_keys = set()
    added_bonds: list[Tuple[int, int, _BondAttributePayload]] = []
    for bond in working_mol.bonds:
        if (
            id(bond.atom1) not in working_atom_indices
            or id(bond.atom2) not in working_atom_indices
        ):
            raise ValueError("The working copy contains a bond to an external atom")
        first, second = _bond_endpoint_indices(bond, working_atom_indices)
        key = tuple(sorted((first, second)))
        if first == second or key in working_bond_keys:
            raise ValueError("The working copy contains an invalid duplicate bond")
        working_bond_keys.add(key)
        if first < original_atom_count and second < original_atom_count:
            working_original_bonds.add(
                _bond_identity(bond, working_atom_indices)
            )
        else:
            attributes = cast(_BondAttributePayload, deepcopy(bond.attr_dict))
            added_bonds.append((first, second, attributes))

    if working_original_bonds != original_bonds:
        raise ValueError("The working copy changed the original bond topology")

    return _WorkingCopyCommit(
        original_atom_attrs=tuple(
            np.array(atom.attrs, copy=True)
            for atom in working_atoms[:original_atom_count]
        ),
        added_atom_attrs=tuple(
            np.array(atom.attrs, copy=True)
            for atom in working_atoms[original_atom_count:]
        ),
        added_bonds=tuple(added_bonds),
        conformer_state=deepcopy(working_mol._conformers.__dict__),
        conformer_index=working_mol._conformers_index,
    )


def _snapshot_molecule_for_commit(mol: "Molecule") -> _MoleculeCommitSnapshot:
    return _MoleculeCommitSnapshot(
        atoms=tuple(mol._atoms),
        bonds=tuple(mol._bonds),
        atom_state=tuple(
            (atom, atom.attrs, atom._neighbours, atom._bonds)
            for atom in mol._atoms
        ),
        graph=mol._graph,
        row_to_index=mol._row2idx,
        angles=mol._angles,
        torsions=mol._torsions,
        rings=mol._rings,
        cycle_basis_rings=mol._cycle_basis_rings,
        ring_indices_cache=mol._ring_indices_cache,
        ligand_rings=mol._ligand_rings,
        ligand_cycle_basis_rings=mol._ligand_cycle_basis_rings,
        ligand_rings_signature=mol._ligand_rings_signature,
        obmol=mol._obmol,
        atom_pair_items=tuple(mol._atom_pairs.items()),
        conformer_state=dict(mol._conformers.__dict__),
        conformer_index=mol._conformers_index,
    )


def _restore_failed_commit(
    mol: "Molecule",
    snapshot: _MoleculeCommitSnapshot,
) -> None:
    mol._atoms[:] = snapshot.atoms
    mol._bonds[:] = snapshot.bonds
    for atom, attrs, neighbours, bonds in snapshot.atom_state:
        object.__setattr__(atom, "attrs", attrs)
        object.__setattr__(atom, "_neighbours", neighbours)
        object.__setattr__(atom, "_bonds", bonds)
    mol._graph = snapshot.graph
    mol._row2idx = snapshot.row_to_index
    mol._angles = snapshot.angles
    mol._torsions = snapshot.torsions
    mol._rings = snapshot.rings
    mol._cycle_basis_rings = snapshot.cycle_basis_rings
    mol._ring_indices_cache = snapshot.ring_indices_cache
    mol._ligand_rings = snapshot.ligand_rings
    mol._ligand_cycle_basis_rings = snapshot.ligand_cycle_basis_rings
    mol._ligand_rings_signature = snapshot.ligand_rings_signature
    mol._obmol = snapshot.obmol
    dict.clear(mol._atom_pairs)
    dict.update(mol._atom_pairs, snapshot.atom_pair_items)
    mol._conformers.__dict__.clear()
    mol._conformers.__dict__.update(snapshot.conformer_state)
    mol._conformers_index = snapshot.conformer_index


def _commit_working_copy(
    original_mol: "Molecule",
    working_mol: "Molecule",
) -> None:
    """Commit accepted geometry while preserving caller-owned object identities."""
    payload = _prepare_working_copy_commit(original_mol, working_mol)
    snapshot = _snapshot_molecule_for_commit(original_mol)
    try:
        for atom, attrs in zip(original_mol._atoms, payload.original_atom_attrs):
            atom.attrs = attrs
        for attrs in payload.added_atom_attrs:
            original_mol._create_atom_from_array(attrs)
        for first, second, attributes in payload.added_bonds:
            original_mol._add_bond(first, second, **attributes)

        original_mol._update_graph(clear_conformers=False)
        original_mol._row2idx = None
        original_mol._atom_pairs.update_pairs()
        original_mol._conformers.__dict__.clear()
        original_mol._conformers.__dict__.update(payload.conformer_state)
        original_mol._conformers_index = payload.conformer_index
    except BaseException:
        _restore_failed_commit(original_mol, snapshot)
        raise


def _perturbed_coordinates(
    coordinates: np.ndarray,
    *,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    displacement = rng.normal(0.0, sigma, np.asarray(coordinates).shape)
    displacement = np.clip(displacement, -2.0 * sigma, 2.0 * sigma)
    return np.asarray(coordinates, dtype=float) + displacement


# Stateful Open Babel optimization engine.


class _OpenBabelOptimizer:
    """One stateful Open Babel optimizer used by every public workflow."""

    def __init__(
        self,
        requested_forcefield: Optional[str],
        effective_forcefield: str,
        *,
        algorithm: OptimizationAlgorithm,
        epochs: int,
        steps_per_epoch: int,
        perturb_interval: Optional[int],
        perturb_sigma: float,
        save_movie: bool,
        increasing_vdw: bool,
        vdw_cutoff_start: float,
        vdw_cutoff_end: float,
        seed: Optional[int],
        stop_on_ring_piercing: bool = False,
        energy_tolerance: float = 1.0e-6,
    ) -> None:
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        if steps_per_epoch < 1:
            raise ValueError("steps_per_epoch must be at least 1")
        if perturb_interval is not None and perturb_interval < 1:
            raise ValueError("perturb_interval must be at least 1 when provided")
        if perturb_sigma < 0.0:
            raise ValueError("perturb_sigma must be non-negative")
        if increasing_vdw and vdw_cutoff_end < vdw_cutoff_start:
            raise ValueError(
                "vdw_cutoff_end must not be smaller than vdw_cutoff_start"
            )
        self.requested_forcefield = requested_forcefield
        self.effective_forcefield = effective_forcefield
        self.algorithm = algorithm
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.perturb_interval = perturb_interval
        self.perturb_sigma = perturb_sigma
        self.save_movie = save_movie
        self.increasing_vdw = increasing_vdw
        self.vdw_cutoff_start = vdw_cutoff_start
        self.vdw_cutoff_end = vdw_cutoff_end
        self.stop_on_ring_piercing = stop_on_ring_piercing
        self.energy_tolerance = energy_tolerance
        self.rng = np.random.default_rng(seed)
        self.backend = _get_forcefield(effective_forcefield)

    def _setup(self, mol: "Molecule", obmol: ob.OBMol) -> None:
        _setup_forcefield_backend(
            self.backend,
            mol,
            obmol,
            requested_forcefield=self.requested_forcefield,
            effective_forcefield=self.effective_forcefield,
        )
        if self.increasing_vdw:
            self.backend.UpdatePairsSimple()

    def _set_vdw_cutoff(self, cutoff: float) -> None:
        self.backend.EnableCutOff(True)
        self.backend.SetVDWCutOff(cutoff)
        # Open Babel enables VDW and electrostatic cutoffs together.  Keep the
        # electrostatic term effectively untruncated when only VDW annealing
        # was requested.
        self.backend.SetElectrostaticCutOff(1.0e6)

    def _optimizer_methods(
        self,
    ) -> Tuple[Callable[[int, float], object], Callable[[int], bool]]:
        if self.algorithm == "conjugate":
            return (
                self.backend.ConjugateGradientsInitialize,
                self.backend.ConjugateGradientsTakeNSteps,
            )
        if self.algorithm == "steepest":
            return (
                self.backend.SteepestDescentInitialize,
                self.backend.SteepestDescentTakeNSteps,
            )
        raise ValueError(f"Unknown optimization algorithm: {self.algorithm!r}")

    def _initialize_with_budget(
        self,
        initialize: Callable[[int, float], object],
        remaining_steps: int,
    ) -> int:
        initialization_steps = int(self.algorithm == "conjugate")
        take_step_capacity = remaining_steps - initialization_steps
        # Open Babel returns False both for convergence and for reaching the
        # limit supplied to Initialize().  Keep that private limit one counter
        # step beyond every TakeNSteps() call Hotpot can submit.  Conjugate
        # initialization performs one physical step without incrementing the
        # backend counter; steepest-descent initialization performs none.
        initialize(take_step_capacity + 1, self.energy_tolerance)
        return initialization_steps

    def _gradients(self, obmol: ob.OBMol, factor: float) -> Tuple[float, float]:
        vectors = []
        for atom in ob.OBMolAtomIter(obmol):
            gradient = self.backend.GetGradient(atom)
            vectors.append((gradient.GetX(), gradient.GetY(), gradient.GetZ()))
        norms = np.linalg.norm(np.asarray(vectors, dtype=float) * factor, axis=1)
        return float(np.sqrt(np.mean(norms**2))), float(np.max(norms))

    def _observe_frame(
        self,
        mol: "Molecule",
        obmol: ob.OBMol,
        *,
        factor: float,
        converged: bool,
        epochs_completed: int,
        segment_epochs_completed: int,
        previous_coordinates: Optional[np.ndarray],
        previous_energy: Optional[float],
        energy_changes: deque[float],
        max_displacements: deque[float],
        quality_level: AcceptanceLevel,
        topology_reference: TopologyReference,
        quality_thresholds: Optional[StructureAcceptanceThresholds],
    ) -> _ObservedFrame:
        self.backend.GetCoordinates(obmol)
        coordinates = extract_obmol_coordinates(obmol)
        mol.coordinates = coordinates
        energy = _forcefield_energy_in_kj(self.backend)
        rms_gradient, max_gradient = self._gradients(obmol, factor)
        exploded = bool(self.backend.DetectExplosion())
        if previous_energy is not None:
            energy_changes.append(abs(energy - previous_energy))
        if previous_coordinates is not None:
            displacements = np.linalg.norm(
                coordinates - previous_coordinates,
                axis=1,
            )
            max_displacements.append(float(np.max(displacements)))
        quality_report = evaluate_structure_acceptance(
            mol,
            level=quality_level,
            topology_reference=topology_reference,
            forcefield_report={
                "setup_succeeded": True,
                "converged": converged,
                "final_energy": energy,
                "energy_unit": "kJ/mol",
                "rms_gradient": rms_gradient,
                "max_gradient": max_gradient,
                "exploded": exploded,
                "energy_changes": tuple(energy_changes),
                "max_displacements": tuple(max_displacements),
                "epochs_completed": epochs_completed,
                "segment_epochs_completed": segment_epochs_completed,
            },
            forcefield_stage="final",
            thresholds=quality_thresholds,
        )
        return _ObservedFrame(
            coordinates=coordinates.copy(),
            energy=energy,
            rms_gradient=rms_gradient,
            max_gradient=max_gradient,
            exploded=exploded,
            converged=converged,
            quality_report=quality_report,
            energy_changes=tuple(energy_changes),
            max_displacements=tuple(max_displacements),
        )

    @_serialized_forcefield_call
    def optimize(
        self,
        mol: "Molecule",
        *,
        quality_level: AcceptanceLevel,
        topology_reference: TopologyReference,
        quality_thresholds: Optional[StructureAcceptanceThresholds],
        trajectory: ForceFieldTrajectory,
        trajectory_stage: TrajectoryStage = TrajectoryStage.FINAL_OPTIMIZATION,
        trajectory_attempt: Optional[int] = None,
    ) -> ForceFieldRunReport:
        records_trajectory = trajectory.records(trajectory_stage)
        if records_trajectory:
            trajectory.record_molecule(
                mol,
                stage=trajectory_stage,
                event=TrajectoryEvent.INITIAL,
                attempt=trajectory_attempt,
            )
        obmol, _ = mol2obmol(mol)
        if self.increasing_vdw:
            self._set_vdw_cutoff(self.vdw_cutoff_end)
        else:
            self.backend.EnableCutOff(False)
        self._setup(mol, obmol)
        initialize, take_steps = self._optimizer_methods()
        total_steps = self.epochs * self.steps_per_epoch
        backend_unit = self.backend.GetUnit()
        factor = _energy_factor_to_kj(backend_unit)
        if self.increasing_vdw:
            first_cutoff = self.vdw_cutoff_start + (
                self.vdw_cutoff_end - self.vdw_cutoff_start
            ) / self.epochs
            self._set_vdw_cutoff(first_cutoff)
            self._setup(mol, obmol)
        epoch_initialization_steps = self._initialize_with_budget(
            initialize,
            total_steps,
        )

        best_frame = None
        best_epoch = -1
        best_frame_index: Optional[int] = None
        last_frame = None
        last_epoch = -1
        thresholds = _resolve_acceptance_thresholds(quality_thresholds)
        history_window = thresholds.strict_stability_window
        energy_changes = deque(maxlen=history_window)
        max_displacements = deque(maxlen=history_window)
        epoch_energies = []
        epoch_quality_reports = []
        previous_coordinates = None
        previous_energy = None
        epochs_completed = 0
        segment_epochs_completed = 0
        steps_submitted = 0
        initialization_steps = 0
        terminal_converged = False
        termination_reason: TerminationReason = "budget_exhausted"
        segment_active = True
        stopped_on_ring_piercing = False

        for epoch in range(self.epochs):
            reset_history = (
                self.perturb_interval is not None
                and epoch > 0
                and epoch % self.perturb_interval == 0
            )
            if reset_history:
                coordinates = _perturbed_coordinates(
                    extract_obmol_coordinates(obmol),
                    sigma=self.perturb_sigma,
                    rng=self.rng,
                )
                set_obmol_coordinates(obmol, coordinates)

            if self.increasing_vdw and epoch > 0:
                cutoff = self.vdw_cutoff_start + ((epoch + 1) / self.epochs) * (
                    self.vdw_cutoff_end - self.vdw_cutoff_start
                )
                self._set_vdw_cutoff(cutoff)

            restart_segment = reset_history or (self.increasing_vdw and epoch > 0)
            if restart_segment:
                self._setup(mol, obmol)
                energy_changes.clear()
                max_displacements.clear()
                previous_coordinates = None
                previous_energy = None
                segment_epochs_completed = 0
                remaining_steps = (self.epochs - epoch) * self.steps_per_epoch
                epoch_initialization_steps = self._initialize_with_budget(
                    initialize,
                    remaining_steps,
                )
                segment_active = True

            if not segment_active:
                continue

            steps_to_take = self.steps_per_epoch - epoch_initialization_steps
            initialization_steps += epoch_initialization_steps
            backend_continues = (
                bool(take_steps(steps_to_take)) if steps_to_take else True
            )
            steps_submitted += steps_to_take
            epoch_initialization_steps = 0
            epochs_completed += 1
            segment_epochs_completed += 1
            backend_converged = not backend_continues
            segment_active = backend_continues
            self.backend.GetCoordinates(obmol)
            frame_converged = backend_converged
            terminal_converged = frame_converged
            termination_reason = (
                "converged"
                if terminal_converged
                else "budget_exhausted"
            )

            if self.increasing_vdw and epoch < self.epochs - 1:
                self._set_vdw_cutoff(self.vdw_cutoff_end)
                self._setup(mol, obmol)

            quality_converged = frame_converged and (
                not self.increasing_vdw or epoch == self.epochs - 1
            )
            frame = self._observe_frame(
                mol,
                obmol,
                factor=factor,
                converged=quality_converged,
                epochs_completed=epochs_completed,
                segment_epochs_completed=segment_epochs_completed,
                previous_coordinates=previous_coordinates,
                previous_energy=previous_energy,
                energy_changes=energy_changes,
                max_displacements=max_displacements,
                quality_level=quality_level,
                topology_reference=topology_reference,
                quality_thresholds=quality_thresholds,
            )
            trajectory_frame: Optional[ForceFieldFrame] = None
            if records_trajectory:
                trajectory_frame = trajectory.record_molecule(
                    mol,
                    stage=trajectory_stage,
                    event=TrajectoryEvent.EPOCH_COMPLETE,
                    energy_kj_mol=frame.energy,
                    attempt=trajectory_attempt,
                    step=epoch,
                    evidence=OptimizationFrameEvidence(
                        accepted=frame.quality_report.passed,
                        converged=frame.converged,
                        rms_gradient_kj_mol_angstrom=frame.rms_gradient,
                        max_gradient_kj_mol_angstrom=frame.max_gradient,
                        failed_checks=tuple(
                            check.name
                            for check in frame.quality_report.checks
                            if not check.passed
                        ),
                    ),
                )
            last_frame = frame
            last_epoch = epoch
            if (
                frame.quality_report.passed
                and (best_frame is None or frame.energy < best_frame.energy)
            ):
                best_frame = frame
                best_epoch = epoch
                best_frame_index = (
                    None if trajectory_frame is None else trajectory_frame.index
                )
            if self.save_movie:
                epoch_energies.append(frame.energy)
                epoch_quality_reports.append(frame.quality_report)
            previous_coordinates = frame.coordinates
            previous_energy = frame.energy

            if self.stop_on_ring_piercing:
                piercing_count = frame.quality_report.metrics.get(
                    "bond_ring_piercing_count"
                )
                if piercing_count is None:
                    piercing_state = geo.determine_bond_ring_piercing_state(
                        mol,
                        ring_scope="ligand_skeleton",
                        max_ring_size=_BOND_RING_MAX_SIZE,
                    )
                    stopped_on_ring_piercing = (
                        piercing_state is geo.PiercingState.PIERCES
                    )
                else:
                    stopped_on_ring_piercing = bool(piercing_count)
                if stopped_on_ring_piercing:
                    termination_reason = "ring_piercing"
                    terminal_converged = False
                    break

            if (
                backend_converged
                and not self.increasing_vdw
                and self.perturb_interval is None
            ):
                break

        if last_frame is None:
            raise GeometryQualityError(None)
        if stopped_on_ring_piercing:
            if _has_unreturnable_frame_failure(last_frame.quality_report):
                raise GeometryQualityError(last_frame.quality_report)
            best_frame = last_frame
            best_epoch = last_epoch
            best_frame_index = (
                None if trajectory_frame is None else trajectory_frame.index
            )
        elif not last_frame.quality_report.passed:
            if _has_unreturnable_frame_failure(last_frame.quality_report):
                raise GeometryQualityError(last_frame.quality_report)
            warnings.warn(
                _format_geometry_checks(
                    "The terminal optimization frame failed structure "
                    "acceptance; retaining the last finite-topology frame",
                    tuple(last_frame.quality_report.failures),
                ),
                GeometryQualityWarning,
                stacklevel=2,
            )
            best_frame = last_frame
            best_epoch = last_epoch
            best_frame_index = (
                None if trajectory_frame is None else trajectory_frame.index
            )
            termination_reason = "quality_gate_failed"

        mol.coordinates = best_frame.coordinates
        if best_frame_index is not None:
            trajectory.select(best_frame_index)

        return ForceFieldRunReport(
            requested_forcefield=self.requested_forcefield,
            effective_forcefield=self.effective_forcefield,
            setup_succeeded=True,
            converged=best_frame.converged,
            epochs_completed=epochs_completed,
            steps_submitted=steps_submitted,
            initialization_steps=initialization_steps,
            steps_completed=None,
            final_energy=float(last_frame.energy),
            best_energy=float(best_frame.energy),
            energy_unit="kJ/mol",
            rms_gradient=float(best_frame.rms_gradient),
            max_gradient=float(best_frame.max_gradient),
            exploded=best_frame.exploded,
            quality_report=best_frame.quality_report,
            backend_energy_unit=backend_unit,
            gradient_unit="kJ/(mol*angstrom)",
            energy_changes=best_frame.energy_changes,
            max_displacements=best_frame.max_displacements,
            best_epoch=best_epoch,
            epoch_energies=tuple(epoch_energies),
            epoch_quality_reports=tuple(epoch_quality_reports),
            termination_reason=termination_reason,
            terminal_converged=terminal_converged,
        )


# Ligand-proxy construction and geometric untangling.


def _ligand_candidate_sort_key(
    candidate: _LigandCandidate,
) -> Tuple[int, bool, float, int]:
    """Rank usable fallback starts by piercing count and finite energy."""
    finite_energy = bool(np.isfinite(candidate.energy))
    return (
        candidate.untangling.final_piercing_count,
        not finite_energy,
        candidate.energy if finite_energy else float("inf"),
        candidate.attempt,
    )


def _build_ligand_proxies(
    mol: "Molecule",
    *,
    max_attempts: int,
    candidate_warmup_steps: int,
    candidate_score_steps: int,
    best_candidate_refine_steps: int,
    effective_forcefield: str,
    ligand_untangling_attempts: int = 20,
    perturb_sigma: float = 0.5,
    seed: Optional[int] = None,
    trajectory_attempts: Optional[list[ForceFieldTrajectory]] = None,
) -> Tuple[np.ndarray, ComplexBuildDiagnostics]:
    started = time.monotonic()
    clone_mol = copy(mol)
    _copy_molecule_metadata(mol, clone_mol)
    clone_mol.hide_metal_ligand_bonds(clear_conformers=False)
    rng = np.random.default_rng(seed)
    total_attempts = 0
    total_accepted = 0
    rejections: list[CandidateRejection] = []
    warning_messages: list[str] = []
    selected_untangling_reports: list[RingUntanglingReport] = []

    for component_index, component_mol in enumerate(clone_mol.components):
        if component_mol.has_metal:
            continue

        component_reference = capture_topology(
            component_mol,
            allow_added_hydrogens=False,
        )
        accepted_candidate: Optional[_LigandCandidate] = None
        fallback_candidates: list[_LigandCandidate] = []
        component_attempts = 0
        while accepted_candidate is None and component_attempts < max_attempts:
            component_attempts += 1
            total_attempts += 1
            attempt_trajectory: Optional[ForceFieldTrajectory] = None
            if trajectory_attempts is not None:
                attempt_trajectory = ForceFieldTrajectory.from_molecule(
                    component_mol,
                    start=TrajectoryStart.LIGAND_BUILD,
                )
                trajectory_attempts.append(attempt_trajectory)
                attempt_trajectory.record_molecule(
                    component_mol,
                    stage=TrajectoryStage.LIGAND_BUILD,
                    event=TrajectoryEvent.INITIAL,
                    component_index=component_index,
                    attempt=component_attempts,
                )
            try:
                _ob_build(component_mol)
                if attempt_trajectory is not None:
                    attempt_trajectory.record_molecule(
                        component_mol,
                        stage=TrajectoryStage.LIGAND_BUILD,
                        event=TrajectoryEvent.BUILD_COMPLETE,
                        component_index=component_index,
                        attempt=component_attempts,
                    )
                warmed = _single_ob_optimization(
                    component_mol,
                    effective_forcefield,
                    candidate_warmup_steps,
                )
                if attempt_trajectory is not None:
                    attempt_trajectory.record_molecule(
                        component_mol,
                        stage=TrajectoryStage.LIGAND_BUILD,
                        event=TrajectoryEvent.WARMUP_COMPLETE,
                        energy_kj_mol=float(warmed.energy),
                        component_index=component_index,
                        attempt=component_attempts,
                    )
                untangling = _untangle_ring_piercings(
                    component_mol,
                    effective_forcefield,
                    attempt_limit=ligand_untangling_attempts,
                    short_steps=candidate_warmup_steps,
                    settling_steps=candidate_score_steps,
                    perturb_sigma=perturb_sigma,
                    rng=rng,
                    ring_scope="ligand_skeleton",
                    initial_energy=float(warmed.energy),
                    trajectory=attempt_trajectory,
                    trajectory_stage=TrajectoryStage.LIGAND_BUILD,
                )
            except ForceFieldError as exc:
                if attempt_trajectory is not None:
                    terminal_frame = attempt_trajectory.record_molecule(
                        component_mol,
                        stage=TrajectoryStage.LIGAND_BUILD,
                        event=TrajectoryEvent.TERMINAL,
                        component_index=component_index,
                        attempt=component_attempts,
                    )
                    attempt_trajectory.select(terminal_frame.index)
                rejections.append(
                    CandidateRejection(component_index, component_attempts, str(exc))
                )
                continue

            candidate_quality = evaluate_structure_acceptance(
                component_mol,
                level="basic",
                topology_reference=component_reference,
                forcefield_report={
                    "setup_succeeded": True,
                    "final_energy": untangling.energy,
                    "energy_unit": "kJ/mol",
                    "exploded": False,
                },
                forcefield_stage="candidate",
            )
            candidate = _LigandCandidate(
                coordinates=_copy_coordinates(component_mol.coordinates),
                energy=float(untangling.energy),
                attempt=component_attempts,
                untangling=untangling.report,
                trajectory=attempt_trajectory,
            )
            if not _has_unreturnable_frame_failure(candidate_quality):
                fallback_candidates.append(candidate)
            if not candidate_quality.passed:
                rejections.append(
                    CandidateRejection(
                        component_index,
                        component_attempts,
                        _format_geometry_checks(
                            "candidate geometry gate",
                            tuple(candidate_quality.failures),
                        ),
                        tuple(candidate_quality.failures),
                    )
                )
                continue

            accepted_candidate = candidate
            total_accepted += 1

        if accepted_candidate is None and not fallback_candidates:
            diagnostics = ComplexBuildDiagnostics(
                attempt_count=total_attempts,
                accepted_candidates=total_accepted,
                rejected_candidates=tuple(rejections),
                elapsed_seconds=time.monotonic() - started,
                warning_messages=tuple(warning_messages),
                ligand_untangling=tuple(selected_untangling_reports),
            )
            raise ComplexBuildError(
                f"Component {component_index} produced no usable candidate "
                f"after {component_attempts} attempts",
                diagnostics,
            )

        if accepted_candidate is None:
            selected_candidate = min(
                fallback_candidates,
                key=_ligand_candidate_sort_key,
            )
            component_mol.coordinates = selected_candidate.coordinates
            warning_messages.append(
                f"Component {component_index}: no candidate passed the basic "
                f"geometry gate after {component_attempts} attempts; retaining "
                "the usable attempted geometry with the lowest confirmed "
                "bond-ring piercing count and energy as the next-stage start"
            )
        else:
            selected_candidate = accepted_candidate
            component_mol.coordinates = selected_candidate.coordinates
            try:
                refined = _single_ob_optimization(
                    component_mol,
                    effective_forcefield,
                    best_candidate_refine_steps,
                )
            except ForceFieldError as exc:
                rejections.append(
                    CandidateRejection(
                        component_index,
                        selected_candidate.attempt,
                        f"refined candidate: {exc}",
                    )
                )
                component_mol.coordinates = selected_candidate.coordinates
                if selected_candidate.trajectory is not None:
                    rollback_frame = selected_candidate.trajectory.record_molecule(
                        component_mol,
                        stage=TrajectoryStage.LIGAND_BUILD,
                        event=TrajectoryEvent.ROLLED_BACK,
                        component_index=component_index,
                        attempt=selected_candidate.attempt,
                    )
                    selected_candidate.trajectory.select(rollback_frame.index)
                warning_messages.append(
                    f"Component {component_index}: long refinement failed; "
                    "retaining the medium-optimized candidate"
                )
            else:
                refined_frame: Optional[ForceFieldFrame] = None
                if selected_candidate.trajectory is not None:
                    refined_frame = selected_candidate.trajectory.record_molecule(
                        component_mol,
                        stage=TrajectoryStage.LIGAND_BUILD,
                        event=TrajectoryEvent.OPTIMIZED,
                        energy_kj_mol=float(refined.energy),
                        component_index=component_index,
                        attempt=selected_candidate.attempt,
                    )
                refined_state, refined_report = _scan_confirmed_ring_piercings(
                    component_mol,
                    ring_scope="ligand_skeleton",
                )
                refined_piercing_count = _piercing_count(refined_report)
                refined_quality = evaluate_structure_acceptance(
                    component_mol,
                    level="basic",
                    topology_reference=component_reference,
                    forcefield_report={
                        "setup_succeeded": True,
                        "final_energy": refined.energy,
                        "energy_unit": refined.energy_unit,
                        "exploded": refined.exploded,
                    },
                    forcefield_stage="candidate",
                )
                if not refined_quality.passed:
                    intersection_failures = (
                        _bond_ring_acceptance_checks(
                            component_mol,
                            refined_report,
                        )
                        if refined_report is not None
                        else ()
                    )
                    failures = (
                        tuple(intersection_failures)
                        + tuple(refined_quality.failures)
                    )
                    rejections.append(CandidateRejection(
                        component_index,
                        selected_candidate.attempt,
                        _format_geometry_checks(
                            "refined candidate geometry gate",
                            failures,
                        ),
                        failures,
                    ))
                    component_mol.coordinates = selected_candidate.coordinates
                    if selected_candidate.trajectory is not None:
                        rollback_frame = (
                            selected_candidate.trajectory.record_molecule(
                                component_mol,
                                stage=TrajectoryStage.LIGAND_BUILD,
                                event=TrajectoryEvent.ROLLED_BACK,
                                component_index=component_index,
                                attempt=selected_candidate.attempt,
                            )
                        )
                        selected_candidate.trajectory.select(rollback_frame.index)
                    warning_messages.append(
                        f"Component {component_index}: long refinement failed "
                        "the basic geometry gate; retaining the "
                        "medium-optimized candidate"
                    )
                elif (
                    refined_piercing_count
                    <= selected_candidate.untangling.final_piercing_count
                ):
                    selected_candidate = _LigandCandidate(
                        coordinates=_copy_coordinates(component_mol.coordinates),
                        energy=float(refined.energy),
                        attempt=selected_candidate.attempt,
                        untangling=replace(
                            selected_candidate.untangling,
                            final_piercing_count=refined_piercing_count,
                            minimum_piercing_count=min(
                                selected_candidate.untangling.minimum_piercing_count,
                                refined_piercing_count,
                            ),
                            resolved=(
                                refined_state is not geo.PiercingState.PIERCES
                            ),
                        ),
                        trajectory=selected_candidate.trajectory,
                    )
                    if (
                        refined_frame is not None
                        and selected_candidate.trajectory is not None
                    ):
                        selected_candidate.trajectory.select(refined_frame.index)
                else:
                    component_mol.coordinates = selected_candidate.coordinates
                    if selected_candidate.trajectory is not None:
                        rollback_frame = (
                            selected_candidate.trajectory.record_molecule(
                                component_mol,
                                stage=TrajectoryStage.LIGAND_BUILD,
                                event=TrajectoryEvent.ROLLED_BACK,
                                component_index=component_index,
                                attempt=selected_candidate.attempt,
                            )
                        )
                        selected_candidate.trajectory.select(rollback_frame.index)
                    warning_messages.append(
                        f"Component {component_index}: long refinement increased "
                        "the confirmed piercing count; retaining the "
                        "pre-refinement closed-topology frame"
                    )

        warning_messages.extend(
            f"Component {component_index}: {message}"
            for message in selected_candidate.untangling.warning_messages
        )
        selected_untangling_reports.append(selected_candidate.untangling)
        component_mol.coordinates = selected_candidate.coordinates
        clone_mol.update_atoms_attrs_from_id_dict(
            {
                atom.id: {"coordinates": atom.coordinates}
                for atom in component_mol.atoms
            }
        )

    diagnostics = ComplexBuildDiagnostics(
        attempt_count=total_attempts,
        accepted_candidates=total_accepted,
        rejected_candidates=tuple(rejections),
        elapsed_seconds=time.monotonic() - started,
        warning_messages=tuple(warning_messages),
        ligand_untangling=tuple(selected_untangling_reports),
    )
    return clone_mol.coordinates, diagnostics


# Spawn-worker entry points and IPC lifecycle management.


def _build_ligand_proxies_worker(
    mol: "Molecule",
    connection: Connection,
    max_attempts: int,
    candidate_warmup_steps: int,
    candidate_score_steps: int,
    best_candidate_refine_steps: int,
    effective_forcefield: str,
    seed: Optional[int],
    ligand_untangling_attempts: int = 20,
    perturb_sigma: float = 0.5,
    record_ligand_trajectories: bool = False,
) -> None:
    """Child-process boundary that always sends one structured envelope."""
    _run_ligand_proxy_worker(
        mol,
        connection,
        max_attempts,
        candidate_warmup_steps,
        candidate_score_steps,
        best_candidate_refine_steps,
        effective_forcefield,
        seed,
        ligand_untangling_attempts,
        perturb_sigma,
        record_ligand_trajectories,
        seed_initializer=_seed_openbabel_random,
    )


def _run_ligand_proxy_worker(
    mol: "Molecule",
    connection: Connection,
    max_attempts: int,
    candidate_warmup_steps: int,
    candidate_score_steps: int,
    best_candidate_refine_steps: int,
    effective_forcefield: str,
    seed: Optional[int],
    ligand_untangling_attempts: int = 20,
    perturb_sigma: float = 0.5,
    record_ligand_trajectories: bool = False,
    *,
    seed_initializer: _SeedInitializer,
) -> None:
    """Run the shared ligand-proxy worker body with an explicit RNG adapter."""
    trajectory_attempts: Optional[list[ForceFieldTrajectory]] = (
        [] if record_ligand_trajectories else None
    )
    try:
        if seed is not None:
            seed_initializer(seed)
        coordinates, diagnostics = _build_ligand_proxies(
            mol,
            max_attempts=max_attempts,
            candidate_warmup_steps=candidate_warmup_steps,
            candidate_score_steps=candidate_score_steps,
            best_candidate_refine_steps=best_candidate_refine_steps,
            effective_forcefield=effective_forcefield,
            seed=seed,
            ligand_untangling_attempts=ligand_untangling_attempts,
            perturb_sigma=perturb_sigma,
            trajectory_attempts=trajectory_attempts,
        )
        result = BuildWorkerResult(
            status="ok",
            coordinates=coordinates,
            diagnostics=diagnostics,
            ligand_build_attempts=tuple(trajectory_attempts or ()),
        )
    except Exception as exc:
        result = BuildWorkerResult(
            status="error",
            diagnostics=getattr(exc, "diagnostics", None),
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback=traceback_module.format_exc(),
            ligand_build_attempts=tuple(trajectory_attempts or ()),
        )
    try:
        connection.send(result)
    finally:
        connection.close()


def _seeded_ob_build_worker(
    mol: "Molecule",
    connection: Connection,
    seed: int,
) -> None:
    """Run OBBuilder in a fresh process whose static RNG starts from ``seed``."""
    _run_seeded_ob_build_worker(
        mol,
        connection,
        seed,
        seed_initializer=_seed_openbabel_random,
    )


def _run_seeded_ob_build_worker(
    mol: "Molecule",
    connection: Connection,
    seed: int,
    *,
    seed_initializer: _SeedInitializer,
) -> None:
    """Run the shared OBBuilder worker body with an explicit RNG adapter."""
    try:
        seed_initializer(seed)
        _ob_build(mol)
        result = BuildWorkerResult(
            status="ok",
            coordinates=mol.coordinates,
        )
    except Exception as exc:
        result = BuildWorkerResult(
            status="error",
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback=traceback_module.format_exc(),
        )
    try:
        connection.send(result)
    finally:
        connection.close()


def _receive_worker_result(
    process: mp.Process,
    receive_connection: Connection,
    send_connection: Connection,
    *,
    timeout: float,
    seed: Optional[int] = None,
    require_diagnostics: bool = True,
    worker_error_type: Union[
        type[BuildWorkerError], type[ComplexBuildWorkerError]
    ] = ComplexBuildWorkerError,
    timeout_error_type: Union[
        type[BuildTimeoutError], type[ComplexBuildTimeoutError]
    ] = ComplexBuildTimeoutError,
    operation: str = "building complex geometry",
) -> BuildWorkerResult:
    result = None
    started = False
    try:
        with _WORKER_LIFECYCLE_LOCK:
            previous_seed = os.environ.get("OB_RANDOM_SEED")
            if seed is not None:
                os.environ["OB_RANDOM_SEED"] = str(seed)
            try:
                process.start()
            finally:
                if seed is not None:
                    if previous_seed is None:
                        os.environ.pop("OB_RANDOM_SEED", None)
                    else:
                        os.environ["OB_RANDOM_SEED"] = previous_seed
        started = True
        send_connection.close()
        if not receive_connection.poll(timeout):
            raise timeout_error_type(
                f"Timed out after {timeout:g} seconds while {operation}"
            )
        try:
            result = receive_connection.recv()
        except EOFError as exc:
            raise worker_error_type(
                "WorkerProtocolError",
                "The build worker closed its pipe without a result",
                None,
            ) from exc
        exited = wait_for_connections(
            (process.sentinel,),
            timeout=_WORKER_EXIT_GRACE_SECONDS,
        )
        if not exited:
            raise worker_error_type(
                "WorkerShutdownError",
                "The build worker sent a result but did not terminate",
                None,
            )
        # ``Process.start()`` runs multiprocessing's global child cleanup.
        # Reap under the same lock so another thread cannot win waitpid() and
        # leave this Process object briefly reporting ``exitcode is None``.
        with _WORKER_LIFECYCLE_LOCK:
            process.join(timeout=_WORKER_EXIT_GRACE_SECONDS)
            exitcode = process.exitcode
        if exitcode is None:
            raise worker_error_type(
                "WorkerShutdownError",
                "The build worker did not expose an exit code after termination",
                None,
            )
        if exitcode != 0:
            raise worker_error_type(
                "WorkerExitError",
                f"The build worker exited with code {exitcode}",
                None,
            )
        if not isinstance(result, BuildWorkerResult):
            raise worker_error_type(
                "WorkerProtocolError",
                "The build worker returned an invalid result envelope",
                None,
            )
        if result.status not in ("ok", "error"):
            raise worker_error_type(
                "WorkerProtocolError",
                f"The build worker returned an invalid status: {result.status!r}",
                None,
            )
        if result.status == "error":
            raise worker_error_type(
                result.error_type or "WorkerError",
                result.error_message or "Unknown build worker failure",
                result.traceback,
                result.diagnostics,
            )
        if result.coordinates is None or (
            require_diagnostics and result.diagnostics is None
        ):
            required_fields = "coordinates and diagnostics"
            if not require_diagnostics:
                required_fields = "coordinates"
            raise worker_error_type(
                "WorkerProtocolError",
                f"A successful build worker result requires {required_fields}",
                None,
            )
        return result
    finally:
        if started:
            with _WORKER_LIFECYCLE_LOCK:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5.0)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5.0)
        receive_connection.close()
        send_connection.close()


def _validated_worker_coordinates(
    result: BuildWorkerResult,
    *,
    expected_atom_count: int,
    worker_error_type: Union[
        type[BuildWorkerError], type[ComplexBuildWorkerError]
    ] = ComplexBuildWorkerError,
) -> np.ndarray:
    coordinates = np.asarray(result.coordinates, dtype=float)
    expected_shape = (expected_atom_count, 3)
    if coordinates.shape != expected_shape or not np.all(np.isfinite(coordinates)):
        raise worker_error_type(
            "WorkerProtocolError",
            "A successful build worker result must contain finite coordinates "
            f"with shape {expected_shape}, got {coordinates.shape}",
            None,
            result.diagnostics,
        )
    return coordinates


def _seeded_ob_build_coordinates(
    mol: "Molecule",
    seed: int,
    *,
    timeout: float,
    worker_target: _SeededBuildWorker,
) -> np.ndarray:
    """Build coordinates in an isolated process for repeatable Open Babel RNG."""
    worker_mol = _make_worker_mol(mol)
    context = mp.get_context("spawn")
    receive_connection, send_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=worker_target,
        args=(worker_mol, send_connection, seed),
    )
    result = _receive_worker_result(
        process,
        receive_connection,
        send_connection,
        timeout=timeout,
        seed=seed,
        require_diagnostics=False,
        worker_error_type=BuildWorkerError,
        timeout_error_type=BuildTimeoutError,
        operation="building seeded 3D coordinates",
    )
    return _validated_worker_coordinates(
        result,
        expected_atom_count=len(mol.atoms),
        worker_error_type=BuildWorkerError,
    )


# Non-committing workflow stages with explicit worker injection.


def _finalize_trajectory(
    mol: "Molecule",
    trajectory: ForceFieldTrajectory,
    *,
    ligand_build_attempts: Sequence[ForceFieldTrajectory] = (),
    save_movie: bool,
    trajectory_path: Optional[TrajectoryPath],
) -> ForceFieldTrajectoryArchive:
    """Persist and expose one completed trajectory without choosing its frame."""
    archive = ForceFieldTrajectoryArchive(
        main=trajectory,
        ligand_build_attempts=tuple(ligand_build_attempts),
    )
    if trajectory_path is not None:
        archive.write(trajectory_path)
    trajectory.materialize(mol, keep_all=save_movie)
    return archive


def _prepare_complex_working_mol(
    mol: "Molecule",
    *,
    effective_forcefield: str,
    max_attempts: int,
    candidate_warmup_steps: int,
    candidate_score_steps: int,
    best_candidate_refine_steps: int,
    ligand_untangling_attempts: int = 20,
    coordination_restoration_attempts: int = 20,
    coordination_relaxation_steps: int = 100,
    timeout: float,
    add_hydrogens: bool,
    seed: Optional[int],
    perturb_sigma: float = 0.5,
    trajectory_start: TrajectoryStart = TrajectoryStart.COORDINATION_RESTORATION,
    coordination_geometry: Optional[str],
    worker_target: _ComplexBuildWorker,
) -> _PreparedComplex:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if ligand_untangling_attempts < 1:
        raise ValueError("ligand_untangling_attempts must be at least 1")
    if coordination_restoration_attempts < 1:
        raise ValueError("coordination_restoration_attempts must be at least 1")
    if min(
        candidate_warmup_steps,
        candidate_score_steps,
        best_candidate_refine_steps,
    ) < 1:
        raise ValueError("all candidate optimization step counts must be at least 1")
    if coordination_relaxation_steps < 1:
        raise ValueError("coordination_relaxation_steps must be at least 1")
    if perturb_sigma < 0.0:
        raise ValueError("perturb_sigma must be non-negative")
    if timeout <= 0.0:
        raise ValueError("timeout must be positive")
    working_mol = _hydrogenated_working_copy(
        mol,
        add_hydrogens=add_hydrogens,
        seed=seed,
    )
    trajectory = ForceFieldTrajectory.from_molecule(
        working_mol,
        start=trajectory_start,
    )
    worker_mol = _make_worker_mol(working_mol)
    context = mp.get_context("spawn")
    receive_connection, send_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=worker_target,
        args=(
            worker_mol,
            send_connection,
            max_attempts,
            candidate_warmup_steps,
            candidate_score_steps,
            best_candidate_refine_steps,
            effective_forcefield,
            seed,
            ligand_untangling_attempts,
            perturb_sigma,
            trajectory_start is TrajectoryStart.LIGAND_BUILD,
        ),
    )
    result = _receive_worker_result(
        process,
        receive_connection,
        send_connection,
        timeout=timeout,
        seed=seed,
    )
    working_mol.coordinates = _validated_worker_coordinates(
        result,
        expected_atom_count=len(working_mol.atoms),
    )
    diagnostics = cast(ComplexBuildDiagnostics, result.diagnostics)
    for message in diagnostics.warning_messages:
        warnings.warn(message, ComplexBuildWarning, stacklevel=3)
    if coordination_geometry is not None:
        prepare_coordination_geometry(
            working_mol, strategy=coordination_geometry, seed=seed
        )
    restoration_started = time.monotonic()
    restoration = _restore_coordination_bonds_incrementally(
        working_mol,
        effective_forcefield,
        attempt_limit=coordination_restoration_attempts,
        relaxation_steps=coordination_relaxation_steps,
        perturb_sigma=perturb_sigma,
        rng=np.random.default_rng(seed),
        trajectory=trajectory,
    )
    diagnostics = replace(
        diagnostics,
        elapsed_seconds=(
            diagnostics.elapsed_seconds
            + time.monotonic()
            - restoration_started
        ),
        warning_messages=(
            diagnostics.warning_messages
            + restoration.report.warning_messages
        ),
        coordination_restoration=restoration.report,
    )
    for message in restoration.report.warning_messages:
        warnings.warn(message, ComplexBuildWarning, stacklevel=3)
    return _PreparedComplex(
        mol=working_mol,
        diagnostics=diagnostics,
        trajectory=trajectory,
        ligand_build_attempts=result.ligand_build_attempts,
    )


def _optimize_working_mol(
    working_mol: "Molecule",
    *,
    requested_forcefield: Optional[str],
    effective_forcefield: str,
    algorithm: OptimizationAlgorithm,
    epochs: int,
    steps_per_epoch: int,
    quality_level: AcceptanceLevel,
    topology_reference: TopologyReference,
    quality_thresholds: Optional[StructureAcceptanceThresholds],
    seed: Optional[int],
    perturb_interval: Optional[int],
    perturb_sigma: float,
    save_movie: bool,
    increasing_vdw: bool,
    vdw_cutoff_start: float,
    vdw_cutoff_end: float,
    stop_on_ring_piercing: bool = False,
    trajectory: ForceFieldTrajectory,
    trajectory_stage: TrajectoryStage = TrajectoryStage.FINAL_OPTIMIZATION,
    trajectory_attempt: Optional[int] = None,
) -> ForceFieldRunReport:
    optimizer = _OpenBabelOptimizer(
        requested_forcefield,
        effective_forcefield,
        algorithm=algorithm,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        perturb_interval=perturb_interval,
        perturb_sigma=perturb_sigma,
        save_movie=save_movie,
        increasing_vdw=increasing_vdw,
        vdw_cutoff_start=vdw_cutoff_start,
        vdw_cutoff_end=vdw_cutoff_end,
        seed=seed,
        stop_on_ring_piercing=stop_on_ring_piercing,
    )
    return optimizer.optimize(
        working_mol,
        quality_level=quality_level,
        topology_reference=topology_reference,
        quality_thresholds=quality_thresholds,
        trajectory=trajectory,
        trajectory_stage=trajectory_stage,
        trajectory_attempt=trajectory_attempt,
    )


def _combine_forcefield_run_reports(
    reports: Sequence[ForceFieldRunReport],
) -> ForceFieldRunReport:
    """Combine sequential optimizer segments around topology repairs."""
    final_report = reports[-1]
    preceding_epochs = sum(report.epochs_completed for report in reports[:-1])
    return replace(
        final_report,
        epochs_completed=sum(report.epochs_completed for report in reports),
        steps_submitted=sum(report.steps_submitted for report in reports),
        initialization_steps=sum(
            report.initialization_steps for report in reports
        ),
        best_epoch=preceding_epochs + final_report.best_epoch,
        epoch_energies=tuple(
            energy
            for report in reports
            for energy in report.epoch_energies
        ),
        epoch_quality_reports=tuple(
            quality_report
            for report in reports
            for quality_report in report.epoch_quality_reports
        ),
    )


def _summarize_complex_untangling(
    reports: Sequence[RingUntanglingReport],
    *,
    attempt_limit: int,
    final_state: geo.PiercingState,
    final_piercing_count: int,
) -> RingUntanglingReport:
    """Summarize every repair pass against the final optimized coordinates."""
    warning_messages = [
        message
        for report in reports
        for message in report.warning_messages
    ]
    if final_state is geo.PiercingState.PIERCES:
        warning_messages.append(
            "Confirmed bond-ring piercing remains after full-complex "
            "untangling; retaining the final optimized frame"
        )
    elif final_state is geo.PiercingState.UNDETERMINED:
        warning_messages.append(
            "The final complex contains a mathematically undetermined "
            "bond-ring relation"
        )
    minimum_count = min(
        (report.minimum_piercing_count for report in reports),
        default=final_piercing_count,
    )
    return RingUntanglingReport(
        attempt_limit=attempt_limit,
        attempts_completed=sum(report.attempts_completed for report in reports),
        initial_piercing_count=reports[0].initial_piercing_count,
        final_piercing_count=final_piercing_count,
        minimum_piercing_count=min(minimum_count, final_piercing_count),
        resolved=final_state is not geo.PiercingState.PIERCES,
        warning_messages=_unique_messages(warning_messages),
    )


def _optimize_complex_working_mol(
    working_mol: "Molecule",
    *,
    requested_forcefield: Optional[str],
    effective_forcefield: str,
    algorithm: OptimizationAlgorithm,
    epochs: int,
    steps_per_epoch: int,
    complex_untangling_attempts: int,
    quality_level: AcceptanceLevel,
    topology_reference: TopologyReference,
    quality_thresholds: Optional[StructureAcceptanceThresholds],
    seed: Optional[int],
    perturb_interval: Optional[int],
    perturb_sigma: float,
    save_movie: bool,
    increasing_vdw: bool,
    vdw_cutoff_start: float,
    vdw_cutoff_end: float,
    trajectory: ForceFieldTrajectory,
) -> ForceFieldRunReport:
    """Interleave bounded untangling with complete-complex relaxation."""
    if complex_untangling_attempts < 1:
        raise ValueError("complex_untangling_attempts must be at least 1")
    rng = np.random.default_rng(seed)
    remaining_attempts = complex_untangling_attempts
    remaining_epochs = epochs
    untangling_reports = []
    optimization_reports = []
    consecutive_stalled_repairs = 0

    while True:
        untangling = _untangle_ring_piercings(
            working_mol,
            effective_forcefield,
            attempt_limit=remaining_attempts,
            short_steps=steps_per_epoch,
            settling_steps=0,
            perturb_sigma=perturb_sigma,
            rng=rng,
            ring_scope="ligand_skeleton",
            initial_energy=(
                optimization_reports[-1].best_energy
                if optimization_reports
                else float("nan")
            ),
            trajectory=trajectory,
        )
        untangling_reports.append(untangling.report)
        remaining_attempts -= untangling.report.attempts_completed

        segment_epochs = remaining_epochs if remaining_epochs else 1
        trajectory_stage = (
            TrajectoryStage.FINAL_OPTIMIZATION
            if untangling.report.resolved
            else TrajectoryStage.COMPLEX_UNTANGLING
        )
        report = _optimize_working_mol(
            working_mol,
            requested_forcefield=requested_forcefield,
            effective_forcefield=effective_forcefield,
            algorithm=algorithm,
            epochs=segment_epochs,
            steps_per_epoch=steps_per_epoch,
            quality_level=quality_level,
            topology_reference=topology_reference,
            quality_thresholds=quality_thresholds,
            seed=seed,
            perturb_interval=perturb_interval,
            perturb_sigma=perturb_sigma,
            save_movie=save_movie,
            increasing_vdw=increasing_vdw,
            vdw_cutoff_start=vdw_cutoff_start,
            vdw_cutoff_end=vdw_cutoff_end,
            stop_on_ring_piercing=True,
            trajectory=trajectory,
            trajectory_stage=trajectory_stage,
            trajectory_attempt=len(optimization_reports),
        )
        optimization_reports.append(report)
        remaining_epochs = max(remaining_epochs - report.epochs_completed, 0)
        final_state, final_scan = _scan_confirmed_ring_piercings(
            working_mol,
            ring_scope="ligand_skeleton",
        )
        final_piercing_count = _piercing_count(final_scan)
        if final_state is not geo.PiercingState.PIERCES:
            break
        if remaining_attempts == 0:
            break
        consecutive_stalled_repairs = (
            consecutive_stalled_repairs + 1
            if untangling.report.attempts_completed == 0
            else 0
        )
        if consecutive_stalled_repairs >= 2:
            break

    untangling_report = _summarize_complex_untangling(
        untangling_reports,
        attempt_limit=complex_untangling_attempts,
        final_state=final_state,
        final_piercing_count=final_piercing_count,
    )
    for message in untangling_report.warning_messages:
        warnings.warn(message, GeometryQualityWarning, stacklevel=3)
    return replace(
        _combine_forcefield_run_reports(optimization_reports),
        untangling=untangling_report,
    )


def _complexes_build_workflow(
    mol: "Molecule",
    forcefield: Optional[str] = None,
    *,
    algorithm: OptimizationAlgorithm = "conjugate",
    epochs: int = 100,
    steps_per_epoch: int = 100,
    max_attempts: int = 50,
    candidate_warmup_steps: int = 500,
    candidate_score_steps: int = 1000,
    best_candidate_refine_steps: int = 3000,
    ligand_untangling_attempts: int = 20,
    coordination_restoration_attempts: int = 20,
    coordination_relaxation_steps: int = 100,
    complex_untangling_attempts: int = 30,
    timeout: float = 1000.0,
    add_hydrogens: bool = True,
    quality_level: AcceptanceLevel = "standard",
    quality_thresholds: Optional[StructureAcceptanceThresholds] = None,
    seed: Optional[int] = None,
    perturb_interval: Optional[int] = None,
    perturb_sigma: float = 0.5,
    save_movie: bool = False,
    trajectory_start: TrajectoryStart = TrajectoryStart.COORDINATION_RESTORATION,
    trajectory_path: Optional[TrajectoryPath] = None,
    increasing_vdw: bool = False,
    vdw_cutoff_start: float = 0.0,
    vdw_cutoff_end: float = 12.5,
    coordination_geometry: Optional[str] = None,
    worker_target: _ComplexBuildWorker,
) -> ComplexBuildReport:
    """Build, optimize, validate, and atomically commit a complete complex."""
    _require_explicit_complex(mol)
    topology_reference = capture_topology(
        mol,
        allow_added_hydrogens=add_hydrogens,
    )
    effective_forcefield = _resolve_complex_forcefield(forcefield)
    prepared = _prepare_complex_working_mol(
        mol,
        effective_forcefield=effective_forcefield,
        max_attempts=max_attempts,
        candidate_warmup_steps=candidate_warmup_steps,
        candidate_score_steps=candidate_score_steps,
        best_candidate_refine_steps=best_candidate_refine_steps,
        ligand_untangling_attempts=ligand_untangling_attempts,
        coordination_restoration_attempts=coordination_restoration_attempts,
        coordination_relaxation_steps=coordination_relaxation_steps,
        timeout=timeout,
        add_hydrogens=add_hydrogens,
        seed=seed,
        perturb_sigma=perturb_sigma,
        trajectory_start=trajectory_start,
        coordination_geometry=coordination_geometry,
        worker_target=worker_target,
    )
    optimization_report = _optimize_complex_working_mol(
        prepared.mol,
        requested_forcefield=forcefield,
        effective_forcefield=effective_forcefield,
        algorithm=algorithm,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        complex_untangling_attempts=complex_untangling_attempts,
        quality_level=quality_level,
        topology_reference=topology_reference,
        quality_thresholds=quality_thresholds,
        seed=seed,
        perturb_interval=perturb_interval,
        perturb_sigma=perturb_sigma,
        save_movie=save_movie,
        increasing_vdw=increasing_vdw,
        vdw_cutoff_start=vdw_cutoff_start,
        vdw_cutoff_end=vdw_cutoff_end,
        trajectory=prepared.trajectory,
    )
    trajectory_archive = _finalize_trajectory(
        prepared.mol,
        prepared.trajectory,
        ligand_build_attempts=prepared.ligand_build_attempts,
        save_movie=save_movie,
        trajectory_path=trajectory_path,
    )
    optimization_report = replace(
        optimization_report,
        trajectory=trajectory_archive,
    )
    report = ComplexBuildReport(
        requested_forcefield=forcefield,
        effective_forcefield=effective_forcefield,
        build=prepared.diagnostics,
        optimization=optimization_report,
        quality_report=optimization_report.quality_report,
        trajectory=trajectory_archive,
    )
    _commit_working_copy(mol, prepared.mol)
    return report


# Public force-field and coordination interfaces, ordered from primitives to workflows.


def capture_topology(
    mol: "Molecule",
    *,
    allow_added_hydrogens: bool = True,
) -> TopologyReference:
    """Capture immutable topology expected to survive a force-field workflow."""
    atoms = tuple(mol.atoms)
    atom_indices = _atom_index_map(atoms)
    atom_signatures = tuple(
        AtomTopologySignature(
            index=index,
            atom_id=int(atom.id),
            atomic_number=int(atom.atomic_number),
            formal_charge=int(atom.formal_charge),
        )
        for index, atom in enumerate(atoms)
    )
    bond_signatures = tuple(sorted(
        (
            _topology_bond_signature(bond, atom_indices)
            for bond in mol.bonds
        ),
        key=lambda signature: signature.atom_indices,
    ))
    return TopologyReference(
        atom_signatures,
        bond_signatures,
        allow_added_hydrogens=allow_added_hydrogens,
    )


def evaluate_structure_acceptance(
    mol: "Molecule",
    *,
    level: AcceptanceLevel = "standard",
    topology_reference: Optional[TopologyReference] = None,
    forcefield_report: Optional[ForceFieldAcceptanceEvidence] = None,
    forcefield_stage: ForceFieldStage = "final",
    thresholds: Optional[StructureAcceptanceThresholds] = None,
) -> ForceFieldValidationReport:
    """Apply chemistry and force-field acceptance policy to geometry facts."""
    if level not in ("off", "basic", "standard", "strict"):
        raise ValueError(f"Unknown structure acceptance level: {level!r}")
    if forcefield_stage not in ("candidate", "final"):
        raise ValueError(f"Unknown force-field stage: {forcefield_stage!r}")

    limits = _resolve_acceptance_thresholds(thresholds)
    atoms = tuple(mol.atoms)
    coordinates = np.asarray(mol.coordinates, dtype=float)
    coordinate_checks, finite_ok, metrics = _coordinate_acceptance_section(
        mol,
        atoms,
        coordinates,
    )
    checks = list(coordinate_checks)

    if topology_reference is not None:
        checks.extend(_topology_checks(mol, topology_reference))
    checks.extend(_forcefield_acceptance_checks(
        forcefield_report,
        level,
        limits,
        forcefield_stage,
    ))

    if not finite_ok:
        passed = all(
            check.passed or check.severity != "error" for check in checks
        )
        return ForceFieldValidationReport(level, passed, tuple(checks), metrics)

    atom_pair_checks, atom_pair_metrics = (
        _atom_pair_distance_acceptance_section(mol, level, limits)
    )
    checks.extend(atom_pair_checks)
    metrics.update(atom_pair_metrics)

    if level == "off":
        passed = all(
            check.passed or check.severity != "error" for check in checks
        )
        return ForceFieldValidationReport(level, passed, tuple(checks), metrics)

    bond_checks, bond_metrics = _bond_geometry_acceptance_section(
        mol,
        atoms,
        coordinates,
        level,
        limits,
    )
    checks.extend(bond_checks)
    metrics.update(bond_metrics)

    if level in ("standard", "strict"):
        bond_ring_checks, bond_ring_metrics = (
            _bond_ring_coordination_acceptance_section(
                mol,
                atoms,
                coordinates,
            )
        )
        checks.extend(bond_ring_checks)
        metrics.update(bond_ring_metrics)

    passed = all(check.passed or check.severity != "error" for check in checks)
    return ForceFieldValidationReport(level, passed, tuple(checks), metrics)


def is_structure_accepted(
    mol: "Molecule",
    *,
    level: AcceptanceLevel = "standard",
    topology_reference: Optional[TopologyReference] = None,
    forcefield_report: Optional[ForceFieldAcceptanceEvidence] = None,
    forcefield_stage: ForceFieldStage = "final",
    thresholds: Optional[StructureAcceptanceThresholds] = None,
) -> bool:
    """Return the result of :func:`evaluate_structure_acceptance`."""
    return evaluate_structure_acceptance(
        mol,
        level=level,
        topology_reference=topology_reference,
        forcefield_report=forcefield_report,
        forcefield_stage=forcefield_stage,
        thresholds=thresholds,
    ).passed


def perturb(
    mol: "Molecule",
    *,
    sigma: float = 0.5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Perturb current coordinates in place with a local random generator."""
    coordinates = _perturbed_coordinates(
        mol.coordinates,
        sigma=sigma,
        rng=np.random.default_rng(seed),
    )
    mol.coordinates = coordinates
    return coordinates


def collect_coordination_environments(
    mol: "Molecule",
) -> Tuple[CoordinationEnvironment, ...]:
    """Describe explicit metal--donor connectivity without assigning geometry."""
    metal_donor_pairs = tuple(_iter_metal_donor_pairs(mol))
    ligand_graph = mol.graph.copy()
    ligand_graph.remove_edges_from(
        (metal.idx, donor.idx) for metal, donor in metal_donor_pairs
    )
    component_by_atom = {}
    for component_index, nodes in enumerate(nx.connected_components(ligand_graph)):
        for atom_idx in nodes:
            component_by_atom[atom_idx] = component_index

    donors_by_metal = {metal.idx: [] for metal in mol.metals}
    for metal, donor in metal_donor_pairs:
        donors_by_metal[metal.idx].append(donor.idx)

    environments = []
    for metal in mol.metals:
        donors = sorted(donors_by_metal[metal.idx])
        grouped = {}
        for donor_idx in donors:
            grouped.setdefault(component_by_atom[donor_idx], []).append(donor_idx)
        environments.append(
            CoordinationEnvironment(
                metal_idx=metal.idx,
                donor_indices=tuple(donors),
                coordination_number=len(donors),
                metal_atomic_number=metal.atomic_number,
                metal_formal_charge=metal.formal_charge,
                donor_atomic_numbers=tuple(
                    mol.atoms[index].atomic_number for index in donors
                ),
                chelate_groups=tuple(
                    tuple(indices) for _, indices in sorted(grouped.items())
                ),
            )
        )
    return tuple(environments)


def prepare_coordination_geometry(
    mol: "Molecule",
    *,
    environments: Optional[Tuple[CoordinationEnvironment, ...]] = None,
    strategy: Optional[str] = None,
    seed: Optional[int] = None,
) -> CoordinationGeometryResult:
    """Reserved hook for coordination-number-aware initial placement."""
    _require_explicit_complex(mol)
    raise NotImplementedError(
        "Coordination-number-aware placement is reserved but not implemented"
    )


def _build3d_workflow(
    mol: "Molecule",
    *,
    add_hydrogens: bool = True,
    seed: Optional[int] = None,
    timeout: float = 1000.0,
    worker_target: _SeededBuildWorker,
) -> Build3DReport:
    """Generate initial 3D coordinates with OBBuilder, without optimization."""
    topology_reference = capture_topology(
        mol,
        allow_added_hydrogens=add_hydrogens,
    )
    initial_hydrogens = len(mol.hydrogens)
    working_mol = _hydrogenated_working_copy(
        mol,
        add_hydrogens=add_hydrogens,
        seed=seed,
    )
    if seed is None:
        _ob_build(working_mol)
    else:
        working_mol.coordinates = _seeded_ob_build_coordinates(
            working_mol,
            seed,
            timeout=timeout,
            worker_target=worker_target,
        )
    quality_report = evaluate_structure_acceptance(
        working_mol,
        level="off",
        topology_reference=topology_reference,
    )
    if not quality_report.passed:
        raise GeometryQualityError(quality_report)
    report = Build3DReport(
        atom_count=len(working_mol.atoms),
        added_hydrogen_count=len(working_mol.hydrogens) - initial_hydrogens,
        quality_report=quality_report,
    )
    _commit_working_copy(mol, working_mol)
    return report


def build3d(
    mol: "Molecule",
    *,
    add_hydrogens: bool = True,
    seed: Optional[int] = None,
    timeout: float = 1000.0,
) -> Build3DReport:
    """Generate initial 3D coordinates with OBBuilder, without optimization."""
    return _build3d_workflow(
        mol,
        add_hydrogens=add_hydrogens,
        seed=seed,
        timeout=timeout,
        worker_target=_seeded_ob_build_worker,
    )


def optimize(
    mol: "Molecule",
    forcefield: Optional[str] = "UFF",
    *,
    algorithm: OptimizationAlgorithm = "conjugate",
    epochs: int = 1,
    steps_per_epoch: int = 100,
    add_hydrogens: bool = True,
    quality_level: AcceptanceLevel = "standard",
    quality_thresholds: Optional[StructureAcceptanceThresholds] = None,
    seed: Optional[int] = None,
    perturb_interval: Optional[int] = None,
    perturb_sigma: float = 0.5,
    save_movie: bool = False,
    trajectory_start: TrajectoryStart = TrajectoryStart.FINAL_OPTIMIZATION,
    trajectory_path: Optional[TrajectoryPath] = None,
    increasing_vdw: bool = False,
    vdw_cutoff_start: float = 0.0,
    vdw_cutoff_end: float = 12.5,
) -> ForceFieldRunReport:
    """Run the ordinary Open Babel optimizer, including on explicit complexes."""
    topology_reference = capture_topology(
        mol,
        allow_added_hydrogens=add_hydrogens,
    )
    working_mol = _hydrogenated_working_copy(
        mol,
        add_hydrogens=add_hydrogens,
        seed=seed,
    )
    trajectory = ForceFieldTrajectory.from_molecule(
        working_mol,
        start=trajectory_start,
    )
    effective_forcefield = _resolve_organic_forcefield(forcefield)
    report = _optimize_working_mol(
        working_mol,
        requested_forcefield=forcefield,
        effective_forcefield=effective_forcefield,
        algorithm=algorithm,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        quality_level=quality_level,
        topology_reference=topology_reference,
        quality_thresholds=quality_thresholds,
        seed=seed,
        perturb_interval=perturb_interval,
        perturb_sigma=perturb_sigma,
        save_movie=save_movie,
        increasing_vdw=increasing_vdw,
        vdw_cutoff_start=vdw_cutoff_start,
        vdw_cutoff_end=vdw_cutoff_end,
        trajectory=trajectory,
    )
    trajectory_archive = _finalize_trajectory(
        working_mol,
        trajectory,
        save_movie=save_movie,
        trajectory_path=trajectory_path,
    )
    report = replace(report, trajectory=trajectory_archive)
    _commit_working_copy(mol, working_mol)
    return report


def _build_complex3d_workflow(
    mol: "Molecule",
    forcefield: Optional[str] = None,
    *,
    max_attempts: int = 50,
    candidate_warmup_steps: int = 500,
    candidate_score_steps: int = 1000,
    best_candidate_refine_steps: int = 3000,
    ligand_untangling_attempts: int = 20,
    coordination_restoration_attempts: int = 20,
    coordination_relaxation_steps: int = 100,
    timeout: float = 1000.0,
    add_hydrogens: bool = True,
    seed: Optional[int] = None,
    perturb_sigma: float = 0.5,
    save_movie: bool = False,
    trajectory_start: TrajectoryStart = TrajectoryStart.COORDINATION_RESTORATION,
    trajectory_path: Optional[TrajectoryPath] = None,
    coordination_geometry: Optional[str] = None,
    worker_target: _ComplexBuildWorker,
) -> ComplexBuildReport:
    """Build ligand proxies and restore the complete complex topology."""
    _require_explicit_complex(mol)
    topology_reference = capture_topology(
        mol,
        allow_added_hydrogens=add_hydrogens,
    )
    effective_forcefield = _resolve_complex_forcefield(forcefield)
    prepared = _prepare_complex_working_mol(
        mol,
        effective_forcefield=effective_forcefield,
        max_attempts=max_attempts,
        candidate_warmup_steps=candidate_warmup_steps,
        candidate_score_steps=candidate_score_steps,
        best_candidate_refine_steps=best_candidate_refine_steps,
        ligand_untangling_attempts=ligand_untangling_attempts,
        coordination_restoration_attempts=coordination_restoration_attempts,
        coordination_relaxation_steps=coordination_relaxation_steps,
        timeout=timeout,
        add_hydrogens=add_hydrogens,
        seed=seed,
        perturb_sigma=perturb_sigma,
        trajectory_start=trajectory_start,
        coordination_geometry=coordination_geometry,
        worker_target=worker_target,
    )
    quality_report = evaluate_structure_acceptance(
        prepared.mol,
        level="off",
        topology_reference=topology_reference,
    )
    if not quality_report.passed:
        raise GeometryQualityError(quality_report)
    trajectory_archive = _finalize_trajectory(
        prepared.mol,
        prepared.trajectory,
        ligand_build_attempts=prepared.ligand_build_attempts,
        save_movie=save_movie,
        trajectory_path=trajectory_path,
    )
    report = ComplexBuildReport(
        requested_forcefield=forcefield,
        effective_forcefield=effective_forcefield,
        build=prepared.diagnostics,
        optimization=None,
        quality_report=quality_report,
        trajectory=trajectory_archive,
    )
    _commit_working_copy(mol, prepared.mol)
    return report


def build_complex3d(
    mol: "Molecule",
    forcefield: Optional[str] = None,
    *,
    candidate_count: Optional[int] = None,
    max_attempts: int = 50,
    candidate_warmup_steps: int = 500,
    candidate_score_steps: int = 1000,
    best_candidate_refine_steps: int = 3000,
    ligand_untangling_attempts: int = 20,
    coordination_restoration_attempts: int = 20,
    coordination_relaxation_steps: int = 100,
    timeout: float = 1000.0,
    add_hydrogens: bool = True,
    seed: Optional[int] = None,
    perturb_sigma: float = 0.5,
    save_movie: bool = False,
    trajectory_start: TrajectoryStart = TrajectoryStart.COORDINATION_RESTORATION,
    trajectory_path: Optional[TrajectoryPath] = None,
    coordination_geometry: Optional[str] = None,
) -> ComplexBuildReport:
    """Build ligand proxies and restore the complete complex topology.

    ``candidate_count`` is reserved and currently has no effect.  The current
    workflow builds one ligand starting geometry.  A future multi-conformer
    implementation will generate independent starting conformers, optimize
    and gate them uniformly, deduplicate or cluster them by geometry, rank
    them using topology, geometry, and energy evidence, and refine the
    selected conformer.
    """
    _ = candidate_count
    return _build_complex3d_workflow(
        mol,
        forcefield,
        max_attempts=max_attempts,
        candidate_warmup_steps=candidate_warmup_steps,
        candidate_score_steps=candidate_score_steps,
        best_candidate_refine_steps=best_candidate_refine_steps,
        ligand_untangling_attempts=ligand_untangling_attempts,
        coordination_restoration_attempts=coordination_restoration_attempts,
        coordination_relaxation_steps=coordination_relaxation_steps,
        timeout=timeout,
        add_hydrogens=add_hydrogens,
        seed=seed,
        perturb_sigma=perturb_sigma,
        save_movie=save_movie,
        trajectory_start=trajectory_start,
        trajectory_path=trajectory_path,
        coordination_geometry=coordination_geometry,
        worker_target=_build_ligand_proxies_worker,
    )


def optimize_complex(
    mol: "Molecule",
    forcefield: Optional[str] = None,
    *,
    algorithm: OptimizationAlgorithm = "conjugate",
    epochs: int = 100,
    steps_per_epoch: int = 100,
    complex_untangling_attempts: int = 30,
    add_hydrogens: bool = True,
    quality_level: AcceptanceLevel = "standard",
    quality_thresholds: Optional[StructureAcceptanceThresholds] = None,
    seed: Optional[int] = None,
    perturb_interval: Optional[int] = None,
    perturb_sigma: float = 0.5,
    save_movie: bool = False,
    trajectory_start: TrajectoryStart = TrajectoryStart.COMPLEX_UNTANGLING,
    trajectory_path: Optional[TrajectoryPath] = None,
    increasing_vdw: bool = False,
    vdw_cutoff_start: float = 0.0,
    vdw_cutoff_end: float = 12.5,
) -> ForceFieldRunReport:
    """Optimize existing complex coordinates with the complex force-field policy."""
    _require_explicit_complex(mol)
    topology_reference = capture_topology(
        mol,
        allow_added_hydrogens=add_hydrogens,
    )
    working_mol = _hydrogenated_working_copy(
        mol,
        add_hydrogens=add_hydrogens,
        seed=seed,
    )
    trajectory = ForceFieldTrajectory.from_molecule(
        working_mol,
        start=trajectory_start,
    )
    effective_forcefield = _resolve_complex_forcefield(forcefield)
    report = _optimize_complex_working_mol(
        working_mol,
        requested_forcefield=forcefield,
        effective_forcefield=effective_forcefield,
        algorithm=algorithm,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        complex_untangling_attempts=complex_untangling_attempts,
        quality_level=quality_level,
        topology_reference=topology_reference,
        quality_thresholds=quality_thresholds,
        seed=seed,
        perturb_interval=perturb_interval,
        perturb_sigma=perturb_sigma,
        save_movie=save_movie,
        increasing_vdw=increasing_vdw,
        vdw_cutoff_start=vdw_cutoff_start,
        vdw_cutoff_end=vdw_cutoff_end,
        trajectory=trajectory,
    )
    trajectory_archive = _finalize_trajectory(
        working_mol,
        trajectory,
        save_movie=save_movie,
        trajectory_path=trajectory_path,
    )
    report = replace(report, trajectory=trajectory_archive)
    _commit_working_copy(mol, working_mol)
    return report


def _build_and_optimize_workflow(
    mol: "Molecule",
    forcefield: Optional[str] = "UFF",
    *,
    algorithm: OptimizationAlgorithm = "conjugate",
    epochs: int = 100,
    steps_per_epoch: int = 100,
    add_hydrogens: bool = True,
    quality_level: AcceptanceLevel = "standard",
    quality_thresholds: Optional[StructureAcceptanceThresholds] = None,
    seed: Optional[int] = None,
    timeout: float = 1000.0,
    perturb_interval: Optional[int] = None,
    perturb_sigma: float = 0.5,
    save_movie: bool = False,
    trajectory_start: Optional[TrajectoryStart] = None,
    trajectory_path: Optional[TrajectoryPath] = None,
    increasing_vdw: bool = False,
    vdw_cutoff_start: float = 0.0,
    vdw_cutoff_end: float = 12.5,
    max_attempts: int = 50,
    candidate_warmup_steps: int = 500,
    candidate_score_steps: int = 1000,
    best_candidate_refine_steps: int = 3000,
    ligand_untangling_attempts: int = 20,
    coordination_restoration_attempts: int = 20,
    coordination_relaxation_steps: int = 100,
    complex_untangling_attempts: int = 30,
    coordination_geometry: Optional[str] = None,
    seeded_build_worker: _SeededBuildWorker,
    complex_build_worker: _ComplexBuildWorker,
) -> ForceFieldWorkflowReport:
    """Build and optimize through the organic or complex workflow."""
    if mol.has_metal:
        return _complexes_build_workflow(
            mol,
            forcefield,
            algorithm=algorithm,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            max_attempts=max_attempts,
            candidate_warmup_steps=candidate_warmup_steps,
            candidate_score_steps=candidate_score_steps,
            best_candidate_refine_steps=best_candidate_refine_steps,
            ligand_untangling_attempts=ligand_untangling_attempts,
            coordination_restoration_attempts=coordination_restoration_attempts,
            coordination_relaxation_steps=coordination_relaxation_steps,
            complex_untangling_attempts=complex_untangling_attempts,
            timeout=timeout,
            add_hydrogens=add_hydrogens,
            quality_level=quality_level,
            quality_thresholds=quality_thresholds,
            seed=seed,
            perturb_interval=perturb_interval,
            perturb_sigma=perturb_sigma,
            save_movie=save_movie,
            trajectory_start=(
                trajectory_start
                if trajectory_start is not None
                else TrajectoryStart.COORDINATION_RESTORATION
            ),
            trajectory_path=trajectory_path,
            increasing_vdw=increasing_vdw,
            vdw_cutoff_start=vdw_cutoff_start,
            vdw_cutoff_end=vdw_cutoff_end,
            coordination_geometry=coordination_geometry,
            worker_target=complex_build_worker,
        )

    working_mol = _hydrogenated_working_copy(mol, add_hydrogens=False)
    build_report = _build3d_workflow(
        working_mol,
        add_hydrogens=add_hydrogens,
        seed=seed,
        timeout=timeout,
        worker_target=seeded_build_worker,
    )
    optimization_report = optimize(
        working_mol,
        forcefield,
        algorithm=algorithm,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        add_hydrogens=False,
        quality_level=quality_level,
        quality_thresholds=quality_thresholds,
        seed=seed,
        perturb_interval=perturb_interval,
        perturb_sigma=perturb_sigma,
        save_movie=save_movie,
        trajectory_start=(
            trajectory_start
            if trajectory_start is not None
            else TrajectoryStart.FINAL_OPTIMIZATION
        ),
        trajectory_path=trajectory_path,
        increasing_vdw=increasing_vdw,
        vdw_cutoff_start=vdw_cutoff_start,
        vdw_cutoff_end=vdw_cutoff_end,
    )
    _commit_working_copy(mol, working_mol)
    return BuildAndOptimizeReport(
        requested_forcefield=optimization_report.requested_forcefield,
        effective_forcefield=optimization_report.effective_forcefield,
        build=build_report,
        optimization=optimization_report,
        quality_report=optimization_report.quality_report,
        trajectory=optimization_report.trajectory,
    )


def complexes_build(
    mol: "Molecule",
    forcefield: Optional[str] = None,
    *,
    algorithm: OptimizationAlgorithm = "conjugate",
    epochs: int = 100,
    steps_per_epoch: int = 100,
    candidate_count: Optional[int] = None,
    max_attempts: int = 50,
    candidate_warmup_steps: int = 500,
    candidate_score_steps: int = 1000,
    best_candidate_refine_steps: int = 3000,
    ligand_untangling_attempts: int = 20,
    coordination_restoration_attempts: int = 20,
    coordination_relaxation_steps: int = 100,
    complex_untangling_attempts: int = 30,
    timeout: float = 1000.0,
    add_hydrogens: bool = True,
    quality_level: AcceptanceLevel = "standard",
    quality_thresholds: Optional[StructureAcceptanceThresholds] = None,
    seed: Optional[int] = None,
    perturb_interval: Optional[int] = None,
    perturb_sigma: float = 0.5,
    save_movie: bool = False,
    trajectory_start: TrajectoryStart = TrajectoryStart.COORDINATION_RESTORATION,
    trajectory_path: Optional[TrajectoryPath] = None,
    increasing_vdw: bool = False,
    vdw_cutoff_start: float = 0.0,
    vdw_cutoff_end: float = 12.5,
    coordination_geometry: Optional[str] = None,
) -> ComplexBuildReport:
    """Build, optimize, validate, and atomically commit a complete complex.

    ``candidate_count`` is reserved and currently has no effect.  The current
    workflow builds one ligand starting geometry.  A future multi-conformer
    implementation will generate independent starting conformers, optimize
    and gate them uniformly, deduplicate or cluster them by geometry, rank
    them using topology, geometry, and energy evidence, and refine the
    selected conformer.
    """
    _ = candidate_count
    return _complexes_build_workflow(
        mol,
        forcefield,
        algorithm=algorithm,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        max_attempts=max_attempts,
        candidate_warmup_steps=candidate_warmup_steps,
        candidate_score_steps=candidate_score_steps,
        best_candidate_refine_steps=best_candidate_refine_steps,
        ligand_untangling_attempts=ligand_untangling_attempts,
        coordination_restoration_attempts=coordination_restoration_attempts,
        coordination_relaxation_steps=coordination_relaxation_steps,
        complex_untangling_attempts=complex_untangling_attempts,
        timeout=timeout,
        add_hydrogens=add_hydrogens,
        quality_level=quality_level,
        quality_thresholds=quality_thresholds,
        seed=seed,
        perturb_interval=perturb_interval,
        perturb_sigma=perturb_sigma,
        save_movie=save_movie,
        trajectory_start=trajectory_start,
        trajectory_path=trajectory_path,
        increasing_vdw=increasing_vdw,
        vdw_cutoff_start=vdw_cutoff_start,
        vdw_cutoff_end=vdw_cutoff_end,
        coordination_geometry=coordination_geometry,
        worker_target=_build_ligand_proxies_worker,
    )


def build_and_optimize(
    mol: "Molecule",
    forcefield: Optional[str] = "UFF",
    *,
    algorithm: OptimizationAlgorithm = "conjugate",
    epochs: int = 100,
    steps_per_epoch: int = 100,
    add_hydrogens: bool = True,
    quality_level: AcceptanceLevel = "standard",
    quality_thresholds: Optional[StructureAcceptanceThresholds] = None,
    seed: Optional[int] = None,
    timeout: float = 1000.0,
    perturb_interval: Optional[int] = None,
    perturb_sigma: float = 0.5,
    save_movie: bool = False,
    trajectory_start: Optional[TrajectoryStart] = None,
    trajectory_path: Optional[TrajectoryPath] = None,
    increasing_vdw: bool = False,
    vdw_cutoff_start: float = 0.0,
    vdw_cutoff_end: float = 12.5,
    candidate_count: Optional[int] = None,
    max_attempts: int = 50,
    candidate_warmup_steps: int = 500,
    candidate_score_steps: int = 1000,
    best_candidate_refine_steps: int = 3000,
    ligand_untangling_attempts: int = 20,
    coordination_restoration_attempts: int = 20,
    coordination_relaxation_steps: int = 100,
    complex_untangling_attempts: int = 30,
    coordination_geometry: Optional[str] = None,
) -> ForceFieldWorkflowReport:
    """Build and optimize through the organic or complex workflow.

    ``candidate_count`` is reserved and currently has no effect.  The current
    complex workflow builds one ligand starting geometry.  A future
    multi-conformer implementation will generate independent starting
    conformers, optimize and gate them uniformly, deduplicate or cluster them
    by geometry, rank them using topology, geometry, and energy evidence, and
    refine the selected conformer.
    """
    _ = candidate_count
    return _build_and_optimize_workflow(
        mol,
        forcefield,
        algorithm=algorithm,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        add_hydrogens=add_hydrogens,
        quality_level=quality_level,
        quality_thresholds=quality_thresholds,
        seed=seed,
        timeout=timeout,
        perturb_interval=perturb_interval,
        perturb_sigma=perturb_sigma,
        save_movie=save_movie,
        trajectory_start=trajectory_start,
        trajectory_path=trajectory_path,
        increasing_vdw=increasing_vdw,
        vdw_cutoff_start=vdw_cutoff_start,
        vdw_cutoff_end=vdw_cutoff_end,
        max_attempts=max_attempts,
        candidate_warmup_steps=candidate_warmup_steps,
        candidate_score_steps=candidate_score_steps,
        best_candidate_refine_steps=best_candidate_refine_steps,
        ligand_untangling_attempts=ligand_untangling_attempts,
        coordination_restoration_attempts=coordination_restoration_attempts,
        coordination_relaxation_steps=coordination_relaxation_steps,
        complex_untangling_attempts=complex_untangling_attempts,
        coordination_geometry=coordination_geometry,
        seeded_build_worker=_seeded_ob_build_worker,
        complex_build_worker=_build_ligand_proxies_worker,
    )


def auto_optimize(
    mol: "Molecule",
    forcefield: Optional[str] = None,
    *,
    algorithm: OptimizationAlgorithm = "conjugate",
    epochs: int = 100,
    steps_per_epoch: int = 100,
    complex_untangling_attempts: int = 30,
    add_hydrogens: bool = True,
    quality_level: AcceptanceLevel = "standard",
    quality_thresholds: Optional[StructureAcceptanceThresholds] = None,
    seed: Optional[int] = None,
    perturb_interval: Optional[int] = None,
    perturb_sigma: float = 0.5,
    save_movie: bool = False,
    trajectory_start: Optional[TrajectoryStart] = None,
    trajectory_path: Optional[TrajectoryPath] = None,
    increasing_vdw: bool = False,
    vdw_cutoff_start: float = 0.0,
    vdw_cutoff_end: float = 12.5,
) -> ForceFieldRunReport:
    """Optimize existing coordinates through the appropriate workflow."""
    if mol.has_metal:
        return optimize_complex(
            mol,
            forcefield,
            algorithm=algorithm,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            complex_untangling_attempts=complex_untangling_attempts,
            add_hydrogens=add_hydrogens,
            quality_level=quality_level,
            quality_thresholds=quality_thresholds,
            seed=seed,
            perturb_interval=perturb_interval,
            perturb_sigma=perturb_sigma,
            save_movie=save_movie,
            trajectory_start=(
                trajectory_start
                if trajectory_start is not None
                else TrajectoryStart.COMPLEX_UNTANGLING
            ),
            trajectory_path=trajectory_path,
            increasing_vdw=increasing_vdw,
            vdw_cutoff_start=vdw_cutoff_start,
            vdw_cutoff_end=vdw_cutoff_end,
        )
    return optimize(
        mol,
        forcefield,
        algorithm=algorithm,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        add_hydrogens=add_hydrogens,
        quality_level=quality_level,
        quality_thresholds=quality_thresholds,
        seed=seed,
        perturb_interval=perturb_interval,
        perturb_sigma=perturb_sigma,
        save_movie=save_movie,
        trajectory_start=(
            trajectory_start
            if trajectory_start is not None
            else TrajectoryStart.FINAL_OPTIMIZATION
        ),
        trajectory_path=trajectory_path,
        increasing_vdw=increasing_vdw,
        vdw_cutoff_start=vdw_cutoff_start,
        vdw_cutoff_end=vdw_cutoff_end,
    )
