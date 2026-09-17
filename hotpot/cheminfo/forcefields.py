"""Transactional force-field construction and optimization workflows."""

from __future__ import annotations

import ctypes
import multiprocessing as mp
import os
import threading
import time
import traceback as traceback_module
from collections import deque
from copy import copy, deepcopy
from dataclasses import dataclass
from functools import wraps
from multiprocessing.connection import wait as wait_for_connections
from typing import Any, Literal, Mapping, Optional, Tuple

import networkx as nx
import numpy as np
from openbabel import openbabel as ob

from . import geometry as geo
from .obconvert import extract_obmol_coordinates, mol2obmol, set_obmol_coordinates

OptimizationAlgorithm = Literal["steepest", "conjugate"]
TerminationReason = Literal["converged", "budget_exhausted"]

_SUPPORTED_FORCEFIELDS = frozenset({"UFF", "MMFF94", "MMFF94s", "GAFF", "Ghemical"})
_NEUTRAL_DONOR_ATOMIC_NUMBERS = frozenset({7, 8, 15, 16, 33, 34})


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
    quality_report: Any = None
    backend_energy_unit: Optional[str] = None
    gradient_unit: str = "kJ/(mol*angstrom)"
    energy_changes: Tuple[float, ...] = ()
    max_displacements: Tuple[float, ...] = ()
    best_epoch: int = 0
    epoch_energies: Tuple[float, ...] = ()
    epoch_quality_reports: Tuple[Any, ...] = ()
    termination_reason: TerminationReason = "budget_exhausted"
    terminal_converged: bool = False


@dataclass(frozen=True)
class Build3DReport:
    atom_count: int
    added_hydrogen_count: int
    quality_report: Any


@dataclass(frozen=True)
class CandidateRejection:
    component_index: int
    attempt: int
    reason: str
    quality_failures: Tuple[Any, ...] = ()


@dataclass(frozen=True)
class ComplexBuildDiagnostics:
    attempt_count: int
    accepted_candidates: int
    rejected_candidates: Tuple[CandidateRejection, ...]
    elapsed_seconds: float


@dataclass(frozen=True)
class BuildWorkerResult:
    status: Literal["ok", "error"]
    coordinates: Optional[np.ndarray] = None
    diagnostics: Optional[ComplexBuildDiagnostics] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    traceback: Optional[str] = None


@dataclass(frozen=True)
class ComplexBuildReport:
    requested_forcefield: Optional[str]
    effective_forcefield: str
    build: ComplexBuildDiagnostics
    optimization: Optional[ForceFieldRunReport]
    quality_report: Any


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


class ForceFieldSetupError(ForceFieldError):
    """Raised when Open Babel cannot initialize a requested force field."""


class BuildWorkerError(ForceFieldError):
    """Raised when a generic coordinate-build worker fails."""

    def __init__(
        self,
        error_type: str,
        error_message: str,
        worker_traceback: Optional[str],
        diagnostics: Optional[ComplexBuildDiagnostics] = None,
    ):
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
    ):
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
    ):
        super().__init__(f"{error_type}: {error_message}", diagnostics)
        self.error_type = error_type
        self.error_message = error_message
        self.worker_traceback = worker_traceback


class ComplexBuildTimeoutError(ComplexBuildError, TimeoutError):
    """Raised after a proxy-build worker exceeds its allotted wall time."""


class GeometryQualityError(ForceFieldError):
    """Raised when no generated force-field frame passes the geometry gate."""

    def __init__(self, report: Any):
        super().__init__(
            "The generated geometry did not pass the requested quality gate"
        )
        self.report = report


def _format_geometry_rejection(prefix: str, report: Any) -> str:
    """Render failed geometry checks without discarding measured evidence."""
    details = "; ".join(
        f"{check.name}(measured={check.measured!r}, "
        f"threshold={check.threshold!r}, "
        f"atom_indices={check.atom_indices!r}, "
        f"bond_indices={check.bond_indices!r})"
        for check in report.failures
    )
    return f"{prefix}: {details}"


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
    quality_report: Any
    energy_changes: Tuple[float, ...]
    max_displacements: Tuple[float, ...]


_WORKER_LIFECYCLE_LOCK = threading.Lock()
_OPENBABEL_FORCEFIELD_LOCK = threading.RLock()
_WORKER_EXIT_GRACE_SECONDS = 30.0
_SEEDED_BUILD_TIMEOUT_SECONDS = 1000.0


def _serialized_forcefield_call(function):
    @wraps(function)
    def synchronized(*args, **kwargs):
        with _OPENBABEL_FORCEFIELD_LOCK:
            return function(*args, **kwargs)

    return synchronized


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


def _require_explicit_complex(mol: Any) -> None:
    """Require a metal center with at least one explicit metal--ligand bond."""
    if not mol.has_metal or not any(
        bond.is_metal_ligand_bond for bond in mol.bonds
    ):
        raise ValueError(
            "The complex workflow requires a molecule with at least one "
            "explicit metal-ligand bond"
        )


def _make_constraints(mol: Any) -> ob.OBFFConstraints:
    """Return the intentionally empty force-field constraint adapter."""
    return ob.OBFFConstraints()


def _energy_factor_to_kj(unit: str) -> float:
    normalized = unit.strip().lower().replace(" ", "")
    if normalized in {"kj/mol", "kjmol-1", "kjmol^-1"}:
        return 1.0
    if normalized in {"kcal/mol", "kcalmol-1", "kcalmol^-1"}:
        return 4.184
    raise ValueError(f"Unsupported Open Babel energy unit: {unit!r}")


@_serialized_forcefield_call
def _get_forcefield(name: str) -> ob.OBForceField:
    """Create an independent force-field instance from an Open Babel plugin."""
    prototype = ob.OBForceField.FindType(name)
    if prototype is None:
        raise ForceFieldSetupError(f"Unknown Open Babel force field: {name!r}")
    return prototype.MakeNewInstance()


@_serialized_forcefield_call
def _single_ob_optimization(
    mol: Any, forcefield: str, steps: int
) -> _CandidateOptimizationResult:
    backend = _get_forcefield(forcefield)
    backend.EnableCutOff(False)
    obmol, _ = mol2obmol(mol)
    if not backend.Setup(obmol, _make_constraints(mol)):
        raise ForceFieldSetupError(
            f"Open Babel could not initialize force field {forcefield!r}"
        )
    backend.SteepestDescent(steps)
    backend.GetCoordinates(obmol)
    mol.coordinates = extract_obmol_coordinates(obmol)
    backend_unit = backend.GetUnit()
    energy = float(backend.Energy()) * _energy_factor_to_kj(backend_unit)
    return _CandidateOptimizationResult(
        energy=energy,
        energy_unit="kJ/mol",
        exploded=bool(backend.DetectExplosion()),
    )


def _copy_molecule_metadata(source: Any, target: Any) -> None:
    target.charge = source.charge
    target.properties = dict(source.properties)
    target._model = source._model
    target._environ = source._environ
    target._crystal = source._crystal


def _recalculate_neutral_donor_valence(
    mol: Any,
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
    mol: Any,
    *,
    add_hydrogens: bool,
    seed: Optional[int] = None,
) -> Any:
    """Copy ``mol`` and infer H atoms against its ligand covalent skeleton."""
    working = copy(mol)
    _copy_molecule_metadata(mol, working)
    original_atom_count = len(working.atoms)
    if add_hydrogens:
        if working.has_metal:
            donor_indices = {
                bond.atom2.idx if bond.atom1.is_metal else bond.atom1.idx
                for bond in working.bonds
                if bond.is_metal_ligand_bond
            }
            working.hide_metal_ligand_bonds(clear_conformers=False)
            _recalculate_neutral_donor_valence(working, donor_indices)
            working.add_hydrogens(
                rm_polar_hs=False,
                rng=np.random.default_rng(seed),
            )
            working.recover_hided_metal_ligand_bonds(clear_conformers=False)
        else:
            working.add_hydrogens(
                rm_polar_hs=False,
                rng=np.random.default_rng(seed),
            )
    used_ids = {int(atom.id) for atom in working.atoms[:original_atom_count]}
    next_id = max(used_ids, default=-1) + 1
    for atom in working.atoms[original_atom_count:]:
        while next_id in used_ids:
            next_id += 1
        atom.id = next_id
        used_ids.add(next_id)
        next_id += 1
    return working


def _capture_workflow_topology(
    mol: Any,
    *,
    allow_added_hydrogens: bool,
) -> Any:
    """Capture the caller's input topology without rewriting atom identifiers."""
    return geo.capture_topology(
        mol,
        allow_added_hydrogens=allow_added_hydrogens,
    )


def _structure_worker_proxy(mol: Any) -> Any:
    """Return a structure-only clone with private positional IDs for a worker."""
    proxy = copy(mol)
    proxy.charge = mol.charge
    proxy.refresh_atom_id()
    return proxy


def _seed_openbabel_random(seed: int) -> None:
    """Seed both Open Babel RNG implementations before using ``OBBuilder``."""
    os.environ["OB_RANDOM_SEED"] = str(seed)

    # Open Babel 3.1 uses a function-local OBRandom backed by the process C
    # RNG and time-seeds it on first use.  Initializing that singleton before
    # resetting srand makes the legacy implementation deterministic.  Newer
    # Open Babel builds use OB_RANDOM_SEED through OBRandomMT; the extra C RNG
    # seed is harmless and keeps one code path across supported versions.
    probe = ob.vector3()
    probe.randomUnitVector()
    process_c_library = ctypes.CDLL(None)
    process_c_library.srand.argtypes = (ctypes.c_uint,)
    process_c_library.srand.restype = None
    process_c_library.srand(ctypes.c_uint(seed))


def _commit_working_copy(mol: Any, working: Any) -> None:
    """Commit accepted geometry while preserving caller-owned object identities."""
    original_atom_count = len(mol._atoms)
    working_atoms = tuple(working.atoms)

    for atom, source in zip(mol._atoms, working_atoms[:original_atom_count]):
        atom.attrs = np.array(source.attrs, copy=True)

    for source in working_atoms[original_atom_count:]:
        mol._create_atom_from_array(np.array(source.attrs, copy=True))

    working_positions = {id(atom): index for index, atom in enumerate(working_atoms)}
    for source_bond in working.bonds:
        first = working_positions[id(source_bond.atom1)]
        second = working_positions[id(source_bond.atom2)]
        if first >= original_atom_count or second >= original_atom_count:
            mol._add_bond(first, second, **source_bond.attr_dict)

    mol._update_graph(clear_conformers=False)
    mol._row2idx = None
    mol._atom_pairs.update_pairs()
    mol._conformers.__dict__.clear()
    mol._conformers.__dict__.update(deepcopy(working._conformers.__dict__))
    mol._conformers_index = working._conformers_index


def _perturbed_coordinates(
    coordinates: np.ndarray,
    *,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    displacement = rng.normal(0.0, sigma, np.asarray(coordinates).shape)
    displacement = np.clip(displacement, -2.0 * sigma, 2.0 * sigma)
    return np.asarray(coordinates, dtype=float) + displacement


def perturb(mol: Any, *, sigma: float = 0.5, seed: Optional[int] = None) -> np.ndarray:
    """Perturb current coordinates in place with a local random generator."""
    coordinates = _perturbed_coordinates(
        mol.coordinates,
        sigma=sigma,
        rng=np.random.default_rng(seed),
    )
    mol.coordinates = coordinates
    return coordinates


def collect_coordination_environments(mol: Any) -> Tuple[CoordinationEnvironment, ...]:
    """Describe explicit metal--donor connectivity without assigning geometry."""
    ligand_graph = mol.graph.copy()
    ligand_graph.remove_edges_from(
        (bond.a1idx, bond.a2idx) for bond in mol.bonds if bond.is_metal_ligand_bond
    )
    component_by_atom = {}
    for component_index, nodes in enumerate(nx.connected_components(ligand_graph)):
        for atom_idx in nodes:
            component_by_atom[atom_idx] = component_index

    environments = []
    for metal in mol.metals:
        donors = sorted(
            bond.atom2.idx if bond.atom1.idx == metal.idx else bond.atom1.idx
            for bond in mol.bonds
            if bond.is_metal_ligand_bond and metal.idx in (bond.a1idx, bond.a2idx)
        )
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
    mol: Any,
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
        energy_tolerance: float = 1.0e-6,
    ):
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
        self.energy_tolerance = energy_tolerance
        self.rng = np.random.default_rng(seed)
        self.backend = _get_forcefield(effective_forcefield)

    def _setup(self, mol: Any, obmol: Any) -> None:
        if not self.backend.Setup(obmol, _make_constraints(mol)):
            raise ForceFieldSetupError(
                f"Open Babel could not initialize force field {self.effective_forcefield!r}"
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

    def _optimizer_methods(self):
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

    def _initialize_with_budget(self, initialize, remaining_steps: int) -> int:
        initialization_steps = int(self.algorithm == "conjugate")
        take_step_capacity = remaining_steps - initialization_steps
        # Open Babel returns False both for convergence and for reaching the
        # limit supplied to Initialize().  Keep that private limit one counter
        # step beyond every TakeNSteps() call Hotpot can submit.  Conjugate
        # initialization performs one physical step without incrementing the
        # backend counter; steepest-descent initialization performs none.
        initialize(take_step_capacity + 1, self.energy_tolerance)
        return initialization_steps

    def _gradients(self, obmol: Any, factor: float) -> Tuple[float, float]:
        vectors = []
        for atom in ob.OBMolAtomIter(obmol):
            gradient = self.backend.GetGradient(atom)
            vectors.append((gradient.GetX(), gradient.GetY(), gradient.GetZ()))
        norms = np.linalg.norm(np.asarray(vectors, dtype=float) * factor, axis=1)
        return float(np.sqrt(np.mean(norms**2))), float(np.max(norms))

    def _observe_frame(
        self,
        mol: Any,
        obmol: Any,
        *,
        factor: float,
        converged: bool,
        epochs_completed: int,
        segment_epochs_completed: int,
        previous_coordinates: Optional[np.ndarray],
        previous_energy: Optional[float],
        energy_changes: deque[float],
        max_displacements: deque[float],
        quality_level: str,
        topology_reference: Any,
        quality_thresholds: Optional[Mapping[str, float]],
    ) -> _ObservedFrame:
        self.backend.GetCoordinates(obmol)
        coordinates = extract_obmol_coordinates(obmol)
        mol.coordinates = coordinates
        energy = float(self.backend.Energy(True)) * factor
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
        quality_report = geo.evaluate_geometry_quality(
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
        mol: Any,
        *,
        quality_level: str,
        topology_reference: Any,
        quality_thresholds: Optional[Mapping[str, float]],
    ) -> ForceFieldRunReport:
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
        best_frame_index = -1
        last_frame = None
        history_window = int(
            (quality_thresholds or {}).get("strict_stability_window", 5)
        )
        energy_changes = deque(maxlen=history_window)
        max_displacements = deque(maxlen=history_window)
        movie_coordinates = []
        movie_energies = []
        movie_quality_reports = []
        previous_coordinates = None
        previous_energy = None
        epochs_completed = 0
        segment_epochs_completed = 0
        steps_submitted = 0
        initialization_steps = 0
        terminal_converged = False
        termination_reason: TerminationReason = "budget_exhausted"
        segment_active = True

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
            last_frame = frame
            if frame.quality_report.passed and (
                best_frame is None or frame.energy < best_frame.energy
            ):
                best_frame = frame
                best_epoch = epoch
                best_frame_index = len(movie_coordinates)
            if self.save_movie:
                movie_coordinates.append(frame.coordinates)
                movie_energies.append(frame.energy)
                movie_quality_reports.append(frame.quality_report)
            previous_coordinates = frame.coordinates
            previous_energy = frame.energy

            if (
                backend_converged
                and not self.increasing_vdw
                and self.perturb_interval is None
            ):
                break

        if best_frame is None:
            raise GeometryQualityError(
                None if last_frame is None else last_frame.quality_report
            )

        mol.coordinates = best_frame.coordinates
        mol.conformer_clear()
        if self.save_movie:
            mol.conformer_add(np.asarray(movie_coordinates), np.asarray(movie_energies))
            mol.conformer_load(best_frame_index)
        else:
            mol.conformer_add(best_frame.coordinates, float(best_frame.energy))
            mol.conformer_load(0)

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
            epoch_energies=tuple(movie_energies),
            epoch_quality_reports=tuple(movie_quality_reports),
            termination_reason=termination_reason,
            terminal_converged=terminal_converged,
        )


def _build_ligand_proxies(
    mol: Any,
    *,
    candidate_count: int,
    max_attempts: int,
    candidate_warmup_steps: int,
    candidate_score_steps: int,
    best_candidate_refine_steps: int,
    effective_forcefield: str,
) -> Tuple[np.ndarray, ComplexBuildDiagnostics]:
    started = time.monotonic()
    clone = copy(mol)
    _copy_molecule_metadata(mol, clone)
    clone.hide_metal_ligand_bonds(clear_conformers=False)
    total_attempts = 0
    total_accepted = 0
    rejections = []

    for component_index, component in enumerate(clone.components):
        if component.has_metal:
            continue

        component_reference = geo.capture_topology(
            component,
            allow_added_hydrogens=False,
        )
        candidate_coordinates = []
        candidate_energies = []
        candidate_attempts = []
        component_attempts = 0
        while (
            len(candidate_coordinates) < candidate_count
            and component_attempts < max_attempts
        ):
            component_attempts += 1
            total_attempts += 1
            try:
                ob_build(component)
                _single_ob_optimization(
                    component,
                    effective_forcefield,
                    candidate_warmup_steps,
                )
            except ForceFieldError as exc:
                component.recover_hided_covalent_bonds(clear_conformers=False)
                rejections.append(
                    CandidateRejection(component_index, component_attempts, str(exc))
                )
                continue

            component.recover_hided_covalent_bonds(clear_conformers=False)
            try:
                scored = _single_ob_optimization(
                    component,
                    effective_forcefield,
                    candidate_score_steps,
                )
            except ForceFieldSetupError as exc:
                rejections.append(
                    CandidateRejection(component_index, component_attempts, str(exc))
                )
                continue

            intersections = geo.find_bond_ring_intersections(
                component,
                ring_scope="ligand_skeleton",
            )
            if intersections:
                bonds_to_hide = {}
                for ring, bond in intersections:
                    ring_edge = geo.closest_ring_opening_edge(
                        component,
                        ring,
                        bond,
                    )
                    if ring_edge is None:
                        continue
                    endpoint_key = tuple(sorted((ring_edge.a1idx, ring_edge.a2idx)))
                    bonds_to_hide[endpoint_key] = ring_edge
                if bonds_to_hide:
                    component.hide_bonds(
                        *(bonds_to_hide[key] for key in sorted(bonds_to_hide)),
                        clear_conformers=False,
                    )
                rejections.append(
                    CandidateRejection(
                        component_index,
                        component_attempts,
                        "bond-ring intersection",
                    )
                )
                continue

            candidate_quality = geo.evaluate_geometry_quality(
                component,
                level="basic",
                topology_reference=component_reference,
                forcefield_report={
                    "setup_succeeded": True,
                    "converged": False,
                    "final_energy": scored.energy,
                    "energy_unit": scored.energy_unit,
                    "rms_gradient": None,
                    "max_gradient": None,
                    "exploded": scored.exploded,
                },
            )
            if not candidate_quality.passed:
                rejections.append(
                    CandidateRejection(
                        component_index,
                        component_attempts,
                        _format_geometry_rejection(
                            "candidate geometry gate",
                            candidate_quality,
                        ),
                        tuple(candidate_quality.failures),
                    )
                )
                continue

            candidate_coordinates.append(component.coordinates.copy())
            candidate_energies.append(scored.energy)
            candidate_attempts.append(component_attempts)
            total_accepted += 1

        if len(candidate_coordinates) < candidate_count:
            diagnostics = ComplexBuildDiagnostics(
                attempt_count=total_attempts,
                accepted_candidates=total_accepted,
                rejected_candidates=tuple(rejections),
                elapsed_seconds=time.monotonic() - started,
            )
            raise ComplexBuildError(
                f"Component {component_index} accepted "
                f"{len(candidate_coordinates)}/{candidate_count} candidates after "
                f"{component_attempts} attempts",
                diagnostics,
            )

        refined_candidate_found = False
        for candidate_index in np.argsort(candidate_energies):
            component.coordinates = candidate_coordinates[int(candidate_index)]
            attempt = candidate_attempts[int(candidate_index)]
            try:
                refined = _single_ob_optimization(
                    component,
                    effective_forcefield,
                    best_candidate_refine_steps,
                )
            except ForceFieldError as exc:
                rejections.append(
                    CandidateRejection(
                        component_index,
                        attempt,
                        f"refined candidate: {exc}",
                    )
                )
                continue

            refined_intersections = geo.find_bond_ring_intersections(
                component,
                ring_scope="ligand_skeleton",
            )
            refined_quality = geo.evaluate_geometry_quality(
                component,
                level="basic",
                topology_reference=component_reference,
                forcefield_report={
                    "setup_succeeded": True,
                    "converged": False,
                    "final_energy": refined.energy,
                    "energy_unit": refined.energy_unit,
                    "rms_gradient": None,
                    "max_gradient": None,
                    "exploded": refined.exploded,
                },
            )
            if refined_intersections or not refined_quality.passed:
                if refined_intersections:
                    reason = "refined candidate bond-ring intersection"
                    failures = ()
                else:
                    reason = _format_geometry_rejection(
                        "refined candidate geometry gate",
                        refined_quality,
                    )
                    failures = tuple(refined_quality.failures)
                rejections.append(CandidateRejection(
                    component_index,
                    attempt,
                    reason,
                    failures,
                ))
                continue
            refined_candidate_found = True
            break

        if not refined_candidate_found:
            diagnostics = ComplexBuildDiagnostics(
                attempt_count=total_attempts,
                accepted_candidates=total_accepted,
                rejected_candidates=tuple(rejections),
                elapsed_seconds=time.monotonic() - started,
            )
            raise ComplexBuildError(
                f"Candidate refinement failed for every candidate of component {component_index}",
                diagnostics,
            )
        clone.update_atoms_attrs_from_id_dict(
            {atom.id: {"coordinates": atom.coordinates} for atom in component.atoms}
        )

    clone.recover_hided_metal_ligand_bonds(clear_conformers=False)
    diagnostics = ComplexBuildDiagnostics(
        attempt_count=total_attempts,
        accepted_candidates=total_accepted,
        rejected_candidates=tuple(rejections),
        elapsed_seconds=time.monotonic() - started,
    )
    return clone.coordinates, diagnostics


def _run_complexes_build(
    mol: Any,
    connection: Any,
    candidate_count: int,
    max_attempts: int,
    candidate_warmup_steps: int,
    candidate_score_steps: int,
    best_candidate_refine_steps: int,
    effective_forcefield: str,
    seed: Optional[int],
) -> None:
    """Child-process boundary that always sends one structured envelope."""
    try:
        if seed is not None:
            _seed_openbabel_random(seed)
        coordinates, diagnostics = _build_ligand_proxies(
            mol,
            candidate_count=candidate_count,
            max_attempts=max_attempts,
            candidate_warmup_steps=candidate_warmup_steps,
            candidate_score_steps=candidate_score_steps,
            best_candidate_refine_steps=best_candidate_refine_steps,
            effective_forcefield=effective_forcefield,
        )
        result = BuildWorkerResult(
            status="ok",
            coordinates=coordinates,
            diagnostics=diagnostics,
        )
    except Exception as exc:
        result = BuildWorkerResult(
            status="error",
            diagnostics=getattr(exc, "diagnostics", None),
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback=traceback_module.format_exc(),
        )
    try:
        connection.send(result)
    finally:
        connection.close()


def _run_seeded_ob_build(
    mol: Any,
    connection: Any,
    seed: int,
) -> None:
    """Run OBBuilder in a fresh process whose static RNG starts from ``seed``."""
    try:
        _seed_openbabel_random(seed)
        ob_build(mol)
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
    receive_connection: Any,
    send_connection: Any,
    *,
    timeout: float,
    seed: Optional[int] = None,
    require_diagnostics: bool = True,
    worker_error_type: Any = ComplexBuildWorkerError,
    timeout_error_type: Any = ComplexBuildTimeoutError,
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
    worker_error_type: Any = ComplexBuildWorkerError,
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


def _seeded_ob_build_coordinates(mol: Any, seed: int) -> np.ndarray:
    """Build coordinates in an isolated process for repeatable Open Babel RNG."""
    worker_proxy = _structure_worker_proxy(mol)
    context = mp.get_context("spawn")
    receive_connection, send_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_run_seeded_ob_build,
        args=(worker_proxy, send_connection, seed),
    )
    result = _receive_worker_result(
        process,
        receive_connection,
        send_connection,
        timeout=_SEEDED_BUILD_TIMEOUT_SECONDS,
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


def _build_complex_working(
    mol: Any,
    *,
    effective_forcefield: str,
    candidate_count: int,
    max_attempts: int,
    candidate_warmup_steps: int,
    candidate_score_steps: int,
    best_candidate_refine_steps: int,
    timeout: float,
    add_hydrogens: bool,
    seed: Optional[int],
    coordination_geometry: Optional[str],
) -> Tuple[Any, ComplexBuildDiagnostics]:
    if candidate_count < 1:
        raise ValueError("candidate_count must be at least 1")
    if max_attempts < candidate_count:
        raise ValueError("max_attempts must be at least candidate_count")
    if min(
        candidate_warmup_steps,
        candidate_score_steps,
        best_candidate_refine_steps,
    ) < 1:
        raise ValueError("all candidate optimization step counts must be at least 1")
    if timeout <= 0.0:
        raise ValueError("timeout must be positive")
    working = _hydrogenated_working_copy(
        mol,
        add_hydrogens=add_hydrogens,
        seed=seed,
    )
    worker_proxy = _structure_worker_proxy(working)
    context = mp.get_context("spawn")
    receive_connection, send_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_run_complexes_build,
        args=(
            worker_proxy,
            send_connection,
            candidate_count,
            max_attempts,
            candidate_warmup_steps,
            candidate_score_steps,
            best_candidate_refine_steps,
            effective_forcefield,
            seed,
        ),
    )
    result = _receive_worker_result(
        process,
        receive_connection,
        send_connection,
        timeout=timeout,
        seed=seed,
    )
    working.coordinates = _validated_worker_coordinates(
        result,
        expected_atom_count=len(working.atoms),
    )
    if coordination_geometry is not None:
        prepare_coordination_geometry(
            working, strategy=coordination_geometry, seed=seed
        )
    return working, result.diagnostics


def _run_optimizer_on_working(
    working: Any,
    *,
    requested_forcefield: Optional[str],
    effective_forcefield: str,
    algorithm: OptimizationAlgorithm,
    epochs: int,
    steps_per_epoch: int,
    quality_level: str,
    topology_reference: Any,
    quality_thresholds: Optional[Mapping[str, float]],
    seed: Optional[int],
    perturb_interval: Optional[int],
    perturb_sigma: float,
    save_movie: bool,
    increasing_vdw: bool,
    vdw_cutoff_start: float,
    vdw_cutoff_end: float,
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
    )
    return optimizer.optimize(
        working,
        quality_level=quality_level,
        topology_reference=topology_reference,
        quality_thresholds=quality_thresholds,
    )


def build3d(
    mol: Any,
    *,
    add_hydrogens: bool = True,
    seed: Optional[int] = None,
) -> Build3DReport:
    """Generate initial 3D coordinates with OBBuilder, without optimization."""
    topology_reference = _capture_workflow_topology(
        mol,
        allow_added_hydrogens=add_hydrogens,
    )
    initial_hydrogens = len(mol.hydrogens)
    working = _hydrogenated_working_copy(
        mol,
        add_hydrogens=add_hydrogens,
        seed=seed,
    )
    if seed is None:
        ob_build(working)
    else:
        working.coordinates = _seeded_ob_build_coordinates(working, seed)
    quality_report = geo.evaluate_geometry_quality(
        working,
        level="off",
        topology_reference=topology_reference,
    )
    if not quality_report.passed:
        raise GeometryQualityError(quality_report)
    report = Build3DReport(
        atom_count=len(working.atoms),
        added_hydrogen_count=len(working.hydrogens) - initial_hydrogens,
        quality_report=quality_report,
    )
    _commit_working_copy(mol, working)
    return report


def optimize(
    mol: Any,
    forcefield: Optional[str] = "UFF",
    *,
    algorithm: OptimizationAlgorithm = "conjugate",
    epochs: int = 1,
    steps_per_epoch: int = 100,
    add_hydrogens: bool = True,
    quality_level: str = "standard",
    quality_thresholds: Optional[Mapping[str, float]] = None,
    seed: Optional[int] = None,
    perturb_interval: Optional[int] = None,
    perturb_sigma: float = 0.5,
    save_movie: bool = False,
    increasing_vdw: bool = False,
    vdw_cutoff_start: float = 0.0,
    vdw_cutoff_end: float = 12.5,
) -> ForceFieldRunReport:
    """Run the ordinary Open Babel optimizer, including on explicit complexes."""
    topology_reference = _capture_workflow_topology(
        mol,
        allow_added_hydrogens=add_hydrogens,
    )
    working = _hydrogenated_working_copy(
        mol,
        add_hydrogens=add_hydrogens,
        seed=seed,
    )
    effective_forcefield = _resolve_organic_forcefield(forcefield)
    report = _run_optimizer_on_working(
        working,
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
    )
    _commit_working_copy(mol, working)
    return report


def build_complex3d(
    mol: Any,
    forcefield: Optional[str] = None,
    *,
    candidate_count: int = 5,
    max_attempts: int = 50,
    candidate_warmup_steps: int = 500,
    candidate_score_steps: int = 1000,
    best_candidate_refine_steps: int = 3000,
    timeout: float = 1000.0,
    add_hydrogens: bool = True,
    seed: Optional[int] = None,
    coordination_geometry: Optional[str] = None,
) -> ComplexBuildReport:
    """Build ligand proxies and restore the complete complex topology."""
    _require_explicit_complex(mol)
    topology_reference = _capture_workflow_topology(
        mol,
        allow_added_hydrogens=add_hydrogens,
    )
    effective_forcefield = _resolve_complex_forcefield(forcefield)
    working, diagnostics = _build_complex_working(
        mol,
        effective_forcefield=effective_forcefield,
        candidate_count=candidate_count,
        max_attempts=max_attempts,
        candidate_warmup_steps=candidate_warmup_steps,
        candidate_score_steps=candidate_score_steps,
        best_candidate_refine_steps=best_candidate_refine_steps,
        timeout=timeout,
        add_hydrogens=add_hydrogens,
        seed=seed,
        coordination_geometry=coordination_geometry,
    )
    quality_report = geo.evaluate_geometry_quality(
        working,
        level="off",
        topology_reference=topology_reference,
    )
    if not quality_report.passed:
        raise GeometryQualityError(quality_report)
    report = ComplexBuildReport(
        requested_forcefield=forcefield,
        effective_forcefield=effective_forcefield,
        build=diagnostics,
        optimization=None,
        quality_report=quality_report,
    )
    _commit_working_copy(mol, working)
    return report


def optimize_complex(
    mol: Any,
    forcefield: Optional[str] = None,
    *,
    algorithm: OptimizationAlgorithm = "conjugate",
    epochs: int = 100,
    steps_per_epoch: int = 100,
    add_hydrogens: bool = True,
    quality_level: str = "standard",
    quality_thresholds: Optional[Mapping[str, float]] = None,
    seed: Optional[int] = None,
    perturb_interval: Optional[int] = None,
    perturb_sigma: float = 0.5,
    save_movie: bool = False,
    increasing_vdw: bool = False,
    vdw_cutoff_start: float = 0.0,
    vdw_cutoff_end: float = 12.5,
) -> ForceFieldRunReport:
    """Optimize existing complex coordinates with the complex force-field policy."""
    _require_explicit_complex(mol)
    topology_reference = _capture_workflow_topology(
        mol,
        allow_added_hydrogens=add_hydrogens,
    )
    working = _hydrogenated_working_copy(
        mol,
        add_hydrogens=add_hydrogens,
        seed=seed,
    )
    effective_forcefield = _resolve_complex_forcefield(forcefield)
    report = _run_optimizer_on_working(
        working,
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
    )
    _commit_working_copy(mol, working)
    return report


def _complexes_build_impl(
    mol: Any,
    forcefield: Optional[str] = None,
    *,
    algorithm: OptimizationAlgorithm = "conjugate",
    epochs: int = 100,
    steps_per_epoch: int = 100,
    candidate_count: int = 5,
    max_attempts: int = 50,
    candidate_warmup_steps: int = 500,
    candidate_score_steps: int = 1000,
    best_candidate_refine_steps: int = 3000,
    timeout: float = 1000.0,
    add_hydrogens: bool = True,
    quality_level: str = "standard",
    quality_thresholds: Optional[Mapping[str, float]] = None,
    seed: Optional[int] = None,
    perturb_interval: Optional[int] = None,
    perturb_sigma: float = 0.5,
    save_movie: bool = False,
    increasing_vdw: bool = False,
    vdw_cutoff_start: float = 0.0,
    vdw_cutoff_end: float = 12.5,
    coordination_geometry: Optional[str] = None,
) -> ComplexBuildReport:
    """Build, optimize, validate, and atomically commit a complete complex."""
    _require_explicit_complex(mol)
    topology_reference = _capture_workflow_topology(
        mol,
        allow_added_hydrogens=add_hydrogens,
    )
    effective_forcefield = _resolve_complex_forcefield(forcefield)
    working, diagnostics = _build_complex_working(
        mol,
        effective_forcefield=effective_forcefield,
        candidate_count=candidate_count,
        max_attempts=max_attempts,
        candidate_warmup_steps=candidate_warmup_steps,
        candidate_score_steps=candidate_score_steps,
        best_candidate_refine_steps=best_candidate_refine_steps,
        timeout=timeout,
        add_hydrogens=add_hydrogens,
        seed=seed,
        coordination_geometry=coordination_geometry,
    )
    optimization_report = _run_optimizer_on_working(
        working,
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
    )
    report = ComplexBuildReport(
        requested_forcefield=forcefield,
        effective_forcefield=effective_forcefield,
        build=diagnostics,
        optimization=optimization_report,
        quality_report=optimization_report.quality_report,
    )
    _commit_working_copy(mol, working)
    return report


_LEGACY_COMPLEX_BUILD_OPTIONS = {
    "steps": "epochs",
    "step_size": "steps_per_epoch",
    "perturb_steps": "perturb_interval",
    "save_screenshot": "save_movie",
    "build_times": "candidate_count",
    "init_opt_steps": "candidate_warmup_steps",
    "second_opt_steps": "candidate_score_steps",
    "min_energy_opt_steps": "best_candidate_refine_steps",
    "increasing_Vdw": "increasing_vdw",
    "Vdw_cutoff_start": "vdw_cutoff_start",
    "Vdw_cutoff_end": "vdw_cutoff_end",
}


def _translate_legacy_complex_build_options(options: Mapping[str, Any]) -> dict:
    """Translate historical names once without changing workflow semantics."""
    translated = dict(options)
    for legacy_name, current_name in _LEGACY_COMPLEX_BUILD_OPTIONS.items():
        if legacy_name not in translated:
            continue
        legacy_value = translated.pop(legacy_name)
        if current_name in translated and translated[current_name] != legacy_value:
            raise TypeError(
                f"Conflicting values for {legacy_name!r} and {current_name!r}"
            )
        translated[current_name] = legacy_value
    return translated


def complexes_build(
    mol: Any,
    forcefield: Optional[str] = None,
    **options: Any,
) -> ComplexBuildReport:
    """Compatibility entry for the complete transactional complex workflow."""
    return _complexes_build_impl(
        mol,
        forcefield,
        **_translate_legacy_complex_build_options(options),
    )


def build_and_optimize(
    mol: Any,
    forcefield: Optional[str] = "UFF",
    *,
    algorithm: OptimizationAlgorithm = "conjugate",
    epochs: int = 100,
    steps_per_epoch: int = 100,
    add_hydrogens: bool = True,
    quality_level: str = "standard",
    quality_thresholds: Optional[Mapping[str, float]] = None,
    seed: Optional[int] = None,
    timeout: float = 1000.0,
    perturb_interval: Optional[int] = None,
    perturb_sigma: float = 0.5,
    save_movie: bool = False,
    increasing_vdw: bool = False,
    vdw_cutoff_start: float = 0.0,
    vdw_cutoff_end: float = 12.5,
    candidate_count: int = 5,
    max_attempts: int = 50,
    candidate_warmup_steps: int = 500,
    candidate_score_steps: int = 1000,
    best_candidate_refine_steps: int = 3000,
    coordination_geometry: Optional[str] = None,
) -> Any:
    """Build and optimize through the organic or complex workflow."""
    if mol.has_metal:
        return complexes_build(
            mol,
            forcefield,
            algorithm=algorithm,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            candidate_count=candidate_count,
            max_attempts=max_attempts,
            candidate_warmup_steps=candidate_warmup_steps,
            candidate_score_steps=candidate_score_steps,
            best_candidate_refine_steps=best_candidate_refine_steps,
            timeout=timeout,
            add_hydrogens=add_hydrogens,
            quality_level=quality_level,
            quality_thresholds=quality_thresholds,
            seed=seed,
            perturb_interval=perturb_interval,
            perturb_sigma=perturb_sigma,
            save_movie=save_movie,
            increasing_vdw=increasing_vdw,
            vdw_cutoff_start=vdw_cutoff_start,
            vdw_cutoff_end=vdw_cutoff_end,
            coordination_geometry=coordination_geometry,
        )

    working = _hydrogenated_working_copy(mol, add_hydrogens=False)
    build3d(working, add_hydrogens=add_hydrogens, seed=seed)
    report = optimize(
        working,
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
        increasing_vdw=increasing_vdw,
        vdw_cutoff_start=vdw_cutoff_start,
        vdw_cutoff_end=vdw_cutoff_end,
    )
    _commit_working_copy(mol, working)
    return report


def auto_optimize(
    mol: Any,
    forcefield: Optional[str] = None,
    *,
    algorithm: OptimizationAlgorithm = "conjugate",
    epochs: int = 100,
    steps_per_epoch: int = 100,
    add_hydrogens: bool = True,
    quality_level: str = "standard",
    quality_thresholds: Optional[Mapping[str, float]] = None,
    seed: Optional[int] = None,
    perturb_interval: Optional[int] = None,
    perturb_sigma: float = 0.5,
    save_movie: bool = False,
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
            add_hydrogens=add_hydrogens,
            quality_level=quality_level,
            quality_thresholds=quality_thresholds,
            seed=seed,
            perturb_interval=perturb_interval,
            perturb_sigma=perturb_sigma,
            save_movie=save_movie,
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
        increasing_vdw=increasing_vdw,
        vdw_cutoff_start=vdw_cutoff_start,
        vdw_cutoff_end=vdw_cutoff_end,
    )


@_serialized_forcefield_call
def ob_build(mol: Any) -> None:
    """Compatibility primitive: run OBBuilder directly on ``mol``."""
    builder = ob.OBBuilder()
    obmol, _ = mol2obmol(mol)
    if not builder.Build(obmol):
        raise ForceFieldError("Open Babel could not build initial 3D coordinates")
    mol.coordinates = extract_obmol_coordinates(obmol)


def ob_optimize(mol: Any, ff: str = "UFF", steps: int = 100) -> float:
    """Compatibility primitive returning energy in kJ/mol."""
    return _single_ob_optimization(mol, ff, steps).energy
