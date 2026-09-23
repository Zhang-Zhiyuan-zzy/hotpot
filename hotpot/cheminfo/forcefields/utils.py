"""Transactional force-field construction and optimization workflows."""

from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback as traceback_module
import warnings
from dataclasses import dataclass, replace
from multiprocessing.connection import Connection, wait as wait_for_connections
from typing import (
    TYPE_CHECKING,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np

from .. import geometry as geo
from . import backend as _backend
from .acceptance import (
    evaluate_structure_acceptance,
    is_structure_accepted,
)
from .backend import (
    _CandidateOptimizationResult as _CandidateOptimizationResult,
    _energy_factor_to_kj as _energy_factor_to_kj,
    _find_forcefield_prototype as _find_forcefield_prototype,
    _forcefield_energy_in_kj as _forcefield_energy_in_kj,
    _get_forcefield as _get_forcefield,
    _make_constraints as _make_constraints,
    _ob_build,
    _resolve_complex_forcefield,
    _resolve_organic_forcefield,
    _seed_openbabel_random as _seed_openbabel_random,
    _serialized_forcefield_call as _serialized_forcefield_call,
    _setup_forcefield_backend as _setup_forcefield_backend,
)
from .contracts import (
    AcceptanceCheck,
    AcceptanceLevel,
    Build3DReport,
    BuildAndOptimizeReport,
    BuildTimeoutError,
    BuildWorkerError,
    BuildWorkerResult,
    CandidateRejection,
    ComplexBuildDiagnostics,
    ComplexBuildError,
    ComplexBuildReport,
    ComplexBuildTimeoutError,
    ComplexBuildWarning,
    ComplexBuildWorkerError,
    CoordinationBondRestorationReport,
    CoordinationEnvironment,
    CoordinationGeometryCandidate,
    CoordinationGeometryResult,
    ForceFieldAcceptanceEvidence,
    ForceFieldDiagnosticValue,
    ForceFieldError,
    ForceFieldRunReport,
    ForceFieldSetupError,
    ForceFieldSetupReport,
    ForceFieldStage as ForceFieldStage,
    ForceFieldValidationReport,
    ForceFieldWorkflowReport,
    GeometryQualityError,
    GeometryQualityWarning,
    OptimizationAlgorithm,
    RingUntanglingReport,
    StructureAcceptanceThresholds,
    TerminationReason,
    TrajectoryPath,
)
from .coordination import (
    _require_explicit_complex,
    collect_coordination_environments,
    prepare_coordination_geometry,
)
from .coordinates import perturb
from .ligand import _build_ligand_proxies
from .optimizer import _combine_forcefield_run_reports, _optimize_working_mol
from .repair import (
    _piercing_count,
    _restore_coordination_bonds_incrementally,
    _scan_confirmed_ring_piercings,
    _unique_messages,
    _untangle_ring_piercings,
)
from .topology import (
    AtomTopologySignature,
    BondTopologySignature,
    TopologyReference,
    capture_topology,
)
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
from .working_copy import (
    _commit_working_copy,
    _hydrogenated_working_copy,
    _make_worker_mol,
)


if TYPE_CHECKING:
    from ..core import Molecule


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

_BOND_RING_MAX_SIZE = 16

# Compatibility aliases for private names historically imported from utils.
_SUPPORTED_FORCEFIELDS = _backend._SUPPORTED_FORCEFIELDS
_WORKER_LIFECYCLE_LOCK = _backend._WORKER_LIFECYCLE_LOCK
_OPENBABEL_FORCEFIELD_LOCK = _backend._OPENBABEL_FORCEFIELD_LOCK
_WORKER_EXIT_GRACE_SECONDS = _backend._WORKER_EXIT_GRACE_SECONDS

# Internal workflow data contracts.






@dataclass(frozen=True)
class _PreparedComplex:
    mol: "Molecule"
    diagnostics: ComplexBuildDiagnostics
    trajectory: ForceFieldTrajectory
    ligand_build_attempts: Tuple[ForceFieldTrajectory, ...] = ()


# Stateful Open Babel optimization engine.




# Ligand-proxy construction and geometric untangling.




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
        seed_initializer=_backend._seed_openbabel_random,
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
        seed_initializer=_backend._seed_openbabel_random,
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
        with _backend._WORKER_LIFECYCLE_LOCK:
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
            timeout=_backend._WORKER_EXIT_GRACE_SECONDS,
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
        with _backend._WORKER_LIFECYCLE_LOCK:
            process.join(timeout=_backend._WORKER_EXIT_GRACE_SECONDS)
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
            error = worker_error_type(
                result.error_type or "WorkerError",
                result.error_message or "Unknown build worker failure",
                result.traceback,
                result.diagnostics,
            )
            error.ligand_build_attempts = result.ligand_build_attempts
            raise error
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
            with _backend._WORKER_LIFECYCLE_LOCK:
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


def _preserve_failed_trajectory(
    error: ForceFieldError,
    trajectory: ForceFieldTrajectory,
    *,
    ligand_build_attempts: Sequence[ForceFieldTrajectory] = (),
    trajectory_path: Optional[TrajectoryPath],
) -> None:
    """Attach and optionally persist the facts recorded before a failure."""
    archive = ForceFieldTrajectoryArchive(
        main=trajectory,
        ligand_build_attempts=tuple(ligand_build_attempts),
    )
    error.trajectory = archive
    if trajectory_path is not None:
        archive.write(trajectory_path)


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
    trajectory_path: Optional[TrajectoryPath] = None,
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
    ligand_build_attempts: Tuple[ForceFieldTrajectory, ...] = ()
    try:
        result = _receive_worker_result(
            process,
            receive_connection,
            send_connection,
            timeout=timeout,
            seed=seed,
        )
        ligand_build_attempts = result.ligand_build_attempts
        working_mol.coordinates = _validated_worker_coordinates(
            result,
            expected_atom_count=len(working_mol.atoms),
        )
    except ForceFieldError as error:
        _preserve_failed_trajectory(
            error,
            trajectory,
            ligand_build_attempts=(
                ligand_build_attempts or error.ligand_build_attempts
            ),
            trajectory_path=trajectory_path,
        )
        raise
    diagnostics = cast(ComplexBuildDiagnostics, result.diagnostics)
    for message in diagnostics.warning_messages:
        warnings.warn(message, ComplexBuildWarning, stacklevel=3)
    if coordination_geometry is not None:
        prepare_coordination_geometry(
            working_mol, strategy=coordination_geometry, seed=seed
        )
    restoration_started = time.monotonic()
    try:
        restoration = _restore_coordination_bonds_incrementally(
            working_mol,
            effective_forcefield,
            attempt_limit=coordination_restoration_attempts,
            relaxation_steps=coordination_relaxation_steps,
            perturb_sigma=perturb_sigma,
            rng=np.random.default_rng(seed),
            trajectory=trajectory,
        )
    except ForceFieldError as error:
        _preserve_failed_trajectory(
            error,
            trajectory,
            ligand_build_attempts=ligand_build_attempts,
            trajectory_path=trajectory_path,
        )
        raise
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
        ligand_build_attempts=ligand_build_attempts,
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
    retain_epoch_history: bool,
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
            retain_epoch_history=retain_epoch_history,
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
        trajectory_path=trajectory_path,
        coordination_geometry=coordination_geometry,
        worker_target=worker_target,
    )
    try:
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
            retain_epoch_history=save_movie,
            increasing_vdw=increasing_vdw,
            vdw_cutoff_start=vdw_cutoff_start,
            vdw_cutoff_end=vdw_cutoff_end,
            trajectory=prepared.trajectory,
        )
    except ForceFieldError as error:
        _preserve_failed_trajectory(
            error,
            prepared.trajectory,
            ligand_build_attempts=prepared.ligand_build_attempts,
            trajectory_path=trajectory_path,
        )
        raise
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
    try:
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
            retain_epoch_history=save_movie,
            increasing_vdw=increasing_vdw,
            vdw_cutoff_start=vdw_cutoff_start,
            vdw_cutoff_end=vdw_cutoff_end,
            trajectory=trajectory,
        )
    except ForceFieldError as error:
        _preserve_failed_trajectory(
            error,
            trajectory,
            trajectory_path=trajectory_path,
        )
        raise
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
        trajectory_path=trajectory_path,
        coordination_geometry=coordination_geometry,
        worker_target=worker_target,
    )
    quality_report = evaluate_structure_acceptance(
        prepared.mol,
        level="off",
        topology_reference=topology_reference,
    )
    if not quality_report.passed:
        error = GeometryQualityError(quality_report)
        _preserve_failed_trajectory(
            error,
            prepared.trajectory,
            ligand_build_attempts=prepared.ligand_build_attempts,
            trajectory_path=trajectory_path,
        )
        raise error
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
    try:
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
            retain_epoch_history=save_movie,
            increasing_vdw=increasing_vdw,
            vdw_cutoff_start=vdw_cutoff_start,
            vdw_cutoff_end=vdw_cutoff_end,
            trajectory=trajectory,
        )
    except ForceFieldError as error:
        _preserve_failed_trajectory(
            error,
            trajectory,
            trajectory_path=trajectory_path,
        )
        raise
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
