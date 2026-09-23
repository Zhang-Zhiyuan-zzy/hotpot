"""Spawn-safe worker entry points and multiprocessing lifecycle helpers."""

from __future__ import annotations

import multiprocessing as mp
import os
import traceback as traceback_module
from multiprocessing.connection import Connection, wait as wait_for_connections
from typing import Optional, Protocol, Union, TYPE_CHECKING

import numpy as np

from . import backend as _backend
from .backend import _ob_build
from .contracts import (
    BuildTimeoutError,
    BuildWorkerError,
    BuildWorkerResult,
    ComplexBuildTimeoutError,
    ComplexBuildWorkerError,
)
from .ligand import _build_ligand_proxies
from .trajectory import ForceFieldTrajectory
from .working_copy import _make_worker_mol


if TYPE_CHECKING:
    from ..core import Molecule


__all__ = ()


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
