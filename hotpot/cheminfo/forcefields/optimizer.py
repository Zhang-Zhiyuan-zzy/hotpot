"""Stateful numerical Open Babel optimization and run reporting."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
from openbabel import openbabel as ob

from ..obconvert import extract_obmol_coordinates, mol2obmol, set_obmol_coordinates
from .backend import (
    _energy_factor_to_kj,
    _forcefield_energy_in_kj,
    _get_forcefield,
    _serialized_forcefield_call,
    _setup_forcefield_backend,
)
from .contracts import (
    ForceFieldRunReport,
    GeometryQualityError,
    OptimizationAlgorithm,
    OptimizationStoppingCriteria,
    TerminationReason,
)
from .coordinates import _perturbed_coordinates
from .trajectory import (
    ForceFieldFrame,
    ForceFieldTrajectory,
    OptimizationFrameEvidence,
    TrajectoryEvent,
    TrajectoryStage,
)


if TYPE_CHECKING:
    from ..core import Molecule


__all__ = ()


@dataclass(frozen=True)
class _ObservedFrame:
    coordinates: np.ndarray
    energy: float
    rms_gradient: float
    max_gradient: float
    exploded: bool
    converged: bool
    segment_epochs_completed: int
    segment_index: int
    history_length: int

    @property
    def has_finite_coordinates(self) -> bool:
        return bool(np.all(np.isfinite(self.coordinates)))

    @property
    def has_finite_energy(self) -> bool:
        return bool(np.isfinite(self.energy))

    @property
    def has_finite_gradients(self) -> bool:
        return bool(
            np.isfinite(self.rms_gradient)
            and np.isfinite(self.max_gradient)
        )

    def has_returnable_coordinates(
        self,
        expected_shape: Tuple[int, int],
    ) -> bool:
        return bool(
            self.coordinates.shape == expected_shape
            and self.has_finite_coordinates
        )

    def is_numerically_usable(
        self,
        expected_shape: Tuple[int, int],
    ) -> bool:
        return bool(
            self.has_returnable_coordinates(expected_shape)
            and self.has_finite_energy
            and self.has_finite_gradients
            and not self.exploded
        )


def _segment_satisfies_stopping_criteria(
    frame: _ObservedFrame,
    energy_changes: Sequence[float],
    max_displacements: Sequence[float],
    rms_gradients: Sequence[float],
    max_gradients: Sequence[float],
    criteria: OptimizationStoppingCriteria,
) -> bool:
    """Return whether the current numerical segment is stably stationary."""
    if frame.exploded or not (
        frame.has_finite_coordinates
        and frame.has_finite_energy
        and frame.has_finite_gradients
    ):
        return False
    if (
        len(energy_changes) < criteria.window
        or len(max_displacements) < criteria.window
        or len(rms_gradients) < criteria.window
        or len(max_gradients) < criteria.window
    ):
        return False
    recent_energy_changes = energy_changes[-criteria.window:]
    recent_displacements = max_displacements[-criteria.window:]
    recent_rms_gradients = rms_gradients[-criteria.window:]
    recent_max_gradients = max_gradients[-criteria.window:]
    return bool(
        np.all(np.isfinite(recent_energy_changes))
        and np.all(np.isfinite(recent_displacements))
        and np.all(np.isfinite(recent_rms_gradients))
        and np.all(np.isfinite(recent_max_gradients))
        and max(recent_energy_changes)
        <= criteria.maximum_energy_change_kj_mol
        and max(recent_displacements)
        <= criteria.maximum_atom_displacement_angstrom
        and max(recent_rms_gradients)
        <= criteria.maximum_rms_gradient_kj_mol_angstrom
        and max(recent_max_gradients)
        <= criteria.maximum_gradient_kj_mol_angstrom
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
        retain_epoch_history: bool,
        increasing_vdw: bool,
        vdw_cutoff_start: float,
        vdw_cutoff_end: float,
        seed: Optional[int],
        stopping_criteria: Optional[OptimizationStoppingCriteria] = None,
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
        self.retain_epoch_history = retain_epoch_history
        self.increasing_vdw = increasing_vdw
        self.vdw_cutoff_start = vdw_cutoff_start
        self.vdw_cutoff_end = vdw_cutoff_end
        self.stopping_criteria = stopping_criteria
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
        obmol: ob.OBMol,
        *,
        factor: float,
        converged: bool,
        segment_epochs_completed: int,
        segment_index: int,
        previous_coordinates: Optional[np.ndarray],
        previous_energy: Optional[float],
        energy_changes: list[float],
        max_displacements: list[float],
    ) -> _ObservedFrame:
        self.backend.GetCoordinates(obmol)
        coordinates = extract_obmol_coordinates(obmol)
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
        return _ObservedFrame(
            coordinates=coordinates.copy(),
            energy=energy,
            rms_gradient=rms_gradient,
            max_gradient=max_gradient,
            exploded=exploded,
            converged=converged,
            segment_epochs_completed=segment_epochs_completed,
            segment_index=segment_index,
            history_length=len(energy_changes),
        )

    @_serialized_forcefield_call
    def optimize(
        self,
        mol: "Molecule",
        *,
        trajectory: ForceFieldTrajectory,
        trajectory_stage: TrajectoryStage = TrajectoryStage.FINAL_OPTIMIZATION,
        trajectory_attempt: Optional[int] = None,
    ) -> ForceFieldRunReport:
        records_trajectory = trajectory.records(trajectory_stage)
        expected_coordinate_shape = (len(mol.atoms), 3)
        initial_coordinates = np.asarray(mol.coordinates, dtype=float).copy()
        initial_frame = _ObservedFrame(
            coordinates=initial_coordinates,
            energy=float("nan"),
            rms_gradient=float("nan"),
            max_gradient=float("nan"),
            exploded=False,
            converged=False,
            segment_epochs_completed=0,
            segment_index=0,
            history_length=0,
        )
        initial_frame_index: Optional[int] = None
        if records_trajectory:
            initial_frame_index = trajectory.record_molecule(
                mol,
                stage=trajectory_stage,
                event=TrajectoryEvent.INITIAL,
                attempt=trajectory_attempt,
            ).index
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

        best_frame: Optional[_ObservedFrame] = None
        best_epoch = -1
        best_frame_index: Optional[int] = None
        initial_is_returnable = initial_frame.has_returnable_coordinates(
            expected_coordinate_shape
        )
        latest_returnable_frame = initial_frame if initial_is_returnable else None
        latest_returnable_epoch = -1
        latest_returnable_frame_index = (
            initial_frame_index if initial_is_returnable else None
        )
        last_frame: Optional[_ObservedFrame] = None
        energy_change_segments: list[list[float]] = [[]]
        displacement_segments: list[list[float]] = [[]]
        rms_gradient_segments: list[list[float]] = [[]]
        max_gradient_segments: list[list[float]] = [[]]
        segment_index = 0
        epoch_energies: list[float] = []
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
                energy_change_segments.append([])
                displacement_segments.append([])
                rms_gradient_segments.append([])
                max_gradient_segments.append([])
                segment_index += 1
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

            reported_converged = frame_converged and (
                not self.increasing_vdw or epoch == self.epochs - 1
            )
            frame = self._observe_frame(
                obmol,
                factor=factor,
                converged=reported_converged,
                segment_epochs_completed=segment_epochs_completed,
                segment_index=segment_index,
                previous_coordinates=previous_coordinates,
                previous_energy=previous_energy,
                energy_changes=energy_change_segments[segment_index],
                max_displacements=displacement_segments[segment_index],
            )
            mol.coordinates = frame.coordinates
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
                        converged=frame.converged,
                        exploded=frame.exploded,
                        finite_coordinates=frame.has_finite_coordinates,
                        finite_energy=frame.has_finite_energy,
                        finite_gradients=frame.has_finite_gradients,
                        rms_gradient_kj_mol_angstrom=frame.rms_gradient,
                        max_gradient_kj_mol_angstrom=frame.max_gradient,
                        energy_change_kj_mol=(
                            energy_change_segments[segment_index][-1]
                            if energy_change_segments[segment_index]
                            else None
                        ),
                        max_displacement_angstrom=(
                            displacement_segments[segment_index][-1]
                            if displacement_segments[segment_index]
                            else None
                        ),
                    ),
                )
            last_frame = frame
            rms_gradient_segments[segment_index].append(frame.rms_gradient)
            max_gradient_segments[segment_index].append(frame.max_gradient)
            if frame.has_returnable_coordinates(expected_coordinate_shape):
                latest_returnable_frame = frame
                latest_returnable_epoch = epoch
                latest_returnable_frame_index = (
                    None if trajectory_frame is None else trajectory_frame.index
                )
            if frame.is_numerically_usable(expected_coordinate_shape) and (
                best_frame is None or frame.energy < best_frame.energy
            ):
                best_frame = frame
                best_epoch = epochs_completed - 1
                best_frame_index = (
                    None if trajectory_frame is None else trajectory_frame.index
                )
            if self.retain_epoch_history:
                epoch_energies.append(frame.energy)
            previous_coordinates = frame.coordinates
            previous_energy = frame.energy

            stability_reached = bool(
                not backend_converged
                and not self.increasing_vdw
                and self.stopping_criteria is not None
                and _segment_satisfies_stopping_criteria(
                    frame,
                    energy_change_segments[segment_index],
                    displacement_segments[segment_index],
                    rms_gradient_segments[segment_index],
                    max_gradient_segments[segment_index],
                    self.stopping_criteria,
                )
            )
            if stability_reached:
                segment_active = False
                termination_reason = "stability_reached"

            if (
                (backend_converged or stability_reached)
                and not self.increasing_vdw
                and self.perturb_interval is None
            ):
                break

        if last_frame is None or latest_returnable_frame is None:
            raise GeometryQualityError(None)

        if best_frame is None:
            best_frame = latest_returnable_frame
            best_epoch = latest_returnable_epoch
            best_frame_index = latest_returnable_frame_index

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
            backend_energy_unit=backend_unit,
            gradient_unit="kJ/(mol*angstrom)",
            energy_changes=tuple(
                energy_change_segments[best_frame.segment_index][
                    :best_frame.history_length
                ]
            ),
            max_displacements=tuple(
                displacement_segments[best_frame.segment_index][
                    :best_frame.history_length
                ]
            ),
            best_epoch=best_epoch,
            selected_segment_epochs_completed=(
                best_frame.segment_epochs_completed
            ),
            epoch_energies=tuple(epoch_energies),
            termination_reason=termination_reason,
            terminal_converged=terminal_converged,
        )
def _optimize_working_mol(
    working_mol: "Molecule",
    *,
    requested_forcefield: Optional[str],
    effective_forcefield: str,
    algorithm: OptimizationAlgorithm,
    epochs: int,
    steps_per_epoch: int,
    seed: Optional[int],
    perturb_interval: Optional[int],
    perturb_sigma: float,
    retain_epoch_history: bool,
    increasing_vdw: bool,
    vdw_cutoff_start: float,
    vdw_cutoff_end: float,
    trajectory: ForceFieldTrajectory,
    trajectory_stage: TrajectoryStage = TrajectoryStage.FINAL_OPTIMIZATION,
    trajectory_attempt: Optional[int] = None,
    stopping_criteria: Optional[OptimizationStoppingCriteria] = None,
) -> ForceFieldRunReport:
    optimizer = _OpenBabelOptimizer(
        requested_forcefield,
        effective_forcefield,
        algorithm=algorithm,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        perturb_interval=perturb_interval,
        perturb_sigma=perturb_sigma,
        retain_epoch_history=retain_epoch_history,
        increasing_vdw=increasing_vdw,
        vdw_cutoff_start=vdw_cutoff_start,
        vdw_cutoff_end=vdw_cutoff_end,
        seed=seed,
        stopping_criteria=stopping_criteria,
    )
    return optimizer.optimize(
        working_mol,
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
        best_epoch=(
            -1
            if final_report.best_epoch < 0
            else preceding_epochs + final_report.best_epoch
        ),
        epoch_energies=tuple(
            energy
            for report in reports
            for energy in report.epoch_energies
        ),
    )
