"""Native Open Babel optimization adaptation and run reporting."""

from __future__ import annotations

from dataclasses import replace
from typing import cast, Optional, Sequence, TYPE_CHECKING

import numpy as np

from ..obWrappers import optimize as _native_optimize
from ..obWrappers.native import _native_module
from .backend import _raise_forcefield_setup_error
from .contracts import (
    ForceFieldRunReport,
    OptimizationAlgorithm,
    OptimizationStoppingCriteria,
    TerminationReason,
)
from .coordinates import _perturbed_coordinates
from .trajectory import (
    ForceFieldTrajectory,
    OptimizationFrameEvidence,
    TrajectoryEvent,
    TrajectoryStage,
)


if TYPE_CHECKING:
    from ..core import Molecule


__all__ = ()


class _OpenBabelOptimizer:
    """Adapt one native Open Babel numerical run to Hotpot contracts."""

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

    def _perturbation_offsets(self, atom_count: int) -> Optional[np.ndarray]:
        """Precompute the legacy NumPy perturbations for one native run."""
        if self.perturb_interval is None:
            return None
        count = (self.epochs - 1) // self.perturb_interval
        if count == 0:
            return np.empty((0, atom_count, 3), dtype=np.float64)
        origin = np.zeros((atom_count, 3), dtype=np.float64)
        return np.ascontiguousarray(
            [
                _perturbed_coordinates(
                    origin,
                    sigma=self.perturb_sigma,
                    rng=self.rng,
                )
                for _ in range(count)
            ],
            dtype=np.float64,
        )

    def optimize(
        self,
        mol: "Molecule",
        *,
        trajectory: ForceFieldTrajectory,
        trajectory_stage: TrajectoryStage = TrajectoryStage.FINAL_OPTIMIZATION,
        trajectory_attempt: Optional[int] = None,
    ) -> ForceFieldRunReport:
        """Run the native optimizer and translate its batch result."""
        records_trajectory = trajectory.records(trajectory_stage)
        initial_frame_index: Optional[int] = None
        if records_trajectory:
            initial_frame_index = trajectory.record_molecule(
                mol,
                stage=trajectory_stage,
                event=TrajectoryEvent.INITIAL,
                attempt=trajectory_attempt,
            ).index

        stopping = self.stopping_criteria
        native = _native_module()
        try:
            result = _native_optimize(
                mol,
                self.effective_forcefield,
                algorithm=self.algorithm,
                epochs=self.epochs,
                steps_per_epoch=self.steps_per_epoch,
                perturb_interval=self.perturb_interval,
                perturbation_offsets=self._perturbation_offsets(len(mol.atoms)),
                retain_frames=records_trajectory,
                retain_epoch_history=self.retain_epoch_history,
                increasing_vdw=self.increasing_vdw,
                vdw_cutoff_start=self.vdw_cutoff_start,
                vdw_cutoff_end=self.vdw_cutoff_end,
                energy_tolerance=self.energy_tolerance,
                stopping_window=None if stopping is None else stopping.window,
                maximum_energy_change_kj_mol=(
                    1.0e-4
                    if stopping is None
                    else stopping.maximum_energy_change_kj_mol
                ),
                maximum_atom_displacement_angstrom=(
                    1.0e-4
                    if stopping is None
                    else stopping.maximum_atom_displacement_angstrom
                ),
                maximum_rms_gradient_kj_mol_angstrom=(
                    1.0
                    if stopping is None
                    else stopping.maximum_rms_gradient_kj_mol_angstrom
                ),
                maximum_gradient_kj_mol_angstrom=(
                    5.0
                    if stopping is None
                    else stopping.maximum_gradient_kj_mol_angstrom
                ),
            )
        except native.ForceFieldSetupError as error:
            _raise_forcefield_setup_error(
                error,
                requested_forcefield=self.requested_forcefield,
                effective_forcefield=self.effective_forcefield,
                stage=error.stage,
            )

        trajectory_frame_indices: list[int] = []
        if records_trajectory:
            for frame in result.frames:
                mol.coordinates = frame.coordinates
                trajectory_frame_indices.append(
                    trajectory.record_molecule(
                        mol,
                        stage=trajectory_stage,
                        event=TrajectoryEvent.EPOCH_COMPLETE,
                        energy_kj_mol=frame.energy,
                        attempt=trajectory_attempt,
                        step=frame.epoch_index,
                        evidence=OptimizationFrameEvidence(
                            converged=frame.converged,
                            exploded=frame.exploded,
                            finite_coordinates=bool(
                                np.all(np.isfinite(frame.coordinates))
                            ),
                            finite_energy=bool(np.isfinite(frame.energy)),
                            finite_gradients=bool(
                                np.isfinite(frame.rms_gradient)
                                and np.isfinite(frame.max_gradient)
                            ),
                            rms_gradient_kj_mol_angstrom=frame.rms_gradient,
                            max_gradient_kj_mol_angstrom=frame.max_gradient,
                            energy_change_kj_mol=frame.energy_change,
                            max_displacement_angstrom=frame.max_displacement,
                        ),
                    ).index
                )

        mol.coordinates = result.coordinates
        if records_trajectory:
            if result.selected_frame_index < 0:
                if initial_frame_index is not None:
                    trajectory.select(initial_frame_index)
            else:
                trajectory.select(
                    trajectory_frame_indices[result.selected_frame_index]
                )

        return ForceFieldRunReport(
            requested_forcefield=self.requested_forcefield,
            effective_forcefield=self.effective_forcefield,
            setup_succeeded=True,
            converged=result.converged,
            epochs_completed=result.epochs_completed,
            steps_submitted=result.steps_submitted,
            initialization_steps=result.initialization_steps,
            steps_completed=None,
            final_energy=result.final_energy,
            best_energy=result.best_energy,
            energy_unit=result.energy_unit,
            rms_gradient=result.rms_gradient,
            max_gradient=result.max_gradient,
            exploded=result.exploded,
            backend_energy_unit=result.backend_energy_unit,
            gradient_unit="kJ/(mol*angstrom)",
            energy_changes=result.energy_changes,
            max_displacements=result.max_displacements,
            best_epoch=result.best_epoch,
            selected_segment_epochs_completed=(
                result.selected_segment_epochs_completed
            ),
            epoch_energies=result.epoch_energies,
            termination_reason=cast(
                TerminationReason,
                result.termination_reason,
            ),
            terminal_converged=result.terminal_converged,
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
