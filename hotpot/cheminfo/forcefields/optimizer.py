"""Stateful Open Babel optimization and force-field run reporting."""

from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass, replace
from typing import Callable, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
from openbabel import openbabel as ob

from .. import geometry as geo
from ..obconvert import extract_obmol_coordinates, mol2obmol, set_obmol_coordinates
from .acceptance import (
    _format_geometry_checks,
    _has_unreturnable_frame_failure,
    _resolve_acceptance_thresholds,
    evaluate_structure_acceptance,
)
from .backend import (
    _energy_factor_to_kj,
    _forcefield_energy_in_kj,
    _get_forcefield,
    _serialized_forcefield_call,
    _setup_forcefield_backend,
)
from .contracts import (
    AcceptanceLevel,
    ForceFieldRunReport,
    ForceFieldValidationReport,
    GeometryQualityError,
    GeometryQualityWarning,
    OptimizationAlgorithm,
    StructureAcceptanceThresholds,
    TerminationReason,
)
from .coordinates import _perturbed_coordinates
from .settings import _BOND_RING_MAX_SIZE
from .topology import TopologyReference
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
    quality_report: ForceFieldValidationReport
    energy_changes: Tuple[float, ...]
    max_displacements: Tuple[float, ...]


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
        self.retain_epoch_history = retain_epoch_history
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
            if self.retain_epoch_history:
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
    retain_epoch_history: bool,
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
        retain_epoch_history=retain_epoch_history,
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
