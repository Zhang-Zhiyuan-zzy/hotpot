from types import SimpleNamespace

import numpy as np
import pytest

from hotpot import read_mol
from hotpot.cheminfo import geometry
from hotpot.cheminfo.forcefields import acceptance as acceptance_impl
from hotpot.cheminfo.forcefields import backend as ob_backend
from hotpot.cheminfo.forcefields import coordinates as coordinate_utils
from hotpot.cheminfo.forcefields import optimizer as optimizer_impl
from hotpot.cheminfo.forcefields import utils as ff
from hotpot.cheminfo.forcefields import workflows
from hotpot.cheminfo.forcefields.trajectory import (
    ForceFieldTrajectory,
    OptimizationFrameEvidence,
    TrajectoryEvent,
    TrajectoryStage,
    TrajectoryStart,
)


class _Vector:
    def GetX(self):
        return 0.0

    def GetY(self):
        return 0.0

    def GetZ(self):
        return 0.0


class _Backend:
    def __init__(self, energies, unit="kcal/mol"):
        self.energies = energies
        self.unit = unit
        self.index = -1
        self.cutoffs = []
        self.electrostatic_cutoffs = []
        self.current_cutoff = None
        self.initialized_cutoffs = []
        self.cutoff_enabled = []
        self.pair_updates = 0
        self.initializations = []
        self.take_calls = []
        self.frames = None
        self.obmol = None
        self.has_new_coordinates = False

    def MakeNewInstance(self):
        return self

    def Setup(self, obmol, constraints):
        self.obmol = obmol
        return True

    def ConjugateGradientsInitialize(self, steps, tolerance):
        self.initializations.append((steps, tolerance))
        self.initialized_cutoffs.append(self.current_cutoff)

    def ConjugateGradientsTakeNSteps(self, steps):
        self.take_calls.append(steps)
        self.index += 1
        self.has_new_coordinates = True
        return self.index < len(self.energies) - 1

    SteepestDescentInitialize = ConjugateGradientsInitialize
    SteepestDescentTakeNSteps = ConjugateGradientsTakeNSteps

    def SetVDWCutOff(self, cutoff):
        self.cutoffs.append(cutoff)
        self.current_cutoff = cutoff

    def EnableCutOff(self, enabled):
        self.cutoff_enabled.append(enabled)

    def SetElectrostaticCutOff(self, cutoff):
        self.electrostatic_cutoffs.append(cutoff)

    def UpdatePairsSimple(self):
        self.pair_updates += 1

    def GetCoordinates(self, obmol):
        if self.frames is not None and self.index >= 0 and self.has_new_coordinates:
            obmol.coordinates = np.asarray(self.frames[self.index]).copy()
            self.has_new_coordinates = False

    def Energy(self, gradients=False):
        if self.obmol is not None and hasattr(self.obmol, "coordinates"):
            marker = float(np.mean(self.obmol.coordinates))
            for frame, energy in zip(self.frames or (), self.energies):
                if np.allclose(marker, np.mean(frame)):
                    return energy
        return self.energies[self.index]

    def GetGradient(self, atom):
        return _Vector()

    def DetectExplosion(self):
        return False

    def GetUnit(self):
        return self.unit


class _CutoffDependentBackend(_Backend):
    def __init__(self):
        super().__init__([1.0, 3.0, 2.0], unit="kJ/mol")
        self.scored_cutoffs = []

    def Energy(self, gradients=False):
        if self.obmol is not None and self.frames is not None:
            marker = float(np.mean(self.obmol.coordinates))
            for frame, energy in zip(self.frames, self.energies):
                if np.allclose(marker, np.mean(frame)):
                    self.scored_cutoffs.append(self.current_cutoff)
                    return energy
        return 10.0


class _BudgetBackend(_Backend):
    def ConjugateGradientsTakeNSteps(self, steps):
        self.take_calls.append(steps)
        self.index += 1
        self.has_new_coordinates = True
        return True

    SteepestDescentTakeNSteps = ConjugateGradientsTakeNSteps


class _LimitAwareBackend(_BudgetBackend):
    def ConjugateGradientsInitialize(self, steps, tolerance):
        super().ConjugateGradientsInitialize(steps, tolerance)
        self.maximum_counter = steps
        self.current_counter = 0

    SteepestDescentInitialize = ConjugateGradientsInitialize

    def ConjugateGradientsTakeNSteps(self, steps):
        super().ConjugateGradientsTakeNSteps(steps)
        self.current_counter += steps
        return self.current_counter < self.maximum_counter

    SteepestDescentTakeNSteps = ConjugateGradientsTakeNSteps


class _SegmentConvergingBackend(_Backend):
    def ConjugateGradientsTakeNSteps(self, steps):
        self.take_calls.append(steps)
        self.index += 1
        self.has_new_coordinates = True
        return False

    SteepestDescentTakeNSteps = ConjugateGradientsTakeNSteps


class _ScriptedBackend(_Backend):
    def __init__(self, energies, continues, unit="kJ/mol"):
        super().__init__(energies, unit=unit)
        self.continues = continues

    def ConjugateGradientsTakeNSteps(self, steps):
        self.take_calls.append(steps)
        self.index += 1
        self.has_new_coordinates = True
        return self.continues[self.index]

    SteepestDescentTakeNSteps = ConjugateGradientsTakeNSteps


class _ExplodingBackend(_BudgetBackend):
    def DetectExplosion(self):
        return True


class _OptimizerMolecule:
    def __init__(self):
        self.coordinates = np.zeros((2, 3), dtype=float)
        self.atoms = (
            SimpleNamespace(
                idx=0,
                id=1,
                atomic_number=6,
                formal_charge=0,
                symbol="C",
            ),
            SimpleNamespace(
                idx=1,
                id=2,
                atomic_number=6,
                formal_charge=0,
                symbol="C",
            ),
        )
        self.bonds = ()
        self.energy = None
        self.frames = []
        self.frame_energies = []
        self._conformers_index = 0
        self.obmol = SimpleNamespace()

    def to_obmol(self):
        raise AssertionError("forcefields must rebuild a fresh OBMol")

    def conformer_clear(self):
        self.frames.clear()
        self.frame_energies.clear()

    def conformer_add(self, coordinates, energies):
        coordinates = np.asarray(coordinates)
        if coordinates.ndim == 2:
            coordinates = coordinates[None, ...]
        self.frames.extend(coordinates)
        self.frame_energies.extend(np.asarray(energies).reshape(-1))

    def conformer_load(self, index):
        self.coordinates = np.asarray(self.frames[index]).copy()
        self.energy = self.frame_energies[index]
        self._conformers_index = index


def _optimizer(monkeypatch, backend, frames, **kwargs):
    backend.frames = frames
    obmol = SimpleNamespace(coordinates=np.zeros_like(frames[0], dtype=float))
    monkeypatch.setattr(optimizer_impl, "_get_forcefield", lambda _: backend)
    monkeypatch.setattr(ob_backend, "_make_constraints", lambda _: object())
    monkeypatch.setattr(
        optimizer_impl.ob,
        "OBMolAtomIter",
        lambda _: (object(), object()),
    )
    monkeypatch.setattr(
        optimizer_impl,
        "mol2obmol",
        lambda mol: (obmol, {0: 1, 1: 2}),
    )
    monkeypatch.setattr(
        optimizer_impl,
        "extract_obmol_coordinates",
        lambda current: np.asarray(current.coordinates, dtype=float).copy(),
    )
    monkeypatch.setattr(
        optimizer_impl,
        "set_obmol_coordinates",
        lambda current, coordinates: setattr(
            current, "coordinates", np.asarray(coordinates, dtype=float).copy()
        ),
    )
    return optimizer_impl._OpenBabelOptimizer(
        "MMFF94s",
        "MMFF94s",
        algorithm="conjugate",
        epochs=3,
        steps_per_epoch=7,
        perturb_interval=None,
        perturb_sigma=0.5,
        retain_epoch_history=True,
        increasing_vdw=True,
        vdw_cutoff_start=0.0,
        vdw_cutoff_end=12.0,
        seed=17,
        **kwargs,
    )


def _run_optimizer(optimizer, molecule):
    trajectory = ForceFieldTrajectory.from_molecule(
        molecule,
        start=TrajectoryStart.FINAL_OPTIMIZATION,
    )
    report = optimizer.optimize(molecule, trajectory=trajectory)
    trajectory.materialize(molecule, keep_all=optimizer.retain_epoch_history)
    return report


def test_optimizer_uses_segmented_steps_vdw_interpolation_and_best_frame(monkeypatch):
    frames = [
        np.full((2, 3), 3.0),
        np.full((2, 3), 1.0),
        np.full((2, 3), 2.0),
    ]
    backend = _Backend([3.0, 1.0, 2.0])
    optimizer = _optimizer(monkeypatch, backend, frames)
    molecule = _OptimizerMolecule()

    report = _run_optimizer(optimizer, molecule)

    assert backend.initializations == [
        (21, pytest.approx(1.0e-6)),
        (14, pytest.approx(1.0e-6)),
        (7, pytest.approx(1.0e-6)),
    ]
    assert backend.take_calls == [6, 6, 6]
    assert backend.initialized_cutoffs == pytest.approx([4.0, 8.0, 12.0])
    assert backend.cutoff_enabled == [True] * 6
    assert backend.cutoffs == pytest.approx([12.0, 4.0, 12.0, 8.0, 12.0, 12.0])
    assert backend.electrostatic_cutoffs == [1.0e6] * 6
    assert backend.pair_updates == 6
    assert report.epochs_completed == 3
    assert report.steps_submitted == 18
    assert report.initialization_steps == 3
    assert report.steps_completed is None
    assert report.converged is False
    assert report.terminal_converged is True
    assert report.termination_reason == "converged"
    assert report.best_energy == pytest.approx(4.184)
    assert report.final_energy == pytest.approx(8.368)
    assert report.energy_unit == "kJ/mol"
    assert report.backend_energy_unit == "kcal/mol"
    assert report.gradient_unit == "kJ/(mol*angstrom)"
    assert report.energy_changes == ()
    assert report.max_displacements == ()
    assert np.array_equal(molecule.coordinates, frames[1])
    assert molecule.energy == pytest.approx(report.best_energy)
    assert report.best_epoch == 1
    assert report.selected_segment_epochs_completed == 1
    assert molecule._conformers_index == 2
    assert len(molecule.frames) == 4


def test_optimizer_records_into_shared_trajectory_without_materializing(monkeypatch):
    frames = [
        np.full((2, 3), 3.0),
        np.full((2, 3), 1.0),
        np.full((2, 3), 2.0),
    ]
    optimizer = _optimizer(monkeypatch, _Backend([3.0, 1.0, 2.0]), frames)
    optimizer.increasing_vdw = False
    molecule = _OptimizerMolecule()
    trajectory = ForceFieldTrajectory.from_molecule(
        molecule,
        start=TrajectoryStart.COORDINATION_RESTORATION,
    )

    report = optimizer.optimize(
        molecule,
        trajectory=trajectory,
    )

    assert molecule.frames == []
    assert tuple(frame.event for frame in trajectory) == (
        TrajectoryEvent.INITIAL,
        TrajectoryEvent.EPOCH_COMPLETE,
        TrajectoryEvent.EPOCH_COMPLETE,
        TrajectoryEvent.EPOCH_COMPLETE,
    )
    assert all(
        frame.stage is TrajectoryStage.FINAL_OPTIMIZATION
        for frame in trajectory
    )
    assert trajectory[0].energy_kj_mol is None
    assert trajectory[0].evidence is None
    assert tuple(frame.step for frame in trajectory) == (None, 0, 1, 2)
    assert all(
        isinstance(frame.evidence, OptimizationFrameEvidence)
        for frame in trajectory.frames[1:]
    )
    evidence = tuple(frame.evidence for frame in trajectory.frames[1:])
    assert evidence[0].energy_change_kj_mol is None
    assert evidence[0].max_displacement_angstrom is None
    assert evidence[1].energy_change_kj_mol == pytest.approx(8.368)
    assert evidence[1].max_displacement_angstrom == pytest.approx(np.sqrt(12.0))
    assert evidence[2].energy_change_kj_mol == pytest.approx(4.184)
    assert evidence[2].max_displacement_angstrom == pytest.approx(np.sqrt(3.0))
    assert trajectory.selected_index == 2
    assert report.best_epoch == 1

    trajectory.materialize(molecule, keep_all=True)
    assert len(molecule.frames) == 4
    assert molecule._conformers_index == 2
    assert molecule.energy == pytest.approx(report.best_energy)


def test_vdw_frames_are_ranked_only_under_the_final_cutoff(monkeypatch):
    frames = [
        np.full((2, 3), 3.0),
        np.full((2, 3), 1.0),
        np.full((2, 3), 2.0),
    ]
    backend = _CutoffDependentBackend()
    optimizer = _optimizer(monkeypatch, backend, frames)
    molecule = _OptimizerMolecule()

    report = _run_optimizer(optimizer, molecule)

    assert backend.scored_cutoffs == [12.0, 12.0, 12.0]
    assert report.best_epoch == 0
    assert np.array_equal(molecule.coordinates, frames[0])


def test_optimizer_reports_early_backend_stop_as_converged(monkeypatch):
    frames = [np.zeros((2, 3))]
    backend = _Backend([1.0], unit="kJ/mol")
    optimizer = _optimizer(monkeypatch, backend, frames)
    optimizer.increasing_vdw = False

    report = _run_optimizer(optimizer, _OptimizerMolecule())

    assert report.epochs_completed == 1
    assert report.converged is True
    assert report.termination_reason == "converged"


def test_optimizer_reports_external_step_budget_exhaustion(monkeypatch):
    frames = [
        np.zeros((2, 3)),
        np.ones((2, 3)),
        np.full((2, 3), 2.0),
    ]
    backend = _BudgetBackend([3.0, 2.0, 1.0], unit="kJ/mol")
    optimizer = _optimizer(monkeypatch, backend, frames)
    optimizer.increasing_vdw = False

    report = _run_optimizer(optimizer, _OptimizerMolecule())

    assert report.steps_submitted == 20
    assert report.initialization_steps == 1
    assert report.steps_completed is None
    assert report.terminal_converged is False
    assert report.termination_reason == "budget_exhausted"


def test_opt_in_stability_stopping_requires_a_full_current_segment_window(
    monkeypatch,
):
    frames = [
        np.full((2, 3), index * 1.0e-6)
        for index in range(6)
    ]
    energies = [1.0 - index * 1.0e-6 for index in range(6)]
    backend = _BudgetBackend(energies, unit="kJ/mol")
    optimizer = _optimizer(
        monkeypatch,
        backend,
        frames,
        stopping_criteria=ff.OptimizationStoppingCriteria(window=2),
    )
    optimizer.epochs = 6
    optimizer.increasing_vdw = False

    report = _run_optimizer(optimizer, _OptimizerMolecule())

    assert len(backend.take_calls) == 3
    assert report.epochs_completed == 3
    assert report.terminal_converged is False
    assert report.termination_reason == "stability_reached"


def test_stability_stopping_requires_every_numerical_threshold():
    criteria = ff.OptimizationStoppingCriteria(
        window=2,
        maximum_energy_change_kj_mol=0.1,
        maximum_atom_displacement_angstrom=0.2,
        maximum_rms_gradient_kj_mol_angstrom=0.3,
        maximum_gradient_kj_mol_angstrom=0.4,
    )

    def frame(*, rms_gradient=0.3, max_gradient=0.4):
        return optimizer_impl._ObservedFrame(
            coordinates=np.zeros((2, 3)),
            energy=1.0,
            rms_gradient=rms_gradient,
            max_gradient=max_gradient,
            exploded=False,
            converged=False,
            segment_epochs_completed=3,
            segment_index=0,
            history_length=2,
        )

    assert optimizer_impl._segment_satisfies_stopping_criteria(
        frame(),
        (0.1, 0.1),
        (0.2, 0.2),
        (0.3, 0.3),
        (0.4, 0.4),
        criteria,
    )
    assert not optimizer_impl._segment_satisfies_stopping_criteria(
        frame(),
        (0.1, 0.100001),
        (0.2, 0.2),
        (0.3, 0.3),
        (0.4, 0.4),
        criteria,
    )
    assert not optimizer_impl._segment_satisfies_stopping_criteria(
        frame(),
        (0.1, 0.1),
        (0.2, 0.200001),
        (0.3, 0.3),
        (0.4, 0.4),
        criteria,
    )
    assert not optimizer_impl._segment_satisfies_stopping_criteria(
        frame(),
        (0.1, 0.1),
        (0.2, 0.2),
        (0.300001, 0.3),
        (0.4, 0.4),
        criteria,
    )
    assert not optimizer_impl._segment_satisfies_stopping_criteria(
        frame(),
        (0.1, 0.1),
        (0.2, 0.2),
        (0.3, 0.3),
        (0.400001, 0.4),
        criteria,
    )


def test_none_stopping_criteria_preserves_the_full_optimizer_budget(monkeypatch):
    frames = [
        np.full((2, 3), index * 1.0e-6)
        for index in range(4)
    ]
    backend = _BudgetBackend(
        [1.0 - index * 1.0e-6 for index in range(4)],
        unit="kJ/mol",
    )
    optimizer = _optimizer(monkeypatch, backend, frames)
    optimizer.epochs = 4
    optimizer.increasing_vdw = False

    report = _run_optimizer(optimizer, _OptimizerMolecule())

    assert len(backend.take_calls) == 4
    assert report.epochs_completed == 4
    assert report.termination_reason == "budget_exhausted"


def test_stability_stopping_is_disabled_during_increasing_vdw(
    monkeypatch,
):
    frames = [np.full((2, 3), index * 1.0e-6) for index in range(3)]
    backend = _BudgetBackend([1.0, 1.0, 1.0], unit="kJ/mol")
    optimizer = _optimizer(
        monkeypatch,
        backend,
        frames,
        stopping_criteria=ff.OptimizationStoppingCriteria(window=1),
    )
    monkeypatch.setattr(
        optimizer_impl,
        "_segment_satisfies_stopping_criteria",
        lambda *args: pytest.fail("stability stopping ran during VDW annealing"),
    )

    report = _run_optimizer(optimizer, _OptimizerMolecule())

    assert report.epochs_completed == 3


def test_later_perturbation_segment_replaces_an_earlier_stability_reason(
    monkeypatch,
):
    frames = [
        np.zeros((2, 3)),
        np.full((2, 3), 1.0e-6),
        np.full((2, 3), 2.0e-6),
    ]
    backend = _ScriptedBackend(
        [1.0, 1.0 - 1.0e-6, 0.5],
        [True, True, False],
    )
    optimizer = _optimizer(
        monkeypatch,
        backend,
        frames,
        stopping_criteria=ff.OptimizationStoppingCriteria(window=1),
    )
    optimizer.epochs = 5
    optimizer.perturb_interval = 3
    optimizer.increasing_vdw = False

    report = _run_optimizer(optimizer, _OptimizerMolecule())

    assert len(backend.take_calls) == 3
    assert report.epochs_completed == 3
    assert report.terminal_converged is True
    assert report.termination_reason == "converged"


@pytest.mark.parametrize(
    "backend",
    (
        _BudgetBackend([1.0, float("nan"), 1.0], unit="kJ/mol"),
        _ExplodingBackend([1.0, 1.0, 1.0], unit="kJ/mol"),
    ),
)
def test_nonfinite_or_exploded_frames_take_priority_over_stability_stopping(
    monkeypatch,
    backend,
):
    frames = [
        np.full((2, 3), index * 1.0e-6)
        for index in range(3)
    ]
    optimizer = _optimizer(
        monkeypatch,
        backend,
        frames,
        stopping_criteria=ff.OptimizationStoppingCriteria(window=1),
    )
    optimizer.increasing_vdw = False

    report = _run_optimizer(optimizer, _OptimizerMolecule())

    assert report.epochs_completed == 3
    assert report.termination_reason == "budget_exhausted"


def test_optimizer_epochs_do_not_run_acceptance_or_topology_scans(monkeypatch):
    frames = [
        np.zeros((2, 3)),
        np.ones((2, 3)),
        np.full((2, 3), 2.0),
    ]
    backend = _BudgetBackend([3.0, 2.0, 1.0], unit="kJ/mol")
    optimizer = _optimizer(monkeypatch, backend, frames)
    optimizer.increasing_vdw = False
    monkeypatch.setattr(
        acceptance_impl,
        "evaluate_structure_acceptance",
        lambda *args, **kwargs: pytest.fail("acceptance ran inside optimizer"),
    )
    monkeypatch.setattr(
        geometry,
        "determine_bond_ring_piercing_state",
        lambda *args, **kwargs: pytest.fail("topology scan ran inside optimizer"),
    )
    monkeypatch.setattr(
        geometry,
        "screen_bond_ring_relations",
        lambda *args, **kwargs: pytest.fail("topology scan ran inside optimizer"),
    )
    molecule = _OptimizerMolecule()

    report = _run_optimizer(optimizer, molecule)

    assert report.epochs_completed == 3
    assert report.quality_report is None
    assert report.termination_reason == "budget_exhausted"
    assert report.terminal_converged is False
    np.testing.assert_array_equal(molecule.coordinates, frames[-1])


@pytest.mark.parametrize(
    ("algorithm", "expected_limit", "expected_submitted", "expected_initialization"),
    (
        ("conjugate", 21, 20, 1),
        ("steepest", 22, 21, 0),
    ),
)
def test_backend_limit_sentinel_does_not_masquerade_as_convergence(
    monkeypatch,
    algorithm,
    expected_limit,
    expected_submitted,
    expected_initialization,
):
    frames = [
        np.zeros((2, 3)),
        np.ones((2, 3)),
        np.full((2, 3), 2.0),
    ]
    backend = _LimitAwareBackend([3.0, 2.0, 1.0], unit="kJ/mol")
    optimizer = _optimizer(monkeypatch, backend, frames)
    optimizer.algorithm = algorithm
    optimizer.increasing_vdw = False

    report = _run_optimizer(optimizer, _OptimizerMolecule())

    assert backend.initializations == [(expected_limit, pytest.approx(1.0e-6))]
    assert report.steps_submitted == expected_submitted
    assert report.initialization_steps == expected_initialization
    assert report.terminal_converged is False
    assert report.termination_reason == "budget_exhausted"


def test_selected_and_terminal_convergence_are_reported_separately(monkeypatch):
    frames = [np.zeros((2, 3)), np.ones((2, 3))]
    backend = _Backend([1.0, 2.0], unit="kJ/mol")
    optimizer = _optimizer(monkeypatch, backend, frames)
    optimizer.increasing_vdw = False

    report = _run_optimizer(optimizer, _OptimizerMolecule())

    assert report.best_epoch == 0
    assert report.converged is False
    assert report.terminal_converged is True
    assert report.termination_reason == "converged"


def test_scheduled_perturbations_restart_converged_segments(monkeypatch):
    frames = [
        np.full((2, 3), 3.0),
        np.full((2, 3), 2.0),
        np.full((2, 3), 1.0),
    ]
    backend = _SegmentConvergingBackend([3.0, 2.0, 1.0], unit="kJ/mol")
    optimizer = _optimizer(monkeypatch, backend, frames)
    optimizer.epochs = 7
    optimizer.perturb_interval = 3
    optimizer.increasing_vdw = False
    molecule = _OptimizerMolecule()

    report = _run_optimizer(optimizer, molecule)

    assert backend.initializations == [
        (49, pytest.approx(1.0e-6)),
        (28, pytest.approx(1.0e-6)),
        (7, pytest.approx(1.0e-6)),
    ]
    assert backend.take_calls == [6, 6, 6]
    assert report.epochs_completed == 3
    assert report.best_epoch == 2
    assert report.epoch_energies[report.best_epoch] == report.best_energy
    assert report.selected_segment_epochs_completed == 1
    assert len(molecule.frames) == 4
    assert molecule._conformers_index == 3
    assert np.array_equal(molecule.coordinates, frames[2])


def _numerical_report(
    *,
    epochs_completed,
    best_epoch,
    best_energy,
    epoch_energies,
    selected_segment_epochs_completed=1,
):
    return ff.ForceFieldRunReport(
        requested_forcefield="UFF",
        effective_forcefield="UFF",
        setup_succeeded=True,
        converged=False,
        epochs_completed=epochs_completed,
        steps_submitted=epochs_completed,
        initialization_steps=0,
        steps_completed=None,
        final_energy=best_energy,
        best_energy=best_energy,
        energy_unit="kJ/mol",
        rms_gradient=1.0,
        max_gradient=2.0,
        exploded=False,
        best_epoch=best_epoch,
        selected_segment_epochs_completed=selected_segment_epochs_completed,
        epoch_energies=epoch_energies,
    )


def test_combined_report_offsets_observed_best_epoch():
    first = _numerical_report(
        epochs_completed=2,
        best_epoch=1,
        best_energy=2.0,
        epoch_energies=(3.0, 2.0),
    )
    final = _numerical_report(
        epochs_completed=3,
        best_epoch=1,
        best_energy=0.5,
        epoch_energies=(1.0, 0.5, 0.75),
        selected_segment_epochs_completed=2,
    )

    combined = optimizer_impl._combine_forcefield_run_reports((first, final))

    assert combined.best_epoch == 3
    assert combined.epoch_energies[combined.best_epoch] == combined.best_energy
    assert combined.selected_segment_epochs_completed == 2


def test_combined_report_preserves_initial_frame_sentinel():
    first = _numerical_report(
        epochs_completed=2,
        best_epoch=1,
        best_energy=2.0,
        epoch_energies=(3.0, 2.0),
    )
    final = _numerical_report(
        epochs_completed=1,
        best_epoch=-1,
        best_energy=float("nan"),
        epoch_energies=(float("nan"),),
        selected_segment_epochs_completed=0,
    )

    combined = optimizer_impl._combine_forcefield_run_reports((first, final))

    assert combined.best_epoch == -1
    assert np.isnan(combined.best_energy)
    assert combined.selected_segment_epochs_completed == 0


def test_default_output_keeps_only_one_frame_and_numerical_history(monkeypatch):
    frames = [np.full((2, 3), float(index + 1)) for index in range(50)]
    backend = _BudgetBackend(
        [float(value) for value in range(50, 0, -1)],
        unit="kJ/mol",
    )
    optimizer = _optimizer(monkeypatch, backend, frames)
    optimizer.epochs = 50
    optimizer.steps_per_epoch = 1
    optimizer.algorithm = "steepest"
    optimizer.increasing_vdw = False
    optimizer.retain_epoch_history = False
    molecule = _OptimizerMolecule()

    report = _run_optimizer(optimizer, molecule)

    assert len(molecule.frames) == 1
    assert report.epoch_energies == ()
    assert len(report.energy_changes) == 49
    assert len(report.max_displacements) == 49
    assert report.selected_segment_epochs_completed == 50


def test_single_step_conjugate_budget_is_consumed_by_initialization(monkeypatch):
    frames = [np.zeros((2, 3))]
    backend = _BudgetBackend([1.0], unit="kJ/mol")
    optimizer = _optimizer(monkeypatch, backend, frames)
    optimizer.epochs = 1
    optimizer.steps_per_epoch = 1
    optimizer.increasing_vdw = False

    report = _run_optimizer(optimizer, _OptimizerMolecule())

    assert backend.initializations == [(1, pytest.approx(1.0e-6))]
    assert backend.take_calls == []
    assert report.steps_submitted == 0
    assert report.initialization_steps == 1
    assert report.termination_reason == "budget_exhausted"


def test_optimizer_selects_lowest_energy_numerically_usable_frame(monkeypatch):
    frames = [
        np.full((2, 3), 3.0),
        np.full((2, 3), 1.0),
        np.full((2, 3), 2.0),
    ]
    backend = _Backend([3.0, 1.0, 2.0], unit="kJ/mol")
    optimizer = _optimizer(monkeypatch, backend, frames)
    optimizer.retain_epoch_history = False
    molecule = _OptimizerMolecule()

    report = _run_optimizer(optimizer, molecule)

    assert report.best_energy == pytest.approx(1.0)
    assert report.quality_report is None
    assert np.array_equal(molecule.coordinates, frames[1])
    assert len(molecule.frames) == 1


def test_optimizer_skips_lower_energy_frame_with_nonfinite_gradients(monkeypatch):
    frames = [
        np.full((2, 3), 3.0),
        np.full((2, 3), 1.0),
        np.full((2, 3), 2.0),
    ]
    backend = _Backend([3.0, 1.0, 2.0], unit="kJ/mol")
    optimizer = _optimizer(monkeypatch, backend, frames)
    optimizer.increasing_vdw = False

    def gradients(obmol, factor):
        if backend.index == 1:
            return float("nan"), float("nan")
        return 0.0, 0.0

    monkeypatch.setattr(optimizer, "_gradients", gradients)
    molecule = _OptimizerMolecule()

    report = _run_optimizer(optimizer, molecule)

    assert report.best_epoch == 2
    assert report.best_energy == pytest.approx(2.0)
    np.testing.assert_array_equal(molecule.coordinates, frames[2])


def test_optimizer_uses_latest_finite_coordinate_when_no_frame_is_usable(
    monkeypatch,
):
    frames = [
        np.zeros((2, 3)),
        np.ones((2, 3)),
        np.full((2, 3), 2.0),
    ]
    backend = _Backend([float("nan")] * 3, unit="kJ/mol")
    optimizer = _optimizer(monkeypatch, backend, frames)
    optimizer.increasing_vdw = False
    molecule = _OptimizerMolecule()

    report = _run_optimizer(optimizer, molecule)

    assert report.quality_report is None
    assert report.best_epoch == 2
    assert np.isnan(report.best_energy)
    np.testing.assert_array_equal(molecule.coordinates, frames[-1])


def test_optimizer_rejects_run_without_any_finite_coordinate_frame(monkeypatch):
    frames = [
        np.full((2, 3), np.nan),
        np.full((2, 3), np.inf),
        np.full((2, 3), np.nan),
    ]
    optimizer = _optimizer(
        monkeypatch,
        _BudgetBackend([3.0, 2.0, 1.0], unit="kJ/mol"),
        frames,
    )
    optimizer.increasing_vdw = False
    molecule = _OptimizerMolecule()
    molecule.coordinates[:] = np.nan

    with pytest.raises(ff.GeometryQualityError) as caught:
        _run_optimizer(optimizer, molecule)

    assert caught.value.report is None


def test_optimizer_restores_finite_initial_frame_when_all_epochs_are_nonreturnable(
    monkeypatch,
):
    frames = [
        np.full((2, 3), np.nan),
        np.full((2, 3), np.inf),
        np.full((2, 3), np.nan),
    ]
    optimizer = _optimizer(
        monkeypatch,
        _BudgetBackend([3.0, 2.0, 1.0], unit="kJ/mol"),
        frames,
    )
    optimizer.increasing_vdw = False
    molecule = _OptimizerMolecule()
    initial_coordinates = molecule.coordinates.copy()

    report = _run_optimizer(optimizer, molecule)

    assert report.best_epoch == -1
    assert report.selected_segment_epochs_completed == 0
    assert report.converged is False
    assert report.exploded is False
    assert report.final_energy == pytest.approx(1.0)
    assert np.isnan(report.best_energy)
    assert np.isnan(report.rms_gradient)
    assert np.isnan(report.max_gradient)
    assert report.energy_changes == ()
    assert report.max_displacements == ()
    np.testing.assert_array_equal(molecule.coordinates, initial_coordinates)


def test_local_perturbation_is_reproducible_without_changing_global_rng():
    coordinates = np.zeros((100, 3))
    np.random.seed(2026)
    expected_global = np.random.random()
    np.random.seed(2026)

    first = coordinate_utils._perturbed_coordinates(
        coordinates,
        sigma=0.2,
        rng=np.random.default_rng(4),
    )
    second = coordinate_utils._perturbed_coordinates(
        coordinates,
        sigma=0.2,
        rng=np.random.default_rng(4),
    )

    assert np.array_equal(first, second)
    assert np.max(first) <= 0.4
    assert np.min(first) >= -0.4
    assert np.random.random() == expected_global


@pytest.mark.parametrize(
    ("requested", "expected"),
    [(None, "UFF"), ("UFF", "UFF"), ("MMFF94s", "UFF"), ("GAFF", "UFF")],
)
def test_complex_forcefield_resolution_is_centralized(requested, expected):
    assert ob_backend._resolve_complex_forcefield(requested) == expected


def test_empty_constraint_adapter_does_not_consume_molecule_flags():
    molecule = SimpleNamespace(
        atoms=property(lambda _: (_ for _ in ()).throw(AssertionError)),
    )
    assert ob_backend._make_constraints(molecule).Size() == 0


def test_energy_conversion_is_explicit():
    assert ob_backend._energy_factor_to_kj("kJ/mol") == 1.0
    assert ob_backend._energy_factor_to_kj("kcal/mol") == pytest.approx(4.184)
    with pytest.raises(ValueError, match="Unsupported Open Babel energy unit"):
        ob_backend._energy_factor_to_kj("hartree")


@pytest.mark.parametrize(
    ("calc_grad", "unit", "expected"),
    [(True, "kJ/mol", 2.5), (False, "kcal/mol", 10.46)],
)
def test_forcefield_energy_in_kj_uses_backend_unit_and_gradient_flag(
    calc_grad,
    unit,
    expected,
):
    calls = []
    backend = SimpleNamespace(
        Energy=lambda requested: calls.append(requested) or 2.5,
        GetUnit=lambda: unit,
    )

    assert ob_backend._forcefield_energy_in_kj(backend, calc_grad) == pytest.approx(
        expected
    )
    assert calls == [calc_grad]


def test_unknown_forcefield_fails_before_setup(monkeypatch):
    monkeypatch.setattr(
        ob_backend,
        "_find_forcefield_prototype",
        lambda name: None,
    )

    with pytest.raises(
        ff.ForceFieldSetupError,
        match="Unknown Open Babel force field",
    ) as caught:
        ob_backend._get_forcefield("not-a-forcefield")

    assert caught.value.report == ff.ForceFieldSetupReport(
        requested_forcefield="not-a-forcefield",
        effective_forcefield="not-a-forcefield",
        stage="lookup",
    )


def test_optimizer_setup_failure_has_structured_diagnostics(monkeypatch):
    backend = _Backend([0.0], unit="kJ/mol")
    backend.Setup = lambda obmol, constraints: False
    optimizer = _optimizer(
        monkeypatch,
        backend,
        [np.zeros((2, 3))],
    )

    with pytest.raises(ff.ForceFieldSetupError) as caught:
        _run_optimizer(optimizer, _OptimizerMolecule())

    assert caught.value.report == ff.ForceFieldSetupReport(
        requested_forcefield="MMFF94s",
        effective_forcefield="MMFF94s",
        stage="setup",
    )


def test_forcefield_lookup_returns_the_serialized_plugin(monkeypatch):
    backend = object()
    monkeypatch.setattr(
        ob_backend,
        "_find_forcefield_prototype",
        lambda _: backend,
    )

    assert ob_backend._get_forcefield("UFF") is backend


@pytest.mark.parametrize(
    ("options", "message"),
    (
        ({"epochs": 0}, "epochs"),
        ({"steps_per_epoch": 0}, "steps_per_epoch"),
        ({"perturb_interval": 0}, "perturb_interval"),
        ({"perturb_sigma": -0.1}, "perturb_sigma"),
        (
            {"increasing_vdw": True, "vdw_cutoff_start": 8.0, "vdw_cutoff_end": 4.0},
            "vdw_cutoff_end",
        ),
    ),
)
def test_optimizer_rejects_invalid_control_parameters(options, message):
    defaults = {
        "algorithm": "conjugate",
        "epochs": 1,
        "steps_per_epoch": 1,
        "perturb_interval": None,
        "perturb_sigma": 0.5,
        "retain_epoch_history": False,
        "increasing_vdw": False,
        "vdw_cutoff_start": 0.0,
        "vdw_cutoff_end": 12.5,
        "seed": None,
    }
    defaults.update(options)

    with pytest.raises(ValueError, match=message):
        optimizer_impl._OpenBabelOptimizer("UFF", "UFF", **defaults)


@pytest.mark.parametrize(
    ("options", "message"),
    (
        ({"window": 0}, "positive integer"),
        ({"window": 2.5}, "positive integer"),
        ({"window": True}, "positive integer"),
        ({"maximum_energy_change_kj_mol": -1.0}, "non-negative"),
        ({"maximum_atom_displacement_angstrom": -1.0}, "non-negative"),
        ({"maximum_rms_gradient_kj_mol_angstrom": -1.0}, "non-negative"),
        ({"maximum_gradient_kj_mol_angstrom": -1.0}, "non-negative"),
        ({"maximum_gradient_kj_mol_angstrom": float("nan")}, "non-negative"),
        ({"maximum_gradient_kj_mol_angstrom": float("inf")}, "finite"),
    ),
)
def test_stopping_criteria_reject_invalid_values(options, message):
    with pytest.raises(ValueError, match=message):
        ff.OptimizationStoppingCriteria(**options)


def test_stopping_criteria_defaults_are_the_documented_numerical_limits():
    criteria = ff.OptimizationStoppingCriteria()

    assert criteria.window == 5
    assert criteria.maximum_energy_change_kj_mol == pytest.approx(1.0e-4)
    assert criteria.maximum_atom_displacement_angstrom == pytest.approx(1.0e-4)
    assert criteria.maximum_rms_gradient_kj_mol_angstrom == pytest.approx(1.0)
    assert criteria.maximum_gradient_kj_mol_angstrom == pytest.approx(5.0)
    assert criteria.__dataclass_params__.frozen is True


def test_ordinary_none_forcefield_is_reported_as_mmff94s(monkeypatch):
    molecule = read_mol("CCO", "smi")
    captured = {}
    stopping_criteria = ff.OptimizationStoppingCriteria(window=3)

    def fake_run(working, **options):
        captured.update(options)
        return ff.ForceFieldRunReport(
            requested_forcefield=options["requested_forcefield"],
            effective_forcefield=options["effective_forcefield"],
            setup_succeeded=True,
            converged=True,
            epochs_completed=1,
            steps_submitted=1,
            initialization_steps=1,
            steps_completed=None,
            final_energy=1.0,
            best_energy=1.0,
            energy_unit="kJ/mol",
            rms_gradient=0.0,
            max_gradient=0.0,
            exploded=False,
        )

    monkeypatch.setattr(workflows, "_optimize_working_mol", fake_run)

    ff.optimize(
        molecule,
        forcefield=None,
        add_hydrogens=False,
        stopping_criteria=stopping_criteria,
    )

    assert captured["requested_forcefield"] is None
    assert captured["effective_forcefield"] == "MMFF94s"
    assert captured["stopping_criteria"] is stopping_criteria


def test_organic_build_and_optimize_integration():
    molecule = read_mol("CCO", "smi")

    report = ff.build_and_optimize(
        molecule,
        forcefield="MMFF94s",
        epochs=2,
        steps_per_epoch=20,
        quality_level="standard",
        seed=7,
    )

    assert report.effective_forcefield == "MMFF94s"
    assert isinstance(report.build, ff.Build3DReport)
    assert report.optimization.energy_unit == "kJ/mol"
    assert report.optimization.epochs_completed <= 2
    assert len(molecule.atoms) == 9
    assert np.all(np.isfinite(molecule.coordinates))
