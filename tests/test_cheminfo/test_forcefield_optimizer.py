from types import SimpleNamespace

import numpy as np
import pytest

from hotpot import read_mol
from hotpot.cheminfo.forcefields import utils as ff


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


class _OptimizerMolecule:
    def __init__(self):
        self.coordinates = np.zeros((2, 3), dtype=float)
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


def _acceptance_report(passed=True, checks=()):
    return ff.ForceFieldValidationReport(
        level="standard",
        passed=passed,
        checks=tuple(checks),
    )


def _optimizer(monkeypatch, backend, frames, **kwargs):
    backend.frames = frames
    obmol = SimpleNamespace(coordinates=np.zeros_like(frames[0], dtype=float))
    monkeypatch.setattr(ff, "_get_forcefield", lambda _: backend)
    monkeypatch.setattr(ff, "_make_constraints", lambda _: object())
    monkeypatch.setattr(ff.ob, "OBMolAtomIter", lambda _: (object(), object()))
    monkeypatch.setattr(ff, "mol2obmol", lambda mol: (obmol, {0: 1, 1: 2}))
    monkeypatch.setattr(
        ff,
        "extract_obmol_coordinates",
        lambda current: np.asarray(current.coordinates, dtype=float).copy(),
    )
    monkeypatch.setattr(
        ff,
        "set_obmol_coordinates",
        lambda current, coordinates: setattr(
            current, "coordinates", np.asarray(coordinates, dtype=float).copy()
        ),
    )
    def evaluate_quality(*args, **options):
        assert options["forcefield_stage"] == "final"
        return _acceptance_report()

    monkeypatch.setattr(ff, "evaluate_structure_acceptance", evaluate_quality)
    return ff._OpenBabelOptimizer(
        "MMFF94s",
        "MMFF94s",
        algorithm="conjugate",
        epochs=3,
        steps_per_epoch=7,
        perturb_interval=None,
        perturb_sigma=0.5,
        save_movie=True,
        increasing_vdw=True,
        vdw_cutoff_start=0.0,
        vdw_cutoff_end=12.0,
        seed=17,
        **kwargs,
    )


def test_optimizer_uses_segmented_steps_vdw_interpolation_and_best_frame(monkeypatch):
    frames = [
        np.full((2, 3), 3.0),
        np.full((2, 3), 1.0),
        np.full((2, 3), 2.0),
    ]
    backend = _Backend([3.0, 1.0, 2.0])
    optimizer = _optimizer(monkeypatch, backend, frames)
    molecule = _OptimizerMolecule()

    report = optimizer.optimize(
        molecule,
        quality_level="standard",
        topology_reference=object(),
        quality_thresholds=None,
    )

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
    assert molecule._conformers_index == report.best_epoch == 1
    assert len(molecule.frames) == 3


def test_vdw_frames_are_ranked_only_under_the_final_cutoff(monkeypatch):
    frames = [
        np.full((2, 3), 3.0),
        np.full((2, 3), 1.0),
        np.full((2, 3), 2.0),
    ]
    backend = _CutoffDependentBackend()
    optimizer = _optimizer(monkeypatch, backend, frames)
    molecule = _OptimizerMolecule()

    report = optimizer.optimize(
        molecule,
        quality_level="standard",
        topology_reference=object(),
        quality_thresholds=None,
    )

    assert backend.scored_cutoffs == [12.0, 12.0, 12.0]
    assert report.best_epoch == 0
    assert np.array_equal(molecule.coordinates, frames[0])


def test_optimizer_reports_early_backend_stop_as_converged(monkeypatch):
    frames = [np.zeros((2, 3))]
    backend = _Backend([1.0], unit="kJ/mol")
    optimizer = _optimizer(monkeypatch, backend, frames)
    optimizer.increasing_vdw = False

    report = optimizer.optimize(
        _OptimizerMolecule(),
        quality_level="standard",
        topology_reference=object(),
        quality_thresholds=None,
    )

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

    report = optimizer.optimize(
        _OptimizerMolecule(),
        quality_level="standard",
        topology_reference=object(),
        quality_thresholds=None,
    )

    assert report.steps_submitted == 20
    assert report.initialization_steps == 1
    assert report.steps_completed is None
    assert report.terminal_converged is False
    assert report.termination_reason == "budget_exhausted"


def test_optimizer_stops_at_first_ring_piercing_and_retains_that_frame(
    monkeypatch,
):
    frames = [
        np.zeros((2, 3)),
        np.ones((2, 3)),
        np.full((2, 3), 2.0),
    ]
    backend = _BudgetBackend([3.0, 2.0, 1.0], unit="kJ/mol")
    optimizer = _optimizer(monkeypatch, backend, frames)
    optimizer.increasing_vdw = False
    optimizer.stop_on_ring_piercing = True

    def evaluate_quality(molecule, **options):
        piercing_count = int(float(molecule.coordinates[0, 0]) == 1.0)
        return ff.ForceFieldValidationReport(
            level="standard",
            passed=not piercing_count,
            checks=(),
            metrics={"bond_ring_piercing_count": piercing_count},
        )

    monkeypatch.setattr(ff, "evaluate_structure_acceptance", evaluate_quality)
    molecule = _OptimizerMolecule()

    report = optimizer.optimize(
        molecule,
        quality_level="standard",
        topology_reference=object(),
        quality_thresholds=None,
    )

    assert report.epochs_completed == 2
    assert report.termination_reason == "ring_piercing"
    assert report.terminal_converged is False
    np.testing.assert_array_equal(molecule.coordinates, frames[1])


def test_ring_piercing_stop_retains_finite_failed_frame_for_repair(monkeypatch):
    frames = [np.zeros((2, 3))]
    backend = _BudgetBackend([1.0], unit="kJ/mol")
    optimizer = _optimizer(monkeypatch, backend, frames)
    optimizer.increasing_vdw = False
    optimizer.stop_on_ring_piercing = True
    rejected = ff.ForceFieldValidationReport(
        level="standard",
        passed=False,
        checks=(
            ff.AcceptanceCheck(name="backend_explosion", passed=False),
        ),
        metrics={"bond_ring_piercing_count": 1},
    )
    monkeypatch.setattr(
        ff,
        "evaluate_structure_acceptance",
        lambda *args, **kwargs: rejected,
    )

    molecule = _OptimizerMolecule()
    report = optimizer.optimize(
        molecule,
        quality_level="standard",
        topology_reference=object(),
        quality_thresholds=None,
    )

    assert report.quality_report is rejected
    assert report.termination_reason == "ring_piercing"
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

    report = optimizer.optimize(
        _OptimizerMolecule(),
        quality_level="standard",
        topology_reference=object(),
        quality_thresholds=None,
    )

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

    report = optimizer.optimize(
        _OptimizerMolecule(),
        quality_level="standard",
        topology_reference=object(),
        quality_thresholds=None,
    )

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

    report = optimizer.optimize(
        molecule,
        quality_level="standard",
        topology_reference=object(),
        quality_thresholds=None,
    )

    assert backend.initializations == [
        (49, pytest.approx(1.0e-6)),
        (28, pytest.approx(1.0e-6)),
        (7, pytest.approx(1.0e-6)),
    ]
    assert backend.take_calls == [6, 6, 6]
    assert report.epochs_completed == 3
    assert report.best_epoch == 6
    assert len(molecule.frames) == 3
    assert molecule._conformers_index == 2
    assert np.array_equal(molecule.coordinates, frames[2])


def test_default_output_keeps_only_one_frame_and_bounded_scalar_history(monkeypatch):
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
    optimizer.save_movie = False
    molecule = _OptimizerMolecule()

    report = optimizer.optimize(
        molecule,
        quality_level="standard",
        topology_reference=object(),
        quality_thresholds=None,
    )

    assert len(molecule.frames) == 1
    assert report.epoch_energies == ()
    assert len(report.energy_changes) == 5
    assert len(report.max_displacements) == 5


def test_single_step_conjugate_budget_is_consumed_by_initialization(monkeypatch):
    frames = [np.zeros((2, 3))]
    backend = _BudgetBackend([1.0], unit="kJ/mol")
    optimizer = _optimizer(monkeypatch, backend, frames)
    optimizer.epochs = 1
    optimizer.steps_per_epoch = 1
    optimizer.increasing_vdw = False

    report = optimizer.optimize(
        _OptimizerMolecule(),
        quality_level="standard",
        topology_reference=object(),
        quality_thresholds=None,
    )

    assert backend.initializations == [(1, pytest.approx(1.0e-6))]
    assert backend.take_calls == []
    assert report.steps_submitted == 0
    assert report.initialization_steps == 1
    assert report.termination_reason == "budget_exhausted"


def test_optimizer_selects_lowest_energy_frame_that_passes_gate(monkeypatch):
    frames = [
        np.full((2, 3), 3.0),
        np.full((2, 3), 1.0),
        np.full((2, 3), 2.0),
    ]
    backend = _Backend([3.0, 1.0, 2.0], unit="kJ/mol")
    optimizer = _optimizer(monkeypatch, backend, frames)
    optimizer.save_movie = False
    monkeypatch.setattr(
        ff,
        "evaluate_structure_acceptance",
        lambda mol, **options: _acceptance_report(
            passed=float(mol.coordinates[0, 0]) != 1.0,
        ),
    )
    molecule = _OptimizerMolecule()

    report = optimizer.optimize(
        molecule,
        quality_level="standard",
        topology_reference=object(),
        quality_thresholds=None,
    )

    assert report.best_energy == pytest.approx(2.0)
    assert np.array_equal(molecule.coordinates, frames[2])
    assert len(molecule.frames) == 1


@pytest.mark.parametrize("save_movie", (False, True))
def test_optimizer_warns_and_retains_finite_frames_when_none_passes_gate(
    monkeypatch,
    save_movie,
):
    frames = [
        np.zeros((2, 3)),
        np.ones((2, 3)),
        np.full((2, 3), 2.0),
    ]
    backend = _Backend([3.0, 2.0, 1.0], unit="kJ/mol")
    optimizer = _optimizer(monkeypatch, backend, frames)
    optimizer.save_movie = save_movie
    rejected = ff.ForceFieldValidationReport(
        level="standard",
        passed=False,
        checks=(
            ff.AcceptanceCheck(
                name="atom_too_close",
                passed=False,
                measured=0.2,
                threshold=0.4,
            ),
        ),
    )
    monkeypatch.setattr(
        ff,
        "evaluate_structure_acceptance",
        lambda *args, **options: rejected,
    )
    molecule = _OptimizerMolecule()

    with pytest.warns(ff.GeometryQualityWarning, match="acceptance"):
        report = optimizer.optimize(
            molecule,
            quality_level="standard",
            topology_reference=object(),
            quality_thresholds=None,
        )

    assert report.quality_report is rejected
    assert report.quality_report.passed is False
    assert report.best_epoch == 2
    assert report.best_energy == pytest.approx(1.0)
    assert report.termination_reason == "quality_gate_failed"
    np.testing.assert_array_equal(molecule.coordinates, frames[-1])
    assert len(molecule.frames) == (3 if save_movie else 1)
    assert molecule._conformers_index == (2 if save_movie else 0)
    assert len(report.epoch_energies) == (3 if save_movie else 0)


@pytest.mark.parametrize(
    "warning_name",
    ("bond_ring_piercing", "bond_ring_scope_coverage"),
)
def test_optimizer_selects_best_frame_despite_bond_ring_warning(
    monkeypatch,
    warning_name,
):
    frames = [np.zeros((2, 3)), np.ones((2, 3))]
    optimizer = _optimizer(
        monkeypatch,
        _Backend([1.0, 2.0], unit="kJ/mol"),
        frames,
    )
    optimizer.epochs = 2
    undetermined = ff.ForceFieldValidationReport(
        level="standard",
        passed=True,
        checks=(
            ff.AcceptanceCheck(
                name=warning_name,
                passed=False,
                severity="warning",
            ),
        ),
    )
    monkeypatch.setattr(
        ff,
        "evaluate_structure_acceptance",
        lambda *args, **options: undetermined,
    )
    molecule = _OptimizerMolecule()

    report = optimizer.optimize(
        molecule,
        quality_level="standard",
        topology_reference=object(),
        quality_thresholds=None,
    )

    assert report.quality_report is undetermined
    assert report.best_epoch == 0
    np.testing.assert_array_equal(molecule.coordinates, frames[0])


@pytest.mark.parametrize(
    "failure_name",
    ("coordinate_shape", "finite_coordinates", "topology_atom_identity"),
)
def test_optimizer_raises_for_unreturnable_frame_failures(
    monkeypatch,
    failure_name,
):
    frames = [np.zeros((2, 3))]
    optimizer = _optimizer(
        monkeypatch,
        _Backend([1.0], unit="kJ/mol"),
        frames,
    )
    optimizer.increasing_vdw = False
    rejected = ff.ForceFieldValidationReport(
        level="standard",
        passed=False,
        checks=(ff.AcceptanceCheck(name=failure_name, passed=False),),
    )
    monkeypatch.setattr(
        ff,
        "evaluate_structure_acceptance",
        lambda *args, **options: rejected,
    )

    with pytest.raises(ff.GeometryQualityError) as caught:
        optimizer.optimize(
            _OptimizerMolecule(),
            quality_level="standard",
            topology_reference=object(),
            quality_thresholds=None,
        )

    assert caught.value.report is rejected


@pytest.mark.parametrize(
    "failure_name",
    (
        "finite_final_energy",
        "finite_rms_gradient",
        "finite_max_gradient",
        "backend_explosion",
        "atom_too_close",
        "bond_length_ratio",
    ),
)
def test_optimizer_retains_finite_frame_with_diagnostic_failure(
    monkeypatch,
    failure_name,
):
    frames = [np.ones((2, 3))]
    optimizer = _optimizer(
        monkeypatch,
        _Backend([1.0], unit="kJ/mol"),
        frames,
    )
    optimizer.increasing_vdw = False
    rejected = ff.ForceFieldValidationReport(
        level="standard",
        passed=False,
        checks=(ff.AcceptanceCheck(name=failure_name, passed=False),),
    )
    monkeypatch.setattr(
        ff,
        "evaluate_structure_acceptance",
        lambda *args, **options: rejected,
    )
    molecule = _OptimizerMolecule()

    with pytest.warns(ff.GeometryQualityWarning, match=failure_name):
        report = optimizer.optimize(
            molecule,
            quality_level="standard",
            topology_reference=object(),
            quality_thresholds=None,
        )

    assert report.quality_report is rejected
    assert report.termination_reason == "quality_gate_failed"
    np.testing.assert_array_equal(molecule.coordinates, frames[-1])
    assert len(molecule.frames) == 1


def test_local_perturbation_is_reproducible_without_changing_global_rng():
    coordinates = np.zeros((100, 3))
    np.random.seed(2026)
    expected_global = np.random.random()
    np.random.seed(2026)

    first = ff._perturbed_coordinates(
        coordinates,
        sigma=0.2,
        rng=np.random.default_rng(4),
    )
    second = ff._perturbed_coordinates(
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
    assert ff._resolve_complex_forcefield(requested) == expected


def test_empty_constraint_adapter_does_not_consume_molecule_flags():
    molecule = SimpleNamespace(
        atoms=property(lambda _: (_ for _ in ()).throw(AssertionError)),
    )
    assert ff._make_constraints(molecule).Size() == 0


def test_energy_conversion_is_explicit():
    assert ff._energy_factor_to_kj("kJ/mol") == 1.0
    assert ff._energy_factor_to_kj("kcal/mol") == pytest.approx(4.184)
    with pytest.raises(ValueError, match="Unsupported Open Babel energy unit"):
        ff._energy_factor_to_kj("hartree")


def test_unknown_forcefield_fails_before_setup(monkeypatch):
    monkeypatch.setattr(ff, "_find_forcefield_prototype", lambda name: None)

    with pytest.raises(
        ff.ForceFieldSetupError,
        match="Unknown Open Babel force field",
    ) as caught:
        ff._get_forcefield("not-a-forcefield")

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
        optimizer.optimize(
            _OptimizerMolecule(),
            quality_level="standard",
            topology_reference=object(),
            quality_thresholds=None,
        )

    assert caught.value.report == ff.ForceFieldSetupReport(
        requested_forcefield="MMFF94s",
        effective_forcefield="MMFF94s",
        stage="setup",
    )


def test_forcefield_lookup_returns_the_serialized_plugin(monkeypatch):
    backend = object()
    monkeypatch.setattr(ff, "_find_forcefield_prototype", lambda _: backend)

    assert ff._get_forcefield("UFF") is backend


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
        "save_movie": False,
        "increasing_vdw": False,
        "vdw_cutoff_start": 0.0,
        "vdw_cutoff_end": 12.5,
        "seed": None,
    }
    defaults.update(options)

    with pytest.raises(ValueError, match=message):
        ff._OpenBabelOptimizer("UFF", "UFF", **defaults)


def test_ordinary_none_forcefield_is_reported_as_mmff94s(monkeypatch):
    molecule = read_mol("CCO", "smi")
    captured = {}

    def fake_run(working, **options):
        captured.update(options)
        return object()

    monkeypatch.setattr(ff, "_optimize_working_mol", fake_run)

    ff.optimize(molecule, forcefield=None, add_hydrogens=False)

    assert captured["requested_forcefield"] is None
    assert captured["effective_forcefield"] == "MMFF94s"


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
