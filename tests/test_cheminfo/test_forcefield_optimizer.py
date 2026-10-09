from types import SimpleNamespace

import numpy as np
import pytest

from hotpot import read_mol
from hotpot.cheminfo import geometry
from hotpot.cheminfo.forcefields import acceptance as acceptance_impl
from hotpot.cheminfo.forcefields import attempts
from hotpot.cheminfo.forcefields import backend as ob_backend
from hotpot.cheminfo.forcefields import coordinates as coordinate_utils
from hotpot.cheminfo.forcefields import optimizer as optimizer_impl
from hotpot.cheminfo.forcefields import ff
from hotpot.cheminfo.forcefields import workflows
from hotpot.cheminfo.forcefields.trajectory import (
    ForceFieldTrajectory,
    OptimizationFrameEvidence,
    TrajectoryEvent,
    TrajectoryStage,
    TrajectoryStart,
)
from hotpot.cheminfo.obWrappers.contracts import (
    OptimizationFrame,
    OptimizationReport,
    RuleExecutionReport,
    RuleStage,
)


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


def _rule_report():
    return RuleExecutionReport(RuleStage.PRE_FORCEFIELD_SETUP)


def _native_frame(
    coordinates,
    energy,
    *,
    epoch_index,
    converged=False,
    exploded=False,
    rms_gradient=1.0,
    max_gradient=2.0,
    energy_change=None,
    max_displacement=None,
    segment_epochs_completed=None,
    segment_index=0,
):
    return OptimizationFrame(
        coordinates=np.asarray(coordinates, dtype=np.float64),
        energy=energy,
        rms_gradient=rms_gradient,
        max_gradient=max_gradient,
        exploded=exploded,
        converged=converged,
        epoch_index=epoch_index,
        segment_epochs_completed=(
            epoch_index + 1
            if segment_epochs_completed is None
            else segment_epochs_completed
        ),
        segment_index=segment_index,
        energy_change=energy_change,
        max_displacement=max_displacement,
    )


def _native_result(
    *,
    frames=(),
    coordinates=None,
    terminal_coordinates=None,
    selected_frame_index=-1,
    best_epoch=-1,
    final_energy=float("nan"),
    best_energy=float("nan"),
    rms_gradient=float("nan"),
    max_gradient=float("nan"),
    exploded=False,
    converged=False,
    epochs_completed=0,
    steps_submitted=0,
    initialization_steps=0,
    selected_segment_epochs_completed=0,
    backend_energy_unit="kcal/mol",
    termination_reason="budget_exhausted",
    terminal_converged=False,
    energy_changes=(),
    max_displacements=(),
    epoch_energies=(),
):
    selected_coordinates = (
        np.zeros((2, 3), dtype=np.float64)
        if coordinates is None
        else np.asarray(coordinates, dtype=np.float64)
    )
    terminal = (
        selected_coordinates
        if terminal_coordinates is None
        else np.asarray(terminal_coordinates, dtype=np.float64)
    )
    return OptimizationReport(
        coordinates=selected_coordinates,
        terminal_coordinates=terminal,
        frames=tuple(frames),
        selected_frame_index=selected_frame_index,
        best_epoch=best_epoch,
        final_energy=final_energy,
        best_energy=best_energy,
        rms_gradient=rms_gradient,
        max_gradient=max_gradient,
        exploded=exploded,
        converged=converged,
        epochs_completed=epochs_completed,
        steps_submitted=steps_submitted,
        initialization_steps=initialization_steps,
        selected_segment_epochs_completed=selected_segment_epochs_completed,
        energy_unit="kJ/mol",
        backend_energy_unit=backend_energy_unit,
        termination_reason=termination_reason,
        terminal_converged=terminal_converged,
        energy_changes=tuple(energy_changes),
        max_displacements=tuple(max_displacements),
        epoch_energies=tuple(epoch_energies),
        rules=_rule_report(),
    )


def _optimizer(monkeypatch, native_result, **options):
    calls = []

    def fake_native_optimize(mol, forcefield, **native_options):
        calls.append((mol, forcefield, native_options))
        return native_result

    monkeypatch.setattr(optimizer_impl, "_native_optimize", fake_native_optimize)
    defaults = {
        "algorithm": "conjugate",
        "epochs": 3,
        "steps_per_epoch": 7,
        "perturb_interval": None,
        "perturb_sigma": 0.5,
        "retain_epoch_history": True,
        "increasing_vdw": True,
        "vdw_cutoff_start": 0.0,
        "vdw_cutoff_end": 12.0,
        "seed": 17,
    }
    defaults.update(options)
    return (
        optimizer_impl._OpenBabelOptimizer(
            "MMFF94s",
            "MMFF94s",
            **defaults,
        ),
        calls,
    )


def _run_optimizer(optimizer, molecule):
    trajectory = ForceFieldTrajectory.from_molecule(
        molecule,
        start=TrajectoryStart.FINAL_OPTIMIZATION,
    )
    report = optimizer.optimize(molecule, trajectory=trajectory)
    trajectory.materialize(molecule, keep_all=optimizer.retain_epoch_history)
    return report, trajectory


def test_optimizer_forwards_native_controls_and_translates_report(monkeypatch):
    coordinates = np.full((2, 3), 1.0)
    terminal_coordinates = np.full((2, 3), 2.0)
    result = _native_result(
        frames=(
            _native_frame(coordinates, 4.184, epoch_index=1),
        ),
        coordinates=coordinates,
        terminal_coordinates=terminal_coordinates,
        selected_frame_index=0,
        best_epoch=1,
        final_energy=8.368,
        best_energy=4.184,
        rms_gradient=0.5,
        max_gradient=1.5,
        exploded=False,
        converged=True,
        epochs_completed=3,
        steps_submitted=18,
        initialization_steps=3,
        selected_segment_epochs_completed=2,
        backend_energy_unit="kcal/mol",
        termination_reason="converged",
        terminal_converged=True,
        energy_changes=(2.0, 1.0),
        max_displacements=(0.3, 0.2),
        epoch_energies=(12.552, 4.184, 8.368),
    )
    optimizer, calls = _optimizer(
        monkeypatch,
        result,
        convergence_level=ff.ConvergenceLevel.FAST,
    )
    molecule = _OptimizerMolecule()

    report, _ = _run_optimizer(optimizer, molecule)

    assert len(calls) == 1
    called_mol, forcefield, options = calls[0]
    assert called_mol is molecule
    assert forcefield == "MMFF94s"
    assert options == {
        "algorithm": "conjugate",
        "epochs": 3,
        "steps_per_epoch": 7,
        "perturb_interval": None,
        "perturbation_offsets": None,
        "retain_frames": True,
        "retain_epoch_history": True,
        "increasing_vdw": True,
        "vdw_cutoff_start": 0.0,
        "vdw_cutoff_end": 12.0,
        "energy_tolerance": 1.0e-6,
        "convergence_level": ff.ConvergenceLevel.FAST,
        "stopping_window": None,
        "maximum_energy_change_kj_mol": 1.0e-4,
        "maximum_atom_displacement_angstrom": 1.0e-4,
        "maximum_rms_gradient_kj_mol_angstrom": 1.0,
        "maximum_gradient_kj_mol_angstrom": 5.0,
    }
    assert report.requested_forcefield == "MMFF94s"
    assert report.effective_forcefield == "MMFF94s"
    assert report.epochs_completed == 3
    assert report.steps_submitted == 18
    assert report.initialization_steps == 3
    assert report.steps_completed is None
    assert report.converged is True
    assert report.terminal_converged is True
    assert report.convergence_level is ff.ConvergenceLevel.FAST
    assert report.termination_reason == "converged"
    assert report.best_energy == pytest.approx(4.184)
    assert report.final_energy == pytest.approx(8.368)
    assert report.energy_unit == "kJ/mol"
    assert report.backend_energy_unit == "kcal/mol"
    assert report.gradient_unit == "kJ/(mol*angstrom)"
    assert report.energy_changes == (2.0, 1.0)
    assert report.max_displacements == (0.3, 0.2)
    assert report.epoch_energies == (12.552, 4.184, 8.368)
    np.testing.assert_array_equal(molecule.coordinates, coordinates)


def test_optimizer_maps_native_frames_to_shared_trajectory(monkeypatch):
    frames = (
        _native_frame(np.full((2, 3), 3.0), 3.0, epoch_index=0),
        _native_frame(
            np.full((2, 3), 1.0),
            1.0,
            epoch_index=1,
            energy_change=2.0,
            max_displacement=np.sqrt(12.0),
        ),
        _native_frame(
            np.full((2, 3), 2.0),
            2.0,
            epoch_index=2,
            converged=True,
            energy_change=1.0,
            max_displacement=np.sqrt(3.0),
        ),
    )
    result = _native_result(
        frames=frames,
        coordinates=frames[1].coordinates,
        terminal_coordinates=frames[2].coordinates,
        selected_frame_index=1,
        best_epoch=1,
        final_energy=2.0,
        best_energy=1.0,
        rms_gradient=1.0,
        max_gradient=2.0,
        converged=False,
        epochs_completed=3,
        steps_submitted=18,
        initialization_steps=3,
        selected_segment_epochs_completed=2,
        termination_reason="converged",
        terminal_converged=True,
        epoch_energies=(3.0, 1.0, 2.0),
    )
    optimizer, _ = _optimizer(monkeypatch, result)
    molecule = _OptimizerMolecule()

    report, trajectory = _run_optimizer(optimizer, molecule)

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
    assert tuple(frame.step for frame in trajectory) == (None, 0, 1, 2)
    assert trajectory[0].energy_kj_mol is None
    assert trajectory[0].evidence is None
    assert all(
        isinstance(frame.evidence, OptimizationFrameEvidence)
        for frame in trajectory.frames[1:]
    )
    evidence = tuple(frame.evidence for frame in trajectory.frames[1:])
    assert evidence[0].energy_change_kj_mol is None
    assert evidence[1].energy_change_kj_mol == pytest.approx(2.0)
    assert evidence[1].max_displacement_angstrom == pytest.approx(np.sqrt(12.0))
    assert evidence[2].converged is True
    assert trajectory.selected_index == 2
    assert report.best_epoch == 1
    assert len(molecule.frames) == 4
    assert molecule._conformers_index == 2
    assert molecule.energy == pytest.approx(report.best_energy)


def test_optimizer_forwards_stopping_and_seeded_perturbation_controls(monkeypatch):
    criteria = ff.OptimizationStoppingCriteria(
        window=2,
        maximum_energy_change_kj_mol=0.1,
        maximum_atom_displacement_angstrom=0.2,
        maximum_rms_gradient_kj_mol_angstrom=0.3,
        maximum_gradient_kj_mol_angstrom=0.4,
    )
    result = _native_result(coordinates=np.zeros((2, 3)))
    first, first_calls = _optimizer(
        monkeypatch,
        result,
        epochs=6,
        perturb_interval=2,
        perturb_sigma=0.2,
        stopping_criteria=criteria,
    )
    _run_optimizer(first, _OptimizerMolecule())
    first_options = first_calls[0][2]

    assert first_options["stopping_window"] == 2
    assert first_options["maximum_energy_change_kj_mol"] == pytest.approx(0.1)
    assert first_options["maximum_atom_displacement_angstrom"] == pytest.approx(0.2)
    assert first_options["maximum_rms_gradient_kj_mol_angstrom"] == pytest.approx(0.3)
    assert first_options["maximum_gradient_kj_mol_angstrom"] == pytest.approx(0.4)
    offsets = first_options["perturbation_offsets"]
    assert offsets.shape == (2, 2, 3)
    assert offsets.dtype == np.float64
    assert offsets.flags.c_contiguous
    assert np.max(offsets) <= 0.4
    assert np.min(offsets) >= -0.4

    second, second_calls = _optimizer(
        monkeypatch,
        result,
        epochs=6,
        perturb_interval=2,
        perturb_sigma=0.2,
        stopping_criteria=criteria,
    )
    _run_optimizer(second, _OptimizerMolecule())
    np.testing.assert_array_equal(
        offsets,
        second_calls[0][2]["perturbation_offsets"],
    )


def test_unrecorded_stage_does_not_request_or_materialize_native_frames(monkeypatch):
    selected = np.full((2, 3), 2.0)
    result = _native_result(
        coordinates=selected,
        terminal_coordinates=selected,
        best_epoch=0,
        best_energy=1.0,
        final_energy=1.0,
        epochs_completed=1,
    )
    optimizer, calls = _optimizer(monkeypatch, result)
    molecule = _OptimizerMolecule()
    trajectory = ForceFieldTrajectory.from_molecule(
        molecule,
        start=TrajectoryStart.FINAL_OPTIMIZATION,
    )

    report = optimizer.optimize(
        molecule,
        trajectory=trajectory,
        trajectory_stage=TrajectoryStage.LIGAND_BUILD,
    )

    assert calls[0][2]["retain_frames"] is False
    assert len(trajectory) == 0
    assert molecule.frames == []
    assert report.best_energy == pytest.approx(1.0)
    np.testing.assert_array_equal(molecule.coordinates, selected)


def test_initial_frame_sentinel_selects_the_preoptimization_structure(monkeypatch):
    initial = np.zeros((2, 3))
    result = _native_result(
        frames=(
            _native_frame(
                np.full((2, 3), np.nan),
                float("nan"),
                epoch_index=0,
                rms_gradient=float("nan"),
                max_gradient=float("nan"),
            ),
        ),
        coordinates=initial,
        terminal_coordinates=np.full((2, 3), np.nan),
        selected_frame_index=-1,
        best_epoch=-1,
        final_energy=float("nan"),
        best_energy=float("nan"),
        epochs_completed=1,
    )
    optimizer, _ = _optimizer(monkeypatch, result)
    molecule = _OptimizerMolecule()

    report, trajectory = _run_optimizer(optimizer, molecule)

    assert trajectory.selected_index == 0
    assert report.best_epoch == -1
    assert report.selected_segment_epochs_completed == 0
    np.testing.assert_array_equal(molecule.coordinates, initial)


def test_optimizer_adapter_does_not_run_acceptance_or_topology_scans(monkeypatch):
    result = _native_result(
        coordinates=np.ones((2, 3)),
        best_epoch=0,
        best_energy=1.0,
        final_energy=1.0,
        epochs_completed=1,
    )
    optimizer, _ = _optimizer(monkeypatch, result)
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

    report, _ = _run_optimizer(optimizer, _OptimizerMolecule())

    assert report.epochs_completed == 1
    assert report.quality_report is None


@pytest.mark.parametrize("stage", ("lookup", "setup", "preflight-validation"))
def test_native_setup_failure_is_translated_to_public_diagnostics(
    monkeypatch,
    stage,
):
    class NativeSetupError(RuntimeError):
        def __init__(self):
            super().__init__(f"native {stage} failed")
            self.stage = stage

    def fail(*args, **kwargs):
        raise NativeSetupError()

    monkeypatch.setattr(
        optimizer_impl,
        "_native_module",
        lambda: SimpleNamespace(ForceFieldSetupError=NativeSetupError),
    )
    monkeypatch.setattr(optimizer_impl, "_native_optimize", fail)
    optimizer = optimizer_impl._OpenBabelOptimizer(
        "MMFF94s",
        "MMFF94s",
        algorithm="conjugate",
        epochs=1,
        steps_per_epoch=1,
        perturb_interval=None,
        perturb_sigma=0.5,
        retain_epoch_history=False,
        increasing_vdw=False,
        vdw_cutoff_start=0.0,
        vdw_cutoff_end=12.0,
        seed=17,
    )
    molecule = _OptimizerMolecule()
    trajectory = ForceFieldTrajectory.from_molecule(molecule)

    with pytest.raises(ff.ForceFieldSetupError) as caught:
        optimizer.optimize(molecule, trajectory=trajectory)

    assert caught.value.report == ff.ForceFieldSetupReport(
        requested_forcefield="MMFF94s",
        effective_forcefield="MMFF94s",
        stage=stage,
    )


def test_native_result_controls_single_frame_materialization(monkeypatch):
    selected = np.full((2, 3), 4.0)
    result = _native_result(
        frames=(
            _native_frame(np.ones((2, 3)), 2.0, epoch_index=0),
            _native_frame(selected, 1.0, epoch_index=1),
        ),
        coordinates=selected,
        terminal_coordinates=selected,
        selected_frame_index=1,
        best_epoch=1,
        final_energy=1.0,
        best_energy=1.0,
        epochs_completed=2,
        selected_segment_epochs_completed=2,
        energy_changes=(1.0,),
        max_displacements=(0.5,),
    )
    optimizer, _ = _optimizer(
        monkeypatch,
        result,
        retain_epoch_history=False,
    )
    molecule = _OptimizerMolecule()

    report, _ = _run_optimizer(optimizer, molecule)

    assert len(molecule.frames) == 1
    assert report.energy_changes == (1.0,)
    assert report.max_displacements == (0.5,)
    assert report.selected_segment_epochs_completed == 2
    np.testing.assert_array_equal(molecule.coordinates, selected)


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


@pytest.mark.parametrize(
    ("requested", "expected"),
    [(None, "MMFF94s"), ("UFF", "UFF"), ("GAFF", "GAFF")],
)
def test_organic_forcefield_resolution_preserves_explicit_choices(
    requested,
    expected,
):
    assert ob_backend._resolve_organic_forcefield(requested) == expected


@pytest.mark.parametrize(
    "resolver",
    (ob_backend._resolve_complex_forcefield, ob_backend._resolve_organic_forcefield),
)
def test_forcefield_resolution_rejects_unknown_names(resolver):
    with pytest.raises(ValueError, match="Unsupported force field"):
        resolver("not-a-forcefield")


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

    monkeypatch.setattr(attempts, "_optimize_working_mol", fake_run)

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
    assert report.optimization.backend_energy_unit in {"kJ/mol", "kcal/mol"}
    assert report.optimization.epochs_completed <= 2
    assert len(molecule.atoms) == 9
    assert np.all(np.isfinite(molecule.coordinates))
