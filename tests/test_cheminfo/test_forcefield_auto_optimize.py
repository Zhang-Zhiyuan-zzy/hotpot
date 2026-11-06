"""Behavior fences for native-FAST-first automatic optimization."""

from __future__ import annotations

from copy import copy
from dataclasses import astuple, replace
from typing import Optional

import numpy as np
import pytest

from hotpot import read_mol
from hotpot.cheminfo.structure.bond_kind import BondKind
from hotpot.cheminfo.operations.forcefields import attempts
from hotpot.cheminfo.operations.forcefields import auto as auto_workflow
from hotpot.cheminfo.operations.forcefields import ff


def _complex_molecule():
    molecule = read_mol("[Eu]N", fmt="smi")
    molecule.coordinates = np.asarray(
        ((0.0, 0.0, 0.0), (2.4, 0.0, 0.0)),
        dtype=np.float64,
    )
    return molecule


def _quality(passed: bool) -> ff.ForceFieldValidationReport:
    return ff.ForceFieldValidationReport(
        passed=passed,
        checks=(),
    )


def _trajectory(molecule) -> ff.ForceFieldTrajectory:
    trajectory = ff.ForceFieldTrajectory.from_molecule(
        molecule,
        start=ff.TrajectoryStart.LIGAND_BUILD,
    )
    frame = trajectory.record_molecule(
        molecule,
        stage=ff.TrajectoryStage.FINAL_OPTIMIZATION,
        event=ff.TrajectoryEvent.TERMINAL,
    )
    trajectory.select(frame.index)
    trajectory.set_terminal(frame.index)
    return trajectory


def _run_report(
    passed: bool,
    *,
    trajectory: Optional[ff.ForceFieldTrajectoryArchive] = None,
    termination_reason: ff.TerminationReason = "converged",
) -> ff.ForceFieldRunReport:
    return ff.ForceFieldRunReport(
        effective_forcefield="UFF",
        setup_succeeded=True,
        converged=True,
        epochs_completed=1,
        steps_submitted=100,
        initialization_steps=1,
        final_energy=-1.0,
        best_energy=-1.0,
        energy_unit="kJ/mol",
        rms_gradient=0.1,
        max_gradient=0.2,
        exploded=False,
        quality_report=_quality(passed),
        trajectory=trajectory,
        termination_reason=termination_reason,
    )


def _diagnostics() -> ff.ComplexBuildDiagnostics:
    return ff.ComplexBuildDiagnostics(
        attempt_count=1,
        accepted_candidates=1,
        rejected_candidates=(),
        elapsed_seconds=2.0,
    )


def _complex_workflow_report(molecule) -> ff.ComplexBuildReport:
    trajectory = _trajectory(molecule)
    archive = ff.ForceFieldTrajectoryArchive(trajectory)
    optimization = _run_report(True, trajectory=archive)
    return ff.ComplexBuildReport(
        effective_forcefield="UFF",
        build=_diagnostics(),
        optimization=optimization,
        quality_report=optimization.quality_report,
        trajectory=archive,
    )


def _preliminary_attempt(
    molecule,
    *,
    status: ff.OptimizationAttemptStatus,
    report: Optional[ff.ForceFieldRunReport],
) -> auto_workflow._NativeFastBuildAttempt:
    return auto_workflow._NativeFastBuildAttempt(
        mol=copy(molecule),
        source=auto_workflow._capture_working_source(molecule),
        trajectory=_trajectory(molecule),
        report=report,
        provenance=ff.OptimizationAttemptReport(
            route=ff.OptimizationRoute.NATIVE_FAST,
            status=status,
            quality_report=None if report is None else report.quality_report,
            build_succeeded=status is not ff.OptimizationAttemptStatus.BUILD_FAILED,
        ),
    )


def test_organic_route_calls_ordinary_optimizer_once(monkeypatch):
    molecule = read_mol("CCO", fmt="smi")
    calls = []

    def ordinary(current, forcefield, **options):
        calls.append((current, forcefield, options))
        return _run_report(True)

    monkeypatch.setattr(auto_workflow, "optimize", ordinary)
    monkeypatch.setattr(
        auto_workflow,
        "_run_native_fast_build_attempt",
        lambda *args, **kwargs: pytest.fail("organic route ran a complex attempt"),
    )

    result = auto_workflow.auto_optimize(
        molecule,
        epochs=5,
        steps_per_epoch=19,
        seed=31,
    )

    assert len(calls) == 1
    current, forcefield, options = calls[0]
    assert current is molecule
    assert forcefield is None
    assert options["epochs"] == 5
    assert options["steps_per_epoch"] == 19
    assert options["seed"] == 31
    assert result.routing_report.selected_route is ff.OptimizationRoute.ORDINARY
    assert result.routing_report.attempts[0].elapsed_seconds == pytest.approx(
        result.elapsed_seconds
    )


def test_native_attempt_orders_build_optimize_and_one_gate(monkeypatch):
    molecule = _complex_molecule()
    events = []
    optimized = _run_report(True)

    def build(working):
        events.append(("build", working))
        working.coordinates = working.coordinates + 0.25
        return type("BuildReport", (), {"succeeded": True})()

    def optimize_working(working, **options):
        events.append(("optimize", working, options))
        return replace(optimized, quality_report=None)

    def accept(working, **options):
        events.append(("accept", working, options))
        return optimized.quality_report

    monkeypatch.setattr(auto_workflow, "_native_build", build)
    monkeypatch.setattr(attempts, "_optimize_working_mol", optimize_working)
    monkeypatch.setattr(attempts, "evaluate_structure_acceptance", accept)

    result = auto_workflow._run_native_fast_build_attempt(
        molecule,
        None,
        algorithm="conjugate",
        epochs=3,
        steps_per_epoch=11,
        add_hydrogens=False,
        quality_level="standard",
        quality_thresholds=None,
        seed=7,
        perturb_interval=None,
        perturb_sigma=0.5,
        stopping_criteria=None,
        save_movie=False,
        trajectory_start=ff.TrajectoryStart.LIGAND_BUILD,
        increasing_vdw=False,
        vdw_cutoff_start=0.0,
        vdw_cutoff_end=12.5,
    )

    assert [event[0] for event in events] == ["build", "optimize", "accept"]
    assert events[0][1] is result.mol
    assert events[1][1] is result.mol
    assert events[2][1] is result.mol
    assert events[1][2]["effective_forcefield"] == "UFF"
    assert events[1][2]["convergence_level"] is ff.ConvergenceLevel.FAST
    assert result.provenance.status is ff.OptimizationAttemptStatus.ACCEPTED
    assert [frame.event for frame in result.trajectory.frames[:2]] == [
        ff.TrajectoryEvent.INITIAL,
        ff.TrajectoryEvent.BUILD_COMPLETE,
    ]


def test_unsuccessful_build_skips_optimizer_and_gate(monkeypatch):
    molecule = _complex_molecule()
    monkeypatch.setattr(
        auto_workflow,
        "_native_build",
        lambda working: type("BuildReport", (), {"succeeded": False})(),
    )
    monkeypatch.setattr(
        auto_workflow,
        "_run_ordinary_optimization_attempt",
        lambda *args, **kwargs: pytest.fail("failed build reached optimization"),
    )

    result = auto_workflow._run_native_fast_build_attempt(
        molecule,
        None,
        algorithm="conjugate",
        epochs=3,
        steps_per_epoch=11,
        add_hydrogens=False,
        quality_level="standard",
        quality_thresholds=None,
        seed=7,
        perturb_interval=None,
        perturb_sigma=0.5,
        stopping_criteria=None,
        save_movie=False,
        trajectory_start=ff.TrajectoryStart.LIGAND_BUILD,
        increasing_vdw=False,
        vdw_cutoff_start=0.0,
        vdw_cutoff_end=12.5,
    )

    assert result.report is None
    assert result.provenance.status is ff.OptimizationAttemptStatus.BUILD_FAILED


def test_native_success_commits_once_without_complex_workflow(monkeypatch):
    molecule = _complex_molecule()
    preliminary = _preliminary_attempt(
        molecule,
        status=ff.OptimizationAttemptStatus.ACCEPTED,
        report=_run_report(True),
    )
    archive = ff.ForceFieldTrajectoryArchive(preliminary.trajectory)
    commits = []

    monkeypatch.setattr(
        auto_workflow,
        "_run_native_fast_build_attempt",
        lambda *args, **kwargs: preliminary,
    )
    monkeypatch.setattr(
        auto_workflow,
        "_complexes_build_workflow",
        lambda *args, **kwargs: pytest.fail("accepted native result used fallback"),
    )
    monkeypatch.setattr(
        auto_workflow,
        "_finalize_trajectory",
        lambda *args, **kwargs: archive,
    )
    monkeypatch.setattr(
        auto_workflow,
        "_commit_working_copy",
        lambda original, completed, **options: commits.append((original, completed)),
    )

    result = auto_workflow.auto_optimize(molecule)

    assert commits == [(molecule, preliminary.mol)]
    assert result.trajectory is archive
    assert result.routing_report.selected_route is ff.OptimizationRoute.NATIVE_FAST
    assert tuple(attempt.status for attempt in result.routing_report.attempts) == (
        ff.OptimizationAttemptStatus.ACCEPTED,
    )


def test_rejected_native_attempt_runs_full_workflow_from_pristine_source(monkeypatch):
    molecule = _complex_molecule()
    molecule.charge = 3
    molecule.properties["source"] = "original"
    original_coordinates = molecule.coordinates.copy()
    preliminary = _preliminary_attempt(
        molecule,
        status=ff.OptimizationAttemptStatus.QUALITY_REJECTED,
        report=_run_report(False),
    )
    preliminary.mol.coordinates = preliminary.mol.coordinates + 9.0
    fallback_starts = []
    commits = []

    monkeypatch.setattr(
        auto_workflow,
        "_run_native_fast_build_attempt",
        lambda *args, **kwargs: preliminary,
    )

    def complex_workflow(current, forcefield, **options):
        fallback_starts.append(current.coordinates.copy())
        assert current.charge == 3
        assert current.properties == {"source": "original"}
        assert options["convergence_level"] is ff.ConvergenceLevel.BALANCED
        current.coordinates = current.coordinates + 0.1
        return _complex_workflow_report(current)

    monkeypatch.setattr(auto_workflow, "_complexes_build_workflow", complex_workflow)
    monkeypatch.setattr(
        auto_workflow,
        "_commit_working_copy",
        lambda original, completed, **options: commits.append((original, completed)),
    )

    result = auto_workflow.auto_optimize(
        molecule,
        convergence_level=ff.ConvergenceLevel.BALANCED,
    )

    assert len(fallback_starts) == 1
    assert np.array_equal(fallback_starts[0], original_coordinates)
    assert np.array_equal(molecule.coordinates, original_coordinates)
    assert len(commits) == 1
    assert commits[0][0] is molecule
    assert result.trajectory.preliminary_attempts == (preliminary.trajectory,)
    assert result.routing_report.selected_route is ff.OptimizationRoute.COMPLEX_WORKFLOW
    assert tuple(attempt.status for attempt in result.routing_report.attempts) == (
        ff.OptimizationAttemptStatus.QUALITY_REJECTED,
        ff.OptimizationAttemptStatus.ACCEPTED,
    )
    assert result.routing_report.attempts[1].build_diagnostics == _diagnostics()


@pytest.mark.parametrize(
    "termination_reason",
    (
        "nonfinite_coordinates",
        "nonfinite_energy",
        "nonfinite_gradients",
        "explosion_detected",
    ),
)
def test_numerical_termination_overrides_passing_quality(termination_reason):
    report = _run_report(True, termination_reason=termination_reason)

    assert (
        auto_workflow._optimization_attempt_report(
            report,
            ff.OptimizationRoute.NATIVE_FAST,
        ).status
        is ff.OptimizationAttemptStatus.NUMERICAL_FAILURE
    )


def test_nonfinite_build_coordinates_are_a_controlled_fallback(monkeypatch):
    molecule = _complex_molecule()

    def build(working):
        working.coordinates = np.full_like(working.coordinates, np.nan)
        return type("BuildReport", (), {"succeeded": True})()

    monkeypatch.setattr(auto_workflow, "_native_build", build)
    monkeypatch.setattr(
        auto_workflow,
        "_run_ordinary_optimization_attempt",
        lambda *args, **kwargs: pytest.fail("non-finite build reached optimization"),
    )

    result = auto_workflow._run_native_fast_build_attempt(
        molecule,
        None,
        algorithm="conjugate",
        epochs=3,
        steps_per_epoch=11,
        add_hydrogens=False,
        quality_level="standard",
        quality_thresholds=None,
        seed=7,
        perturb_interval=None,
        perturb_sigma=0.5,
        stopping_criteria=None,
        save_movie=False,
        trajectory_start=ff.TrajectoryStart.LIGAND_BUILD,
        increasing_vdw=False,
        vdw_cutoff_start=0.0,
        vdw_cutoff_end=12.5,
    )

    assert result.provenance.status is ff.OptimizationAttemptStatus.NUMERICAL_FAILURE
    assert result.provenance.termination_reason == "nonfinite_coordinates"


def test_metal_without_explicit_coordination_bond_fails_before_native_build(
    monkeypatch,
):
    molecule = read_mol("[Zn].N", fmt="smi")
    monkeypatch.setattr(
        auto_workflow,
        "_run_native_fast_build_attempt",
        lambda *args, **kwargs: pytest.fail("invalid complex ran native build"),
    )

    with pytest.raises(ValueError, match="explicit metal-ligand bond"):
        auto_workflow.auto_optimize(molecule)


def test_native_programming_exception_is_not_converted_to_fallback(monkeypatch):
    molecule = _complex_molecule()
    original_coordinates = molecule.coordinates.copy()
    failure = RuntimeError("native attempt failed")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(auto_workflow, "_run_native_fast_build_attempt", fail)
    monkeypatch.setattr(
        auto_workflow,
        "_complexes_build_workflow",
        lambda *args, **kwargs: pytest.fail("exception triggered fallback"),
    )

    with pytest.raises(RuntimeError, match="native attempt failed") as caught:
        auto_workflow.auto_optimize(molecule)

    assert caught.value is failure
    assert np.array_equal(molecule.coordinates, original_coordinates)


def _dative_complex_molecule():
    molecule = read_mol("[Eu]NC", fmt="smi")
    molecule.coordinates = np.asarray(
        ((0.0, 0.0, 0.0), (2.4, 0.0, 0.0), (3.8, 0.0, 0.0)),
        dtype=np.float64,
    )
    molecule.bonds[0]._set_bond_metadata(
        bond_kind=BondKind.DATIVE,
        bond_direction="atom2_to_atom1",
        bond_source="cbond",
        bond_source_metadata={"prediction": {"score": 0.9}},
    )
    molecule.charge = 3
    molecule.properties["source"] = "original"
    return molecule


@pytest.mark.parametrize("add_hydrogens", (False, True))
def test_mapped_dative_uses_full_workflow_from_pristine_source(
    monkeypatch, add_hydrogens,
):
    molecule = _dative_complex_molecule()
    original_atoms = molecule.atoms
    original_bonds = molecule.bonds
    original_bond_facts = tuple(
        (bond.a1idx, bond.a2idx, bond.attr_dict) for bond in original_bonds
    )
    original_coordinates = molecule.coordinates.copy()
    calls = []
    monkeypatch.setattr(
        auto_workflow,
        "_native_build",
        lambda *args, **kwargs: pytest.fail("mapped DATIVE reached native build"),
    )
    monkeypatch.setattr(
        auto_workflow,
        "_run_ordinary_optimization_attempt",
        lambda *args, **kwargs: pytest.fail("unsupported FAST reached optimization"),
    )

    def complex_workflow(current, forcefield, **options):
        calls.append(current)
        assert current is not molecule
        assert len(current.atoms) == len(original_atoms)
        assert np.array_equal(current.coordinates, original_coordinates)
        assert current.charge == 3
        assert current.properties == {"source": "original"}
        assert tuple(
            (bond.a1idx, bond.a2idx, bond.attr_dict) for bond in current.bonds
        ) == original_bond_facts
        assert options["add_hydrogens"] is add_hydrogens
        assert options["convergence_level"] is ff.ConvergenceLevel.BALANCED
        current.coordinates = current.coordinates + 0.1
        return _complex_workflow_report(current)

    monkeypatch.setattr(auto_workflow, "_complexes_build_workflow", complex_workflow)

    result = molecule.optimize(
        add_hydrogens=add_hydrogens,
        convergence_level=ff.ConvergenceLevel.BALANCED,
    )

    assert len(calls) == 1
    assert molecule.atoms == original_atoms
    assert all(current is original for current, original in zip(molecule.bonds, original_bonds))
    assert tuple(
        (bond.a1idx, bond.a2idx, bond.attr_dict) for bond in molecule.bonds
    ) == original_bond_facts
    assert np.array_equal(molecule.coordinates, original_coordinates + 0.1)
    assert molecule.charge == 3
    assert molecule.properties == {"source": "original"}
    routing = result.routing_report
    assert routing.selected_route is ff.OptimizationRoute.COMPLEX_WORKFLOW
    assert tuple(attempt.route for attempt in routing.attempts) == (
        ff.OptimizationRoute.NATIVE_FAST,
        ff.OptimizationRoute.COMPLEX_WORKFLOW,
    )
    assert tuple(attempt.status for attempt in routing.attempts) == (
        ff.OptimizationAttemptStatus.BUILD_FAILED,
        ff.OptimizationAttemptStatus.ACCEPTED,
    )
    preliminary = routing.attempts[0]
    assert preliminary.build_succeeded is False
    assert preliminary.quality_report is None
    assert preliminary.message == (
        "Open Babel native FAST build cannot represent metal-nonmetal "
        "DATIVE bonds losslessly"
    )
    assert len(result.trajectory.preliminary_attempts) == 1
    assert [frame.event for frame in result.trajectory.preliminary_attempts[0].frames] == [
        ff.TrajectoryEvent.INITIAL,
    ]


@pytest.mark.parametrize("error_type", (ff.ForceFieldError, RuntimeError))
def test_mapped_dative_fallback_failure_preserves_caller_atomically(
    monkeypatch, error_type,
):
    molecule = _dative_complex_molecule()
    auto_workflow._capture_working_source(molecule)
    original_record, original_metadata = molecule._capture_factual_record(include_frames=False)
    original_values = astuple(original_record)
    original_atoms = molecule.atoms
    original_bond = molecule.bonds[0]
    failure = error_type("full complex workflow failed")
    monkeypatch.setattr(
        auto_workflow,
        "_native_build",
        lambda *args, **kwargs: pytest.fail("mapped DATIVE reached native build"),
    )

    def fail(current, *args, **kwargs):
        current.coordinates = current.coordinates + 9.0
        current.properties["source"] = "failed candidate"
        current.bonds[0]._set_bond_metadata(bond_kind=BondKind.SINGLE)
        raise failure

    monkeypatch.setattr(auto_workflow, "_complexes_build_workflow", fail)
    monkeypatch.setattr(
        auto_workflow,
        "_commit_working_copy",
        lambda *args, **kwargs: pytest.fail("failed fallback committed"),
    )

    with pytest.raises(error_type, match="full complex workflow failed") as caught:
        molecule.optimize(add_hydrogens=False)

    assert caught.value is failure
    current_record, current_metadata = molecule._capture_factual_record(include_frames=False)
    assert astuple(current_record) == original_values
    assert current_metadata == original_metadata
    assert molecule.atoms == original_atoms
    assert molecule.bonds[0] is original_bond
    assert molecule.properties == {"source": "original"}


@pytest.mark.parametrize("mapped_coordination", (False, True))
def test_unmapped_dative_native_rejection_propagates(monkeypatch, mapped_coordination):
    molecule = read_mol("[Eu]NC", fmt="smi")
    if mapped_coordination:
        molecule.bonds[0]._set_bond_metadata(
            bond_kind=BondKind.DATIVE, bond_direction="atom2_to_atom1",
        )
    molecule.bonds[1]._set_bond_metadata(
        bond_kind=BondKind.DATIVE, bond_direction="atom1_to_atom2",
    )
    original_coordinates = molecule.coordinates.copy()
    monkeypatch.setattr(
        auto_workflow,
        "_complexes_build_workflow",
        lambda *args, **kwargs: pytest.fail("unmapped DATIVE triggered fallback"),
    )

    with pytest.raises(ValueError, match="cannot represent dative bonds losslessly"):
        molecule.optimize(add_hydrogens=False)

    assert np.array_equal(molecule.coordinates, original_coordinates)
    assert molecule.bonds[1].bond_kind is BondKind.DATIVE
    assert molecule.bonds[1].bond_direction == "atom1_to_atom2"


def test_unknown_native_build_error_propagates_without_fallback(monkeypatch):
    molecule = _complex_molecule()
    original_coordinates = molecule.coordinates.copy()
    failure = ValueError("unrecognized native representation failure")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(auto_workflow, "_native_build", fail)
    monkeypatch.setattr(
        auto_workflow,
        "_complexes_build_workflow",
        lambda *args, **kwargs: pytest.fail("unknown native error triggered fallback"),
    )

    with pytest.raises(ValueError, match="unrecognized native representation failure") as caught:
        molecule.optimize(add_hydrogens=False)

    assert caught.value is failure
    assert np.array_equal(molecule.coordinates, original_coordinates)
