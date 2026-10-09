"""Behavior fences for the automatic ordinary-first optimization cascade."""

from __future__ import annotations

from copy import copy
from dataclasses import replace
from typing import Optional

import numpy as np
import pytest

from hotpot import read_mol
from hotpot.cheminfo.forcefields import attempts
from hotpot.cheminfo.forcefields import auto as auto_workflow
from hotpot.cheminfo.forcefields import ff


def _complex_molecule():
    molecule = read_mol("[Eu]N", fmt="smi")
    molecule.coordinates = np.asarray(
        ((0.0, 0.0, 0.0), (2.4, 0.0, 0.0)),
        dtype=np.float64,
    )
    return molecule


def _quality(passed: bool) -> ff.ForceFieldValidationReport:
    return ff.ForceFieldValidationReport(
        level="standard",
        passed=passed,
        checks=(),
    )


def _trajectory(molecule) -> ff.ForceFieldTrajectory:
    trajectory = ff.ForceFieldTrajectory.from_molecule(
        molecule,
        start=ff.TrajectoryStart.FINAL_OPTIMIZATION,
    )
    frame = trajectory.record_molecule(
        molecule,
        stage=ff.TrajectoryStage.FINAL_OPTIMIZATION,
        event=ff.TrajectoryEvent.TERMINAL,
    )
    trajectory.select(frame.index)
    return trajectory


def _run_report(
    passed: bool,
    *,
    trajectory: Optional[ff.ForceFieldTrajectoryArchive] = None,
) -> ff.ForceFieldRunReport:
    return ff.ForceFieldRunReport(
        requested_forcefield="UFF",
        effective_forcefield="UFF",
        setup_succeeded=True,
        converged=True,
        epochs_completed=1,
        steps_submitted=100,
        initialization_steps=1,
        steps_completed=None,
        final_energy=-1.0,
        best_energy=-1.0,
        energy_unit="kJ/mol",
        rms_gradient=0.1,
        max_gradient=0.2,
        exploded=False,
        quality_report=_quality(passed),
        trajectory=trajectory,
        convergence_level=ff.ConvergenceLevel.FAST,
        termination_reason="converged",
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
        "optimize_complex",
        lambda *args, **kwargs: pytest.fail("organic route used complex optimizer"),
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
    assert len(result.routing_report.attempts) == 1


def test_fast_complex_attempt_uses_uff_and_one_acceptance_gate(monkeypatch):
    molecule = _complex_molecule()
    optimized = _run_report(True)
    optimizer_calls = []
    acceptance_calls = []

    def optimize_working(current, **options):
        optimizer_calls.append((current, options))
        return replace(optimized, quality_report=None)

    def accept(current, **options):
        acceptance_calls.append((current, options))
        return optimized.quality_report

    monkeypatch.setattr(attempts, "_optimize_working_mol", optimize_working)
    monkeypatch.setattr(attempts, "evaluate_structure_acceptance", accept)

    working, _, report = auto_workflow._run_fast_complex_attempt(
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
        trajectory_start=ff.TrajectoryStart.FINAL_OPTIMIZATION,
        increasing_vdw=False,
        vdw_cutoff_start=0.0,
        vdw_cutoff_end=12.5,
    )

    assert working is not molecule
    assert len(optimizer_calls) == 1
    assert optimizer_calls[0][1]["effective_forcefield"] == "UFF"
    assert optimizer_calls[0][1]["convergence_level"] is ff.ConvergenceLevel.FAST
    assert len(acceptance_calls) == 1
    assert acceptance_calls[0][0] is working
    assert report.quality_report is optimized.quality_report


def test_fast_complex_success_commits_once_without_fallback(monkeypatch):
    molecule = _complex_molecule()
    working = copy(molecule)
    working.coordinates = working.coordinates + np.asarray((0.3, 0.0, 0.0))
    preliminary = _trajectory(working)
    report = _run_report(True)
    commits = []
    archive = ff.ForceFieldTrajectoryArchive(preliminary)

    monkeypatch.setattr(
        auto_workflow,
        "_run_fast_complex_attempt",
        lambda *args, **kwargs: (working, preliminary, report),
    )
    monkeypatch.setattr(
        auto_workflow,
        "optimize_complex",
        lambda *args, **kwargs: pytest.fail("accepted FAST result used fallback"),
    )
    monkeypatch.setattr(
        auto_workflow,
        "_finalize_trajectory",
        lambda *args, **kwargs: archive,
    )
    monkeypatch.setattr(
        auto_workflow,
        "_commit_working_copy",
        lambda original, completed: commits.append((original, completed)),
    )

    result = auto_workflow.auto_optimize(molecule)

    assert commits == [(molecule, working)]
    assert result.trajectory is archive
    assert result.routing_report.selected_route is ff.OptimizationRoute.ORDINARY_FAST
    assert tuple(attempt.status for attempt in result.routing_report.attempts) == (
        ff.OptimizationAttemptStatus.ACCEPTED,
    )


def test_quality_rejected_fast_attempt_falls_back_from_original_coordinates(
    monkeypatch,
):
    molecule = _complex_molecule()
    molecule.charge = 3
    molecule.properties["source"] = "original"
    original_coordinates = molecule.coordinates.copy()
    rejected = copy(molecule)
    rejected.coordinates = rejected.coordinates + np.asarray((9.0, 0.0, 0.0))
    preliminary = _trajectory(rejected)
    fallback_trajectory = _trajectory(molecule)
    fallback_archive = ff.ForceFieldTrajectoryArchive(fallback_trajectory)
    fallback_report = _run_report(True, trajectory=fallback_archive)
    fallback_starts = []
    commits = []

    monkeypatch.setattr(
        auto_workflow,
        "_run_fast_complex_attempt",
        lambda *args, **kwargs: (rejected, preliminary, _run_report(False)),
    )

    def complex_fallback(current, forcefield, **options):
        fallback_starts.append(current.coordinates.copy())
        assert current.charge == 3
        assert current.properties == {"source": "original"}
        current.coordinates = current.coordinates + np.asarray((0.1, 0.0, 0.0))
        return fallback_report

    monkeypatch.setattr(auto_workflow, "optimize_complex", complex_fallback)
    monkeypatch.setattr(
        auto_workflow,
        "_commit_working_copy",
        lambda original, completed: commits.append((original, completed)),
    )

    result = auto_workflow.auto_optimize(molecule)

    assert len(fallback_starts) == 1
    assert np.array_equal(fallback_starts[0], original_coordinates)
    assert np.array_equal(molecule.coordinates, original_coordinates)
    assert len(commits) == 1
    assert commits[0][0] is molecule
    assert result.trajectory.main is fallback_trajectory
    assert result.trajectory.preliminary_optimization_attempts == (preliminary,)
    assert result.routing_report.selected_route is ff.OptimizationRoute.COMPLEX
    assert tuple(attempt.status for attempt in result.routing_report.attempts) == (
        ff.OptimizationAttemptStatus.QUALITY_REJECTED,
        ff.OptimizationAttemptStatus.ACCEPTED,
    )


@pytest.mark.parametrize(
    "termination_reason",
    (
        "nonfinite_coordinates",
        "nonfinite_energy",
        "nonfinite_gradients",
        "explosion_detected",
    ),
)
def test_numerical_termination_overrides_a_passing_quality_report(
    termination_reason,
):
    report = replace(
        _run_report(True),
        termination_reason=termination_reason,
    )

    attempt = auto_workflow._attempt_report(
        report,
        ff.OptimizationRoute.ORDINARY_FAST,
    )

    assert attempt.status is ff.OptimizationAttemptStatus.NUMERICAL_FAILURE


@pytest.mark.parametrize(
    "convergence_level",
    (
        ff.ConvergenceLevel.OPENBABEL,
        ff.ConvergenceLevel.BALANCED,
        ff.ConvergenceLevel.STRICT,
    ),
)
def test_explicit_nonfast_level_preserves_direct_complex_route(
    monkeypatch,
    convergence_level,
):
    molecule = _complex_molecule()
    calls = []

    monkeypatch.setattr(
        auto_workflow,
        "_run_fast_complex_attempt",
        lambda *args, **kwargs: pytest.fail("non-FAST route ran preliminary attempt"),
    )

    def complex_optimize(current, forcefield, **options):
        calls.append((current, forcefield, options))
        return _run_report(True)

    monkeypatch.setattr(auto_workflow, "optimize_complex", complex_optimize)

    result = auto_workflow.auto_optimize(
        molecule,
        convergence_level=convergence_level,
    )

    assert len(calls) == 1
    assert calls[0][0] is molecule
    assert calls[0][2]["convergence_level"] is convergence_level
    assert result.routing_report.selected_route is ff.OptimizationRoute.COMPLEX


def test_metal_without_explicit_coordination_bond_fails_before_any_optimizer(
    monkeypatch,
):
    molecule = read_mol("[Zn].N", fmt="smi")
    monkeypatch.setattr(
        auto_workflow,
        "_run_fast_complex_attempt",
        lambda *args, **kwargs: pytest.fail("invalid complex ran preliminary attempt"),
    )
    monkeypatch.setattr(
        auto_workflow,
        "optimize_complex",
        lambda *args, **kwargs: pytest.fail("invalid complex ran complex optimizer"),
    )

    with pytest.raises(ValueError, match="explicit metal-ligand bond"):
        auto_workflow.auto_optimize(molecule)


def test_preliminary_exception_is_not_converted_into_complex_fallback(monkeypatch):
    molecule = _complex_molecule()
    original_coordinates = molecule.coordinates.copy()
    failure = RuntimeError("preliminary optimizer failed")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(auto_workflow, "_run_fast_complex_attempt", fail)
    monkeypatch.setattr(
        auto_workflow,
        "optimize_complex",
        lambda *args, **kwargs: pytest.fail("exception triggered fallback"),
    )

    with pytest.raises(RuntimeError, match="preliminary optimizer failed") as caught:
        auto_workflow.auto_optimize(molecule)

    assert caught.value is failure
    assert np.array_equal(molecule.coordinates, original_coordinates)
