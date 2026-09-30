"""Public-workflow behavior fence for the native complex stages."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from hotpot import read_mol
from hotpot.cheminfo.forcefields import acceptance as acceptance_policy
from hotpot.cheminfo.forcefields import workflows
from hotpot.cheminfo.forcefields.contracts import (
    ComplexBuildDiagnostics,
    ForceFieldSetupError,
)
from hotpot.cheminfo.forcefields.native import (
    ComplexOptimizationOptions,
    create_optimization_session,
    optimize_complex as run_native_optimization,
)
from hotpot.cheminfo.forcefields.native_adapters import (
    native_optimization_offsets,
)
from hotpot.cheminfo.forcefields.native_packing import (
    pack_complex_session_input,
)
from hotpot.cheminfo.forcefields.trajectory import (
    ForceFieldTrajectory,
)


def _complex():
    mol = read_mol("[Eu]N", fmt="smi")
    mol.coordinates = np.asarray(
        ((0.0, 0.0, 0.0), (2.4, 0.0, 0.0)),
        dtype=np.float64,
    )
    return mol


def _prepared_complex(mol, **kwargs):
    working_mol = mol.copy()
    trajectory = ForceFieldTrajectory.from_molecule(
        working_mol,
        start=kwargs["trajectory_start"],
    )
    return workflows._PreparedComplex(
        mol=working_mol,
        diagnostics=ComplexBuildDiagnostics(
            attempt_count=1,
            accepted_candidates=1,
            rejected_candidates=(),
            elapsed_seconds=0.25,
            ligand_build_elapsed_seconds=0.25,
        ),
        trajectory=trajectory,
    )


def _unexpected_call(name):
    def unexpected(*args, **kwargs):
        raise AssertionError(f"{name} must not be called")

    return unexpected


def _native_optimization_result(mol):
    session_input = pack_complex_session_input(mol)
    options = ComplexOptimizationOptions(
        epochs=1,
        steps_per_epoch=1,
        untangling_attempt_limit=1,
    )
    offsets = native_optimization_offsets(
        len(mol.atoms),
        2026,
        options=options,
    )
    result = run_native_optimization(
        create_optimization_session(session_input),
        offsets.untangling,
        offsets.optimization,
        options=options,
    )
    return session_input, result


def test_build_complex3d_routes_only_through_native_stage_two(
    monkeypatch,
) -> None:
    mol = _complex()
    calls = {"stage_two": 0}
    real_stage_two = workflows._native_restore_coordination

    def stage_two(*args, **kwargs):
        calls["stage_two"] += 1
        return real_stage_two(*args, **kwargs)

    monkeypatch.setattr(
        workflows,
        "_prepare_complex_working_mol",
        _prepared_complex,
    )
    monkeypatch.setattr(workflows, "_native_restore_coordination", stage_two)
    monkeypatch.setattr(
        workflows,
        "_native_optimize_complex",
        _unexpected_call("native Stage 3"),
    )
    monkeypatch.setattr(
        workflows,
        "_native_run_complex_workflow_from_input",
        _unexpected_call("native workflow coordinator"),
    )

    report = workflows.build_complex3d(
        mol,
        add_hydrogens=False,
        coordination_restoration_attempts=1,
        coordination_relaxation_steps=1,
        seed=2026,
    )

    assert calls == {"stage_two": 1}
    assert report.optimization is None
    assert report.build.coordination_restoration is not None
    assert report.build.ligand_build_elapsed_seconds == 0.25
    assert report.build.elapsed_seconds == (
        report.build.ligand_build_elapsed_seconds
        + report.build.coordination_restoration.elapsed_seconds
    )
    assert report.trajectory is not None


def test_build_complex3d_setup_failure_preserves_stage_one_diagnostics(
    monkeypatch,
) -> None:
    mol = _complex()
    original_coordinates = mol.coordinates.copy()
    original_bonds = tuple(mol.bonds)
    prepared = _prepared_complex(
        mol,
        trajectory_start=workflows.TrajectoryStart.COORDINATION_RESTORATION,
    )
    real_coordination_options = workflows._coordination_options

    def unavailable_coordination_forcefield(**kwargs):
        return replace(
            real_coordination_options(**kwargs),
            forcefield="HOTPOT_MISSING_FORCEFIELD",
        )

    monkeypatch.setattr(
        workflows,
        "_prepare_complex_working_mol",
        lambda *args, **kwargs: prepared,
    )
    monkeypatch.setattr(
        workflows,
        "_coordination_options",
        unavailable_coordination_forcefield,
    )

    with pytest.raises(ForceFieldSetupError) as caught:
        workflows.build_complex3d(
            mol,
            add_hydrogens=False,
            coordination_restoration_attempts=1,
            coordination_relaxation_steps=1,
            seed=2026,
        )

    error = caught.value
    assert error.report is not None
    assert error.report.workflow_stage == "coordination_restoration"
    assert error.diagnostics == prepared.diagnostics
    assert error.trajectory is not None
    np.testing.assert_array_equal(mol.coordinates, original_coordinates)
    assert tuple(mol.bonds) == original_bonds


def test_optimize_complex_uses_native_stage_three_and_checkpoint_acceptance(
    monkeypatch,
) -> None:
    mol = _complex()
    calls = {"stage_three": 0, "checkpoint_acceptance": 0}
    real_stage_three = workflows._native_optimize_complex
    real_checkpoint_acceptance = (
        workflows.evaluate_structure_acceptance_at_native_checkpoint
    )

    def stage_three(*args, **kwargs):
        calls["stage_three"] += 1
        return real_stage_three(*args, **kwargs)

    def checkpoint_acceptance(*args, **kwargs):
        calls["checkpoint_acceptance"] += 1
        return real_checkpoint_acceptance(*args, **kwargs)

    monkeypatch.setattr(workflows, "_native_optimize_complex", stage_three)
    monkeypatch.setattr(
        workflows,
        "evaluate_structure_acceptance_at_native_checkpoint",
        checkpoint_acceptance,
    )
    monkeypatch.setattr(
        workflows,
        "_native_restore_coordination",
        _unexpected_call("native Stage 2"),
    )
    monkeypatch.setattr(
        workflows,
        "_native_run_complex_workflow_from_input",
        _unexpected_call("native workflow coordinator"),
    )
    monkeypatch.setattr(
        acceptance_policy.geo,
        "screen_bond_ring_relations",
        _unexpected_call("Python bond--ring scanner"),
    )

    report = workflows.optimize_complex(
        mol,
        epochs=1,
        steps_per_epoch=1,
        complex_untangling_attempts=1,
        add_hydrogens=False,
        quality_level="off",
        seed=2026,
    )

    assert calls == {"stage_three": 1, "checkpoint_acceptance": 1}
    assert report.quality_report is not None
    assert report.trajectory is not None


def test_complexes_build_calls_coordinator_once_and_ingests_once(
    monkeypatch,
) -> None:
    mol = _complex()
    calls = {"coordinator": 0, "ingest": 0}
    real_coordinator = workflows._native_run_complex_workflow_from_input
    real_ingest = workflows.ingest_native_trajectory

    def coordinator(*args, **kwargs):
        calls["coordinator"] += 1
        return real_coordinator(*args, **kwargs)

    def ingest(*args, **kwargs):
        calls["ingest"] += 1
        return real_ingest(*args, **kwargs)

    monkeypatch.setattr(
        workflows,
        "_prepare_complex_working_mol",
        _prepared_complex,
    )
    monkeypatch.setattr(
        workflows,
        "_native_run_complex_workflow_from_input",
        coordinator,
    )
    monkeypatch.setattr(
        workflows,
        "create_coordination_session",
        _unexpected_call("Python-created coordination session"),
    )
    monkeypatch.setattr(workflows, "ingest_native_trajectory", ingest)
    monkeypatch.setattr(
        workflows,
        "_native_restore_coordination",
        _unexpected_call("independent native Stage 2"),
    )
    monkeypatch.setattr(
        workflows,
        "_native_optimize_complex",
        _unexpected_call("independent native Stage 3"),
    )

    report = workflows.complexes_build(
        mol,
        epochs=1,
        steps_per_epoch=1,
        coordination_restoration_attempts=1,
        coordination_relaxation_steps=1,
        complex_untangling_attempts=1,
        add_hydrogens=False,
        quality_level="off",
        seed=2026,
    )

    assert calls == {"coordinator": 1, "ingest": 1}
    assert report.build.coordination_restoration is not None
    assert report.optimization is not None
    assert report.trajectory is not None


def test_optimize_complex_commits_selected_not_terminal_state(
    monkeypatch,
) -> None:
    mol = _complex()
    original_coordination_bond = next(
        bond for bond in mol.bonds if bond.is_metal_ligand_bond
    )
    _, native_result = _native_optimization_result(mol)
    batch = native_result.trajectory
    selected_index = batch.selected_frame_index
    terminal_index = batch.terminal_frame_index
    assert selected_index is not None
    assert terminal_index is not None

    coordinates = np.array(batch.coordinates, copy=True)
    selected_coordinates = coordinates[selected_index] + 0.25
    terminal_coordinates = coordinates[terminal_index] + 4.0
    coordinates[selected_index] = selected_coordinates
    coordinates[terminal_index] = terminal_coordinates
    terminal_topology = replace(
        batch.topology_revisions[0],
        active_coordination_bond_mask=np.zeros(1, dtype=np.uint8),
    )
    frame_topologies = np.array(batch.frame_topology_revisions, copy=True)
    frame_topologies[terminal_index] = 1
    synthetic_batch = replace(
        batch,
        coordinates=coordinates,
        topology_revisions=(batch.topology_revisions[0], terminal_topology),
        frame_topology_revisions=frame_topologies,
    )
    synthetic_result = replace(
        native_result,
        selected_coordinates=selected_coordinates,
        terminal_coordinates=terminal_coordinates,
        final_active_coordination_mask=np.ones(1, dtype=np.uint8),
        trajectory=synthetic_batch,
    )
    calls = {"ingest": 0}
    real_ingest = workflows.ingest_native_trajectory

    def ingest(*args, **kwargs):
        calls["ingest"] += 1
        return real_ingest(*args, **kwargs)

    monkeypatch.setattr(
        workflows,
        "_native_optimize_complex",
        lambda *args, **kwargs: synthetic_result,
    )
    monkeypatch.setattr(workflows, "ingest_native_trajectory", ingest)

    report = workflows.optimize_complex(
        mol,
        epochs=1,
        steps_per_epoch=1,
        complex_untangling_attempts=1,
        add_hydrogens=False,
        quality_level="off",
        seed=2026,
    )

    assert calls == {"ingest": 1}
    np.testing.assert_array_equal(mol.coordinates, selected_coordinates)
    assert not np.array_equal(mol.coordinates, terminal_coordinates)
    assert original_coordination_bond in mol.bonds
    assert report.trajectory is not None
    trajectory = report.trajectory.main
    assert trajectory.selected_index is not None
    assert trajectory.terminal_index is not None
    np.testing.assert_array_equal(
        trajectory.coordinates(trajectory.selected_index),
        selected_coordinates,
    )
    np.testing.assert_array_equal(
        trajectory.coordinates(trajectory.terminal_index),
        terminal_coordinates,
    )
    selected_bonds = trajectory.topology(trajectory.selected_index).bonds
    terminal_bonds = trajectory.topology(trajectory.terminal_index).bonds
    assert any(bond.atom_indices == (0, 1) for bond in selected_bonds)
    assert all(bond.atom_indices != (0, 1) for bond in terminal_bonds)


def test_native_setup_failure_is_typed_and_does_not_commit(
    monkeypatch,
) -> None:
    mol = _complex()
    coordinates_before = mol.coordinates.copy()
    bonds_before = tuple(mol.bonds)
    native_error = workflows._native_module().ForceFieldSetupError(
        "native setup failed"
    )
    native_error.stage = "setup"
    native_error.workflow_stage = "complex_optimization"
    native_error.completed_coordination = None

    def fail_setup(*args, **kwargs):
        raise native_error

    monkeypatch.setattr(workflows, "_native_optimize_complex", fail_setup)

    with pytest.raises(ForceFieldSetupError, match="native setup failed") as caught:
        workflows.optimize_complex(
            mol,
            epochs=1,
            steps_per_epoch=1,
            complex_untangling_attempts=1,
            add_hydrogens=False,
            quality_level="off",
            seed=2026,
        )

    assert caught.value.report is not None
    assert caught.value.report.stage == "setup"
    assert caught.value.report.workflow_stage == "complex_optimization"
    assert caught.value.diagnostics is None
    np.testing.assert_array_equal(mol.coordinates, coordinates_before)
    assert tuple(mol.bonds) == bonds_before


def test_composed_setup_failure_preserves_completed_coordination_facts(
    monkeypatch,
) -> None:
    mol = _complex()
    original_coordinates = mol.coordinates.copy()
    original_bonds = tuple(mol.bonds)
    prepared = _prepared_complex(
        mol,
        trajectory_start=workflows.TrajectoryStart.COORDINATION_RESTORATION,
    )
    real_optimization_options = workflows._optimization_options

    def unavailable_optimization_forcefield(**kwargs):
        return replace(
            real_optimization_options(**kwargs),
            forcefield="HOTPOT_MISSING_FORCEFIELD",
        )

    monkeypatch.setattr(
        workflows,
        "_prepare_complex_working_mol",
        lambda *args, **kwargs: prepared,
    )
    monkeypatch.setattr(
        workflows,
        "_optimization_options",
        unavailable_optimization_forcefield,
    )

    with pytest.raises(ForceFieldSetupError) as caught:
        workflows.complexes_build(
            mol,
            epochs=1,
            steps_per_epoch=1,
            coordination_restoration_attempts=1,
            coordination_relaxation_steps=1,
            complex_untangling_attempts=1,
            add_hydrogens=False,
            quality_level="off",
            seed=2026,
        )

    error = caught.value
    assert error.report is not None
    assert error.report.workflow_stage == "complex_optimization"
    assert error.diagnostics is not None
    assert error.diagnostics.coordination_restoration is not None
    assert error.trajectory is not None
    assert len(error.trajectory.main) > 0
    np.testing.assert_array_equal(mol.coordinates, original_coordinates)
    assert tuple(mol.bonds) == original_bonds
