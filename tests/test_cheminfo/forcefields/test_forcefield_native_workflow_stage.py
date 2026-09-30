"""Behavior fence for the thin native Stage-2/Stage-3 coordinator."""

from __future__ import annotations

import numpy as np
import pytest

from hotpot.cheminfo.forcefields.native import (
    ComplexOptimizationOptions,
    CoordinationStageOptions,
    create_coordination_session,
    run_complex_workflow,
    run_complex_workflow_from_input,
    snapshot_structure,
)
from hotpot.cheminfo.forcefields.native_packing import ComplexSessionInput
from hotpot.cheminfo.forcefields.trajectory import (
    TrajectoryStage,
    TrajectoryStart,
)
from hotpot.cheminfo.obWrappers import _ob_native


def _complex() -> ComplexSessionInput:
    return ComplexSessionInput(
        schema_version=1,
        atomic_numbers=np.asarray((63, 7), dtype=np.int32),
        formal_charges=np.zeros(2, dtype=np.int32),
        partial_charges=np.zeros(2, dtype=np.float64),
        coordinates=np.asarray(
            ((0.0, 0.0, 0.0), (2.4, 0.0, 0.0)),
            dtype=np.float64,
        ),
        atom_aromatic=np.zeros(2, dtype=np.uint8),
        ligand_bond_indices=np.empty((0, 2), dtype=np.int32),
        ligand_bond_orders=np.empty(0, dtype=np.float64),
        ligand_bond_kinds=np.empty(0, dtype=np.uint8),
        ligand_bond_aromatic=np.empty(0, dtype=np.uint8),
        metal_indices=np.asarray((0,), dtype=np.int32),
        intended_coordination_bonds=np.asarray(((0, 1),), dtype=np.int32),
        intended_coordination_orders=np.ones(1, dtype=np.float64),
        intended_coordination_kinds=np.full(1, 6, dtype=np.uint8),
        unit_cell=None,
    )


def _offsets(frame_count: int) -> np.ndarray:
    return np.zeros((frame_count, 2, 3), dtype=np.float64)


def test_workflow_preserves_stage_reports_and_merges_their_trajectories() -> None:
    session = create_coordination_session(_complex())

    result = run_complex_workflow(
        session,
        _offsets(0),
        _offsets(1),
        _offsets(0),
        coordination_options=CoordinationStageOptions(
            attempt_limit=1,
            relaxation_steps=1,
        ),
        optimization_options=ComplexOptimizationOptions(
            epochs=1,
            steps_per_epoch=1,
            untangling_attempt_limit=1,
        ),
    )

    coordination_frames = result.coordination.trajectory.frame_count
    optimization_frames = result.optimization.trajectory.frame_count
    assert result.coordination.elapsed_seconds > 0.0
    assert result.optimization.elapsed_seconds > 0.0
    assert result.trajectory.frame_count == (
        coordination_frames + optimization_frames
    )
    assert result.trajectory.start is TrajectoryStart.COORDINATION_RESTORATION
    assert result.trajectory.stages[:coordination_frames] == (
        result.coordination.trajectory.stages
    )
    assert result.trajectory.stages[coordination_frames:] == (
        result.optimization.trajectory.stages
    )
    assert result.trajectory.selected_frame_index == (
        coordination_frames
        + result.optimization.trajectory.selected_frame_index
    )
    assert result.trajectory.terminal_frame_index == (
        coordination_frames
        + result.optimization.trajectory.terminal_frame_index
    )
    np.testing.assert_array_equal(
        result.selected_coordinates,
        result.optimization.selected_coordinates,
    )
    np.testing.assert_array_equal(
        result.terminal_coordinates,
        result.optimization.terminal_coordinates,
    )
    np.testing.assert_array_equal(
        result.final_active_coordination_mask,
        result.optimization.final_active_coordination_mask,
    )
    np.testing.assert_array_equal(
        snapshot_structure(session).coordinates,
        result.selected_coordinates,
    )
    assert result.warning_codes == tuple(
        dict.fromkeys(
            result.coordination.warning_codes
            + result.optimization.warning_codes
        )
    )


def test_input_owned_workflow_creates_and_consumes_its_native_session() -> None:
    result = run_complex_workflow_from_input(
        _complex(),
        _offsets(0),
        _offsets(1),
        _offsets(0),
        coordination_options=CoordinationStageOptions(
            attempt_limit=1,
            relaxation_steps=1,
        ),
        optimization_options=ComplexOptimizationOptions(
            epochs=1,
            steps_per_epoch=1,
            untangling_attempt_limit=1,
        ),
    )

    assert result.coordination.bond_count == 1
    assert result.optimization.final_active_coordination_mask.tolist() == [1]
    assert result.trajectory.frame_count == (
        result.coordination.trajectory.frame_count
        + result.optimization.trajectory.frame_count
    )


def test_workflow_keeps_independent_entries_and_respects_stage_filtering() -> None:
    session = create_coordination_session(_complex())

    result = run_complex_workflow(
        session,
        _offsets(0),
        _offsets(1),
        _offsets(0),
        coordination_options=CoordinationStageOptions(
            attempt_limit=1,
            relaxation_steps=1,
            trajectory_start=TrajectoryStart.FINAL_OPTIMIZATION,
        ),
        optimization_options=ComplexOptimizationOptions(
            epochs=1,
            steps_per_epoch=1,
            untangling_attempt_limit=1,
        ),
    )

    assert result.coordination.trajectory.frame_count == 0
    assert result.trajectory.frame_count == (
        result.optimization.trajectory.frame_count
    )
    assert result.trajectory.start is TrajectoryStart.COMPLEX_UNTANGLING
    assert set(result.trajectory.stages) <= {
        TrajectoryStage.COMPLEX_UNTANGLING,
        TrajectoryStage.FINAL_OPTIMIZATION,
    }
    assert hasattr(_ob_native, "restore_coordination")
    assert hasattr(_ob_native, "optimize_complex")
    assert hasattr(_ob_native, "run_complex_workflow")
    assert hasattr(_ob_native, "run_complex_workflow_from_input")


def test_invalid_stage_three_offsets_do_not_mutate_the_shared_session() -> None:
    session = create_coordination_session(_complex())
    before = snapshot_structure(session)

    with pytest.raises(ValueError, match="optimization perturbation"):
        run_complex_workflow(
            session,
            _offsets(0),
            _offsets(1),
            _offsets(0),
            coordination_options=CoordinationStageOptions(
                attempt_limit=1,
                relaxation_steps=1,
            ),
            optimization_options=ComplexOptimizationOptions(
                epochs=2,
                steps_per_epoch=1,
                untangling_attempt_limit=1,
                perturb_interval=1,
            ),
        )

    after = snapshot_structure(session)
    np.testing.assert_array_equal(after.coordinates, before.coordinates)
    np.testing.assert_array_equal(
        after.active_ligand_bond_mask,
        before.active_ligand_bond_mask,
    )
    np.testing.assert_array_equal(
        after.active_coordination_mask,
        before.active_coordination_mask,
    )
    assert after.coordinate_revision == before.coordinate_revision
    assert after.topology_revision == before.topology_revision
