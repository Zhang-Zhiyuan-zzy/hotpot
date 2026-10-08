"""Python-boundary behavior fence for native Stage-3 optimization."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from hotpot.cheminfo.forcefields.native import (
    ComplexOptimizationOptions,
    ConvergenceLevel,
    RingScreeningOptions,
    _native_complex_optimization_options,
    create_optimization_session,
    optimize_complex,
    snapshot_structure,
)
from hotpot.cheminfo.forcefields.native_packing import ComplexSessionInput
from hotpot.cheminfo.forcefields.native_reports import (
    NativeRingGraphScope,
    NativeStageStatus,
    _native_ring_checkpoint_report,
)
from hotpot.cheminfo.forcefields.trajectory import (
    TrajectoryEvent,
    TrajectoryStage,
)
from hotpot.cheminfo.geometry import PiercingState, SegmentCycleIndeterminacy
from hotpot.cheminfo.obWrappers import _ob_native


def _assembled_complex() -> ComplexSessionInput:
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


def _options(**changes: object) -> ComplexOptimizationOptions:
    return replace(
        ComplexOptimizationOptions(
            epochs=1,
            steps_per_epoch=1,
            untangling_attempt_limit=1,
        ),
        **changes,
    )


def _offsets(frame_count: int) -> np.ndarray:
    return np.zeros((frame_count, 2, 3), dtype=np.float64)


def test_stage_three_options_forward_ring_and_torsion_policy() -> None:
    options = _options(
        convergence_level=ConvergenceLevel.FAST,
        torsion_singularity_threshold=2.0e-6,
        torsion_repair_angle_radians=2.0e-3,
        ring_screening=RingScreeningOptions(
            maximum_actionable_ring_size=12,
            maximum_relevant_cycle_count=321,
        ),
    )

    native_options = _native_complex_optimization_options(options)

    assert native_options.torsion_singularity_threshold == 2.0e-6
    assert native_options.convergence_level.name == "FAST"
    assert native_options.torsion_repair_angle_radians == 2.0e-3
    assert native_options.ring_screening.maximum_actionable_ring_size == 12
    assert native_options.ring_screening.maximum_relevant_cycle_count == 321
    assert native_options.ring_screening.geometry_absolute_length == 1.0e-8
    assert native_options.ring_screening.surface_maximum_surface_count == 132


def test_native_stage_three_maps_final_checkpoint_and_required_frames() -> None:
    session = create_optimization_session(_assembled_complex())

    result = optimize_complex(
        session,
        _offsets(1),
        _offsets(0),
        options=_options(),
    )

    assert result.status is NativeStageStatus.COMPLETED
    assert result.elapsed_seconds > 0.0
    assert result.final_checkpoint.state is PiercingState.DOES_NOT_PIERCE
    assert result.final_checkpoint.scope is NativeRingGraphScope.FULL_GRAPH
    assert result.final_checkpoint.maximum_actionable_ring_size == 16
    assert result.final_checkpoint.scan_complete
    assert result.final_checkpoint.actionable_findings == ()
    assert result.final_piercing_count == (
        result.final_checkpoint.piercing_pair_count
    )
    assert result.selected_frame_index == result.trajectory.selected_frame_index
    assert result.trajectory.terminal_frame_index is not None
    assert result.trajectory.events[0] is TrajectoryEvent.TOPOLOGY_CHECKPOINT
    assert result.trajectory.events[-1] is TrajectoryEvent.TERMINAL
    assert TrajectoryStage.COMPLEX_UNTANGLING in result.trajectory.stages
    assert TrajectoryStage.FINAL_OPTIMIZATION in result.trajectory.stages
    assert not result.selected_coordinates.flags.writeable
    np.testing.assert_array_equal(
        snapshot_structure(session).active_coordination_mask,
        np.asarray((1,), dtype=np.uint8),
    )


def test_native_checkpoint_maps_typed_actionable_findings() -> None:
    finding = _ob_native.NativeBondRingFinding(
        0,
        [1, 2, 3],
        (4, 5),
        _ob_native.NativePiercingState.UNDETERMINED,
        [_ob_native.NativeSegmentCycleIndeterminacy.NUMERIC_BAND],
        False,
        False,
    )
    native_report = _ob_native.NativeRingCheckpointReport(
        _ob_native.NativePiercingState.UNDETERMINED,
        _ob_native.NativeRingGraphScope.FULL_GRAPH,
        16,
        10000,
        1,
        1,
        0,
        1,
        1,
        0,
        1,
        0,
        0,
        1,
        False,
        [finding],
    )

    report = _native_ring_checkpoint_report(native_report)

    assert report.state is PiercingState.UNDETERMINED
    assert report.scope is NativeRingGraphScope.FULL_GRAPH
    assert report.actionable_findings[0].bond_key == (4, 5)
    assert report.actionable_findings[0].indeterminacy_causes == (
        SegmentCycleIndeterminacy.NUMERIC_BAND,
    )


def test_stage_three_rejects_the_wrong_offset_stream_without_mutation() -> None:
    session = create_optimization_session(_assembled_complex())
    before = snapshot_structure(session)

    with pytest.raises(ValueError, match="untangling perturbation"):
        optimize_complex(
            session,
            _offsets(0),
            _offsets(0),
            options=_options(),
        )

    after = snapshot_structure(session)
    np.testing.assert_array_equal(after.coordinates, before.coordinates)
    np.testing.assert_array_equal(
        after.active_coordination_mask,
        before.active_coordination_mask,
    )


def test_stage_two_and_stage_three_remain_independent_native_entries() -> None:
    assert hasattr(_ob_native, "restore_coordination")
    assert hasattr(_ob_native, "optimize_complex")
