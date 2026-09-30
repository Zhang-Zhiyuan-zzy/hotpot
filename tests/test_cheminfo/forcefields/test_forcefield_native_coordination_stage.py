"""Behavior fence for the native Stage-2 coordination controller."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from hotpot.cheminfo.forcefields.native import (
    CoordinationStageOptions,
    MetalPlacementOptions,
    create_coordination_session,
    restore_coordination,
    snapshot_structure,
)
from hotpot.cheminfo.forcefields.native_packing import ComplexSessionInput
from hotpot.cheminfo.forcefields.native_reports import NativeStageStatus
from hotpot.cheminfo.forcefields.trajectory import (
    CoordinationFrameEvidence,
    TrajectoryEvent,
    TrajectoryStart,
)
from hotpot.cheminfo.obWrappers import _ob_native
from hotpot.cheminfo.obWrappers.settings import (
    TORSION_REPAIR_ANGLE_RADIANS,
    TORSION_SINGULARITY_THRESHOLD,
)


def _session_input(
    atomic_numbers: tuple[int, ...],
    coordinates: tuple[tuple[float, float, float], ...],
    *,
    ligand_bonds: tuple[tuple[int, int], ...] = (),
    coordination_bonds: tuple[tuple[int, int], ...],
) -> ComplexSessionInput:
    atom_count = len(atomic_numbers)
    ligand_bond_count = len(ligand_bonds)
    coordination_bond_count = len(coordination_bonds)
    return ComplexSessionInput(
        schema_version=1,
        atomic_numbers=np.asarray(atomic_numbers, dtype=np.int32),
        formal_charges=np.zeros(atom_count, dtype=np.int32),
        partial_charges=np.zeros(atom_count, dtype=np.float64),
        coordinates=np.asarray(coordinates, dtype=np.float64),
        atom_aromatic=np.zeros(atom_count, dtype=np.uint8),
        ligand_bond_indices=np.asarray(
            ligand_bonds,
            dtype=np.int32,
        ).reshape((-1, 2)),
        ligand_bond_orders=np.ones(ligand_bond_count, dtype=np.float64),
        ligand_bond_kinds=np.ones(ligand_bond_count, dtype=np.uint8),
        ligand_bond_aromatic=np.zeros(ligand_bond_count, dtype=np.uint8),
        metal_indices=np.asarray((0,), dtype=np.int32),
        intended_coordination_bonds=np.asarray(
            coordination_bonds,
            dtype=np.int32,
        ).reshape((-1, 2)),
        intended_coordination_orders=np.ones(
            coordination_bond_count,
            dtype=np.float64,
        ),
        intended_coordination_kinds=np.full(
            coordination_bond_count,
            6,
            dtype=np.uint8,
        ),
        unit_cell=None,
    )


def _target_distance(atomic_number: int) -> float:
    return (
        _ob_native.covalent_radius(63).angstrom
        + _ob_native.covalent_radius(atomic_number).angstrom
    )


def _one_attempt_options(**changes: object) -> CoordinationStageOptions:
    return replace(
        CoordinationStageOptions(attempt_limit=1, relaxation_steps=1),
        **changes,
    )


def _offsets(frame_count: int, atom_count: int) -> np.ndarray:
    return np.zeros((frame_count, atom_count, 3), dtype=np.float64)


def test_empty_stage_returns_before_metal_placement() -> None:
    source = _session_input(
        (63, 7, 6),
        ((0.0, 0.0, 0.0), (2.4, 0.0, 0.0), (3.8, 0.0, 0.0)),
        ligand_bonds=((1, 2),),
        coordination_bonds=(),
    )
    session = create_coordination_session(source)

    result = restore_coordination(
        session,
        _offsets(0, 3),
        options=CoordinationStageOptions(),
    )

    assert result.status is NativeStageStatus.COMPLETED
    assert result.bond_count == 0
    assert result.placement_report.metals == []
    assert result.elapsed_seconds > 0.0
    assert result.trajectory.events == (
        TrajectoryEvent.COORDINATION_READY,
        TrajectoryEvent.TERMINAL,
    )
    assert result.trajectory.selected_frame_index == 1
    assert result.trajectory.terminal_frame_index == 1


def test_safe_bond_has_paired_trial_acceptance_and_immediate_relaxation() -> None:
    target = _target_distance(7)
    source = _session_input(
        (63, 7, 6),
        ((-target, 0.0, 0.0), (0.0, 0.0, 0.0), (1.4, 0.0, 0.0)),
        ligand_bonds=((1, 2),),
        coordination_bonds=((0, 1),),
    )
    session = create_coordination_session(source)

    result = restore_coordination(
        session,
        _offsets(0, 3),
        options=_one_attempt_options(),
    )

    assert result.status is NativeStageStatus.COMPLETED
    assert result.bond_count == 1
    assert result.attempts_completed == 0
    np.testing.assert_array_equal(
        result.final_active_coordination_mask,
        np.asarray((1,), dtype=np.uint8),
    )
    events = result.trajectory.events
    assert events.count(TrajectoryEvent.BOND_TRIAL) == 1
    assert events.count(TrajectoryEvent.BOND_ACCEPTED) == 1
    assert events.index(TrajectoryEvent.BOND_TRIAL) < events.index(
        TrajectoryEvent.BOND_ACCEPTED
    ) < events.index(TrajectoryEvent.OPTIMIZED)
    assert TrajectoryEvent.BOND_ROLLBACK not in events
    assert result.trajectory.attempts[events.index(TrajectoryEvent.OPTIMIZED)] == 0
    assert result.trajectory.selected_frame_index == (
        result.trajectory.terminal_frame_index
    )


def test_actual_metal_relocation_has_adjacent_trial_and_result_frames() -> None:
    source = _session_input(
        (63, 6, 7, 7),
        (
            (0.0, 0.0, 0.0),
            (0.0, 0.6607, 0.0),
            (-1.5, 0.0, 0.0),
            (1.5, 0.0, 0.0),
        ),
        coordination_bonds=((0, 2), (0, 3)),
    )

    result = restore_coordination(
        create_coordination_session(source),
        _offsets(0, 4),
        options=_one_attempt_options(),
    )

    assert result.metal_relocation_attempt_count == 1
    assert result.relocated_metal_indices == (0,)
    placement = result.placement_report.metals[0]
    assert placement.moved
    assert placement.candidates_evaluated > 1
    events = result.trajectory.events
    trial = events.index(TrajectoryEvent.METAL_RELOCATION_TRIAL)
    assert events[trial + 1] is TrajectoryEvent.METAL_RELOCATED
    np.testing.assert_array_equal(
        result.trajectory.coordinates[trial][0],
        source.coordinates[0],
    )
    np.testing.assert_array_equal(
        result.trajectory.coordinates[trial + 1][0],
        placement.selected_coordinates,
    )
    assert trial + 1 < events.index(TrajectoryEvent.BOND_TRIAL)


def test_unsorted_input_bonds_are_tried_in_canonical_order() -> None:
    target = _target_distance(7)
    source = _session_input(
        (63, 7, 7),
        (
            (0.0, 0.0, 0.0),
            (-target, 0.0, 0.0),
            (target, 0.0, 0.0),
        ),
        coordination_bonds=((0, 2), (0, 1)),
    )

    result = restore_coordination(
        create_coordination_session(source),
        _offsets(0, 3),
        options=_one_attempt_options(),
    )

    trial_bonds = tuple(
        evidence.bond_atom_indices
        for event, evidence in zip(
            result.trajectory.events,
            result.trajectory.evidence,
        )
        if event is TrajectoryEvent.BOND_TRIAL
        and isinstance(evidence, CoordinationFrameEvidence)
    )
    assert trial_bonds == ((0, 1), (0, 2))
    assert result.attempts_completed == 0
    assert result.trajectory.events.count(TrajectoryEvent.OPTIMIZED) == 2


def test_undetermined_relation_warns_but_is_accepted() -> None:
    target = _target_distance(7)
    source = _session_input(
        (63, 7, 1, 6, 6, 6, 6),
        (
            (2.0, 3.0, 1.0 - target),
            (2.0, 3.0, 1.0),
            (30.0, 30.0, 30.0),
            (0.0, 0.0, 0.0),
            (10.0, 10.0, 0.0),
            (0.0, 10.0, 0.0),
            (10.0, 0.0, 0.0),
        ),
        ligand_bonds=((3, 4), (4, 5), (5, 6), (6, 3)),
        coordination_bonds=((0, 1),),
    )

    result = restore_coordination(
        create_coordination_session(source),
        _offsets(0, 7),
        options=_one_attempt_options(
            placement=MetalPlacementOptions(maximum_candidate_count=1),
        ),
    )

    assert result.undetermined_trial_count == 1
    assert result.rejected_piercing_trial_count == 0
    assert result.forced_bond_keys == ()
    assert "coordination_relation_undetermined" in result.warning_codes
    assert TrajectoryEvent.BOND_ACCEPTED in result.trajectory.events


def test_invalid_explicit_offset_schedule_does_not_mutate_session() -> None:
    target = _target_distance(7)
    source = _session_input(
        (63, 7),
        ((-target, 0.0, 0.0), (0.0, 0.0, 0.0)),
        coordination_bonds=((0, 1),),
    )
    session = create_coordination_session(source)
    before = snapshot_structure(session)

    with pytest.raises(ValueError, match="attempt_limit - 1"):
        restore_coordination(
            session,
            _offsets(0, 2),
            options=CoordinationStageOptions(
                attempt_limit=2,
                relaxation_steps=1,
            ),
        )

    after = snapshot_structure(session)
    np.testing.assert_array_equal(after.coordinates, before.coordinates)
    np.testing.assert_array_equal(
        after.active_coordination_mask,
        before.active_coordination_mask,
    )


def test_piercing_bond_is_forced_at_limit_without_post_force_optimize() -> None:
    target = _target_distance(7)
    source = _session_input(
        (63, 7, 6, 6, 6, 6),
        (
            (0.0, 0.0, 0.5 * target),
            (0.0, 0.0, -0.5 * target),
            (-3.0, -3.0, 0.0),
            (3.0, -3.0, 0.0),
            (3.0, 3.0, 0.0),
            (-3.0, 3.0, 0.0),
        ),
        ligand_bonds=((2, 3), (3, 4), (4, 5), (5, 2)),
        coordination_bonds=((0, 1),),
    )
    session = create_coordination_session(source)

    result = restore_coordination(
        session,
        _offsets(0, 6),
        options=_one_attempt_options(
            placement=MetalPlacementOptions(maximum_candidate_count=1),
        ),
    )

    assert result.status is NativeStageStatus.PARTIAL
    assert result.forced_bond_keys == ((0, 1),)
    assert result.rejected_piercing_trial_count >= 1
    assert "coordination_bonds_forced" in result.warning_codes
    events = result.trajectory.events
    assert events.count(TrajectoryEvent.BOND_TRIAL) == events.count(
        TrajectoryEvent.BOND_REJECTED
    )
    forced = events.index(TrajectoryEvent.BOND_FORCED)
    assert events[forced + 1] is TrajectoryEvent.TERMINAL
    assert TrajectoryEvent.OPTIMIZED not in events[forced + 1 :]
    np.testing.assert_array_equal(
        result.final_active_coordination_mask,
        np.asarray((1,), dtype=np.uint8),
    )


def test_stalled_sequence_relaxes_before_using_explicit_perturbation() -> None:
    target = _target_distance(7)
    source = _session_input(
        (63, 7, 6, 6, 6, 6),
        (
            (0.0, 0.0, 0.5 * target),
            (0.0, 0.0, -0.5 * target),
            (-3.0, -3.0, 0.0),
            (3.0, -3.0, 0.0),
            (3.0, 3.0, 0.0),
            (-3.0, 3.0, 0.0),
        ),
        ligand_bonds=((2, 3), (3, 4), (4, 5), (5, 2)),
        coordination_bonds=((0, 1),),
    )
    session = create_coordination_session(source)

    result = restore_coordination(
        session,
        _offsets(1, 6),
        options=CoordinationStageOptions(
            attempt_limit=2,
            relaxation_steps=1,
            placement=MetalPlacementOptions(maximum_candidate_count=1),
        ),
    )

    events = result.trajectory.events
    first_rejected = events.index(TrajectoryEvent.BOND_REJECTED)
    first_optimized = events.index(TrajectoryEvent.OPTIMIZED)
    perturbed = events.index(TrajectoryEvent.PERTURBED)
    second_optimized = events.index(
        TrajectoryEvent.OPTIMIZED,
        first_optimized + 1,
    )
    assert first_rejected < first_optimized < perturbed < second_optimized
    assert result.attempts_completed == 2
    assert result.forced_bond_keys == ((0, 1),)


def test_later_trajectory_start_suppresses_stage_two_frames() -> None:
    target = _target_distance(7)
    source = _session_input(
        (63, 7),
        ((-target, 0.0, 0.0), (0.0, 0.0, 0.0)),
        coordination_bonds=((0, 1),),
    )
    result = restore_coordination(
        create_coordination_session(source),
        _offsets(0, 2),
        options=_one_attempt_options(
            trajectory_start=TrajectoryStart.COMPLEX_UNTANGLING,
        ),
    )

    assert result.trajectory.frame_count == 0
    assert result.trajectory.selected_frame_index is None
    assert result.trajectory.terminal_frame_index is None
    np.testing.assert_array_equal(
        result.final_active_coordination_mask,
        np.asarray((1,), dtype=np.uint8),
    )


def test_python_defaults_forward_canonical_torsion_settings() -> None:
    options = CoordinationStageOptions()

    assert options.torsion_singularity_threshold == (
        TORSION_SINGULARITY_THRESHOLD
    )
    assert options.torsion_repair_angle_radians == TORSION_REPAIR_ANGLE_RADIANS
