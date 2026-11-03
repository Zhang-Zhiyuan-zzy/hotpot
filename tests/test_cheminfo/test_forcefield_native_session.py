"""Native force-field session and transport-contract tests."""

from __future__ import annotations

from copy import copy
from dataclasses import fields
import pickle
import subprocess
import sys
from importlib import import_module

import numpy as np
import pytest

from hotpot import read_mol
from hotpot.cheminfo.rules.physics.forcefields.contracts import OptimizationStoppingCriteria
from hotpot.cheminfo.adapters.native.forcefields.native import (
    ComplexOptimizationOptions,
    CoordinationStageOptions,
    FrameDetail,
    _native_complex_optimization_options,
    _native_coordination_stage_options,
    create_coordination_session,
    create_optimization_session,
    set_coordination_active_mask,
    set_ligand_bond_active_mask,
    snapshot_structure,
    update_structure_coordinates,
)
from hotpot.cheminfo.adapters.structure.forcefields import pack_complex_session_input
from hotpot.cheminfo.adapters.native.forcefields.reports import (
    NativeStageStatus,
    _complex_optimization_result,
    _complex_workflow_result,
    _coordination_stage_result,
    _native_trajectory_batch,
)
from hotpot.cheminfo.structure.forcefields import (
    CoordinationFrameEvidence,
    TrajectoryEvent,
    TrajectoryStage,
    TrajectoryStart,
)


native = import_module("hotpot.cheminfo.obWrappers._ob_native")


def _complex():
    mol = read_mol("[Eu]NCC", fmt="smi")
    mol.coordinates = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (2.4, 0.0, 0.0),
            (3.8, 0.0, 0.0),
            (5.2, 0.0, 0.0),
        ),
        dtype=np.float64,
    )
    return mol


def _native_trajectory(coordinates: np.ndarray):
    intact = native.NativeTopologyRevision(
        np.asarray((1, 1), dtype=np.uint8),
        np.asarray((0,), dtype=np.uint8),
    )
    ring_open = native.NativeTopologyRevision(
        np.asarray((0, 1), dtype=np.uint8),
        np.asarray((0,), dtype=np.uint8),
    )
    evidence = native.NativeCoordinationFrameEvidence(
        (0, 1),
        False,
        1,
        False,
        1,
        0,
        0,
        0,
        None,
        4,
        [1],
        0.75,
        0.20,
    )
    selected = native.NativeTrajectoryFrame(
        coordinates,
        native.NativeTrajectoryStage.COORDINATION_RESTORATION,
        native.NativeTrajectoryEvent.BOND_TRIAL,
        None,
        0,
        None,
        None,
        evidence,
        1,
    )
    terminal_coordinates = coordinates.copy()
    terminal_coordinates[0, 2] = 0.25
    terminal = native.NativeTrajectoryFrame(
        terminal_coordinates,
        native.NativeTrajectoryStage.COORDINATION_RESTORATION,
        native.NativeTrajectoryEvent.TERMINAL,
        None,
        1,
        None,
        -10.0,
        None,
        0,
    )
    return native.NativeTrajectoryBatch(
        len(coordinates),
        2,
        1,
        native.NativeTrajectoryStart.COORDINATION_RESTORATION,
        [intact, ring_open],
        [selected, terminal],
        0,
        1,
    )


def test_session_factories_keep_one_opaque_native_structure():
    mol = _complex()
    packed = pack_complex_session_input(mol)
    coordination = create_coordination_session(packed)
    optimization = create_optimization_session(packed)

    assert not hasattr(coordination, "obmol")
    with pytest.raises(TypeError):
        copy(coordination)

    inactive = snapshot_structure(coordination)
    active = snapshot_structure(optimization)
    assert inactive.ligand_bond_count == 2
    assert inactive.active_bond_count == 2
    assert inactive.active_ligand_bond_mask.tolist() == [1, 1]
    assert inactive.active_coordination_mask.tolist() == [0]
    assert inactive.component_ids.tolist() == [0, 1, 1, 1]
    assert active.active_bond_count == 3
    assert active.active_coordination_mask.tolist() == [1]

    packed.coordinates[:] = 99.0
    assert not np.all(snapshot_structure(coordination).coordinates == 99.0)


def test_session_coordinate_and_two_topology_masks_round_trip():
    session = create_coordination_session(_complex())
    initial = snapshot_structure(session)
    moved = initial.coordinates.copy()
    moved[0] = (0.5, 0.25, -0.25)

    update_structure_coordinates(session, moved)
    set_coordination_active_mask(session, np.asarray((1,), dtype=np.uint8))
    set_ligand_bond_active_mask(session, np.asarray((0, 1), dtype=np.uint8))
    changed = snapshot_structure(session)

    assert np.array_equal(changed.coordinates, moved)
    assert changed.active_ligand_bond_mask.tolist() == [0, 1]
    assert changed.active_coordination_mask.tolist() == [1]
    assert changed.active_bond_count == 2
    assert changed.coordinate_revision == 1
    assert changed.topology_revision == 2
    assert not changed.coordinates.flags.writeable
    assert not changed.active_ligand_bond_mask.flags.writeable

    update_structure_coordinates(session, moved)
    set_coordination_active_mask(session, np.asarray((1,), dtype=np.uint8))
    set_ligand_bond_active_mask(session, np.asarray((0, 1), dtype=np.uint8))
    unchanged = snapshot_structure(session)
    assert unchanged.coordinate_revision == 1
    assert unchanged.topology_revision == 2


def test_native_stage_options_preserve_trajectory_start_and_stopping():
    coordination = _native_coordination_stage_options(
        CoordinationStageOptions(
            trajectory_start=TrajectoryStart.LIGAND_BUILD,
            frame_detail=FrameDetail.ALL_ATTEMPTS,
        )
    )
    assert (
        coordination.trajectory_start
        == native.NativeTrajectoryStart.LIGAND_BUILD
    )
    assert coordination.frame_detail == native.FrameDetail.ALL_ATTEMPTS

    optimization = _native_complex_optimization_options(
        ComplexOptimizationOptions(
            trajectory_start=TrajectoryStart.FINAL_OPTIMIZATION,
            stopping=OptimizationStoppingCriteria(window=7),
        )
    )
    assert (
        optimization.trajectory_start
        == native.NativeTrajectoryStart.FINAL_OPTIMIZATION
    )
    assert optimization.stopping.window == 7


def test_perturbation_offsets_have_a_typed_native_contract():
    offsets = np.arange(24, dtype=np.float64).reshape(2, 4, 3)
    batch = native.PerturbationOffsetBatch(offsets)

    assert batch.atom_count == 4
    assert batch.frame_count == 2
    assert np.array_equal(batch.offsets, offsets)

    with pytest.raises(ValueError, match="dimensions"):
        native.PerturbationOffsetBatch(np.empty((2, 4), dtype=np.float64))


def test_trajectory_round_trip_preserves_topology_and_stage_evidence():
    coordinates = _complex().coordinates
    converted = _native_trajectory_batch(_native_trajectory(coordinates))

    assert converted.start is TrajectoryStart.COORDINATION_RESTORATION
    assert converted.stages == (
        TrajectoryStage.COORDINATION_RESTORATION,
        TrajectoryStage.COORDINATION_RESTORATION,
    )
    assert converted.events == (
        TrajectoryEvent.BOND_TRIAL,
        TrajectoryEvent.TERMINAL,
    )
    assert converted.selected_frame_index == 0
    assert converted.terminal_frame_index == 1
    assert converted.frame_topology_revisions.tolist() == [1, 0]
    assert len(converted.topology_revisions) == 2
    assert converted.topology_revisions[1].active_ligand_bond_mask.tolist() == [
        0,
        1,
    ]
    assert converted.topology_revisions[0].active_coordination_bond_mask.tolist() == [
        0
    ]
    assert converted.evidence[0] == CoordinationFrameEvidence(
        bond_atom_indices=(0, 1),
        accepted=False,
        pending_bond_count=1,
        piercing_relation_count=1,
        metal_atom_index=0,
        relocation_candidates_evaluated=4,
        safe_donor_atom_indices=(1,),
        minimum_normalized_clearance=0.75,
        coordination_distance_deviation=0.20,
    )
    assert converted.evidence[1] is None


def test_stage_and_composed_results_round_trip_without_science_placeholders():
    coordinates = _complex().coordinates
    terminal_coordinates = coordinates.copy()
    terminal_coordinates[0, 2] = 0.25
    trajectory = _native_trajectory(coordinates)
    coordination = native.CoordinationStageResult(
        native.NativeStageStatus.PARTIAL,
        coordinates,
        terminal_coordinates,
        np.asarray((0,), dtype=np.uint8),
        20,
        2,
        1,
        [0],
        [],
        [],
        1,
        0,
        0,
        ["coordination_partial"],
        trajectory,
    )
    final_checkpoint = native.NativeRingCheckpointReport(
        native.NativePiercingState.DOES_NOT_PIERCE,
        native.NativeRingGraphScope.FULL_GRAPH,
        16,
        10000,
        0,
        0,
        0,
        3,
        0,
        0,
        0,
        0,
        0,
        0,
        True,
        [],
    )
    optimization = native.ComplexOptimizationResult(
        native.NativeStageStatus.COMPLETED,
        coordinates,
        terminal_coordinates,
        np.asarray((1,), dtype=np.uint8),
        30,
        0,
        0,
        0,
        0,
        True,
        0,
        0,
        -10.0,
        -10.5,
        0.2,
        0.3,
        [0.5],
        [0.1],
        [-10.0],
        False,
        True,
        True,
        1,
        100,
        0,
        1,
        "kJ/mol",
        "converged",
        [],
        trajectory,
        final_checkpoint,
    )
    workflow = native.ComplexWorkflowResult(
        coordination,
        optimization,
        coordinates,
        terminal_coordinates,
        np.asarray((0,), dtype=np.uint8),
        ["workflow_warning"],
        trajectory,
    )

    converted_coordination = _coordination_stage_result(coordination)
    converted_optimization = _complex_optimization_result(optimization)
    converted_workflow = _complex_workflow_result(workflow)
    assert converted_coordination.status is NativeStageStatus.PARTIAL
    assert converted_optimization.energy_changes == (0.5,)
    assert converted_optimization.max_displacements == (0.1,)
    assert converted_optimization.epoch_energies == (-10.0,)
    assert converted_workflow.warning_codes == ("workflow_warning",)
    assert hasattr(native, "restore_coordination")
    assert hasattr(native, "optimize_complex")


def test_current_transport_values_cold_pickle_at_their_actual_owners():
    packed = pack_complex_session_input(_complex())
    options = ComplexOptimizationOptions(
        epochs=3, trajectory_start=TrajectoryStart.FINAL_OPTIMIZATION,
    )
    coordination_options = CoordinationStageOptions(
        trajectory_start=TrajectoryStart.COORDINATION_RESTORATION,
        frame_detail=FrameDetail.ALL_ATTEMPTS,
    )
    snapshot = snapshot_structure(create_coordination_session(packed))
    batch = _native_trajectory_batch(_native_trajectory(packed.coordinates))
    values = (packed, options, coordination_options, snapshot, batch)
    array_specs = tuple(
        (index, field.name, array.dtype.str, array.shape, array.tolist())
        for index, value in enumerate(values)
        for field in fields(value)
        if isinstance(array := getattr(value, field.name), np.ndarray)
    )
    script = """
import pickle
import sys
import numpy as np
from hotpot.cheminfo.adapters.native.forcefields import contracts, native, reports
from hotpot.cheminfo.structure import forcefields as structure
values, array_specs = pickle.loads(sys.stdin.buffer.read())
packed, options, coordination_options, snapshot, batch = values
classes = (
    contracts.ComplexSessionInput, native.ComplexOptimizationOptions,
    native.CoordinationStageOptions, native.StructureSnapshot, reports.NativeTrajectoryBatch,
)
assert tuple(type(value) for value in values) == classes
assert tuple(cls.__module__ for cls in classes) == (
    contracts.__name__, native.__name__, native.__name__, native.__name__, reports.__name__,
)
for index, name, dtype, shape, expected in array_specs:
    array = getattr(values[index], name)
    assert array.dtype.str == dtype and array.shape == shape
    np.testing.assert_array_equal(array, np.asarray(expected, dtype=array.dtype))
assert packed.coordinates.shape == snapshot.coordinates.shape == (4, 3)
assert options.epochs == 3
assert options.trajectory_start is structure.TrajectoryStart.FINAL_OPTIMIZATION
assert coordination_options.frame_detail is native.FrameDetail.ALL_ATTEMPTS
assert batch.start is structure.TrajectoryStart.COORDINATION_RESTORATION
assert batch.stages == (structure.TrajectoryStage.COORDINATION_RESTORATION,) * 2
assert batch.events == (structure.TrajectoryEvent.BOND_TRIAL, structure.TrajectoryEvent.TERMINAL)
assert batch.selected_frame_index == 0 and batch.terminal_frame_index == 1
assert type(batch.evidence[0]) is structure.CoordinationFrameEvidence
assert batch.evidence[0].bond_atom_indices == (0, 1) and batch.evidence[1] is None
assert all(type(revision) is reports.NativeTopologyRevision for revision in batch.topology_revisions)
assert batch.topology_revisions[1].active_ligand_bond_mask.tolist() == [0, 1]
assert batch.frame_topology_revisions.tolist() == [1, 0]
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        input=pickle.dumps((values, array_specs)), capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode()


def test_live_packing_and_translation_arrays_are_detached_with_distinct_write_permissions():
    molecule = _complex()
    packed = pack_complex_session_input(molecule)
    original_coordinates = molecule.coordinates.copy()
    for field in fields(packed):
        array = getattr(packed, field.name)
        if isinstance(array, np.ndarray):
            assert array.flags.writeable
    assert not np.shares_memory(packed.coordinates, molecule.coordinates)

    session = create_coordination_session(packed)
    snapshot = snapshot_structure(session)
    raw_batch = _native_trajectory(packed.coordinates)
    batch = _native_trajectory_batch(raw_batch)
    snapshot_coordinates = snapshot.coordinates.copy()
    batch_coordinates = batch.coordinates.copy()

    molecule.coordinates = original_coordinates + 0.125
    np.testing.assert_array_equal(packed.coordinates, original_coordinates)
    packed.coordinates[0, 2] += 2.0
    np.testing.assert_array_equal(molecule.coordinates, original_coordinates + 0.125)
    update_structure_coordinates(session, original_coordinates + 0.25)
    set_ligand_bond_active_mask(session, np.asarray((0, 1), dtype=np.uint8))
    set_coordination_active_mask(session, np.asarray((1,), dtype=np.uint8))
    changed = snapshot_structure(session)

    np.testing.assert_array_equal(snapshot.coordinates, snapshot_coordinates)
    np.testing.assert_array_equal(batch.coordinates, batch_coordinates)
    assert snapshot.active_ligand_bond_mask.tolist() == [1, 1]
    assert snapshot.active_coordination_mask.tolist() == [0]
    assert changed.active_ligand_bond_mask.tolist() == [0, 1]
    assert changed.active_coordination_mask.tolist() == [1]
    assert not np.shares_memory(snapshot.coordinates, changed.coordinates)
    assert not np.shares_memory(batch.coordinates[0], raw_batch.frames[0].coordinates)
    arrays = (
        snapshot.coordinates, snapshot.active_ligand_bond_mask,
        snapshot.active_coordination_mask, snapshot.component_ids,
        batch.coordinates, batch.energies_kj_mol, batch.frame_topology_revisions,
        *(array for revision in batch.topology_revisions for array in (
            revision.active_ligand_bond_mask, revision.active_coordination_bond_mask,
        )),
    )
    for array in arrays:
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = 0
