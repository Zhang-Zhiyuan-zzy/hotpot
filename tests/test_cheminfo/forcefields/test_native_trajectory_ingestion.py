"""Tests for ingesting batched native force-field trajectory frames."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

from hotpot import read_mol
from hotpot.cheminfo.forcefields.native_packing import (
    pack_complex_session_input,
    unpack_native_topology_revisions,
)
from hotpot.cheminfo.forcefields.native_reports import (
    NativeTopologyRevision,
    NativeTrajectoryBatch,
)
from hotpot.cheminfo.forcefields.trajectory import (
    ForceFieldTrajectory,
    OptimizationFrameEvidence,
    RingFrameEvidence,
    TrajectoryEvent,
    TrajectoryStage,
    TrajectoryStart,
)


def _complex_and_input():
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
    session_input = replace(
        pack_complex_session_input(mol),
        intended_coordination_kinds=np.asarray((6,), dtype=np.uint8),
    )
    return mol, session_input


def _native_batch(coordinates: np.ndarray) -> NativeTrajectoryBatch:
    revisions = (
        NativeTopologyRevision(
            active_ligand_bond_mask=np.asarray((1, 1), dtype=np.uint8),
            active_coordination_bond_mask=np.asarray((0,), dtype=np.uint8),
        ),
        NativeTopologyRevision(
            active_ligand_bond_mask=np.asarray((0, 1), dtype=np.uint8),
            active_coordination_bond_mask=np.asarray((1,), dtype=np.uint8),
        ),
    )
    terminal_coordinates = coordinates.copy()
    terminal_coordinates[0, 2] = 0.25
    return NativeTrajectoryBatch(
        coordinates=np.stack((coordinates, terminal_coordinates)),
        start=TrajectoryStart.COMPLEX_UNTANGLING,
        stages=(
            TrajectoryStage.COMPLEX_UNTANGLING,
            TrajectoryStage.FINAL_OPTIMIZATION,
        ),
        events=(
            TrajectoryEvent.TOPOLOGY_CHECKPOINT,
            TrajectoryEvent.TERMINAL,
        ),
        component_indices=(None, None),
        attempts=(2, None),
        steps=(None, 4),
        energies_kj_mol=np.asarray((np.nan, -12.5), dtype=np.float64),
        evidence=(
            RingFrameEvidence(
                confirmed_piercing_count=1,
                uncertain_relation_count=0,
                ring_scope="full_graph",
                max_ring_size=16,
                selected_ring_count=2,
                excluded_ring_count=0,
                candidate_pair_count=4,
                aabb_separated_pair_count=2,
                exact_pair_count=2,
                does_not_pierce_pair_count=3,
                scan_complete=True,
            ),
            OptimizationFrameEvidence(
                converged=True,
                exploded=False,
                finite_coordinates=True,
                finite_energy=True,
                finite_gradients=True,
                rms_gradient_kj_mol_angstrom=0.25,
                max_gradient_kj_mol_angstrom=0.5,
                energy_change_kj_mol=0.01,
                max_displacement_angstrom=0.02,
            ),
        ),
        topology_revisions=revisions,
        frame_topology_revisions=np.asarray((0, 1), dtype=np.int32),
        selected_frame_index=0,
        terminal_frame_index=1,
    )


def test_native_topology_masks_rebuild_complete_hotpot_bond_tables():
    _, session_input = _complex_and_input()
    topologies = unpack_native_topology_revisions(
        session_input,
        _native_batch(session_input.coordinates).topology_revisions,
    )

    assert tuple(bond.atom_indices for bond in topologies[0]) == ((1, 2), (2, 3))
    assert tuple(bond.atom_indices for bond in topologies[1]) == ((2, 3), (0, 1))
    assert topologies[1][-1].bond_kind == "dative"
    assert topologies[1][-1].bond_order == 1.0


def test_ingestion_maps_native_indices_after_existing_python_frames():
    mol, session_input = _complex_and_input()
    trajectory = ForceFieldTrajectory.from_molecule(
        mol,
        start=TrajectoryStart.COORDINATION_RESTORATION,
    )
    prefix = trajectory.record_molecule(
        mol,
        stage=TrajectoryStage.COORDINATION_RESTORATION,
        event=TrajectoryEvent.COORDINATION_READY,
    )
    trajectory.select(prefix.index)
    trajectory.set_terminal(prefix.index)

    native_batch = _native_batch(session_input.coordinates)
    native_to_global = trajectory.ingest_native_batch(native_batch, session_input)

    assert native_to_global == (1, 2)
    assert trajectory.selected_index == 1
    assert trajectory.terminal_index == 2
    assert trajectory.selected_frame is trajectory[1]
    assert trajectory.terminal_frame is trajectory[2]
    assert trajectory[1].energy_kj_mol is None
    assert trajectory[1].attempt == 2
    assert trajectory[2].energy_kj_mol == -12.5
    assert trajectory[2].step == 4
    assert trajectory[1].evidence == native_batch.evidence[0]
    assert trajectory[2].evidence == native_batch.evidence[1]
    assert tuple(bond.atom_indices for bond in trajectory.topology(1).bonds) == (
        (1, 2),
        (2, 3),
    )
    assert tuple(bond.atom_indices for bond in trajectory.topology(2).bonds) == (
        (0, 1),
        (2, 3),
    )
    assert np.array_equal(
        trajectory.coordinates(2),
        native_batch.coordinates[1],
    )


def test_selected_and_terminal_native_frames_round_trip_independently(tmp_path):
    mol, session_input = _complex_and_input()
    trajectory = ForceFieldTrajectory.from_molecule(mol)
    trajectory.ingest_native_batch(
        _native_batch(session_input.coordinates),
        session_input,
    )
    output = tmp_path / "native_trajectory"

    trajectory.write(output, include_sdf=False)
    restored = ForceFieldTrajectory.read(output)

    assert restored.selected_index == 0
    assert restored.terminal_index == 1
    assert restored.selected_frame is restored[0]
    assert restored.terminal_frame is restored[1]

    manifest_path = output / "trajectory.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["terminal_index"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    legacy_v4 = ForceFieldTrajectory.read(output)
    assert legacy_v4.selected_index == 0
    assert legacy_v4.terminal_index is None

