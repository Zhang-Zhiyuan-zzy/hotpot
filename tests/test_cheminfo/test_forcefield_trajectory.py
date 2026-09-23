"""Tests for topology-aware force-field trajectory records."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from hotpot import read_mol
from hotpot.cheminfo.core import BondKind
from hotpot.cheminfo.forcefields.trajectory import (
    CoordinationFrameEvidence,
    ForceFieldTrajectory,
    ForceFieldTrajectoryArchive,
    OptimizationFrameEvidence,
    RingFrameEvidence,
    TrajectoryEvent,
    TrajectoryStage,
    TrajectoryStart,
)


def _coordination_trajectory():
    molecule = read_mol("[Zn].N", "smi")
    trajectory = ForceFieldTrajectory.from_molecule(molecule)
    first = trajectory.record_molecule(
        molecule,
        stage=TrajectoryStage.COORDINATION_RESTORATION,
        event=TrajectoryEvent.COORDINATION_READY,
        evidence=CoordinationFrameEvidence(None, accepted=True, pending_bond_count=1),
    )
    molecule.add_bond(0, 1, 1.0, bond_kind=BondKind.DATIVE)
    second = trajectory.record_molecule(
        molecule,
        stage=TrajectoryStage.COORDINATION_RESTORATION,
        event=TrajectoryEvent.BOND_ACCEPTED,
        energy_kj_mol=-12.5,
        evidence=CoordinationFrameEvidence((0, 1), accepted=True),
    )
    molecule.coordinates = molecule.coordinates + np.array([0.25, 0.0, 0.0])
    third = trajectory.record_molecule(
        molecule,
        stage=TrajectoryStage.FINAL_OPTIMIZATION,
        event=TrajectoryEvent.EPOCH_COMPLETE,
        energy_kj_mol=-13.0,
        step=1,
        evidence=OptimizationFrameEvidence(
            accepted=True,
            converged=False,
            rms_gradient_kj_mol_angstrom=0.4,
        ),
    )
    trajectory.select(third.index)
    return molecule, trajectory, (first, second, third)


def test_enum_contract_and_default_start():
    molecule = read_mol("CC", "smi")
    trajectory = ForceFieldTrajectory.from_molecule(molecule)

    assert trajectory.start is TrajectoryStart.COORDINATION_RESTORATION
    assert TrajectoryStage.COMPLEX_UNTANGLING.value == "complex_untangling"
    assert TrajectoryEvent.RING_OPENED.value == "ring_opened"


@pytest.mark.parametrize(
    ("start", "expected_stages"),
    (
        (TrajectoryStart.LIGAND_BUILD, tuple(TrajectoryStage)),
        (
            TrajectoryStart.COORDINATION_RESTORATION,
            (
                TrajectoryStage.COORDINATION_RESTORATION,
                TrajectoryStage.COMPLEX_UNTANGLING,
                TrajectoryStage.FINAL_OPTIMIZATION,
            ),
        ),
        (
            TrajectoryStart.COMPLEX_UNTANGLING,
            (
                TrajectoryStage.COMPLEX_UNTANGLING,
                TrajectoryStage.FINAL_OPTIMIZATION,
            ),
        ),
        (
            TrajectoryStart.FINAL_OPTIMIZATION,
            (TrajectoryStage.FINAL_OPTIMIZATION,),
        ),
    ),
)
def test_records_respects_the_configured_start(start, expected_stages):
    molecule = read_mol("CC", "smi")
    trajectory = ForceFieldTrajectory.from_molecule(molecule, start=start)

    assert tuple(stage for stage in TrajectoryStage if trajectory.records(stage)) == (
        expected_stages
    )


def test_coordinate_and_topology_revisions_are_pooled_independently():
    _, trajectory, (first, second, third) = _coordination_trajectory()

    assert trajectory.coordinate_revision_count == 2
    assert trajectory.topology_revision_count == 2
    assert first.coordinate_revision == second.coordinate_revision
    assert first.topology_revision != second.topology_revision
    assert second.coordinate_revision != third.coordinate_revision
    assert second.topology_revision == third.topology_revision
    assert len(trajectory.topology(first.index).bonds) == 0
    assert trajectory.topology(second.index).bonds[0].bond_kind == "dative"


def test_frame_is_immutable_and_returned_coordinates_are_independent():
    _, trajectory, (_, second, _) = _coordination_trajectory()

    with pytest.raises(FrozenInstanceError):
        second.energy_kj_mol = 0.0

    coordinates = trajectory.coordinates(second.index)
    coordinates[0, 0] += 100.0
    assert not np.array_equal(coordinates, trajectory.coordinates(second.index))


@pytest.mark.parametrize("keep_all", (False, True))
def test_materialize_exposes_selected_coordinates(keep_all):
    molecule, trajectory, (_, _, third) = _coordination_trajectory()
    expected = trajectory.coordinates(third.index)
    molecule.coordinates = molecule.coordinates + 10.0

    trajectory.materialize(molecule, keep_all=keep_all)

    assert np.allclose(molecule.coordinates, expected)
    assert molecule.conformers_number == (len(trajectory) if keep_all else 1)


def test_trajectory_archive_round_trip_preserves_frames_and_evidence(tmp_path):
    _, main, _ = _coordination_trajectory()
    branch_molecule = read_mol("CC", "smi")
    branch = ForceFieldTrajectory.from_molecule(
        branch_molecule,
        start=TrajectoryStart.LIGAND_BUILD,
    )
    branch_frame = branch.record_molecule(
        branch_molecule,
        stage=TrajectoryStage.LIGAND_BUILD,
        event=TrajectoryEvent.BUILD_COMPLETE,
        component_index=0,
        attempt=2,
        evidence=RingFrameEvidence(confirmed_piercing_count=1),
    )
    branch.select(branch_frame.index)
    archive = ForceFieldTrajectoryArchive(main, (branch,))

    archive_path = tmp_path / "trajectory_archive"
    archive.write(archive_path)
    restored = ForceFieldTrajectoryArchive.read(archive_path)

    assert restored.main.atoms == main.atoms
    assert restored.main.frames == main.frames
    assert restored.main.topology_revisions == main.topology_revisions
    assert restored.main.selected_index == main.selected_index
    for frame in main.frames:
        assert np.array_equal(
            restored.main.coordinates(frame.index),
            main.coordinates(frame.index),
        )
    assert restored.ligand_build_attempts[0].frames == branch.frames
    assert (archive_path / "main" / "coordinates.npz").is_file()
    assert (archive_path / "main" / "trajectory.json").is_file()
    assert (archive_path / "main" / "trajectory.sdf").is_file()


def test_sdf_uses_the_topology_of_each_frame(tmp_path):
    _, trajectory, _ = _coordination_trajectory()
    sdf_path = tmp_path / "trajectory.sdf"

    trajectory.write_sdf(sdf_path)

    records = sdf_path.read_text(encoding="utf-8").split("$$$$\n")
    first_counts = records[0].splitlines()[3]
    second_counts = records[1].splitlines()[3]
    assert int(first_counts[3:6]) == 0
    assert int(second_counts[3:6]) == 1
    assert '"bond_kind":"dative"' in records[1]
    assert ">  <HOTpot Selected>" in records[2]
    assert "True" in records[2]


def test_record_molecule_rejects_changed_atom_identity():
    molecule = read_mol("CC", "smi")
    trajectory = ForceFieldTrajectory.from_molecule(molecule)
    changed = read_mol("CN", "smi")

    with pytest.raises(ValueError, match="atom identity"):
        trajectory.record_molecule(
            changed,
            stage=TrajectoryStage.FINAL_OPTIMIZATION,
            event=TrajectoryEvent.INITIAL,
        )
