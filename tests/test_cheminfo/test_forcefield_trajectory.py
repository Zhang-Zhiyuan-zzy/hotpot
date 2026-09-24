"""Tests for topology-aware force-field trajectory records."""

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from hotpot import read_mol
from hotpot.cheminfo.core import BondKind
from hotpot.cheminfo.forcefields import utils as forcefield_utils
from hotpot.cheminfo.forcefields.trajectory import (
    CoordinationFrameEvidence,
    ForceFieldTrajectory,
    ForceFieldTrajectoryArchive,
    OptimizationFrameEvidence,
    RingFrameEvidence,
    TrajectoryEvent,
    TrajectoryStage,
    TrajectoryStart,
    _TrajectoryWriter,
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
            converged=False,
            exploded=False,
            finite_coordinates=True,
            finite_energy=True,
            finite_gradients=True,
            rms_gradient_kj_mol_angstrom=0.4,
            energy_change_kj_mol=0.25,
            max_displacement_angstrom=0.10,
        ),
    )
    trajectory.select(third.index)
    return molecule, trajectory, (first, second, third)


def _directory_contents(path: Path) -> dict[str, bytes]:
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in path.rglob("*")
        if item.is_file()
    }


def _publish_temporary_paths(path: Path) -> tuple[Path, ...]:
    return tuple(path.parent.glob(f".{path.name}.*-*"))


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


def test_empty_trajectory_materialization_keeps_existing_conformers():
    molecule = read_mol("CC", "smi")
    original_coordinates = molecule.coordinates.copy()
    trajectory = ForceFieldTrajectory.from_molecule(molecule)

    trajectory.materialize(molecule, keep_all=True)

    assert np.array_equal(molecule.coordinates, original_coordinates)


def test_nonempty_trajectory_requires_controller_selection_before_materializing():
    molecule = read_mol("CC", "smi")
    trajectory = ForceFieldTrajectory.from_molecule(molecule)
    trajectory.record_molecule(
        molecule,
        stage=TrajectoryStage.COORDINATION_RESTORATION,
        event=TrajectoryEvent.INITIAL,
    )

    with pytest.raises(ValueError, match="must be selected"):
        trajectory.materialize(molecule, keep_all=False)


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


def test_nonfinite_energy_is_serialized_as_unknown(tmp_path):
    molecule = read_mol("CC", "smi")
    trajectory = ForceFieldTrajectory.from_molecule(molecule)
    frame = trajectory.record_molecule(
        molecule,
        stage=TrajectoryStage.FINAL_OPTIMIZATION,
        event=TrajectoryEvent.INITIAL,
        energy_kj_mol=float("nan"),
        evidence=OptimizationFrameEvidence(
            converged=False,
            exploded=False,
            finite_coordinates=True,
            finite_energy=False,
            finite_gradients=False,
            rms_gradient_kj_mol_angstrom=float("nan"),
            max_gradient_kj_mol_angstrom=float("inf"),
            energy_change_kj_mol=float("nan"),
            max_displacement_angstrom=float("inf"),
        ),
    )
    trajectory.select(frame.index)

    path = tmp_path / "trajectory"
    trajectory.write(path)
    restored = ForceFieldTrajectory.read(path)

    assert trajectory[0].energy_kj_mol is None
    assert restored[0].energy_kj_mol is None
    restored_evidence = restored[0].evidence
    assert isinstance(restored_evidence, OptimizationFrameEvidence)
    assert restored_evidence.rms_gradient_kj_mol_angstrom is None
    assert restored_evidence.max_gradient_kj_mol_angstrom is None
    assert restored_evidence.energy_change_kj_mol is None
    assert restored_evidence.max_displacement_angstrom is None
    assert restored_evidence.finite_coordinates is True
    assert restored_evidence.finite_energy is False
    assert restored_evidence.finite_gradients is False
    manifest_text = (path / "trajectory.json").read_text(encoding="utf-8")
    assert json.loads(manifest_text)["format_version"] == 2
    assert "NaN" not in manifest_text
    assert "Infinity" not in manifest_text
    json.loads(manifest_text, parse_constant=lambda value: pytest.fail(value))


def test_sdf_omits_nonfinite_coordinate_frames(tmp_path):
    molecule = read_mol("CC", "smi")
    trajectory = ForceFieldTrajectory.from_molecule(molecule)
    frame = trajectory.record(
        np.full((len(molecule.atoms), 3), np.nan),
        (),
        stage=TrajectoryStage.COORDINATION_RESTORATION,
        event=TrajectoryEvent.TERMINAL,
    )
    trajectory.select(frame.index)
    path = tmp_path / "trajectory"

    trajectory.write(path)
    restored = ForceFieldTrajectory.read(path)

    assert (path / "trajectory.sdf").read_text(encoding="utf-8") == ""
    assert json.loads((path / "trajectory.json").read_text(encoding="utf-8"))[
        "sdf_frame_indices"
    ] == []
    assert np.all(np.isnan(restored.coordinates(0)))


def test_rewriting_without_sdf_removes_the_obsolete_export(tmp_path):
    _, trajectory, _ = _coordination_trajectory()
    path = tmp_path / "trajectory"
    trajectory.write(path, include_sdf=True)

    trajectory.write(path, include_sdf=False)

    assert not (path / "trajectory.sdf").exists()


def test_failed_first_write_does_not_publish_a_partial_trajectory(
    tmp_path,
    monkeypatch,
):
    _, trajectory, _ = _coordination_trajectory()
    path = tmp_path / "trajectory"

    def fail_to_write_coordinates(*args, **kwargs):
        raise RuntimeError("coordinate serialization failed")

    monkeypatch.setattr(np, "savez_compressed", fail_to_write_coordinates)

    with pytest.raises(RuntimeError, match="coordinate serialization failed"):
        trajectory.write(path)

    assert not path.exists()
    assert _publish_temporary_paths(path) == ()


def test_archive_rewrite_removes_obsolete_ligand_attempt_directories(tmp_path):
    _, main, _ = _coordination_trajectory()
    branch_molecule = read_mol("CC", "smi")
    branch = ForceFieldTrajectory.from_molecule(
        branch_molecule,
        start=TrajectoryStart.LIGAND_BUILD,
    )
    branch_frame = branch.record_molecule(
        branch_molecule,
        stage=TrajectoryStage.LIGAND_BUILD,
        event=TrajectoryEvent.TERMINAL,
    )
    branch.select(branch_frame.index)
    path = tmp_path / "trajectory_archive"
    ForceFieldTrajectoryArchive(main, (branch, branch)).write(path)
    (path / "obsolete.txt").write_text("old archive", encoding="utf-8")

    ForceFieldTrajectoryArchive(main, (branch,)).write(path)

    assert (path / "ligand_build_attempts" / "0000").is_dir()
    assert not (path / "ligand_build_attempts" / "0001").exists()
    assert not (path / "obsolete.txt").exists()
    assert _publish_temporary_paths(path) == ()


def test_failed_archive_overwrite_preserves_the_complete_existing_archive(
    tmp_path,
    monkeypatch,
):
    _, main, _ = _coordination_trajectory()
    branch_molecule = read_mol("CC", "smi")
    branch = ForceFieldTrajectory.from_molecule(
        branch_molecule,
        start=TrajectoryStart.LIGAND_BUILD,
    )
    branch_frame = branch.record_molecule(
        branch_molecule,
        stage=TrajectoryStage.LIGAND_BUILD,
        event=TrajectoryEvent.TERMINAL,
    )
    branch.select(branch_frame.index)
    path = tmp_path / "trajectory_archive"
    ForceFieldTrajectoryArchive(main, (branch, branch)).write(path)
    original_contents = _directory_contents(path)

    def fail_to_write_coordinates(*args, **kwargs):
        raise RuntimeError("coordinate serialization failed")

    monkeypatch.setattr(np, "savez_compressed", fail_to_write_coordinates)

    with pytest.raises(RuntimeError, match="coordinate serialization failed"):
        ForceFieldTrajectoryArchive(main, (branch,)).write(path)

    assert _directory_contents(path) == original_contents
    assert len(ForceFieldTrajectoryArchive.read(path).ligand_build_attempts) == 2
    assert _publish_temporary_paths(path) == ()


def test_failed_archive_publish_restores_the_existing_archive(
    tmp_path,
    monkeypatch,
):
    _, main, _ = _coordination_trajectory()
    path = tmp_path / "trajectory_archive"
    ForceFieldTrajectoryArchive(main).write(path)
    original_contents = _directory_contents(path)
    original_replace = Path.replace

    def fail_staging_publish(source, target):
        if source.name.startswith(f".{path.name}.staging-") and Path(target) == path:
            raise OSError("publish failed")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_staging_publish)

    with pytest.raises(OSError, match="publish failed"):
        ForceFieldTrajectoryArchive(main).write(path, include_sdf=False)

    assert _directory_contents(path) == original_contents
    assert _publish_temporary_paths(path) == ()


def test_failed_archive_rollback_reports_the_preserved_backup(
    tmp_path,
    monkeypatch,
):
    _, main, _ = _coordination_trajectory()
    path = tmp_path / "trajectory_archive"
    ForceFieldTrajectoryArchive(main).write(path)
    original_contents = _directory_contents(path)
    original_replace = Path.replace

    def fail_publish_and_rollback(source, target):
        source = Path(source)
        target = Path(target)
        if source.name.startswith(f".{path.name}.staging-") and target == path:
            raise OSError("publish failed")
        if source.name.startswith(f".{path.name}.backup-") and target == path:
            raise OSError("rollback failed")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_publish_and_rollback)

    with pytest.raises(RuntimeError, match="publication and rollback both failed") as error:
        ForceFieldTrajectoryArchive(main).write(path, include_sdf=False)

    assert "publication error: OSError('publish failed')" in str(error.value)
    assert "rollback error: OSError('rollback failed')" in str(error.value)
    assert not path.exists()
    backup_paths = tuple(
        item
        for item in _publish_temporary_paths(path)
        if ".backup-" in item.name
    )
    assert len(backup_paths) == 1
    assert _directory_contents(backup_paths[0]) == original_contents


def test_backup_cleanup_failure_does_not_turn_a_successful_publish_into_failure(
    tmp_path,
    monkeypatch,
):
    _, main, _ = _coordination_trajectory()
    path = tmp_path / "trajectory_archive"
    ForceFieldTrajectoryArchive(main).write(path, include_sdf=True)
    original_remove_path = _TrajectoryWriter._remove_path

    def fail_backup_cleanup(temporary_path):
        if ".backup-" in temporary_path.name:
            raise PermissionError("backup cleanup failed")
        original_remove_path(temporary_path)

    monkeypatch.setattr(
        _TrajectoryWriter,
        "_remove_path",
        staticmethod(fail_backup_cleanup),
    )

    with pytest.warns(RuntimeWarning, match="backup cleanup failed"):
        ForceFieldTrajectoryArchive(main).write(path, include_sdf=False)

    assert not (path / "main" / "trajectory.sdf").exists()
    assert ForceFieldTrajectoryArchive.read(path).main.frames == main.frames
    assert any(".backup-" in item.name for item in _publish_temporary_paths(path))


def test_staging_cleanup_failure_does_not_mask_the_serialization_failure(
    tmp_path,
    monkeypatch,
):
    _, trajectory, _ = _coordination_trajectory()
    path = tmp_path / "trajectory"
    original_remove_path = _TrajectoryWriter._remove_path

    def fail_to_write_coordinates(*args, **kwargs):
        raise RuntimeError("coordinate serialization failed")

    def fail_staging_cleanup(temporary_path):
        if ".staging-" in temporary_path.name:
            raise PermissionError("staging cleanup failed")
        original_remove_path(temporary_path)

    monkeypatch.setattr(np, "savez_compressed", fail_to_write_coordinates)
    monkeypatch.setattr(
        _TrajectoryWriter,
        "_remove_path",
        staticmethod(fail_staging_cleanup),
    )

    with pytest.warns(RuntimeWarning, match="staging cleanup failed"):
        with pytest.raises(RuntimeError, match="coordinate serialization failed"):
            trajectory.write(path)

    assert not path.exists()
    assert any(".staging-" in item.name for item in _publish_temporary_paths(path))


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


@pytest.mark.parametrize("save_movie", (False, True))
def test_build_and_optimize_persists_all_frames_before_materializing(
    tmp_path,
    save_movie,
):
    molecule = read_mol("CCO", "smi")
    path = tmp_path / "trajectory_archive"

    report = forcefield_utils.build_and_optimize(
        molecule,
        "MMFF94s",
        epochs=2,
        steps_per_epoch=10,
        seed=7,
        save_movie=save_movie,
        trajectory_path=path,
    )
    restored = ForceFieldTrajectoryArchive.read(path)

    assert report.trajectory is not None
    assert len(report.trajectory.main) >= 2
    assert restored.main.frames == report.trajectory.main.frames
    assert restored.main.selected_index == report.trajectory.main.selected_index
    assert molecule.conformers_number == (
        len(report.trajectory.main) if save_movie else 1
    )
