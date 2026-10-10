"""Strict contracts for pipeline artifact safety and lineage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pytest


def _water():
    import hotpot

    mol = hotpot.read_mol("[H]O[H]", "smi")
    mol.coordinates = np.asarray(
        (
            (-0.75, 0.0, 0.0),
            (0.0, 0.5, 0.0),
            (0.75, 0.0, 0.0),
        )
    )
    return mol


def _stage_factory(operation: Callable):
    class PreparedStage:
        def __init__(self, spec) -> None:
            self.spec = spec

        def execute(self, payload, context):
            return operation(self.spec, payload, context)

    class Stage:
        def prepare(self, spec):
            return PreparedStage(spec)

    return Stage()


def _all_path_values(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"directory", "path", "relative_path", "final"}:
                yield item
            yield from _all_path_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_path_values(item)


def test_payload_hash_covers_geometry_topology_and_electronic_state() -> None:
    from hotpot.cheminfo.calculator.electronic_state import (
        ChargeInferenceSource,
        ElectronicState,
        SpinInferenceSource,
    )
    from hotpot.pipeline.artifacts import payload_sha256
    from hotpot.pipeline.contracts import MolecularPayload, MolecularRecord

    state = ElectronicState(
        charge=0,
        unpaired_electrons=0,
        multiplicity=1,
        fragment_charges=(0,),
        charge_source=ChargeInferenceSource.EXPLICIT,
        spin_source=SpinInferenceSource.EXPLICIT,
        assumptions=("test state",),
    )
    mol = _water()
    payload = MolecularPayload((MolecularRecord(mol, state),))
    identical = MolecularPayload((MolecularRecord(mol.copy(), state),))
    moved_mol = mol.copy()
    moved_mol.atoms[0].coordinates = (-0.70, 0.0, 0.0)
    moved = MolecularPayload((MolecularRecord(moved_mol, state),))
    charged_state = ElectronicState(
        charge=1,
        unpaired_electrons=1,
        multiplicity=2,
        fragment_charges=(1,),
        charge_source=ChargeInferenceSource.EXPLICIT,
        spin_source=SpinInferenceSource.EXPLICIT,
        assumptions=("test state",),
    )
    charged = MolecularPayload((MolecularRecord(mol.copy(), charged_state),))

    assert payload_sha256(payload) == payload_sha256(identical)
    assert payload_sha256(payload) != payload_sha256(moved)
    assert payload_sha256(payload) != payload_sha256(charged)


def test_artifact_from_file_records_one_stage_relative_regular_file(
    tmp_path: Path,
) -> None:
    from hotpot.pipeline.artifacts import artifact_from_file

    stage_directory = tmp_path / "stage"
    artifact_path = stage_directory / "native" / "result.json"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b'{"energy": -1.0}\n')

    artifact = artifact_from_file(stage_directory, artifact_path)

    assert artifact.relative_path == Path("native/result.json")
    assert artifact.sha256 == hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    assert artifact.size_bytes == artifact_path.stat().st_size


def test_successful_stage_is_atomically_committed_with_verified_lineage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from hotpot.cheminfo._io import MolReader
    from hotpot.pipeline.artifacts import payload_sha256
    from hotpot.pipeline.contracts import (
        MolecularPayload,
        MolecularRecord,
        PipelineStatus,
        StageResult,
        StageSpec,
        StageStatus,
    )
    from hotpot.pipeline import runner

    temporary_stage_directories: list[Path] = []

    def execute(spec, payload, context):
        temporary_stage_directories.append(context.stage_directory)
        assert context.stage_directory.exists()
        assert context.stage_directory.parent == context.run_directory / "stages"
        return StageResult(StageStatus.SUCCEEDED, payload)

    monkeypatch.setattr(runner, "get_stage", lambda name: _stage_factory(execute))
    initial_payload = MolecularPayload((MolecularRecord(_water()),))
    results_directory = tmp_path / "atomic"

    result = runner.run_pipeline(
        (StageSpec("identity", ()),),
        results_directory=results_directory,
        initial_payload=initial_payload,
    )

    assert result.status is PipelineStatus.SUCCEEDED
    assert len(temporary_stage_directories) == 1
    assert not temporary_stage_directories[0].exists()
    committed = tuple((results_directory / "stages").glob("00-identity"))
    assert len(committed) == 1
    assert not tuple((results_directory / "stages").glob(".*.tmp-*"))
    manifest = json.loads(
        (results_directory / "manifest.json").read_text(encoding="utf-8")
    )
    expected_payload_hash = payload_sha256(initial_payload)
    assert manifest["status"] == "succeeded"
    assert manifest["stages"][0]["input_sha256"] == expected_payload_hash
    assert manifest["stages"][0]["output_sha256"] == expected_payload_hash
    assert manifest["stages"][0]["directory"] == "stages/00-identity"
    assert manifest["final"] == "final.sdf"
    final_path = results_directory / "final.sdf"
    assert final_path.is_file()
    stage_output = results_directory / "stages" / "00-identity" / "output.sdf"
    assert final_path.read_bytes() == stage_output.read_bytes()
    assert "$$$$" in final_path.read_text(encoding="utf-8")
    assert manifest["final_sha256"] == hashlib.sha256(
        final_path.read_bytes()
    ).hexdigest()
    restored_mol = tuple(MolReader(final_path, fmt="sdf"))
    assert len(restored_mol) == 1
    assert tuple(atom.symbol for atom in restored_mol[0].atoms) == ("H", "O", "H")
    assert all(
        not Path(path_value).is_absolute()
        for path_value in _all_path_values(manifest)
        if isinstance(path_value, str)
    )


def test_existing_or_symlinked_results_root_is_rejected_before_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from hotpot.pipeline.contracts import (
        MolecularPayload,
        MolecularRecord,
        StageResult,
        StageSpec,
        StageStatus,
    )
    from hotpot.pipeline import runner

    executions = 0

    def execute(spec, payload, context):
        nonlocal executions
        executions += 1
        return StageResult(StageStatus.SUCCEEDED, payload)

    monkeypatch.setattr(runner, "get_stage", lambda name: _stage_factory(execute))
    payload = MolecularPayload((MolecularRecord(_water()),))
    existing = tmp_path / "existing"
    existing.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(target, target_is_directory=True)

    with pytest.raises(FileExistsError):
        runner.run_pipeline(
            (StageSpec("identity", ()),),
            results_directory=existing,
            initial_payload=payload,
        )
    with pytest.raises(FileExistsError):
        runner.run_pipeline(
            (StageSpec("identity", ()),),
            results_directory=linked,
            initial_payload=payload,
        )

    assert executions == 0


def test_declared_stage_artifact_hash_and_size_are_verified_and_recorded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from hotpot.pipeline.contracts import (
        Artifact,
        MolecularPayload,
        MolecularRecord,
        StageResult,
        StageSpec,
        StageStatus,
    )
    from hotpot.pipeline import runner

    content = b"native evidence\n"
    content_sha256 = hashlib.sha256(content).hexdigest()

    def execute(spec, payload, context):
        artifact_path = context.stage_directory / "artifacts" / "native.log"
        artifact_path.parent.mkdir()
        artifact_path.write_bytes(content)
        return StageResult(
            StageStatus.SUCCEEDED,
            payload,
            artifacts=(
                Artifact(
                    Path("artifacts/native.log"),
                    content_sha256,
                    len(content),
                ),
            ),
        )

    monkeypatch.setattr(runner, "get_stage", lambda name: _stage_factory(execute))
    results_directory = tmp_path / "verified-artifact"
    runner.run_pipeline(
        (StageSpec("evidence", ()),),
        results_directory=results_directory,
        initial_payload=MolecularPayload((MolecularRecord(_water()),)),
    )

    artifact_path = (
        results_directory / "stages" / "00-evidence" / "artifacts" / "native.log"
    )
    assert artifact_path.read_bytes() == content
    manifest = json.loads(
        (results_directory / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["stages"][0]["artifacts"] == [
        {
            "relative_path": "artifacts/native.log",
            "sha256": content_sha256,
            "size_bytes": len(content),
        }
    ]


def test_declared_stage_artifact_hash_mismatch_fails_the_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from hotpot.pipeline.contracts import (
        Artifact,
        MolecularPayload,
        MolecularRecord,
        PipelineExecutionError,
        StageResult,
        StageSpec,
        StageStatus,
    )
    from hotpot.pipeline import runner

    def execute(spec, payload, context):
        artifact_path = context.stage_directory / "artifact.txt"
        artifact_path.write_text("actual\n", encoding="utf-8")
        return StageResult(
            StageStatus.SUCCEEDED,
            payload,
            artifacts=(Artifact(Path("artifact.txt"), "0" * 64, 7),),
        )

    monkeypatch.setattr(runner, "get_stage", lambda name: _stage_factory(execute))

    with pytest.raises(PipelineExecutionError, match="SHA-256"):
        runner.run_pipeline(
            (StageSpec("bad-hash", ()),),
            results_directory=tmp_path / "bad-hash",
            initial_payload=MolecularPayload((MolecularRecord(_water()),)),
        )


@pytest.mark.parametrize(
    "artifact_path",
    (Path("../escape.log"), Path("/tmp/escape.log")),
)
def test_artifact_path_traversal_is_rejected(
    artifact_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from hotpot.pipeline.contracts import (
        Artifact,
        MolecularPayload,
        MolecularRecord,
        PipelineExecutionError,
        StageResult,
        StageSpec,
        StageStatus,
    )
    from hotpot.pipeline import runner

    def execute(spec, payload, context):
        return StageResult(
            StageStatus.SUCCEEDED,
            payload,
            artifacts=(Artifact(artifact_path, "0" * 64, 0),),
        )

    monkeypatch.setattr(runner, "get_stage", lambda name: _stage_factory(execute))

    with pytest.raises(PipelineExecutionError, match="artifact"):
        runner.run_pipeline(
            (StageSpec("unsafe", ()),),
            results_directory=tmp_path / f"traversal-{artifact_path.name}",
            initial_payload=MolecularPayload((MolecularRecord(_water()),)),
        )


def test_symlinked_stage_artifact_cannot_escape_results_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from hotpot.pipeline.contracts import (
        Artifact,
        MolecularPayload,
        MolecularRecord,
        PipelineExecutionError,
        StageResult,
        StageSpec,
        StageStatus,
    )
    from hotpot.pipeline import runner

    outside = tmp_path / "private.txt"
    outside.write_text("outside\n", encoding="utf-8")

    def execute(spec, payload, context):
        artifact_directory = context.stage_directory / "artifacts"
        artifact_directory.mkdir()
        artifact_path = artifact_directory / "escape.txt"
        artifact_path.symlink_to(outside)
        return StageResult(
            StageStatus.SUCCEEDED,
            payload,
            artifacts=(
                Artifact(
                    Path("artifacts/escape.txt"),
                    hashlib.sha256(outside.read_bytes()).hexdigest(),
                    outside.stat().st_size,
                ),
            ),
        )

    monkeypatch.setattr(runner, "get_stage", lambda name: _stage_factory(execute))

    with pytest.raises(PipelineExecutionError, match="symlink"):
        runner.run_pipeline(
            (StageSpec("unsafe", ()),),
            results_directory=tmp_path / "symlink-artifact",
            initial_payload=MolecularPayload((MolecularRecord(_water()),)),
        )


def test_stage_crash_never_leaves_a_completed_looking_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from hotpot.pipeline.contracts import (
        MolecularPayload,
        MolecularRecord,
        PipelineExecutionError,
        StageExecutionError,
        StageSpec,
    )
    from hotpot.pipeline import runner

    temporary_paths: list[Path] = []

    def crash(spec, payload, context):
        temporary_paths.append(context.stage_directory)
        (context.stage_directory / "partial.log").write_text(
            "partial evidence\n",
            encoding="utf-8",
        )
        raise StageExecutionError("crash")

    monkeypatch.setattr(runner, "get_stage", lambda name: _stage_factory(crash))
    results_directory = tmp_path / "crash"

    with pytest.raises(PipelineExecutionError):
        runner.run_pipeline(
            (StageSpec("crash", ()),),
            results_directory=results_directory,
            initial_payload=MolecularPayload((MolecularRecord(_water()),)),
        )

    assert len(temporary_paths) == 1
    assert not temporary_paths[0].exists()
    assert not (results_directory / "stages" / "00-crash").exists()
    failed = tuple((results_directory / "stages").glob("00-crash*.failed"))
    assert len(failed) == 1
    assert (failed[0] / "partial.log").is_file()
    assert not (results_directory / "final.sdf").exists()
