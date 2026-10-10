"""Strict contract fence for the not-yet-implemented pipeline types."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path

import pytest

from hotpot.cheminfo.calculator.electronic_state import (
    ChargeInferenceSource,
    ElectronicState,
    SpinInferenceSource,
)
from hotpot.cheminfo.core import Molecule


def _state(charge: int, unpaired_electrons: int) -> ElectronicState:
    return ElectronicState(
        charge=charge,
        unpaired_electrons=unpaired_electrons,
        multiplicity=unpaired_electrons + 1,
        fragment_charges=(charge,),
        charge_source=ChargeInferenceSource.EXPLICIT,
        spin_source=SpinInferenceSource.EXPLICIT,
        assumptions=(),
    )


def test_pipeline_status_values_are_stable() -> None:
    from hotpot.pipeline.contracts import PipelineStatus, StageStatus

    assert tuple(status.value for status in StageStatus) == (
        "succeeded",
        "quality-failed",
    )
    assert tuple(status.value for status in PipelineStatus) == (
        "succeeded",
        "failed",
    )


def test_stage_spec_is_frozen_and_preserves_argv_order() -> None:
    from hotpot.pipeline.contracts import StageSpec

    spec = StageSpec(
        name="xtb",
        argv=("--method", "gfnff", "--task", "optimize"),
    )

    assert is_dataclass(spec)
    assert spec.name == "xtb"
    assert spec.argv == ("--method", "gfnff", "--task", "optimize")
    with pytest.raises(FrozenInstanceError):
        spec.name = "ff"


def test_molecular_payload_preserves_record_and_state_order() -> None:
    from hotpot.pipeline.contracts import MolecularPayload, MolecularRecord

    first_mol = Molecule()
    second_mol = Molecule()
    first_state = _state(0, 0)
    second_state = _state(1, 1)
    first_record = MolecularRecord(first_mol, first_state)
    second_record = MolecularRecord(second_mol, second_state)
    payload = MolecularPayload((first_record, second_record))

    assert payload.records == (first_record, second_record)
    assert payload.records[0].molecule is first_mol
    assert payload.records[0].electronic_state is first_state
    assert payload.records[1].molecule is second_mol
    assert payload.records[1].electronic_state is second_state
    with pytest.raises(FrozenInstanceError):
        first_record.electronic_state = second_state
    with pytest.raises(FrozenInstanceError):
        payload.records = ()


def test_stage_context_artifact_and_result_are_frozen_typed_facts(
    tmp_path: Path,
) -> None:
    from hotpot.pipeline.contracts import (
        Artifact,
        MolecularPayload,
        MolecularRecord,
        StageContext,
        StageResult,
        StageSpec,
        StageStatus,
    )

    spec = StageSpec("ff", ("--route", "complex"))
    context = StageContext(
        run_directory=tmp_path,
        stage_directory=tmp_path / "stages" / "00-ff",
        stage_index=0,
        spec=spec,
    )
    artifact = Artifact(
        relative_path=Path("output.sdf"),
        sha256="a" * 64,
        size_bytes=17,
    )
    payload = MolecularPayload((MolecularRecord(Molecule(), None),))
    result = StageResult(
        status=StageStatus.SUCCEEDED,
        payload=payload,
        artifacts=(artifact,),
        report={"route": "complex", "accepted": True},
        stderr="",
    )

    assert context.spec is spec
    assert context.stage_index == 0
    assert result.payload is payload
    assert result.artifacts == (artifact,)
    assert result.report["accepted"] is True
    with pytest.raises(FrozenInstanceError):
        context.stage_index = 1
    with pytest.raises(FrozenInstanceError):
        artifact.size_bytes = 0
    with pytest.raises(FrozenInstanceError):
        result.status = StageStatus.QUALITY_FAILED


def test_pipeline_result_represents_success_and_quality_failure(
    tmp_path: Path,
) -> None:
    from hotpot.pipeline.contracts import (
        MolecularPayload,
        PipelineRunResult,
        PipelineStatus,
    )

    payload = MolecularPayload(())
    succeeded = PipelineRunResult(
        status=PipelineStatus.SUCCEEDED,
        payload=payload,
        results_directory=tmp_path,
        stage_results=(),
    )
    failed = PipelineRunResult(
        status=PipelineStatus.FAILED,
        payload=payload,
        results_directory=tmp_path,
        stage_results=(),
        failed_stage_index=2,
    )

    assert succeeded.failed_stage_index is None
    assert succeeded.payload is payload
    assert failed.failed_stage_index == 2
    with pytest.raises(FrozenInstanceError):
        failed.failed_stage_index = None


def test_stage_protocol_separates_preparation_from_execution() -> None:
    from hotpot.pipeline.contracts import (
        Artifact,
        MolecularPayload,
        MolecularStage,
        PreparedMolecularStage,
        StageContext,
        StageResult,
        StageSpec,
        StageStatus,
    )

    assert getattr(MolecularStage, "_is_protocol", False)
    assert getattr(PreparedMolecularStage, "_is_protocol", False)
    assert callable(MolecularStage.prepare)
    assert callable(PreparedMolecularStage.execute)

    class PreparedStage:
        def execute(
            self,
            payload: MolecularPayload,
            context: StageContext,
        ) -> StageResult:
            return StageResult(
                status=StageStatus.SUCCEEDED,
                payload=payload,
                artifacts=(
                    Artifact(Path("output.sdf"), "b" * 64, 23),
                ),
                report={"stage_index": context.stage_index},
                stderr="",
            )

    class Stage:
        def prepare(self, spec: StageSpec) -> PreparedMolecularStage:
            assert spec.name == "xtb"
            return PreparedStage()

    spec = StageSpec("xtb", ("--method", "gfn2"))
    payload = MolecularPayload(())
    context = StageContext(Path("run"), Path("run/stages/00-xtb"), 0, spec)
    result = Stage().prepare(spec).execute(payload, context)

    assert result.status is StageStatus.SUCCEEDED
    assert result.payload is payload
    assert result.report == {"stage_index": 0}


def test_pipeline_execution_error_retains_partial_result_and_stage_error(
    tmp_path: Path,
) -> None:
    from hotpot.pipeline.contracts import (
        PipelineExecutionError,
        PipelineRunResult,
        PipelineStatus,
        StageExecutionError,
    )

    partial_result = PipelineRunResult(
        status=PipelineStatus.FAILED,
        payload=None,
        results_directory=tmp_path,
        stage_results=(),
        failed_stage_index=0,
    )
    stage_error = StageExecutionError("native process failed")
    error = PipelineExecutionError(partial_result, stage_error)

    assert isinstance(error, RuntimeError)
    assert error.run_result is partial_result
    assert error.stage_error is stage_error
    assert "native process failed" in str(error)


def test_definition_and_execution_errors_are_distinct() -> None:
    from hotpot.pipeline.contracts import (
        PipelineDefinitionError,
        StageExecutionError,
    )

    assert issubclass(PipelineDefinitionError, ValueError)
    assert issubclass(StageExecutionError, RuntimeError)
    assert not issubclass(PipelineDefinitionError, StageExecutionError)

