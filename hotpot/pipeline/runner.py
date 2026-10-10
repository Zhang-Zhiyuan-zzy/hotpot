"""Ordered execution and atomic persistence for molecular pipelines."""

from __future__ import annotations

import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

from .artifacts import (
    _copy_file_atomic,
    _verify_artifacts,
    _write_json_atomic,
    _write_payload,
    _write_text_atomic,
    payload_sha256,
)
from .contracts import (
    JSONValue,
    MolecularPayload,
    PipelineDefinitionError,
    PipelineExecutionError,
    PipelineRunResult,
    PipelineStatus,
    PreparedMolecularStage,
    StageContext,
    StageExecutionError,
    StageResult,
    StageSpec,
    StageStatus,
)
from .registry import get_stage


__all__ = [
    "run_pipeline",
]


# Run and directory facts.


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hotpot_version() -> str:
    from hotpot import version

    return version()


def _stage_label(index: int, name: str) -> str:
    safe_name = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in name
    ).strip("-")
    return f"{index:02d}-{safe_name}"


def _prepare_stages(
    specs: Sequence[StageSpec],
) -> Tuple[PreparedMolecularStage, ...]:
    return tuple(get_stage(spec.name).prepare(spec) for spec in specs)


def _create_run_directory(results_directory: Path) -> None:
    if os.path.lexists(results_directory):
        raise FileExistsError(results_directory)
    results_directory.mkdir(parents=True, exist_ok=False)
    (results_directory / "input").mkdir()
    (results_directory / "stages").mkdir()


def _temporary_stage_directory(
    results_directory: Path,
    index: int,
    name: str,
) -> Path:
    label = _stage_label(index, name)
    stage_directory = (
        results_directory
        / "stages"
        / f".{label}.tmp-{uuid.uuid4().hex}"
    )
    stage_directory.mkdir()
    return stage_directory


# Manifest assembly.


def _input_document(
    initial_payload: MolecularPayload,
    sdf_sha256: str,
) -> Mapping[str, JSONValue]:
    return {
        "path": "input/input.sdf",
        "payload_sha256": payload_sha256(initial_payload),
        "sdf_sha256": sdf_sha256,
    }


def _run_document(
    *,
    status: str,
    started_at: str,
    input_document: Mapping[str, JSONValue],
    stage_documents: Sequence[Mapping[str, JSONValue]],
    failed_stage_index: Optional[int] = None,
    final_sha256: Optional[str] = None,
) -> Mapping[str, JSONValue]:
    updated_at = _utc_timestamp()
    document = {
        "hotpot_version": _hotpot_version(),
        "input": input_document,
        "schema_version": 1,
        "stages": tuple(stage_documents),
        "started_at": started_at,
        "status": status,
        "updated_at": updated_at,
    }
    if status != "running":
        document["finished_at"] = updated_at
    if failed_stage_index is not None:
        document["failed_stage_index"] = failed_stage_index
    if final_sha256 is not None:
        document["final"] = "final.sdf"
        document["final_sha256"] = final_sha256
    return document


def _successful_stage_document(
    *,
    context: StageContext,
    result: StageResult,
    committed_directory: Path,
    input_sha256: str,
    output_sha256: str,
    output_sdf_sha256: str,
    elapsed_seconds: float,
    started_at: str,
    finished_at: str,
    artifacts: Tuple[Mapping[str, JSONValue], ...],
) -> Mapping[str, JSONValue]:
    return {
        "argv": context.spec.argv,
        "artifacts": artifacts,
        "directory": committed_directory.relative_to(
            context.run_directory
        ).as_posix(),
        "elapsed_seconds": elapsed_seconds,
        "finished_at": finished_at,
        "index": context.stage_index,
        "input_sha256": input_sha256,
        "name": context.spec.name,
        "output_sdf_sha256": output_sdf_sha256,
        "output_sha256": output_sha256,
        "report": result.report,
        "started_at": started_at,
        "status": result.status.value,
    }


def _failed_stage_document(
    *,
    context: StageContext,
    failed_directory: Path,
    input_sha256: str,
    elapsed_seconds: float,
    started_at: str,
    finished_at: str,
    error: StageExecutionError,
) -> Mapping[str, JSONValue]:
    return {
        "argv": context.spec.argv,
        "directory": failed_directory.relative_to(
            context.run_directory
        ).as_posix(),
        "elapsed_seconds": elapsed_seconds,
        "error": str(error),
        "finished_at": finished_at,
        "index": context.stage_index,
        "input_sha256": input_sha256,
        "name": context.spec.name,
        "started_at": started_at,
        "status": "execution-failed",
    }


# Stage persistence and failure propagation.


def _persist_stage_result(
    context: StageContext,
    result: StageResult,
    *,
    input_sha256: str,
    elapsed_seconds: float,
    started_at: str,
    finished_at: str,
) -> Mapping[str, JSONValue]:
    output_sdf_sha256 = _write_payload(
        context.stage_directory / "output.sdf",
        result.payload,
    )
    _write_json_atomic(context.stage_directory / "report.json", result.report)
    _write_text_atomic(context.stage_directory / "stderr.log", result.stderr)
    artifact_documents = _verify_artifacts(
        context.stage_directory,
        result.artifacts,
    )

    committed_directory = (
        context.run_directory
        / "stages"
        / _stage_label(context.stage_index, context.spec.name)
    )
    stage_document = _successful_stage_document(
        context=context,
        result=result,
        committed_directory=committed_directory,
        input_sha256=input_sha256,
        output_sha256=payload_sha256(result.payload),
        output_sdf_sha256=output_sdf_sha256,
        elapsed_seconds=elapsed_seconds,
        started_at=started_at,
        finished_at=finished_at,
        artifacts=artifact_documents,
    )
    _write_json_atomic(
        context.stage_directory / "manifest.json",
        stage_document,
    )
    context.stage_directory.rename(committed_directory)
    return stage_document


def _persist_stage_failure(
    context: StageContext,
    *,
    input_sha256: str,
    elapsed_seconds: float,
    started_at: str,
    finished_at: str,
    error: StageExecutionError,
) -> Mapping[str, JSONValue]:
    failed_directory = (
        context.run_directory
        / "stages"
        / f"{_stage_label(context.stage_index, context.spec.name)}.failed"
    )
    stage_document = _failed_stage_document(
        context=context,
        failed_directory=failed_directory,
        input_sha256=input_sha256,
        elapsed_seconds=elapsed_seconds,
        started_at=started_at,
        finished_at=finished_at,
        error=error,
    )
    _write_json_atomic(
        context.stage_directory / "failure.json",
        stage_document,
    )
    context.stage_directory.rename(failed_directory)
    return stage_document


# Public controller.


def run_pipeline(
    specs: Sequence[StageSpec],
    *,
    results_directory: Path,
    initial_payload: Optional[MolecularPayload] = None,
) -> PipelineRunResult:
    """Prepare and execute ordered molecular stages in one exclusive result root."""

    if not specs:
        raise PipelineDefinitionError("A pipeline must contain at least one stage")
    prepared_stages = _prepare_stages(specs)
    source_payload = initial_payload or MolecularPayload(())
    run_directory = Path(results_directory)
    _create_run_directory(run_directory)

    started_at = _utc_timestamp()
    input_sdf_sha256 = _write_payload(
        run_directory / "input" / "input.sdf",
        source_payload,
    )
    input_document = _input_document(source_payload, input_sdf_sha256)
    stage_documents: List[Mapping[str, JSONValue]] = []
    stage_results: List[StageResult] = []
    current_payload = source_payload
    manifest_path = run_directory / "manifest.json"
    _write_json_atomic(
        manifest_path,
        _run_document(
            status="running",
            started_at=started_at,
            input_document=input_document,
            stage_documents=stage_documents,
        ),
    )

    for index, (spec, prepared_stage) in enumerate(
        zip(specs, prepared_stages)
    ):
        temporary_directory = _temporary_stage_directory(
            run_directory,
            index,
            spec.name,
        )
        context = StageContext(
            run_directory=run_directory,
            stage_directory=temporary_directory,
            stage_index=index,
            spec=spec,
        )
        input_sha256 = payload_sha256(current_payload)
        stage_started_at = _utc_timestamp()
        stage_started = time.perf_counter()
        try:
            result = prepared_stage.execute(current_payload, context)
            stage_finished_at = _utc_timestamp()
            stage_document = _persist_stage_result(
                context,
                result,
                input_sha256=input_sha256,
                elapsed_seconds=time.perf_counter() - stage_started,
                started_at=stage_started_at,
                finished_at=stage_finished_at,
            )
        except StageExecutionError as error:
            stage_finished_at = _utc_timestamp()
            stage_document = _persist_stage_failure(
                context,
                input_sha256=input_sha256,
                elapsed_seconds=time.perf_counter() - stage_started,
                started_at=stage_started_at,
                finished_at=stage_finished_at,
                error=error,
            )
            stage_documents.append(stage_document)
            _write_json_atomic(
                manifest_path,
                _run_document(
                    status=PipelineStatus.FAILED.value,
                    started_at=started_at,
                    input_document=input_document,
                    stage_documents=stage_documents,
                    failed_stage_index=index,
                ),
            )
            partial_result = PipelineRunResult(
                status=PipelineStatus.FAILED,
                payload=current_payload,
                results_directory=run_directory,
                stage_results=tuple(stage_results),
                failed_stage_index=index,
            )
            raise PipelineExecutionError(partial_result, error) from error

        stage_documents.append(stage_document)
        stage_results.append(result)
        current_payload = result.payload
        if result.status is StageStatus.QUALITY_FAILED:
            _write_json_atomic(
                manifest_path,
                _run_document(
                    status=PipelineStatus.FAILED.value,
                    started_at=started_at,
                    input_document=input_document,
                    stage_documents=stage_documents,
                    failed_stage_index=index,
                ),
            )
            return PipelineRunResult(
                status=PipelineStatus.FAILED,
                payload=current_payload,
                results_directory=run_directory,
                stage_results=tuple(stage_results),
                failed_stage_index=index,
            )

        _write_json_atomic(
            manifest_path,
            _run_document(
                status="running",
                started_at=started_at,
                input_document=input_document,
                stage_documents=stage_documents,
            ),
        )

    final_source = (
        run_directory
        / "stages"
        / _stage_label(len(specs) - 1, specs[-1].name)
        / "output.sdf"
    )
    final_sha256 = _copy_file_atomic(
        final_source,
        run_directory / "final.sdf",
    )
    _write_json_atomic(
        manifest_path,
        _run_document(
            status=PipelineStatus.SUCCEEDED.value,
            started_at=started_at,
            input_document=input_document,
            stage_documents=stage_documents,
            final_sha256=final_sha256,
        ),
    )
    return PipelineRunResult(
        status=PipelineStatus.SUCCEEDED,
        payload=current_payload,
        results_directory=run_directory,
        stage_results=tuple(stage_results),
    )

