"""Typed molecular pipeline contracts and lazy stage registration."""

from .artifacts import artifact_from_file, payload_sha256
from .contracts import (
    Artifact,
    JSONScalar,
    JSONValue,
    MolecularPayload,
    MolecularRecord,
    MolecularStage,
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
from .registry import (
    DuplicateStageError,
    UnknownStageError,
    builtin_stage_names,
    get_stage,
    register_stage,
    stage_import_path,
)
from .runner import run_pipeline


__all__ = [
    "Artifact",
    "DuplicateStageError",
    "JSONScalar",
    "JSONValue",
    "MolecularPayload",
    "MolecularRecord",
    "MolecularStage",
    "PipelineDefinitionError",
    "PipelineExecutionError",
    "PipelineRunResult",
    "PipelineStatus",
    "PreparedMolecularStage",
    "StageContext",
    "StageExecutionError",
    "StageResult",
    "StageSpec",
    "StageStatus",
    "UnknownStageError",
    "artifact_from_file",
    "builtin_stage_names",
    "get_stage",
    "register_stage",
    "run_pipeline",
    "stage_import_path",
    "payload_sha256",
]
