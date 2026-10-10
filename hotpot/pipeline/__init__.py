"""Typed molecular pipeline contracts and lazy stage registration."""

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
    "builtin_stage_names",
    "get_stage",
    "register_stage",
    "stage_import_path",
]
