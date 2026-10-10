"""Typed contracts shared by molecular pipeline stages and controllers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

if TYPE_CHECKING:
    from hotpot.cheminfo.calculator.electronic_state import ElectronicState
    from hotpot.cheminfo.core import Molecule


__all__ = [
    "Artifact",
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
]


JSONScalar = Union[None, bool, int, float, str]
JSONValue = Union[
    JSONScalar,
    Tuple["JSONValue", ...],
    Mapping[str, "JSONValue"],
]


class StageStatus(str, Enum):
    """Terminal state reported by one successfully executed stage."""

    SUCCEEDED = "succeeded"
    QUALITY_FAILED = "quality-failed"


class PipelineStatus(str, Enum):
    """Terminal state of the ordered pipeline."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


class PipelineDefinitionError(ValueError):
    """Raised when a pipeline or stage definition is invalid."""


class StageExecutionError(RuntimeError):
    """Raised when a prepared stage cannot produce a result."""


@dataclass(frozen=True)
class StageSpec:
    """A stage name and its uninterpreted, ordered command arguments."""

    name: str
    argv: Tuple[str, ...]


@dataclass(frozen=True)
class MolecularRecord:
    """One molecule and the electronic state carried between stages."""

    molecule: "Molecule"
    electronic_state: Optional["ElectronicState"] = None


@dataclass(frozen=True)
class MolecularPayload:
    """An ordered collection of molecular records."""

    records: Tuple[MolecularRecord, ...]


@dataclass(frozen=True)
class StageContext:
    """Filesystem and ordering facts assigned to one stage execution."""

    run_directory: Path
    stage_directory: Path
    stage_index: int
    spec: StageSpec


@dataclass(frozen=True)
class Artifact:
    """A stage-relative artifact with verified content identity."""

    relative_path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class StageResult:
    """The molecular output and evidence returned by a stage."""

    status: StageStatus
    payload: MolecularPayload
    artifacts: Tuple[Artifact, ...] = ()
    report: Mapping[str, JSONValue] = field(default_factory=dict)
    stderr: str = ""


@dataclass(frozen=True)
class PipelineRunResult:
    """The terminal payload and ordered evidence of a pipeline run."""

    status: PipelineStatus
    payload: Optional[MolecularPayload]
    results_directory: Path
    stage_results: Tuple[StageResult, ...]
    failed_stage_index: Optional[int] = None


class PipelineExecutionError(RuntimeError):
    """An execution failure accompanied by the persisted partial run."""

    def __init__(
        self,
        run_result: PipelineRunResult,
        stage_error: StageExecutionError,
    ) -> None:
        super().__init__(str(stage_error))
        self.run_result = run_result
        self.stage_error = stage_error


@runtime_checkable
class PreparedMolecularStage(Protocol):
    """An immutable stage operation prepared from one stage spec."""

    def execute(
        self,
        payload: MolecularPayload,
        context: StageContext,
    ) -> StageResult:
        """Execute this operation for one molecular payload."""


@runtime_checkable
class MolecularStage(Protocol):
    """A registered molecular stage that can prepare independent runs."""

    def prepare(self, spec: StageSpec) -> PreparedMolecularStage:
        """Parse one stage specification into an executable operation."""
