"""Pure Plan-17 A01--A10 force-field acceptance evaluation.

The benchmark runner is responsible for translating mutable Hotpot reports and
trajectory archives into the immutable evidence records defined here.  This
module deliberately evaluates those records without importing Hotpot or
reading files so that the archived decision can be reproduced cheaply.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field, fields, is_dataclass
from math import isfinite, isinf, isnan
from typing import Literal, Optional, Union, cast


JsonValue = Union[
    None,
    bool,
    int,
    float,
    str,
    list["JsonValue"],
    dict[str, "JsonValue"],
]

GATE_IDS = tuple(f"A{index:02d}" for index in range(1, 11))


def _json_value(value: object) -> JsonValue:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("JSON evidence contains a non-finite float")
        return value
    raise TypeError(f"Unsupported JSON evidence value: {type(value).__name__}")


class _JsonRecord:
    def to_dict(self) -> dict[str, JsonValue]:
        """Return a deeply JSON-safe representation of this record."""
        return cast(dict[str, JsonValue], _json_value(self))


@dataclass(frozen=True)
class NumericEvidence(_JsonRecord):
    """One numeric observation with non-finite values safely normalized."""

    value: Optional[float]
    diagnostic: Optional[str] = None
    finite: bool = dataclass_field(default=False, init=False)

    def __post_init__(self) -> None:
        value = self.value
        if value is None:
            object.__setattr__(self, "finite", False)
            object.__setattr__(
                self,
                "diagnostic",
                self.diagnostic or "unavailable",
            )
            return

        number = float(value)
        if isfinite(number):
            object.__setattr__(self, "value", number)
            object.__setattr__(self, "finite", True)
            return

        if isnan(number):
            diagnostic = "nan"
        elif isinf(number) and number > 0.0:
            diagnostic = "positive_infinity"
        else:
            diagnostic = "negative_infinity"
        object.__setattr__(self, "value", None)
        object.__setattr__(self, "finite", False)
        object.__setattr__(self, "diagnostic", self.diagnostic or diagnostic)

    @classmethod
    def from_value(
        cls,
        value: Optional[float],
        *,
        diagnostic: Optional[str] = None,
    ) -> "NumericEvidence":
        """Capture a report value without retaining NaN or infinity."""
        return cls(value=value, diagnostic=diagnostic)


@dataclass(frozen=True)
class CoordinateProposalEvidence(_JsonRecord):
    """A01 evidence for a route's mapped coordinate proposal."""

    returned: bool
    coordinate_shape: Optional[tuple[int, int]]
    atom_row_count: int
    expected_atom_row_count: int
    mapping_preserved: bool
    coordinates_finite: bool
    status: str

    def __post_init__(self) -> None:
        if self.coordinate_shape is not None:
            object.__setattr__(
                self,
                "coordinate_shape",
                tuple(self.coordinate_shape),
            )


@dataclass(frozen=True)
class OptimizationFrameEvidence(_JsonRecord):
    """Selected or exploratory numerical evidence in archive order."""

    frame_id: str
    sequence: int
    stage: str
    event: str
    role: Literal["selected", "exploratory"]
    coordinates_finite: bool
    energy: NumericEvidence
    gradients_evaluated: bool
    rms_gradient: NumericEvidence
    max_gradient: NumericEvidence
    exploded: bool
    optimization_progressed: bool = False


@dataclass(frozen=True)
class RecoveryEvidence(_JsonRecord):
    """Explicit proof that one standard recovery restored optimization."""

    action_name: str
    attempted: bool
    finite_gradient_frame_id: Optional[str]
    optimization_resumed: bool
    resumed_frame_id: Optional[str]


@dataclass(frozen=True)
class UnavailableGradientEvent(_JsonRecord):
    """One observed inability to obtain a finite optimization direction."""

    event_id: str
    frame_id: str
    stage: str
    recovery: Optional[RecoveryEvidence] = None


@dataclass(frozen=True)
class QualityEvidence(_JsonRecord):
    """The existing terminal Hotpot structure-acceptance report."""

    level: str
    passed: bool
    check_names: tuple[str, ...]
    failure_names: tuple[str, ...] = ()
    warning_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "check_names", tuple(self.check_names))
        object.__setattr__(self, "failure_names", tuple(self.failure_names))
        object.__setattr__(self, "warning_names", tuple(self.warning_names))


@dataclass(frozen=True)
class SemanticEvidence(_JsonRecord):
    """A09 coordinate-free graph comparison result."""

    equal: bool
    source_digest: str
    result_digest: str
    differences: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "differences", tuple(self.differences))


@dataclass(frozen=True)
class StageRecord(_JsonRecord):
    """One explicit route-stage outcome for A10."""

    name: str
    status: str
    recovery_actions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "recovery_actions", tuple(self.recovery_actions))


@dataclass(frozen=True)
class StageTraceEvidence(_JsonRecord):
    """Expected and observed stage summaries, including fallback evidence."""

    expected_stages: tuple[str, ...]
    records: tuple[StageRecord, ...]
    provider_fallback_used: bool
    fallback_provider: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "expected_stages", tuple(self.expected_stages))
        object.__setattr__(self, "records", tuple(self.records))


@dataclass(frozen=True)
class AcceptanceEvidence(_JsonRecord):
    """Complete immutable input to the Plan-17 acceptance decision."""

    case_id: str
    route: str
    proposal: CoordinateProposalEvidence
    setup_succeeded: bool
    setup_exception: Optional[str]
    frames: tuple[OptimizationFrameEvidence, ...]
    committed_coordinates_finite: bool
    unavailable_gradient_events: tuple[UnavailableGradientEvent, ...]
    established_recovery_actions: tuple[str, ...]
    quality: QualityEvidence
    semantics: SemanticEvidence
    stage_trace: StageTraceEvidence
    terminal_reason: str
    converged: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "frames", tuple(self.frames))
        object.__setattr__(
            self,
            "unavailable_gradient_events",
            tuple(self.unavailable_gradient_events),
        )
        object.__setattr__(
            self,
            "established_recovery_actions",
            tuple(self.established_recovery_actions),
        )


@dataclass(frozen=True)
class GateDecision(_JsonRecord):
    """One A01--A10 decision with stable diagnostic codes."""

    gate_id: str
    passed: bool
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))


@dataclass(frozen=True)
class AcceptanceResult(_JsonRecord):
    """Serializable case result returned by :func:`evaluate_acceptance`."""

    case_id: str
    route: str
    passed: bool
    gates: tuple[GateDecision, ...]
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "gates", tuple(self.gates))
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    def decision(self, gate_id: str) -> GateDecision:
        """Return one named gate decision."""
        return next(gate for gate in self.gates if gate.gate_id == gate_id)


def _decision(gate_id: str, diagnostics: list[str]) -> GateDecision:
    return GateDecision(
        gate_id=gate_id,
        passed=not diagnostics,
        diagnostics=tuple(diagnostics),
    )


def _selected_frame(
    frames: tuple[OptimizationFrameEvidence, ...],
) -> tuple[Optional[OptimizationFrameEvidence], list[str]]:
    selected = tuple(frame for frame in frames if frame.role == "selected")
    if len(selected) != 1:
        return None, [f"selected_frame_count:{len(selected)}"]
    return selected[0], []


def _evaluate_a06(evidence: AcceptanceEvidence) -> GateDecision:
    diagnostics: list[str] = []
    frames_by_id = {frame.frame_id: frame for frame in evidence.frames}
    if len(frames_by_id) != len(evidence.frames):
        diagnostics.append("duplicate_frame_id")

    unavailable_frames = {
        frame.frame_id: frame
        for frame in evidence.frames
        if frame.gradients_evaluated
        and not (frame.rms_gradient.finite and frame.max_gradient.finite)
    }
    events_by_frame: dict[str, list[UnavailableGradientEvent]] = {}
    event_ids: set[str] = set()
    for event in evidence.unavailable_gradient_events:
        if event.event_id in event_ids:
            diagnostics.append(f"duplicate_gradient_event_id:{event.event_id}")
        event_ids.add(event.event_id)
        events_by_frame.setdefault(event.frame_id, []).append(event)

    for frame_id, frame in unavailable_frames.items():
        events = events_by_frame.get(frame_id, [])
        if len(events) != 1:
            diagnostics.append(
                f"unavailable_gradient_event_count:{frame_id}:{len(events)}"
            )
            continue
        event = events[0]
        if event.stage != frame.stage:
            diagnostics.append(f"gradient_event_stage_mismatch:{event.event_id}")
        recovery = event.recovery
        if recovery is None:
            diagnostics.append(f"recovery_missing:{event.event_id}")
            continue
        if not recovery.action_name:
            diagnostics.append(f"recovery_name_missing:{event.event_id}")
        elif recovery.action_name not in evidence.established_recovery_actions:
            diagnostics.append(
                f"recovery_not_established:{event.event_id}:{recovery.action_name}"
            )
        if not recovery.attempted:
            diagnostics.append(f"recovery_not_attempted:{event.event_id}")

        finite_frame = frames_by_id.get(recovery.finite_gradient_frame_id or "")
        if finite_frame is None:
            diagnostics.append(f"finite_gradient_proof_missing:{event.event_id}")
        elif finite_frame.sequence <= frame.sequence:
            diagnostics.append(f"finite_gradient_not_later:{event.event_id}")
        elif not finite_frame.gradients_evaluated or not (
            finite_frame.rms_gradient.finite and finite_frame.max_gradient.finite
        ):
            diagnostics.append(f"finite_gradient_not_restored:{event.event_id}")

        resumed_frame = frames_by_id.get(recovery.resumed_frame_id or "")
        if not recovery.optimization_resumed:
            diagnostics.append(f"optimization_not_resumed:{event.event_id}")
        elif resumed_frame is None:
            diagnostics.append(f"resume_proof_missing:{event.event_id}")
        elif resumed_frame.sequence <= frame.sequence:
            diagnostics.append(f"resume_not_later:{event.event_id}")
        elif not resumed_frame.optimization_progressed:
            diagnostics.append(f"resume_progress_missing:{event.event_id}")

    for frame_id, events in events_by_frame.items():
        if frame_id not in unavailable_frames:
            diagnostics.extend(
                f"unmatched_unavailable_gradient_event:{event.event_id}"
                for event in events
            )

    return _decision("A06", diagnostics)


def _evaluate_a10(evidence: AcceptanceEvidence) -> GateDecision:
    trace = evidence.stage_trace
    diagnostics: list[str] = []
    observed_stages = tuple(record.name for record in trace.records)
    if not trace.expected_stages:
        diagnostics.append("expected_stages_missing")
    elif observed_stages != trace.expected_stages:
        diagnostics.append("stage_sequence_mismatch")
    if any(not record.status for record in trace.records):
        diagnostics.append("stage_status_missing")
    if trace.provider_fallback_used or trace.fallback_provider is not None:
        diagnostics.append("provider_fallback_used")

    records_by_stage = {record.name: record for record in trace.records}
    for event in evidence.unavailable_gradient_events:
        if event.recovery is None or not event.recovery.action_name:
            continue
        stage_record = records_by_stage.get(event.stage)
        if (
            stage_record is None
            or event.recovery.action_name not in stage_record.recovery_actions
        ):
            diagnostics.append(f"recovery_not_in_stage_trace:{event.event_id}")
    return _decision("A10", diagnostics)


def evaluate_acceptance(evidence: AcceptanceEvidence) -> AcceptanceResult:
    """Evaluate one O0/F1/F2/C1/C2 case against hard gates A01--A10."""
    proposal = evidence.proposal
    a01_diagnostics: list[str] = []
    if not proposal.returned:
        a01_diagnostics.append("proposal_not_returned")
    if proposal.status != "returned":
        a01_diagnostics.append(f"proposal_status:{proposal.status}")
    if proposal.coordinate_shape != (proposal.expected_atom_row_count, 3):
        a01_diagnostics.append("proposal_shape_mismatch")
    if proposal.atom_row_count != proposal.expected_atom_row_count:
        a01_diagnostics.append("proposal_atom_row_count_mismatch")
    if not proposal.mapping_preserved:
        a01_diagnostics.append("proposal_mapping_changed")
    if not proposal.coordinates_finite:
        a01_diagnostics.append("proposal_coordinates_nonfinite")

    a02_diagnostics: list[str] = []
    if not evidence.setup_succeeded:
        a02_diagnostics.append("forcefield_setup_failed")
    if evidence.setup_exception is not None:
        a02_diagnostics.append("forcefield_setup_exception")

    selected, selected_diagnostics = _selected_frame(evidence.frames)
    a03_diagnostics = list(selected_diagnostics)
    if selected is not None and not selected.coordinates_finite:
        a03_diagnostics.append("selected_coordinates_nonfinite")
    if not evidence.committed_coordinates_finite:
        a03_diagnostics.append("committed_coordinates_nonfinite")

    a04_diagnostics = list(selected_diagnostics)
    if selected is not None and not selected.energy.finite:
        a04_diagnostics.append(
            f"selected_best_energy_nonfinite:{selected.energy.diagnostic}"
        )

    a05_diagnostics = list(selected_diagnostics)
    if selected is not None:
        if not selected.gradients_evaluated:
            a05_diagnostics.append("selected_gradients_not_evaluated")
        if not selected.rms_gradient.finite:
            a05_diagnostics.append(
                f"selected_rms_gradient_nonfinite:{selected.rms_gradient.diagnostic}"
            )
        if not selected.max_gradient.finite:
            a05_diagnostics.append(
                f"selected_max_gradient_nonfinite:{selected.max_gradient.diagnostic}"
            )

    a07_diagnostics = list(selected_diagnostics)
    if selected is not None and selected.exploded:
        a07_diagnostics.append("selected_frame_exploded")

    quality = evidence.quality
    a08_diagnostics: list[str] = []
    if quality.level != "standard":
        a08_diagnostics.append(f"quality_level:{quality.level}")
    if not quality.check_names:
        a08_diagnostics.append("quality_checks_missing")
    if not quality.passed or quality.failure_names:
        a08_diagnostics.append("standard_quality_failed")

    semantics = evidence.semantics
    a09_diagnostics: list[str] = []
    if not semantics.source_digest or not semantics.result_digest:
        a09_diagnostics.append("semantic_digest_missing")
    if semantics.source_digest != semantics.result_digest:
        a09_diagnostics.append("semantic_digest_changed")
    if not semantics.equal or semantics.differences:
        a09_diagnostics.append("graph_semantics_changed")

    a06 = _evaluate_a06(evidence)
    gates = (
        _decision("A01", a01_diagnostics),
        _decision("A02", a02_diagnostics),
        _decision("A03", a03_diagnostics),
        _decision("A04", a04_diagnostics),
        _decision("A05", a05_diagnostics),
        a06,
        _decision("A07", a07_diagnostics),
        _decision("A08", a08_diagnostics),
        _decision("A09", a09_diagnostics),
        _evaluate_a10(evidence),
    )

    diagnostics: list[str] = []
    if not evidence.converged:
        diagnostics.append("not_strictly_converged")
    diagnostics.append(f"terminal_reason:{evidence.terminal_reason}")
    for frame in evidence.frames:
        if frame.role != "exploratory":
            continue
        if not frame.energy.finite:
            diagnostics.append(f"exploratory_nonfinite_energy:{frame.frame_id}")
        if frame.exploded:
            diagnostics.append(f"exploratory_explosion:{frame.frame_id}")
    if a06.passed:
        for event in evidence.unavailable_gradient_events:
            diagnostics.append(f"recovered_unavailable_gradient:{event.event_id}")

    return AcceptanceResult(
        case_id=evidence.case_id,
        route=evidence.route,
        passed=all(gate.passed for gate in gates),
        gates=gates,
        diagnostics=tuple(diagnostics),
    )


__all__ = (
    "GATE_IDS",
    "AcceptanceEvidence",
    "AcceptanceResult",
    "CoordinateProposalEvidence",
    "GateDecision",
    "NumericEvidence",
    "OptimizationFrameEvidence",
    "QualityEvidence",
    "RecoveryEvidence",
    "SemanticEvidence",
    "StageRecord",
    "StageTraceEvidence",
    "UnavailableGradientEvent",
    "evaluate_acceptance",
)
