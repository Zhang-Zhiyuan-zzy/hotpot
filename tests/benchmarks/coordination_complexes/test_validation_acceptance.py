"""Synthetic contract tests for Plan-17 force-field acceptance."""

from __future__ import annotations

from dataclasses import replace
import json

import pytest

from .validation_acceptance import (
    GATE_IDS,
    AcceptanceEvidence,
    CoordinateProposalEvidence,
    NumericEvidence,
    OptimizationFrameEvidence,
    QualityEvidence,
    RecoveryEvidence,
    SemanticEvidence,
    StageRecord,
    StageTraceEvidence,
    UnavailableGradientEvent,
    evaluate_acceptance,
)


def _number(value: float) -> NumericEvidence:
    return NumericEvidence.from_value(value)


def _frame(
    frame_id: str,
    sequence: int,
    *,
    role: str = "exploratory",
    energy: float = -10.0,
    rms_gradient: float = 0.2,
    max_gradient: float = 0.8,
    exploded: bool = False,
    optimization_progressed: bool = False,
) -> OptimizationFrameEvidence:
    return OptimizationFrameEvidence(
        frame_id=frame_id,
        sequence=sequence,
        stage="optimization",
        event="epoch_complete",
        role=role,
        coordinates_finite=True,
        energy=_number(energy),
        gradients_evaluated=True,
        rms_gradient=_number(rms_gradient),
        max_gradient=_number(max_gradient),
        exploded=exploded,
        optimization_progressed=optimization_progressed,
    )


def _selected(**changes: object) -> OptimizationFrameEvidence:
    frame = _frame(
        "selected",
        10,
        role="selected",
        optimization_progressed=True,
    )
    return replace(frame, **changes)


def _baseline(**changes: object) -> AcceptanceEvidence:
    stages = (
        "provider",
        "ligand_build",
        "placement",
        "coordination_restoration",
        "optimization",
    )
    evidence = AcceptanceEvidence(
        case_id="extractant-001",
        route="F1",
        proposal=CoordinateProposalEvidence(
            returned=True,
            coordinate_shape=(24, 3),
            atom_row_count=24,
            expected_atom_row_count=24,
            mapping_preserved=True,
            coordinates_finite=True,
            status="returned",
        ),
        setup_succeeded=True,
        setup_exception=None,
        frames=(_selected(),),
        committed_coordinates_finite=True,
        unavailable_gradient_events=(),
        established_recovery_actions=(
            "perturb_close_atoms",
            "perturb_forbidden_angle",
        ),
        quality=QualityEvidence(
            level="standard",
            passed=True,
            check_names=("finite_coordinates", "bond_geometry", "ring_piercing"),
        ),
        semantics=SemanticEvidence(
            equal=True,
            source_digest="semantic-digest",
            result_digest="semantic-digest",
        ),
        stage_trace=StageTraceEvidence(
            expected_stages=stages,
            records=tuple(StageRecord(stage, "succeeded") for stage in stages),
            provider_fallback_used=False,
        ),
        terminal_reason="converged",
        converged=True,
    )
    return replace(evidence, **changes)


def _gradient_recovery(
    *,
    recovery: RecoveryEvidence | None,
    terminal_reason: str = "converged",
) -> AcceptanceEvidence:
    unavailable = _frame(
        "gradient-unavailable",
        2,
        rms_gradient=float("nan"),
        max_gradient=float("inf"),
    )
    restored = _frame(
        "gradient-restored",
        3,
        optimization_progressed=True,
    )
    baseline = _baseline(
        frames=(unavailable, restored, _selected()),
        unavailable_gradient_events=(
            UnavailableGradientEvent(
                event_id="gradient-event-1",
                frame_id=unavailable.frame_id,
                stage="optimization",
                recovery=recovery,
            ),
        ),
        terminal_reason=terminal_reason,
    )
    records = tuple(
        replace(
            record,
            recovery_actions=("perturb_close_atoms",),
        )
        if record.name == "optimization"
        else record
        for record in baseline.stage_trace.records
    )
    return replace(
        baseline,
        stage_trace=replace(baseline.stage_trace, records=records),
    )


def _successful_recovery() -> RecoveryEvidence:
    return RecoveryEvidence(
        action_name="perturb_close_atoms",
        attempted=True,
        finite_gradient_frame_id="gradient-restored",
        optimization_resumed=True,
        resumed_frame_id="gradient-restored",
    )


def test_recovered_infinite_energy_exploratory_frame_is_diagnostic_only() -> None:
    exploratory = _frame(
        "rejected-explosion",
        1,
        energy=float("inf"),
        exploded=True,
    )

    result = evaluate_acceptance(
        _baseline(frames=(exploratory, _selected()))
    )

    assert result.passed
    assert result.decision("A04").passed
    assert result.decision("A07").passed
    assert "exploratory_nonfinite_energy:rejected-explosion" in result.diagnostics
    assert "exploratory_explosion:rejected-explosion" in result.diagnostics


def test_established_recovery_with_finite_gradients_and_progress_passes() -> None:
    result = evaluate_acceptance(
        _gradient_recovery(recovery=_successful_recovery())
    )

    assert result.passed
    assert result.decision("A06").passed
    assert "recovered_unavailable_gradient:gradient-event-1" in result.diagnostics


def test_unavailable_gradient_without_standard_recovery_fails() -> None:
    result = evaluate_acceptance(_gradient_recovery(recovery=None))

    assert not result.passed
    assert result.decision("A06").diagnostics == (
        "recovery_missing:gradient-event-1",
    )


def test_attempted_recovery_that_does_not_restore_optimization_fails() -> None:
    recovery = RecoveryEvidence(
        action_name="perturb_close_atoms",
        attempted=True,
        finite_gradient_frame_id=None,
        optimization_resumed=False,
        resumed_frame_id=None,
    )

    result = evaluate_acceptance(_gradient_recovery(recovery=recovery))

    assert not result.decision("A06").passed
    assert "finite_gradient_proof_missing:gradient-event-1" in (
        result.decision("A06").diagnostics
    )
    assert "optimization_not_resumed:gradient-event-1" in (
        result.decision("A06").diagnostics
    )


def test_infinite_selected_best_energy_fails() -> None:
    selected = _selected(energy=NumericEvidence.from_value(float("inf")))

    result = evaluate_acceptance(_baseline(frames=(selected,)))

    assert not result.passed
    assert result.decision("A04").diagnostics == (
        "selected_best_energy_nonfinite:positive_infinity",
    )


@pytest.mark.parametrize("field", ("rms_gradient", "max_gradient"))
def test_nonfinite_selected_gradient_fails(field: str) -> None:
    selected = _selected(
        **{field: NumericEvidence.from_value(float("nan"))},
    )

    result = evaluate_acceptance(_baseline(frames=(selected,)))

    assert not result.passed
    assert not result.decision("A05").passed
    assert not result.decision("A06").passed


def test_selected_frame_explosion_fails() -> None:
    result = evaluate_acceptance(
        _baseline(frames=(_selected(exploded=True),))
    )

    assert not result.passed
    assert result.decision("A07").diagnostics == ("selected_frame_exploded",)


def test_standard_geometry_failure_fails() -> None:
    quality = QualityEvidence(
        level="standard",
        passed=False,
        check_names=("minimum_distance",),
        failure_names=("minimum_distance",),
    )

    result = evaluate_acceptance(_baseline(quality=quality))

    assert not result.passed
    assert result.decision("A08").diagnostics == ("standard_quality_failed",)


def test_budget_exhaustion_with_valid_selected_structure_passes() -> None:
    result = evaluate_acceptance(
        _baseline(terminal_reason="budget_exhausted", converged=False)
    )

    assert result.passed
    assert "not_strictly_converged" in result.diagnostics
    assert "terminal_reason:budget_exhausted" in result.diagnostics


def test_terminal_reason_cannot_mask_an_unavailable_gradient() -> None:
    evidence = _gradient_recovery(
        recovery=None,
        terminal_reason="nonfinite_energy",
    )

    result = evaluate_acceptance(evidence)

    assert not result.decision("A06").passed
    assert "terminal_reason:nonfinite_energy" in result.diagnostics


def test_unlisted_recovery_name_and_provider_fallback_both_fail() -> None:
    recovery = replace(
        _successful_recovery(),
        action_name="benchmark_only_rescue",
    )
    evidence = _gradient_recovery(recovery=recovery)
    evidence = replace(
        evidence,
        stage_trace=replace(
            evidence.stage_trace,
            provider_fallback_used=True,
            fallback_provider="openbabel",
        ),
    )

    result = evaluate_acceptance(evidence)

    assert "recovery_not_established:gradient-event-1:benchmark_only_rescue" in (
        result.decision("A06").diagnostics
    )
    assert result.decision("A10").diagnostics == (
        "provider_fallback_used",
        "recovery_not_in_stage_trace:gradient-event-1",
    )


def test_semantic_change_is_a09_failure() -> None:
    semantics = SemanticEvidence(
        equal=False,
        source_digest="before",
        result_digest="after",
        differences=("dative_direction",),
    )

    result = evaluate_acceptance(_baseline(semantics=semantics))

    assert not result.decision("A09").passed
    assert result.decision("A09").diagnostics == (
        "semantic_digest_changed",
        "graph_semantics_changed",
    )


def test_evidence_and_result_are_strict_json_and_gate_complete() -> None:
    exploratory = _frame(
        "nonfinite-values",
        1,
        energy=float("-inf"),
        rms_gradient=float("nan"),
        max_gradient=float("inf"),
    )
    evidence = _baseline(
        frames=(exploratory, _selected()),
        unavailable_gradient_events=(
            UnavailableGradientEvent(
                event_id="unrecovered",
                frame_id=exploratory.frame_id,
                stage="optimization",
            ),
        ),
    )
    result = evaluate_acceptance(evidence)

    evidence_json = json.dumps(evidence.to_dict(), allow_nan=False, sort_keys=True)
    result_json = json.dumps(result.to_dict(), allow_nan=False, sort_keys=True)

    assert '"value": null' in evidence_json
    assert '"finite": false' in evidence_json
    assert "negative_infinity" in evidence_json
    assert "positive_infinity" in evidence_json
    assert "nan" in evidence_json
    assert tuple(gate.gate_id for gate in result.gates) == GATE_IDS
    assert '"passed": false' in result_json


def test_missing_or_failed_proposal_setup_and_stage_are_independent_gates() -> None:
    baseline = _baseline()
    proposal = replace(
        baseline.proposal,
        returned=False,
        status="failed",
        coordinate_shape=None,
        atom_row_count=0,
        coordinates_finite=False,
    )
    records = baseline.stage_trace.records[:-1]
    evidence = replace(
        baseline,
        proposal=proposal,
        setup_succeeded=False,
        setup_exception="setup error",
        stage_trace=replace(baseline.stage_trace, records=records),
    )

    result = evaluate_acceptance(evidence)

    assert not result.decision("A01").passed
    assert not result.decision("A02").passed
    assert not result.decision("A10").passed
