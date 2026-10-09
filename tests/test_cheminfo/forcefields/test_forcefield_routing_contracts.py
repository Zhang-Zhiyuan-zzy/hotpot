"""Contracts describing automatic force-field route selection."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import get_args

import pytest

from hotpot.cheminfo.forcefields.contracts import (
    ForceFieldRunReport,
    OptimizationAttemptReport,
    OptimizationAttemptStatus,
    OptimizationRoute,
    OptimizationRoutingReport,
    TerminationReason,
)


def _run_report() -> ForceFieldRunReport:
    return ForceFieldRunReport(
        requested_forcefield="UFF",
        effective_forcefield="UFF",
        setup_succeeded=True,
        converged=True,
        epochs_completed=1,
        steps_submitted=100,
        initialization_steps=1,
        steps_completed=None,
        final_energy=1.0,
        best_energy=1.0,
        energy_unit="kJ/mol",
        rms_gradient=0.1,
        max_gradient=0.2,
        exploded=False,
    )


def test_termination_reason_covers_native_controller_failures() -> None:
    assert set(get_args(TerminationReason)) == {
        "converged",
        "budget_exhausted",
        "stability_reached",
        "topology_blocked",
        "nonfinite_coordinates",
        "nonfinite_energy",
        "nonfinite_gradients",
        "explosion_detected",
    }


def test_direct_forcefield_run_report_has_no_routing_by_default() -> None:
    assert _run_report().routing_report is None


def test_routing_report_preserves_attempt_order_and_selected_route() -> None:
    attempts = (
        OptimizationAttemptReport(
            route=OptimizationRoute.ORDINARY_FAST,
            status=OptimizationAttemptStatus.QUALITY_REJECTED,
            termination_reason="converged",
        ),
        OptimizationAttemptReport(
            route=OptimizationRoute.COMPLEX,
            status=OptimizationAttemptStatus.ACCEPTED,
            termination_reason="converged",
        ),
    )
    routing = OptimizationRoutingReport(
        attempts=attempts,
        selected_route=OptimizationRoute.COMPLEX,
    )

    assert routing.attempts == attempts
    assert routing.selected_route is OptimizationRoute.COMPLEX
    assert routing.selected_route == "complex"

    with pytest.raises(FrozenInstanceError):
        routing.selected_route = OptimizationRoute.ORDINARY_FAST
