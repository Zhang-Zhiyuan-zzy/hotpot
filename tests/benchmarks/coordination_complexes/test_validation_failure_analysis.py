"""Tests for Plan-17 failed-route diagnostics and efficiency summaries."""

from __future__ import annotations

from copy import deepcopy

import pytest

from .validation_failure_analysis import classify_failure, efficiency_rows


def _record(
    *,
    route: str = "O0",
    run_id: str = "formal-o0",
    case_index: int = 1,
    passed: bool = True,
    elapsed: float = 10.0,
    recorded_at: str = "2026-10-23T06:30:10+00:00",
) -> dict[str, object]:
    return {
        "route": route,
        "run_id": run_id,
        "case_index": case_index,
        "passed": passed,
        "status": "automated_passed" if passed else "automated_failed",
        "recorded_at": recorded_at,
        "acceptance": {
            "gates": [
                {
                    "gate_id": f"A{index:02d}",
                    "passed": True,
                    "diagnostics": [],
                }
                for index in range(1, 11)
            ]
        },
        "evidence": {
            "frames": [],
            "quality": {"failure_names": [], "warning_names": []},
            "unavailable_gradient_events": [],
        },
        "error": None,
        "timing": {
            "runner_elapsed_seconds": elapsed,
            "stages": {
                "provider_or_generation_seconds": 1.0,
                "ligand_build_seconds": 2.0,
                "coordination_restoration_seconds": 1.0,
                "optimization_seconds": 4.0,
                "acceptance_seconds": 0.5,
            },
        },
        "resources": {
            "usage": {
                "user_cpu_seconds": elapsed / 2,
                "system_cpu_seconds": 0.1,
                "process_max_rss_kib_at_case_end": 100_000,
            }
        },
    }


@pytest.mark.parametrize(
    ("message", "route", "expected"),
    [
        (
            "RDKit ETKDGv3 returned no conformers",
            "C1",
            "etkdg_no_conformer",
        ),
        (
            "Component 0 failed its single conformer provider call",
            "F1",
            "etkdg_no_conformer",
        ),
        (
            "coordination bounds matrix failed triangle smoothing",
            "C2",
            "c2_bounds_triangle_smoothing",
        ),
    ],
)
def test_execution_failure_categories_are_explicit(message, route, expected):
    record = _record(route=route, passed=False)
    record["status"] = "execution_failed"
    record["error"] = {
        "phase": "route_execution",
        "type": "RuntimeError",
        "message": message,
    }

    assert classify_failure(record)[0] == expected


def test_rejected_geometry_and_unavailable_gradient_remain_distinct():
    geometry = _record(passed=False)
    geometry["acceptance"]["gates"][7]["passed"] = False
    geometry["evidence"]["quality"]["failure_names"] = [
        "atom_too_close",
        "atom_too_close",
        "short_bond",
    ]

    gradient = deepcopy(geometry)
    gradient["evidence"]["unavailable_gradient_events"] = [
        {"event_id": "gradient:main:1", "recovery": None}
    ]

    geometry_category, geometry_reason = classify_failure(geometry)
    assert geometry_category == "a08_geometry_rejected"
    assert "atom_too_close x2" in geometry_reason
    assert classify_failure(gradient)[0] == "unrecovered_unavailable_gradient"


def test_efficiency_summary_uses_elapsed_and_labels_parent_cpu_scope():
    records = [
        _record(case_index=1, elapsed=10.0),
        _record(
            case_index=2,
            elapsed=20.0,
            passed=False,
            recorded_at="2026-10-23T06:30:30+00:00",
        ),
        _record(
            route="F1",
            run_id="formal-f1",
            case_index=1,
            elapsed=5.0,
        ),
        _record(
            route="F1",
            run_id="formal-f1",
            case_index=2,
            elapsed=10.0,
            recorded_at="2026-10-23T06:30:20+00:00",
        ),
    ]

    o0, f1 = efficiency_rows(records)

    assert o0["runner_elapsed_sum_seconds"] == 30.0
    assert o0["accepted_count"] == 1
    assert o0["runner_elapsed_per_accepted_seconds"] == 30.0
    assert f1["runner_elapsed_sum_seconds"] == 15.0
    assert f1["elapsed_sum_relative_to_o0"] == 0.5
    assert f1["runner_elapsed_p50_seconds"] == 7.5
    assert f1["parent_worker_user_cpu_seconds"] == 7.5
    assert f1["parent_worker_user_cpu_to_elapsed_ratio"] == 0.5
