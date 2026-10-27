"""Bounded and full recomputation tests for the 187/182 extractant boundary."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from tests.benchmarks.coordination_complexes.configuration import (
    BUILTIN_SUITES,
    RunProfile,
)
from tests.benchmarks.coordination_complexes.runner import run_benchmark


NO_CBOND_INDICES = {139, 182, 185, 186, 187}


@pytest.mark.model_artifact
def test_one_governed_extractant_covers_the_current_workflow(tmp_path: Path) -> None:
    output = tmp_path / "extractant-smoke"

    summary = run_benchmark(
        BUILTIN_SUITES["extractants-am-187"],
        output,
        profile=RunProfile.SMOKE,
        workers=1,
        limit=1,
        backend="hotpot-auto",
    )

    assert summary["sample_count"] == 1
    assert summary["cbond_success_count"] == 1
    assert summary["quality_passed"] == 1
    record = json.loads((output / "cases/0001/report.json").read_text())
    assert record["status"] == "passed"
    assert record["validation"]["passed"] is True
    assert np.isfinite(record["optimization"]["best_energy"])


@pytest.mark.scientific_boundary
@pytest.mark.high_compute
@pytest.mark.model_artifact
def test_complete_current_workflow_reproduces_the_187_182_boundary(
    tmp_path: Path,
) -> None:
    workers = int(os.environ.get("HOTPOT_TEST_WORKERS", "1"))

    summary = run_benchmark(
        BUILTIN_SUITES["extractants-am-187"],
        tmp_path / "extractants-am-187",
        profile=RunProfile.STANDARD,
        workers=workers,
        backend="hotpot-auto",
    )

    assert summary["sample_count"] == 187
    assert summary["cbond_success_count"] == 182
    assert summary["quality_passed"] == 182
    assert summary["status_counts"] == {"passed": 182, "failed_cbond": 5}
    assert {
        failure["index"] for failure in summary["failure_cases"]
    } == NO_CBOND_INDICES
