"""Execution-record integrity for bounded and complete extractant O0 runs."""

from __future__ import annotations

from importlib import import_module
import os
from pathlib import Path

import pytest

from tests.benchmarks.coordination_complexes.configuration import CANONICAL_EXTRACTANT_SOURCE
from tests.benchmarks.coordination_complexes.io import load_smiles


_runner = import_module("plan.20_fact_rule_architecture.run_scientific_report")
_records = import_module("plan.20_fact_rule_architecture._scientific_report")


@pytest.mark.model_artifact
def test_one_extractant_retains_o0_observations_and_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "extractant-smoke"
    _runner.run_report(output, seeds=(42,), profile="smoke", indices=(1,))

    report, _ = _records.read_report(output)
    assert report["summary"]["execution_error_count"] == 0
    assert report["configuration"]["selected_indices"] == [1]
    assert len(report["trials"]) == 1
    trial = report["trials"][0]
    assert trial["effective_seed"] == 43
    assert trial["provider"]["effective_seed"] is None
    assert trial["artifacts"]
    assert trial["selected_frame"] is not None or trial["missing_evidence"]
    if trial["outcome"] == "completed":
        assert trial["scientific_status"] == "observed"


@pytest.mark.scientific_boundary
@pytest.mark.high_compute
@pytest.mark.model_artifact
def test_complete_extractant_report_covers_all_source_cases_and_seeds(tmp_path: Path) -> None:
    output = tmp_path / "extractants-o0"
    workers = int(os.environ.get("HOTPOT_TEST_WORKERS", "1"))
    _runner.run_report(output, workers=workers)

    report, _ = _records.read_report(output)
    assert report["summary"]["execution_error_count"] == 0
    source = load_smiles(CANONICAL_EXTRACTANT_SOURCE)
    assert report["configuration"]["selected_indices"] == [index for index, _ in source]
    assert report["configuration"]["root_seeds"] == [42, 43, 44]
    assert len(report["trials"]) == len(source) * 3
    assert sum(report["summary"]["trial_outcomes"].values()) == len(report["trials"])
    assert report["source"]["sha256"] == _records.sha256(CANONICAL_EXTRACTANT_SOURCE)
