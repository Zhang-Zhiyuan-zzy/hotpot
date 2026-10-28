"""Report completeness and comparison semantics, independent of scientific truth."""

from __future__ import annotations

from importlib import import_module
import json
from pathlib import Path

import numpy as np
import pytest


_records = import_module("plan.20_fact_rule_architecture._scientific_report")
_compare = import_module("plan.20_fact_rule_architecture.compare_scientific_reports")


def _report(root: Path, *, energy=1.0, failed_seed=None, monitor_passed=True) -> dict:
    root.mkdir()
    source = root / "source.smi"
    source.write_text("O\n")
    case = {"case_id": "water", "index": 1, "smiles": "O", "semantic_graph_sha256": "same-input",
            "eligibility": {"state": "eligible", "reason": "current_rule"}}
    trials = []
    for seed in (42, 43):
        frame_path = root / f"seed-{seed}.npy"
        np.save(frame_path, np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]))
        reference = _records.artifact(frame_path, root)
        frame = {"coordinates": reference, "shape": [3, 3], "coordinates_finite": True,
                 "atom_identity": [[0, 8], [1, 1], [2, 1]]}
        failed = seed == failed_seed
        trials.append({
            "case_id": "water", "case_index": 1, "root_seed": seed, "effective_seed": seed + 1,
            "eligibility": "eligible", "outcome": "execution_error" if failed else "completed",
            "attempted": True, "scientific_status": "unresolved" if failed else "observed",
            "elapsed_seconds": 0.1 if failed else 10.0, "stages": [],
            "observations": {"production_report": {"quality_report": {"passed": monitor_passed}}},
            "optimization": {"best_energy": energy, "energy_unit": "kJ/mol"},
            "work": {"epochs_completed": 0 if failed else 10},
            "selected_frame": None if failed else frame, "last_finite_frames": [],
            "missing_evidence": ["execution failed"] if failed else [],
            "errors": [{"phase": "o0_execution", "message": "reported failure"}] if failed else [],
            "artifacts": [reference],
        })
    document = {
        "schema": _records.SCHEMA,
        "source": {"sha256": _records.sha256(source), "snapshot_artifact": _records.artifact(source, root)},
        "identity": {"rules": {}, "runtime": {}, "cbond_model": {}},
        "configuration": {"root_seeds": [42, 43], "selected_indices": [1], "settings": {}},
        "cases": [case], "trials": trials, "summary": _records.summarize([case], trials), "artifacts": [],
    }
    (root / "report.json").write_text(json.dumps(document))
    return document


def test_large_scientific_difference_and_monitor_flags_do_not_fail_comparison(tmp_path):
    _report(tmp_path / "before")
    _report(tmp_path / "after", energy=1e20, monitor_passed=False)
    exit_code = _compare.main([
        "--baseline", str(tmp_path / "before"), "--candidate", str(tmp_path / "after"),
        "--output", str(tmp_path / "comparison"),
    ])
    comparison = json.loads((tmp_path / "comparison/comparison.json").read_text())
    assert exit_code == 0
    assert comparison["distributions"]["absolute_energy_difference_kj_mol"]["maximum"] > 1e19
    assert comparison["candidate_summary"]["unresolved_trials"] == 0


def test_failed_early_trial_survives_later_output_and_is_excluded_from_work_pair(tmp_path):
    _report(tmp_path / "before")
    _report(tmp_path / "after", failed_seed=42)
    exit_code = _compare.main([
        "--baseline", str(tmp_path / "before"), "--candidate", str(tmp_path / "after"),
        "--output", str(tmp_path / "comparison"),
    ])
    comparison = json.loads((tmp_path / "comparison/comparison.json").read_text())
    assert exit_code == 1
    assert comparison["candidate_summary"]["trial_outcomes"] == {"execution_error": 1, "completed": 1}
    assert comparison["trial_sets"]["paired_count"] == 2
    assert comparison["trial_sets"]["same_recorded_work_count"] == 1


def test_missing_trial_is_an_integrity_error_even_when_summary_was_recomputed(tmp_path):
    document = _report(tmp_path / "report")
    document["trials"].pop()
    document["summary"] = _records.summarize(document["cases"], document["trials"])
    (tmp_path / "report/report.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="missing, extra or duplicate"):
        _records.read_report(tmp_path / "report")


def test_modified_coordinate_artifact_is_rejected(tmp_path):
    _report(tmp_path / "report")
    (tmp_path / "report/seed-42.npy").write_bytes(b"changed bytes")
    with pytest.raises(ValueError, match="artifact identity mismatch"):
        _records.read_report(tmp_path / "report")


@pytest.mark.parametrize("field,value", [("shape", [2, 3]), ("coordinates_finite", False)])
def test_coordinate_metadata_must_describe_the_recorded_bytes(tmp_path, field, value):
    document = _report(tmp_path / "report")
    document["trials"][0]["selected_frame"][field] = value
    (tmp_path / "report/report.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="coordinate"):
        _records.read_report(tmp_path / "report")


def test_nonfinite_json_literal_is_not_a_valid_missing_value_record(tmp_path):
    document = _report(tmp_path / "report")
    document["trials"][0]["elapsed_seconds"] = float("nan")
    (tmp_path / "report/report.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="nonfinite JSON literal"):
        _records.read_report(tmp_path / "report")


def test_nonfinite_coordinates_remain_reportable_when_the_flag_is_accurate(tmp_path):
    document = _report(tmp_path / "report")
    frame_path = tmp_path / "report/seed-42.npy"
    np.save(frame_path, np.full((3, 3), np.nan))
    reference = _records.artifact(frame_path, tmp_path / "report")
    trial = document["trials"][0]
    trial["selected_frame"]["coordinates"] = reference
    trial["selected_frame"]["coordinates_finite"] = False
    trial["artifacts"] = [reference]
    (tmp_path / "report/report.json").write_text(json.dumps(document))
    comparison = _compare.compare_reports(tmp_path / "report", tmp_path / "report")
    assert comparison["pairs"][0]["aligned_coordinate_rmsd_angstrom"] is None
    assert "nonfinite_coordinates" in comparison["pairs"][0]["comparison_exclusions"]
    assert comparison["technical_execution_errors"] == 0


def test_malformed_or_absent_report_has_technical_exit_code(tmp_path):
    assert _compare.main([
        "--baseline", str(tmp_path / "absent"), "--candidate", str(tmp_path / "absent"),
        "--output", str(tmp_path / "comparison"),
    ]) == 2


def test_current_eligibility_distinguishes_no_output_from_execution_error(monkeypatch):
    runner = import_module("plan.20_fact_rule_architecture.run_scientific_report")

    def cbond(index, *args):
        if index == 1:
            return None
        raise RuntimeError("unavailable inference")

    monkeypatch.setattr(runner, "_current_cbond", cbond)
    settings = runner.settings_for_profile(runner.RunProfile.SMOKE)
    assert runner._case(1, "O", "Am", settings)["eligibility"]["state"] == "ineligible"
    unknown = runner._case(2, "N", "Am", settings)["eligibility"]
    assert unknown["state"] == "unknown"
    assert unknown["error"]["message"] == "unavailable inference"
