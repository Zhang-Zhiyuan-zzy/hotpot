"""Prevent incomplete work and failed cleanup from becoming performance gains."""

from __future__ import annotations

from copy import deepcopy
import base64
import csv
import hashlib
from importlib import import_module
import io
import json
from pathlib import Path
import zipfile

import pytest


_runner = import_module("plan.20_fact_rule_architecture.run_performance_baseline")
_comparison = import_module("plan.20_fact_rule_architecture.compare_performance_baselines")
_manifest = json.loads(
    (Path(__file__).parents[1] / "plan/20_fact_rule_architecture/performance_manifest.json").read_text()
)


def _forcefield_observation():
    # Shape of the real organic_forcefield report, with an enlarged audit budget.
    # Open Babel deliberately leaves steps_completed unknown.
    return {
        "completed_calls": 1,
        "value": {
            "atoms": 9,
            "finite_coordinates": True,
            "report": {
                "build": {"atom_count": 9, "added_hydrogen_count": 6},
                "optimization": {
                    "epochs_completed": 20,
                    "selected_segment_epochs_completed": 20,
                    "steps_submitted": 399,
                    "initialization_steps": 1,
                    "steps_completed": None,
                    "converged": False,
                    "final_energy": 19.75,
                    "elapsed_seconds": 10.0,
                    "quality_report": {"passed": True},
                },
            },
        },
    }


def _report(workload_id, observation, *, seconds=10.0):
    manifest = deepcopy(_manifest)
    entry = next(entry for entry in manifest["workloads"] if entry["id"] == workload_id)
    manifest["workloads"] = [entry]
    samples = [
        {
            "index": index,
            "warmup": False,
            "status": "completed",
            "seconds": seconds,
            "peak_rss_bytes": 100_000_000,
            "value": deepcopy(observation),
            "work_counters": _runner.work_counters(observation),
        }
        for index in range(manifest["policies"][entry["tier"]]["samples"])
    ]
    return {
        "metadata": {"smoke": False, "manifest_sha256": "same-frozen-manifest"},
        "manifest": manifest,
        "workloads": [{"id": workload_id, "samples": samples, "summary": _runner.summary(samples)}],
    }


@pytest.mark.parametrize(
    "field,reduced",
    [
        ("epochs_completed", 1),
        ("selected_segment_epochs_completed", 1),
        ("steps_submitted", 19),
        ("initialization_steps", 0),
        ("steps_completed", 1),
    ],
)
def test_faster_forcefield_with_different_actual_work_is_not_a_gain(field, reduced):
    before = _forcefield_observation()
    after = deepcopy(before)
    if field == "steps_completed":
        before["value"]["report"]["optimization"][field] = 400
    after["value"]["report"]["optimization"][field] = reduced
    result = _comparison.compare(
        _report("organic_forcefield", before),
        _report("organic_forcefield", after, seconds=1.0),
    )
    row = result["workloads"][0]
    assert row["status"] == "inconclusive"
    assert not row["comparable"]
    assert row["paired_samples"] == 0
    assert all(sample["reason"] == "different completed work" for sample in row["excluded_samples"])
    assert result["end_to_end"]["geometric_mean_runtime_ratio"] is None


def test_unknown_completed_steps_remain_distinct_from_missing_or_zero():
    before = _forcefield_observation()
    path = "value.report.optimization.steps_completed"
    assert _runner.work_counters(before)[path] is None
    for replacement in ({}, {"steps_completed": 0}):
        after = deepcopy(before)
        after["value"]["report"]["optimization"].pop("steps_completed")
        after["value"]["report"]["optimization"].update(replacement)
        result = _comparison.compare(
            _report("organic_forcefield", before),
            _report("organic_forcefield", after, seconds=1.0),
        )
        assert not result["workloads"][0]["comparable"]


@pytest.mark.parametrize("field", ["ob_atoms", "rd_atoms"])
def test_conversion_with_same_call_count_but_fewer_output_atoms_is_not_a_gain(field):
    before = {"completed_calls": 20, "value": {"ob_atoms": 13, "rd_atoms": 13, "calls": 2}}
    after = deepcopy(before)
    after["value"][field] = 1
    result = _comparison.compare(
        _report("toolkit_conversions", before),
        _report("toolkit_conversions", after, seconds=1.0),
    )
    assert not result["workloads"][0]["comparable"]
    assert result["workloads"][0]["status"] == "inconclusive"
    assert result["families"]["conversions/cpu"]["comparable_members"] == []


@pytest.mark.parametrize("failed_side", ["baseline", "candidate"])
def test_cleanup_failure_prevents_gain_despite_completed_timing_samples(failed_side):
    observation = _forcefield_observation()
    before = _report("organic_forcefield", observation)
    after = _report("organic_forcefield", observation, seconds=1.0)
    failed = before if failed_side == "baseline" else after
    failed["workloads"][0]["teardown_error"] = "RuntimeError('worker shutdown failed')"
    result = _comparison.compare(before, after)
    row = result["workloads"][0]
    assert row["status"] == "inconclusive"
    assert not row["comparable"]
    assert "workload teardown failed" in row["reasons"]
    assert result["end_to_end"]["comparable_members"] == []
    assert result["overall_family_geometric_mean"] is None


def test_fixed_work_retains_parity_when_scientific_observations_change():
    before = _forcefield_observation()
    after = deepcopy(before)
    after["value"]["report"]["optimization"].update(
        final_energy=1e20, elapsed_seconds=0.1, converged=True, quality_report={"passed": False}
    )
    result = _comparison.compare(
        _report("organic_forcefield", before),
        _report("organic_forcefield", after),
    )
    row = result["workloads"][0]
    assert row["status"] == "parity"
    assert row["comparable"]
    assert row["paired_bootstrap_95ci"] == [1.0, 1.0]
    assert result["end_to_end"]["complete"]


def _gpu_report(*, seconds=10.0):
    block = {"gpu_index": 0, "cpu_ids": list(range(16)), "site_cpu_ids": [0], "infer_cpu_ids": list(range(1, 16))}
    observation = {"completed_calls": 1, "value": {
        "requested_molecules": 512, "returned_molecules": 512, "sites": 320, "atom_predictions": 3000,
        "gpu_started": True, "resource_plan": {
            "gpu_indices": [0], "available_cpu_ids": list(range(64)), "site_cpu_per_gpu": 1,
            "infer_cpu_per_gpu": 15, "gpu_cpu_assignments": [block],
        },
    }}
    result = _report("mca_1gpu", observation, seconds=seconds)
    result["metadata"].update(cpu_affinity=list(range(64)), model_identity={"sha256": "verified-model"})
    _set_gpu_snapshot(result, "0, GPU-111, NVIDIA A800, 570.133, 81920, 1000, 0")
    return result


def _set_gpu_snapshot(report, allocated, *, other="1, GPU-999, NVIDIA A800, 570.133, 81920, 1000, 0"):
    snapshot = {"available": True, "readings": [allocated, other]}
    report["metadata"].update(gpu_before=deepcopy(snapshot), gpu_after=deepcopy(snapshot))
    for sample in report["workloads"][0]["samples"]:
        sample.update(gpu_before=deepcopy(snapshot), gpu_after=deepcopy(snapshot))


@pytest.mark.parametrize("changed", [
    "0, GPU-222, NVIDIA A800, 570.133, 81920, 1000, 0",
    "0, GPU-111, NVIDIA H100, 570.133, 81920, 1000, 0",
    "0, GPU-111, NVIDIA A800, 580.65, 81920, 1000, 0",
])
def test_faster_gpu_run_on_changed_uuid_model_or_driver_is_not_a_gain(changed):
    before, after = _gpu_report(), _gpu_report(seconds=1.0)
    _set_gpu_snapshot(after, changed)
    result = _comparison.compare(before, after)
    row = result["workloads"][0]
    assert row["status"] == "inconclusive"
    assert not row["comparable"]
    assert "allocated GPU identity or actual resource allocation differs" in row["reasons"]
    assert result["end_to_end"]["comparable_members"] == []


def test_gpu_load_and_unallocated_device_changes_do_not_prevent_comparison():
    before, after = _gpu_report(), _gpu_report()
    _set_gpu_snapshot(after, "0, GPU-111, NVIDIA A800, 570.133, 81920, 33000, 95",
                      other="1, GPU-333, NVIDIA H100, 580.65, 81920, 9000, 85")
    result = _comparison.compare(before, after)
    assert result["workloads"][0]["status"] == "parity"
    assert result["workloads"][0]["comparable"]


def test_cpu_workload_ignores_unrelated_gpu_and_visibility_changes():
    observation = _forcefield_observation()
    before, after = _report("organic_forcefield", observation), _report("organic_forcefield", observation)
    before["metadata"].update(gpu_before={"available": False}, cuda_visible_devices="0")
    after["metadata"].update(gpu_before={"available": True, "readings": ["changed GPU"]}, cuda_visible_devices="1")
    result = _comparison.compare(before, after)
    assert result["environment_mismatches"] == []
    assert result["workloads"][0]["status"] == "parity"


@pytest.mark.parametrize("change", ["cpu_allocation", "visibility", "midrun_identity"])
def test_same_host_gpu_run_requires_the_same_actual_allocation(change):
    before, after = _gpu_report(), _gpu_report(seconds=1.0)
    if change == "cpu_allocation":
        for sample in after["workloads"][0]["samples"]:
            block = sample["value"]["value"]["resource_plan"]["gpu_cpu_assignments"][0]
            block.update(cpu_ids=list(range(16, 32)), site_cpu_ids=[16], infer_cpu_ids=list(range(17, 32)))
    elif change == "visibility":
        after["metadata"]["cuda_visible_devices"] = "1"
    else:
        after["workloads"][0]["samples"][0]["gpu_after"]["readings"][0] = "0, GPU-222, NVIDIA H100, 570.133, 81920, 1000, 0"
    row = _comparison.compare(before, after)["workloads"][0]
    assert not row["comparable"]
    assert row["status"] == "inconclusive"


def test_cbond_cuda_requires_observed_session_device_not_just_cuda_label():
    before = _report("cbond_cuda", {"completed_calls": 20, "value": {"predictions": 2, "device": "cuda"}})
    before["metadata"]["model_identity"] = {"sha256": "verified-model"}
    _set_gpu_snapshot(before, "0, GPU-111, NVIDIA A800, 570.133, 81920, 1000, 0")
    allocation = {"device": "cuda", "sessions": {
        name: {"providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
               "provider_options": {"CUDAExecutionProvider": {"device_id": "0"}}}
        for name in ("graph", "cbond")
    }}
    for sample in before["workloads"][0]["samples"]:
        sample["runtime_allocation"] = deepcopy(allocation)
    assert _comparison.compare(before, before)["workloads"][0]["status"] == "parity"
    after = deepcopy(before)
    for sample in after["workloads"][0]["samples"]:
        sample["seconds"] = 1.0
        sample["runtime_allocation"]["sessions"]["cbond"]["provider_options"]["CUDAExecutionProvider"]["device_id"] = "1"
    assert not _comparison.compare(before, after)["workloads"][0]["comparable"]


def _wheel_installation(tmp_path):
    root = tmp_path / "site-packages"
    record_name = "hotpot_zzy-1.0.dist-info/RECORD"
    contents = {"hotpot/__init__.py": b"# installed hotpot\n", "hotpot/_native.so": b"frozen native binary",
                "hotpot_zzy-1.0.dist-info/METADATA": b"Name: hotpot-zzy\nVersion: 1.0\n"}
    record = io.StringIO()
    writer = csv.writer(record, lineterminator="\n")
    for name, content in contents.items():
        checksum = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
        writer.writerow([name, "sha256=" + checksum, len(content)])
    writer.writerow([record_name, "", ""])
    contents[record_name] = record.getvalue().encode()
    wheel = tmp_path / "hotpot_zzy-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, content in contents.items():
            archive.writestr(name, content)
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    return wheel, root, root / record_name, root / "hotpot/__init__.py"


def test_wheel_identity_binds_record_installed_native_bytes_and_loaded_origin(tmp_path):
    args = _wheel_installation(tmp_path)
    verified = _runner.verify_wheel_installation(*args)
    assert verified["verified"]
    native = args[1] / "hotpot/_native.so"
    assert _runner.verify_loaded_package_files(verified, [str(args[3]), str(native)])["verified"]
    native.write_bytes(b"different native build in the same installation path")
    assert not _runner.verify_wheel_installation(*args)["verified"]
    assert not _runner.verify_loaded_package_files(verified, [str(native)])["verified"]


def test_matching_bytes_from_another_checkout_do_not_prove_wheel_import(tmp_path):
    args = _wheel_installation(tmp_path)
    verified = _runner.verify_wheel_installation(*args)
    other = tmp_path / "checkout/hotpot/__init__.py"
    other.parent.mkdir(parents=True)
    other.write_bytes(args[3].read_bytes())
    assert not _runner.verify_wheel_installation(*args[:3], other)["verified"]
    assert not _runner.verify_loaded_package_files(verified, [str(other)])["verified"]


def test_installed_record_rewrite_cannot_hide_a_different_distribution(tmp_path):
    args = _wheel_installation(tmp_path)
    args[2].write_text(args[2].read_text().replace("sha256=", "sha512=", 1))
    identity = _runner.verify_wheel_installation(*args)
    assert not identity["verified"]
    assert any("installed RECORD differs" in error for error in identity["errors"])


def test_unverified_installation_cannot_claim_a_wheel_speedup():
    before = _report("organic_forcefield", _forcefield_observation())
    after = _report("organic_forcefield", _forcefield_observation(), seconds=1.0)
    for report in (before, after):
        report["metadata"].update(package_mode="installed", wheel={"sha256": "claimed-wheel"},
                                  installed_wheel={"verified": True, "wheel_sha256": "different-wheel"})
        report["workloads"][0]["loaded_package_identity"] = {"verified": True}
    row = _comparison.compare(before, after)["workloads"][0]
    assert not row["comparable"]
    assert row["status"] == "inconclusive"
