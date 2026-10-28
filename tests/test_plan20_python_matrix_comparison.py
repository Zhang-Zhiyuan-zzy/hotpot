"""Matrix comparisons reject missing, changed, or unbound historical work."""

import base64
import csv
from importlib import import_module
import io
from pathlib import Path
from zipfile import ZipFile

import pytest


_comparison = import_module("plan.20_fact_rule_architecture.compare_python_matrix_baselines")
write_json = _comparison.write_json


def _record(root, path):
    return {"path": path.relative_to(root).as_posix(), "sha256": _comparison.digest(path)}


def _wheel(root):
    path = root / "hotpot.whl"
    payload = b'"""Synthetic installed package evidence."""\n'
    checksum = _comparison.hashlib.sha256(payload).hexdigest()
    encoded = base64.urlsafe_b64encode(bytes.fromhex(checksum)).decode().rstrip("=")
    rows = io.StringIO()
    writer = csv.writer(rows)
    writer.writerow(["hotpot/__init__.py", f"sha256={encoded}", len(payload)])
    writer.writerow(["hotpot_zzy-1.dist-info/RECORD", "", ""])
    record = rows.getvalue().encode()
    with ZipFile(path, "w") as archive:
        archive.writestr("hotpot/__init__.py", payload)
        archive.writestr("hotpot_zzy-1.dist-info/RECORD", record)
    return path, checksum, _comparison.hashlib.sha256(record).hexdigest()


def _group(root, source, definition, preflight, installation, label, *, collect, seconds, skip_first, successful_subtests=0):
    directory = root / preflight["version"] / label / definition["id"]
    directory.mkdir(parents=True)
    nodes = definition["nodeids"]
    skipped = nodes[0 if skip_first else 1]
    reports = []
    if not collect:
        for node in nodes:
            reports.append({"nodeid": node, "when": "setup", "outcome": "skipped" if node == skipped else "passed"})
            if node != skipped:
                reports.append({"nodeid": node, "when": "call", "outcome": "passed"})
            reports.append({"nodeid": node, "when": "teardown", "outcome": "passed"})
        reports += [{"nodeid": next(node for node in nodes if node != skipped), "when": "call", "outcome": "passed"}] * successful_subtests
    package_file = installation["package_file"]
    write_json(directory / "probe.json", {"nodeids": nodes, "exitstatus": 0, "modules": {"hotpot": package_file}, "test_reports": reports})
    write_json(directory / "package-binding.json", {"installed_hotpot": {"verified": True, "errors": [], "files": [{"path": package_file, "sha256": installation["files"][0]["sha256"]}]}, "legacy_plain_source_modules": {}})
    command = [preflight["executable"], "-m", "pytest", "-q", "-p", "_plan20_matrix_probe", "--rootdir", str(source), *definition["options"]]
    command += ["--collect-only"] if collect else [f"--junitxml={directory / 'junit.xml'}"]
    command += [str(source / node.split("::")[0]) + "::" + node.split("::")[1] for node in nodes]
    process = {"command": command, "returncode": 0, "seconds": seconds, "peak_rss_bytes": 100}
    write_json(directory / "pytest-process.json", process)
    (directory / "pytest.stdout.log").write_text("collected evidence\n")
    (directory / "pytest.stderr.log").write_text("")
    result = {**process, "group": definition["id"], "status": "completed", "requested_nodes": len(nodes),
              "collected_nodes": len(nodes), "collection_sha256": _comparison.json_digest(nodes), "errors": [],
              "package_binding_verified": True, "legacy_plain_source_module_count": 0}
    if not collect:
        cases = "".join(f'<testcase classname="{definition["id"]}" name="{node.split("::")[1]}">{"<skipped/>" if node == skipped else ""}</testcase>' for node in nodes)
        (directory / "junit.xml").write_text(f'<testsuites><testsuite tests="{2 + successful_subtests}" failures="0" errors="0" skipped="1">{cases}</testsuite></testsuites>')
        result["junit"] = {"tests": 2 + successful_subtests, "failures": 0, "errors": 0, "skipped": 1}
    result["artifacts"] = [_record(root, path) for path in sorted(directory.iterdir())]
    write_json(directory / "result.json", result)
    return result


def make_report(root, *, candidate=False, ratios=None, skip_first=False, successful_subtests=0):
    root.mkdir()
    source = root.parent / ("candidate-source" if candidate else "baseline-source")
    manifest = {
        "schema": "hotpot-plan20-python-matrix-manifest/v1", "baseline_source_commit": "baseline",
        "versions": [{"version": version, "cpu_affinity": list(range(index * 8, index * 8 + 8))} for index, version in enumerate(_comparison.VERSIONS)],
        "policy": {"warmups": 1, "samples": 5, "thread_environment": {"OMP_NUM_THREADS": "1"}, "python_hash_seed": "0",
                   "neutral_band": [0.97, 1.03], "bootstrap_seed": 42, "bootstrap_samples": 100, "confidence_level": 0.95},
        "pytest_common_options": ["-q"], "total_common_nodes": 4,
        "groups": [{"id": name, "options": [], "nodeids": [f"tests/test_{name}.py::test_one", f"tests/test_{name}.py::test_two"]} for name in ("contracts", "smarts")],
    }
    write_json(root / "manifest.json", manifest)
    write_json(root / "environment-descriptor.json", {"fixture": True})
    write_json(root / "model-identity.json", {"fixture": "same-model-artifact"})
    wheel, package_hash, record_hash = _wheel(root)
    versions = []
    for item in manifest["versions"]:
        version = item["version"]
        python = str(root / f"env-{version}/bin/python")
        directory = root / version / "preflight"
        directory.mkdir(parents=True)
        package_file = str(root / f"env-{version}/site-packages/hotpot/__init__.py")
        installation = {"verified": True, "errors": [], "wheel_sha256": _comparison.digest(wheel), "wheel_record_sha256": record_hash,
                        "package_file": package_file, "files": [{"wheel_path": "hotpot/__init__.py", "installed_path": package_file, "sha256": package_hash}]}
        write_json(directory / "installed-wheel.json", installation)
        packages = {"pytest": "9.2.0", "numpy": "2.5.4", "hotpot-zzy": "2" if candidate else "1"}
        process = {"returncode": 0, "seconds": 1.0, "peak_rss_bytes": 100}
        preflight = {"status": "completed", "version": version, "python": f"{version}.0 fixture", "executable": python,
                     "platform": "fixture-linux", "hostname": "fixture-host", "cpu_affinity": item["cpu_affinity"], "packages": packages,
                     "dependency_lock_sha256": _comparison.json_digest(packages), "models_verify": process,
                     "environment": {"OMP_NUM_THREADS": "1", "PYTHONHASHSEED": "0", "PYTEST_ADDOPTS": "", "PYTHONNOUSERSITE": "1", "PYTHONSAFEPATH": "1",
                                     "PATH": str(Path(python).parent) + ":/usr/bin", "LD_PRELOAD": "/fixed/libstdc++.so", "CUDA_VISIBLE_DEVICES": "",
                                     "HOTPOT_TEST_WORKDIR": str(root / version), "PYTHONPATH": str(root / version / "test-support"),
                                     "HOTPOT_MCA_MODEL_DIR": str(root / "mca"), "HOTPOT_CBOND_MODEL_DIR": str(root / "cbond"), "HOTPOT_MODEL_HOME": str(root / "models")},
                     "wheel": {"path": str(wheel), "sha256": _comparison.digest(wheel), "binding": _record(root, directory / "installed-wheel.json")},
                     "model_identity": {"path": str(root / "model-identity.json"), "sha256": _comparison.digest(root / "model-identity.json")}}
        write_json(directory / "models-verify-process.json", process)
        preflight["collection"] = [_group(root, source, group, preflight, installation, "preflight", collect=True, seconds=1.0, skip_first=skip_first) for group in manifest["groups"]]
        write_json(directory / "result.json", preflight)
        preflight["worker_process"] = process
        write_json(root / version / "preflight-execution.json", process)
        samples = []
        for index in range(6):
            label = f"attempt-{index:02d}"
            seconds = 10.0 * (ratios or {}).get(version, 1.0)
            groups = [_group(root, source, group, preflight, installation, label, collect=False, seconds=seconds, skip_first=skip_first, successful_subtests=successful_subtests) for group in manifest["groups"]]
            sample = {"version": version, "index": index, "warmup": index == 0, "status": "completed", "seconds": 2 * seconds, "peak_rss_bytes": 100, "groups": groups,
                      "work": [{key: group.get(key) for key in ("group", "requested_nodes", "collected_nodes", "collection_sha256", "junit")} for group in groups]}
            write_json(root / version / label / "result.json", sample)
            sample["worker_process"] = process
            write_json(root / version / f"{label}-execution.json", process)
            samples.append(sample)
        versions.append({"version": version, "preflight": preflight, "samples": samples})
    metadata = {"phase": "candidate" if candidate else "baseline", "source": {"commit": "candidate" if candidate else "baseline", "production_tree": "new" if candidate else "old"},
                "source_repository": str(source), "manifest_sha256": _comparison.digest(root / "manifest.json"),
                "environment_descriptor_sha256": _comparison.digest(root / "environment-descriptor.json"),
                "runner_sha256": "runner", "shared_helper_sha256": "shared", "probe_sha256": "probe", "shared_load_note": "fixed isolated test allocation"}
    report = {"schema": "hotpot-plan20-python-matrix-report/v1", "metadata": metadata, "manifest": manifest, "versions": versions}
    write_json(root / "report.json", report)
    return root / "report.json"


def _rewrite_attempt(report_path, report, index):
    sample = report["versions"][0]["samples"][index]
    write_json(report_path.parent / "3.10" / f"attempt-{sample['index']:02d}/result.json", {key: value for key, value in sample.items() if key != "worker_process"})
    write_json(report_path, report)


def test_self_comparison_is_only_internal_consistency(tmp_path):
    report = make_report(tmp_path / "baseline")
    result = _comparison.compare_reports(report, report)
    assert result["conclusion"] == "internally_consistent"
    assert result["mode"] == "internal_consistency"
    assert all(row["paired_indices"] == [1, 2, 3, 4, 5] for row in result["versions"])
    assert "overall_family_geometric_mean" not in result


def test_successful_subtests_are_counted_without_separate_junit_cases(tmp_path):
    report = make_report(tmp_path / "baseline", successful_subtests=49)
    result = _comparison.compare_reports(report, report)
    assert result["conclusion"] == "internally_consistent"
    assert all(row["comparable"] for row in result["versions"])


def test_independent_paths_and_hotpot_version_do_not_hide_one_version_regression(tmp_path):
    before = make_report(tmp_path / "baseline")
    after = make_report(tmp_path / "candidate", candidate=True, ratios={"3.10": 0.5, "3.14": 1.5})
    result = _comparison.compare_reports(before, after)
    assert result["all_versions_comparable"]
    assert result["versions"][0]["status"] == "improved"
    assert result["versions"][-1]["status"] == "regression"
    assert result["conclusion"] == "regression"


def test_changed_skip_identity_with_equal_totals_is_not_equal_work(tmp_path):
    before = make_report(tmp_path / "baseline")
    after = make_report(tmp_path / "candidate", candidate=True, skip_first=True, ratios={version: 0.5 for version in _comparison.VERSIONS})
    result = _comparison.compare_reports(before, after)
    assert result["conclusion"] == "inconclusive"
    assert all(not row["comparable"] for row in result["versions"])
    assert any("skip work differs" in reason for reason in result["versions"][0]["reasons"])


@pytest.mark.parametrize("damage", ["missing_sample", "worker_exit", "corrupt_probe", "unexecuted", "unbound_wheel"])
def test_incomplete_or_unbound_evidence_cannot_establish_speedup(tmp_path, damage):
    before = make_report(tmp_path / "baseline")
    after = make_report(tmp_path / "candidate", candidate=True, ratios={"3.10": 0.5})
    report = _comparison.read_json(after)
    sample = report["versions"][0]["samples"][0]
    if damage == "missing_sample":
        report["versions"][0]["samples"].pop()
    elif damage == "worker_exit":
        sample["worker_process"] = {**sample["worker_process"], "returncode": -11}
        write_json(after.parent / "3.10/attempt-00-execution.json", sample["worker_process"])
    elif damage in {"corrupt_probe", "unexecuted"}:
        group = sample["groups"][0]
        path = after.parent / "3.10/attempt-00/contracts/probe.json"
        probe = _comparison.read_json(path)
        probe["test_reports"] = []
        write_json(path, probe)
        if damage == "unexecuted":
            group["artifacts"] = [_record(after.parent, path) if reference["path"].endswith("/probe.json") else reference for reference in group["artifacts"]]
    else:
        preflight = report["versions"][0]["preflight"]
        path = after.parent / "3.10/preflight/installed-wheel.json"
        installation = _comparison.read_json(path)
        installation["files"][0]["sha256"] = "0" * 64
        write_json(path, installation)
        preflight["wheel"]["binding"] = _record(after.parent, path)
        write_json(after.parent / "3.10/preflight/result.json", {key: value for key, value in preflight.items() if key != "worker_process"})
    _rewrite_attempt(after, report, 0)
    result = _comparison.compare_reports(before, after)
    assert result["versions"][0]["status"] == "inconclusive"
    assert not result["versions"][0]["comparable"]
    assert result["candidate_audit"]["versions"][0]["errors"]


@pytest.mark.parametrize("change", ["dependency", "model", "affinity", "loader"])
def test_environment_changes_are_separate_comparison_failures(tmp_path, change):
    before = make_report(tmp_path / "baseline")
    after = make_report(tmp_path / "candidate", candidate=True)
    report = _comparison.read_json(after)
    preflight = report["versions"][0]["preflight"]
    if change == "dependency":
        preflight["packages"]["numpy"] = "2.6.0"
        preflight["dependency_lock_sha256"] = _comparison.json_digest(preflight["packages"])
    elif change == "model":
        path = after.parent / "other-model-identity.json"
        write_json(path, {"fixture": "different-model"})
        preflight["model_identity"] = {"path": str(path), "sha256": _comparison.digest(path)}
    elif change == "affinity":
        preflight["cpu_affinity"] = list(range(40, 48))
    else:
        preflight["environment"]["LD_PRELOAD"] = "/other/libstdc++.so"
    write_json(after.parent / "3.10/preflight/result.json", {key: value for key, value in preflight.items() if key != "worker_process"})
    write_json(after, report)
    result = _comparison.compare_reports(before, after)
    assert not result["versions"][0]["comparable"]
    assert result["versions"][1]["comparable"]


def test_missing_required_artifact_is_not_a_successful_partial_audit(tmp_path):
    report = make_report(tmp_path / "baseline")
    (report.parent / "3.10/attempt-02/contracts/junit.xml").unlink()
    result = _comparison.compare_reports(report, report)
    assert result["conclusion"] == "inconclusive"
    assert not result["baseline_audit"]["verified"]
    assert len(result["baseline_audit"]["versions"][0]["samples"]) == 6
