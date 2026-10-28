"""Audit and compare the supported-Python matrix, independently per version.

Historical wheel bindings are checked against retained wheel archives and saved
import evidence, not against a venv that may since have installed the candidate.
No cross-version average or original 71-workload weight is used.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET
from zipfile import BadZipFile, ZipFile

if __package__:
    from .compare_performance_baselines import paired_interval
    from .run_performance_baseline import digest, write_json
else:
    from compare_performance_baselines import paired_interval
    from run_performance_baseline import digest, write_json


VERSIONS = ["3.10", "3.11", "3.12", "3.13", "3.14"]
PATH_ROLES = ("PYTHONPATH", "HOTPOT_TEST_WORKDIR", "HOTPOT_MCA_MODEL_DIR",
              "HOTPOT_CBOND_MODEL_DIR", "HOTPOT_MODEL_HOME")


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def artifact_path(root, reference):
    relative = Path(reference["path"])
    path = (root / relative).resolve()
    require(not relative.is_absolute() and path.is_relative_to(root), "artifact escapes report root")
    require(digest(path) == reference["sha256"], f"artifact hash mismatch: {relative}")
    return path


def process_ok(value):
    require(value["returncode"] == 0 and not value.get("process_error"), "process did not exit successfully")
    require(math.isfinite(value["seconds"]) and value["seconds"] > 0, "process elapsed time is invalid")


def wheel_identity(root, preflight):
    wheel = preflight["wheel"]
    installation = read_json(artifact_path(root, wheel["binding"]))
    require(installation["verified"] and not installation["errors"], "historical installed-wheel binding failed")
    require(digest(wheel["path"]) == wheel["sha256"] == installation["wheel_sha256"], "retained wheel hash differs")
    files = installation["files"]
    recorded = {entry["wheel_path"]: entry["sha256"] for entry in files}
    require(len(recorded) == len(files), "duplicate installed-wheel members")
    with ZipFile(wheel["path"]) as archive:
        records = [name for name in archive.namelist() if name.endswith(".dist-info/RECORD")]
        require(len(records) == 1, "wheel must contain one distribution RECORD")
        record = archive.read(records[0])
        require(hashlib.sha256(record).hexdigest() == installation["wheel_record_sha256"], "wheel RECORD hash differs")
        expected = {}
        for name, checksum, _ in csv.reader(io.StringIO(record.decode())):
            if not checksum:
                continue
            algorithm, encoded = checksum.split("=", 1)
            require(algorithm == "sha256", "unsupported wheel RECORD digest")
            checksum = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).hex()
            require(hashlib.sha256(archive.read(name)).hexdigest() == checksum, f"wheel member hash differs: {name}")
            expected[name] = checksum
    require(recorded == expected and "hotpot/__init__.py" in expected, "installed inventory differs from wheel RECORD")
    installed = {entry["installed_path"]: entry["sha256"] for entry in files}
    package_path = next(entry["installed_path"] for entry in files if entry["wheel_path"] == "hotpot/__init__.py")
    require(len(installed) == len(files) and installation["package_file"] == package_path, "installed package path is not bound")
    return {"installed": installed, "members": expected}


def counter_signature(values):
    return [[*key, count] for key, count in sorted(Counter(values).items())]


def audit_group(root, source_root, definition, group, wheel, prefix, common_options, preflight, *, collect):
    require(group["group"] == definition["id"] and group["status"] == "completed", "pytest group did not complete")
    require(not group.get("errors"), "pytest group reported validation errors")
    references = group["artifacts"]
    paths = [artifact_path(root, reference) for reference in references]
    require(len(set(paths)) == len(paths), "duplicate artifact references")
    directory = root / prefix / definition["id"]
    required = ["probe.json", "package-binding.json", "pytest-process.json", "pytest.stdout.log", "pytest.stderr.log"]
    if not collect:
        required.append("junit.xml")
    require(all(directory / name in paths for name in required), "required pytest artifact is missing")
    process = read_json(directory / "pytest-process.json")
    process_ok(process)
    require(all(group[key] == process[key] for key in ("command", "returncode", "seconds", "peak_rss_bytes")), "pytest process record differs from report")
    recorded_root = Path(preflight["environment"]["HOTPOT_TEST_WORKDIR"]).parent
    command = [preflight["executable"], "-m", "pytest", *common_options, "-p", "_plan20_matrix_probe", "--rootdir", str(source_root), *definition["options"]]
    command += ["--collect-only"] if collect else [f"--junitxml={recorded_root / prefix / definition['id'] / 'junit.xml'}"]
    command += [str(source_root / node.split("::", 1)[0]) + "::" + node.split("::", 1)[1] for node in definition["nodeids"]]
    require(group["command"] == command, "pytest command differs from frozen selection/options")
    probe = read_json(directory / "probe.json")
    expected = definition["nodeids"]
    require(probe["exitstatus"] == 0 and Counter(probe["nodeids"]) == Counter(expected), "actual collection differs from frozen nodes")
    require(group["requested_nodes"] == group["collected_nodes"] == len(expected), "reported collection counts differ")
    require(group["collection_sha256"] == json_digest(probe["nodeids"]), "reported collection digest differs")
    binding = read_json(directory / "package-binding.json")
    loaded = binding["installed_hotpot"]
    require(loaded["verified"] and not loaded["errors"] and group["package_binding_verified"], "loaded wheel binding failed")
    modules = probe["modules"]
    require("hotpot" in modules, "pytest did not load the installed package")
    expected_paths = {path for name, path in modules.items() if name == "hotpot" or name.startswith("hotpot.")}
    observed_paths = {entry["path"] for entry in loaded["files"]}
    require(expected_paths == observed_paths, "probe and loaded-file inventory differ")
    for entry in loaded["files"]:
        require(wheel["installed"].get(entry["path"]) == entry["sha256"], "loaded file is not retained wheel content")
    legacy = binding["legacy_plain_source_modules"]
    require(set(legacy) == {name for name in modules if name.split(".")[0] in {"mca", "cbond"}}, "legacy module inventory differs")
    for name, entry in legacy.items():
        path = Path(entry["path"])
        require(entry["path"] == modules[name] and path.is_relative_to(source_root / "hotpot/cheminfo/AImodels" / name.split(".")[0]), "undeclared source import")
        require(wheel["members"].get(path.relative_to(source_root).as_posix()) == entry["sha256"], "legacy source import differs from the corresponding wheel")
    require(group["legacy_plain_source_module_count"] == len(legacy), "legacy import count differs")
    work = {"collection": probe["nodeids"]}
    if not collect:
        reports = probe["test_reports"]
        require(all(item["nodeid"] in expected and item["outcome"] in {"passed", "skipped"} for item in reports), "unexpected or failing executed node")
        executed = {item["nodeid"] for item in reports if item["when"] == "call" or (item["when"] == "setup" and item["outcome"] == "skipped")}
        teardown = {item["nodeid"] for item in reports if item["when"] == "teardown"}
        require(executed == teardown == set(expected), "collected nodes did not all execute and finish teardown")
        xml = ET.parse(directory / "junit.xml").getroot()
        totals = {key: sum(int(suite.get(key, 0)) for suite in xml.iter("testsuite")) for key in ("tests", "failures", "errors", "skipped")}
        cases = [(case.get("classname"), case.get("name"), "skipped" if case.find("skipped") is not None else "failed" if case.find("failure") is not None else "error" if case.find("error") is not None else "passed") for case in xml.iter("testcase")]
        terminal_reports = [item for item in reports if item["when"] == "call" or (item["when"] == "setup" and item["outcome"] == "skipped")]
        # Pytest counts successful subtests in the suite total without emitting
        # separate testcase elements. Their call reports still record the work.
        require(totals == group["junit"] and totals["tests"] == len(terminal_reports)
                and len(expected) <= len(cases) <= totals["tests"], "JUnit actual work differs from report/selection")
        require(not totals["failures"] and not totals["errors"] and all(case[2] in {"passed", "skipped"} for case in cases), "JUnit contains failing tests")
        require(totals["skipped"] == sum(case[2] == "skipped" for case in cases), "JUnit skipped count differs")
        work.update(reports=counter_signature((item["nodeid"], item["when"], item["outcome"]) for item in reports),
                    junit=totals, cases=counter_signature(cases))
    return work


def worker_record(root, version, label, value):
    execution = read_json(root / version / f"{label}-execution.json")
    require(value["worker_process"] == execution, "worker execution evidence differs")
    process_ok(execution)
    require(read_json(root / version / label / "result.json") == {key: item for key, item in value.items() if key != "worker_process"}, "worker result artifact differs from report")


def audit_version(root, report, entry):
    version = entry["version"]
    result = {"version": version, "errors": [], "samples": []}
    errors = result["errors"]
    definitions = report["manifest"]["groups"]
    preflight = entry.get("preflight", {})
    wheel = {"installed": {}, "members": {}}
    try:
        require(preflight["status"] == "completed" and preflight["version"] == version, "preflight did not complete")
        require(preflight["python"].split()[0].startswith(version + "."), "Python version differs")
        worker_record(root, version, "preflight", preflight)
        require(preflight["dependency_lock_sha256"] == json_digest(preflight["packages"]), "dependency version digest differs")
        allocation = next(item["cpu_affinity"] for item in report["manifest"]["versions"] if item["version"] == version)
        require(preflight["cpu_affinity"] == allocation, "actual CPU allocation differs from manifest")
        environment = preflight["environment"]
        require(all(environment[key] == value for key, value in report["manifest"]["policy"]["thread_environment"].items()), "helper thread allocation differs")
        require(environment["PYTEST_ADDOPTS"] == "" and environment["PYTHONHASHSEED"] == report["manifest"]["policy"]["python_hash_seed"], "pytest/hash configuration differs")
        workdir = Path(environment["HOTPOT_TEST_WORKDIR"])
        require(workdir.name == version and not workdir.is_relative_to(Path(report["metadata"]["source_repository"])) and Path(environment["PYTHONPATH"]) == workdir / "test-support", "installed test paths are not isolated")
        require(preflight["model_identity"]["sha256"] == digest(preflight["model_identity"]["path"]), "model identity artifact differs")
        process_ok(preflight["models_verify"])
        require(read_json(root / version / "preflight/models-verify-process.json") == preflight["models_verify"], "model verification process evidence differs")
        wheel = wheel_identity(root, preflight)
    except (KeyError, ValueError, TypeError, OSError, BadZipFile) as error:
        errors.append(f"preflight: {error}")
    for definition, group in zip(definitions, preflight.get("collection", [])):
        try:
            audit_group(root, Path(report["metadata"]["source_repository"]), definition, group, wheel, f"{version}/preflight", report["manifest"]["pytest_common_options"], preflight, collect=True)
        except (KeyError, ValueError, TypeError, OSError, ET.ParseError) as error:
            errors.append(f"preflight/{definition['id']}: {error}")
    if len(preflight.get("collection", [])) != len(definitions):
        errors.append("preflight collection groups are incomplete")
    samples = entry.get("samples", [])
    if Counter(sample.get("index") for sample in samples) != Counter(range(6)):
        errors.append("the complete warmup plus five measurement indices are required")
    for sample in samples:
        checked = {"index": sample.get("index"), "warmup": sample.get("warmup"), "reported_status": sample.get("status"),
                   "seconds": sample.get("seconds"), "peak_rss_bytes": sample.get("peak_rss_bytes"), "errors": [], "work": []}
        result["samples"].append(checked)
        try:
            require(sample["version"] == version and sample["status"] == "completed", "attempt did not complete")
            require(sample["warmup"] == (sample["index"] == 0), "warmup classification differs")
            worker_record(root, version, f"attempt-{sample['index']:02d}", sample)
            require(len(sample["groups"]) == len(definitions), "attempt groups are incomplete")
            require(math.isfinite(sample["seconds"]) and sample["seconds"] > 0 and sample["peak_rss_bytes"] > 0, "attempt timing/RSS is invalid")
            require(math.isclose(sample["seconds"], sum(group["seconds"] for group in sample["groups"]), rel_tol=1e-12), "attempt time does not include both pytest processes")
            require(sample["peak_rss_bytes"] == max(group["peak_rss_bytes"] for group in sample["groups"]), "attempt RSS differs from subprocess evidence")
            work = [{key: group.get(key) for key in ("group", "requested_nodes", "collected_nodes", "collection_sha256", "junit")} for group in sample["groups"]]
            require(sample["work"] == work, "attempt work summary differs from groups")
        except (KeyError, ValueError, TypeError, OSError) as error:
            checked["errors"].append(str(error))
        for definition, group in zip(definitions, sample.get("groups", [])):
            try:
                work = audit_group(root, Path(report["metadata"]["source_repository"]), definition, group, wheel, f"{version}/attempt-{sample['index']:02d}", report["manifest"]["pytest_common_options"], preflight, collect=False)
                checked["work"].append({"group": definition["id"], "sha256": json_digest(work), "junit": work["junit"]})
            except (KeyError, ValueError, TypeError, OSError, ET.ParseError) as error:
                checked["errors"].append(f"{definition['id']}: {error}")
        if checked["errors"]:
            errors.append(f"attempt {sample.get('index')}: " + "; ".join(checked["errors"]))
    result["verified"] = not errors
    return result


def audit_report(path):
    path = Path(path).resolve()
    report = read_json(path)
    root = path.parent
    errors = []
    try:
        require(report["schema"] == "hotpot-plan20-python-matrix-report/v1", "unsupported matrix report schema")
        manifest, metadata = report["manifest"], report["metadata"]
        require(read_json(root / "manifest.json") == manifest, "copied manifest differs from report")
        require(digest(root / "manifest.json") == metadata["manifest_sha256"], "manifest artifact hash differs")
        require(digest(root / "environment-descriptor.json") == metadata["environment_descriptor_sha256"], "environment descriptor artifact differs")
        require([item["version"] for item in manifest["versions"]] == VERSIONS, "all five supported versions are required")
        require(Counter(entry["version"] for entry in report["versions"]) == Counter(VERSIONS), "report versions are absent or duplicated")
        require(manifest["policy"]["warmups"] == 1 and manifest["policy"]["samples"] == 5, "frozen sampling policy requires one warmup and five measurements")
        nodes = [node for group in manifest["groups"] for node in group["nodeids"]]
        require(len(nodes) == len(set(nodes)) == manifest["total_common_nodes"] and len(manifest["groups"]) == 2, "common node inventory is inconsistent")
        cpus = [cpu for item in manifest["versions"] for cpu in item["cpu_affinity"]]
        require(len(cpus) == len(set(cpus)) and all(len(item["cpu_affinity"]) == 8 for item in manifest["versions"]), "CPU allocations overlap or differ from the frozen eight CPUs")
        if metadata["phase"] == "baseline":
            require(metadata["source"]["commit"] == manifest["baseline_source_commit"], "baseline source commit differs")
        require(metadata.get("shared_load_note") and "not assessed" not in metadata["shared_load_note"], "shared host load was not assessed")
    except (KeyError, ValueError, TypeError, OSError) as error:
        errors.append(str(error))
    versions = []
    for entry in report.get("versions", []):
        try:
            versions.append(audit_version(root, report, entry))
        except (KeyError, ValueError, TypeError, OSError) as error:
            versions.append({"version": entry.get("version"), "verified": False, "samples": [], "errors": [str(error)]})
    return report, {"report": str(path), "sha256": digest(path), "errors": errors, "versions": versions,
                    "verified": not errors and len(versions) == 5 and all(version["verified"] for version in versions)}


def environment_identity(preflight):
    environment = dict(preflight["environment"])
    for key in PATH_ROLES:
        environment.pop(key)
    python_bin = str(Path(preflight["executable"]).parent)
    environment["PATH"] = ["<python-bin>" if part == python_bin else part for part in environment["PATH"].split(":")]
    return {"python": preflight["python"], "platform": preflight["platform"], "hostname": preflight["hostname"],
            "cpu_affinity": preflight["cpu_affinity"], "environment": environment,
            "dependencies": {name: version for name, version in preflight["packages"].items() if name.lower().replace("_", "-") != "hotpot-zzy"},
            "model_identity_sha256": preflight["model_identity"]["sha256"]}


def compare_reports(baseline_path, candidate_path):
    before, ba = audit_report(baseline_path)
    after, ca = (before, ba) if Path(baseline_path).resolve() == Path(candidate_path).resolve() else audit_report(candidate_path)
    policy = before["manifest"]["policy"]
    global_reasons = [f"baseline: {reason}" for reason in ba["errors"]] + [f"candidate: {reason}" for reason in ca["errors"]]
    for key in ("manifest_sha256", "runner_sha256", "shared_helper_sha256", "probe_sha256"):
        if not before["metadata"].get(key) or before["metadata"].get(key) != after["metadata"].get(key):
            global_reasons.append(f"{key} differs or is absent")
    rows = []
    for version in VERSIONS:
        row = {"version": version, "comparable": False, "status": "inconclusive", "reasons": list(global_reasons), "paired_indices": []}
        rows.append(row)
        entries = [next((entry for entry in report["versions"] if entry["version"] == version), None) for report in (before, after)]
        audits = [next((entry for entry in audit["versions"] if entry["version"] == version), None) for audit in (ba, ca)]
        if any(entry is None for entry in (*entries, *audits)):
            row["reasons"].append("version report or evidence is missing")
            continue
        for label, audit in zip(("baseline", "candidate"), audits):
            row["reasons"].extend(f"{label}: {error}" for error in audit["errors"])
        try:
            old_identity, new_identity = (environment_identity(entry["preflight"]) for entry in entries)
            row["environment_mismatches"] = [key for key in old_identity if old_identity[key] != new_identity[key]]
            row["reasons"].extend(f"environment {key} differs" for key in row["environment_mismatches"])
        except (KeyError, TypeError) as error:
            row["reasons"].append(f"environment identity incomplete: {error}")
        pairs = []
        indexed = [{sample["index"]: sample for sample in audit["samples"]} for audit in audits]
        for index in range(6):
            old, new = (samples.get(index) for samples in indexed)
            if old is None or new is None or old["errors"] or new["errors"]:
                row["reasons"].append(f"attempt {index} has incomplete/failed evidence")
            elif old["work"] != new["work"]:
                row["reasons"].append(f"attempt {index} collection/execution/skip work differs")
            elif index:
                row["paired_indices"].append(index)
                pairs.append((old["seconds"], new["seconds"]))
        if len(pairs) == 5:
            old, new = zip(*pairs)
            ratio = statistics.median(new) / statistics.median(old)
            interval = paired_interval(old, new, policy)
            row.update(runtime_ratio=ratio, paired_bootstrap_95ci=interval, paired_seconds=pairs,
                       peak_rss_ratio=max(sample["peak_rss_bytes"] for sample in audits[1]["samples"] if not sample["warmup"]) / max(sample["peak_rss_bytes"] for sample in audits[0]["samples"] if not sample["warmup"]))
            if not row["reasons"]:
                row["comparable"] = True
                low, high = policy["neutral_band"]
                row["status"] = "improved" if interval[1] < low else "regression" if interval[0] > high else "parity" if low <= ratio <= high and interval[0] >= low and interval[1] <= high else "inconclusive"
                if row["status"] == "inconclusive":
                    row["reasons"].append("uncertainty does not establish improvement, regression, or parity")
    self_comparison = Path(baseline_path).resolve() == Path(candidate_path).resolve() or before["metadata"]["source"]["production_tree"] == after["metadata"]["source"]["production_tree"]
    complete = all(row["comparable"] for row in rows)
    conclusion = "internally_consistent" if self_comparison and complete else "regression" if any(row["status"] == "regression" for row in rows) else "no_regression_established" if complete and all(row["status"] in {"improved", "parity"} for row in rows) else "inconclusive"
    return {"schema": "hotpot-plan20-python-matrix-comparison/v1", "mode": "internal_consistency" if self_comparison else "baseline_candidate",
            "baseline_audit": ba, "candidate_audit": ca, "policy": policy, "versions": rows,
            "all_versions_comparable": complete, "conclusion": conclusion,
            "path_comparison": {"excluded_after_own_binding": [*PATH_ROLES, "executable", "wheel.path", "installed_path", "source_repository"],
                                "normalized": "PATH interpreter-bin entries; other PATH/loader entries remain configuration identity",
                                "dependency_scope": "Exact recorded package versions excluding the tested hotpot-zzy version; no claim that equal versions establish equal dependency binaries."},
            "interpretation": "Self/source-identical comparison establishes internal consistency only, never F20 improvement. Every Python version is a separate gate; improvements cannot offset another version's regression. No 71-workload weight or cross-version average is applied. Historical loader/dependency binary and shared-load evidence still require review."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        result = compare_reports(args.baseline, args.candidate)
    except (KeyError, ValueError, TypeError, OSError) as error:
        result = {"schema": "hotpot-plan20-python-matrix-comparison/v1", "conclusion": "inconclusive", "input_error": str(error), "versions": []}
    write_json(args.output / "comparison.json", result)
    lines = ["# Plan20 supported-Python comparison", "", result.get("interpretation", result.get("input_error", "")), "",
             "| Python | Pairs | Candidate / baseline | Paired 95% CI | Status |", "|---|---:|---:|---|---|"]
    for row in result["versions"]:
        lines.append(f"| {row['version']} | {len(row['paired_indices'])} | {row.get('runtime_ratio', '—')} | {row.get('paired_bootstrap_95ci', '—')} | {row['status']} |")
    lines += ["", f"Conclusion: {result['conclusion']}.", "", "All missing/failed evidence and exclusions are retained in comparison.json."]
    (args.output / "comparison.md").write_text("\n".join(lines) + "\n")
    print(args.output / "comparison.json")
    return int(result["conclusion"] not in {"internally_consistent", "no_regression_established"})


if __name__ == "__main__":
    raise SystemExit(main())
