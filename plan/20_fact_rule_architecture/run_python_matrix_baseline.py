"""Measure the frozen supported-Python contracts with prepared installed wheels.

No environments, packages or models are installed by this tool. Example input::

    {"model_identity": "/artifacts/model-identity.json",
     "environment": {"HOTPOT_MCA_MODEL_DIR": "/models/mca",
                     "HOTPOT_CBOND_MODEL_DIR": "/models/cbond",
                     "HOTPOT_MODEL_HOME": "/artifacts/model-cache"},
     "versions": [{"version": "3.10", "python": "/env310/bin/python",
                   "wheel": "/wheels/hotpot-cp310.whl"}, ...]}

Run outside the checkout using --repository, --environments and --output.
All five interpreters complete preflight before six synchronized matrix rounds.
Only the two pytest subprocess times enter each version's measurement.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET


HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "_plan20_performance_report", HERE / "run_performance_baseline.py"
)
_shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared)
digest, write_json, percentile = _shared.digest, _shared.write_json, _shared.percentile

PROBE = '''"""Record actual pytest collection and import paths without hashing in timing."""
import json
import os
from pathlib import Path
import sys

_reports = []

def pytest_runtest_logreport(report):
    _reports.append({"nodeid": report.nodeid, "when": report.when,
                     "outcome": report.outcome})

def pytest_sessionfinish(session, exitstatus):
    modules = {name: str(Path(module.__file__).resolve())
               for name, module in tuple(sys.modules.items())
               if getattr(module, "__file__", None)
               and name.split(".")[0] in {"hotpot", "mca", "cbond"}}
    Path(os.environ["HOTPOT_MATRIX_PROBE"]).write_text(json.dumps({
        "nodeids": [item.nodeid for item in session.items],
        "exitstatus": int(exitstatus), "modules": modules,
        "test_reports": _reports,
    }, indent=2) + "\\n")
'''


def read_json(path):
    return json.loads(Path(path).read_text())


def artifact(path, root):
    path = Path(path)
    return {"path": path.relative_to(root).as_posix(), "sha256": digest(path)}


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def git_identity(repository):
    def git(*parts):
        return subprocess.check_output(["git", "-C", str(repository), *parts], text=True).strip()
    return {"commit": git("rev-parse", "HEAD"), "production_tree": git("rev-parse", "HEAD:hotpot"),
            "status": git("status", "--porcelain")}


def process(command, directory, environment, *, name):
    """wait4 gives this subprocess's resource record, not a running cumulative peak."""
    result = {"command": command, "returncode": None, "peak_rss_bytes": None}
    with (directory / f"{name}.stdout.log").open("w") as stdout, (directory / f"{name}.stderr.log").open("w") as stderr:
        started = time.perf_counter()
        try:
            child = subprocess.Popen(command, cwd=directory, env=environment, stdout=stdout, stderr=stderr)
            _, status, usage = os.wait4(child.pid, 0)
            elapsed = time.perf_counter() - started
            child.returncode = os.waitstatus_to_exitcode(status)
            result.update(returncode=child.returncode, peak_rss_bytes=usage.ru_maxrss * 1024,
                          user_cpu_seconds=usage.ru_utime, system_cpu_seconds=usage.ru_stime)
        except OSError as error:
            elapsed = time.perf_counter() - started
            result["process_error"] = repr(error)
        result["seconds"] = elapsed
    write_json(directory / f"{name}-process.json", result)
    return result


def configured_environment(args, manifest, descriptor, version):
    item = next(item for item in descriptor["versions"] if item["version"] == version)
    environment = os.environ.copy()
    environment.update(descriptor.get("environment", {}))
    environment.update(item.get("environment", {}))
    environment.update(manifest["policy"]["thread_environment"])
    directory = args.output / version
    environment.update({
        "PATH": str(Path(item["python"]).absolute().parent) + os.pathsep + environment.get("PATH", ""),
        "PYTHONPATH": str(directory / "test-support"), "PYTHONSAFEPATH": "1",
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": manifest["policy"]["python_hash_seed"],
        "HOTPOT_TEST_INSTALLED_PACKAGE": "1", "HOTPOT_TEST_WHEEL": str(Path(item["wheel"]).resolve()),
        "HOTPOT_TEST_WORKDIR": str(directory), "PYTEST_ADDOPTS": "",
        "HOTPOT_MODEL_SOURCE": "local", "MPLCONFIGDIR": str(directory / "matplotlib"),
        "TMPDIR": str(directory / "tmp"),
    })
    return item, environment


def run_pytest(args, manifest, group, directory, environment, installation, *, collect=False):
    directory.mkdir(parents=True, exist_ok=False)
    expected = group["nodeids"]
    targets = [str(args.repository / node.split("::", 1)[0]) + "::" + node.split("::", 1)[1]
               for node in expected]
    probe_path = directory / "probe.json"
    environment = {**environment, "HOTPOT_MATRIX_PROBE": str(probe_path)}
    command = [sys.executable, "-m", "pytest", *manifest["pytest_common_options"],
               "-p", "_plan20_matrix_probe", "--rootdir", str(args.repository), *group["options"]]
    command += ["--collect-only"] if collect else [f"--junitxml={directory / 'junit.xml'}"]
    result = process(command + targets, directory, environment, name="pytest")
    result.update(group=group["id"], requested_nodes=len(expected), status="error")
    errors = []
    try:
        if not probe_path.is_file():
            errors.append("pytest did not return collection/import evidence")
        else:
            probe = read_json(probe_path)
            observed = probe["nodeids"]
            result.update(collected_nodes=len(observed), collection_sha256=json_digest(observed),
                          missing_nodes=list((Counter(expected) - Counter(observed)).elements()),
                          extra_nodes=list((Counter(observed) - Counter(expected)).elements()))
            if result["missing_nodes"] or result["extra_nodes"]:
                errors.append("actual collection differs from the frozen common selection")
            if not collect:
                executed = {report["nodeid"] for report in probe["test_reports"]
                            if report["when"] == "call" or
                            (report["when"] == "setup" and report["outcome"] in {"skipped", "failed"})}
                result["unexecuted_nodes"] = sorted(set(expected) - executed)
                if result["unexecuted_nodes"]:
                    errors.append("collected tests did not all execute or report a setup outcome")
            modules = probe["modules"]
            loaded = [path for name, path in modules.items() if name == "hotpot" or name.startswith("hotpot.")]
            binding = _shared.verify_loaded_package_files(installation, loaded)
            if "hotpot" not in modules or not binding["verified"]:
                errors.append("pytest Hotpot imports are not bound to the verified wheel")
            legacy = {name: {"path": path, "sha256": digest(path)} for name, path in modules.items()
                      if name.split(".")[0] in {"mca", "cbond"}}
            if any(not Path(value["path"]).is_relative_to(args.repository / "hotpot/cheminfo/AImodels" / name.split(".")[0])
                   for name, value in legacy.items()):
                errors.append("plain mca/cbond import is outside its declared source exception")
            write_json(directory / "package-binding.json", {
                "installed_hotpot": binding, "legacy_plain_source_modules": legacy,
                "note": "The original mca/cbond test conftests inject source module paths; these are explicit baseline exceptions.",
            })
            result["package_binding_verified"] = "hotpot" in modules and binding["verified"]
            result["legacy_plain_source_module_count"] = len(legacy)
        if not collect:
            xml = directory / "junit.xml"
            if not xml.is_file():
                errors.append("pytest did not write its JUnit report")
            else:
                suites = list(ET.parse(xml).getroot().iter("testsuite"))
                result["junit"] = {key: sum(int(suite.get(key, 0)) for suite in suites)
                                   for key in ("tests", "failures", "errors", "skipped")}
                if result["junit"]["failures"] or result["junit"]["errors"]:
                    errors.append("JUnit reports failing technical tests")
                if result["junit"]["tests"] < len(expected):
                    errors.append("JUnit reports fewer tests than the frozen selection")
    except Exception as error:
        errors.append(f"report validation failed: {error!r}")
        result["validation_traceback"] = traceback.format_exc()
    result["errors"] = errors
    if result["returncode"] == 0 and not errors:
        result["status"] = "completed"
    result["artifacts"] = [artifact(path, args.output) for path in sorted(directory.iterdir()) if path.is_file()]
    write_json(directory / "result.json", result)
    return result


def preflight(args, manifest, descriptor, version, environment):
    directory = args.output / version / "preflight"
    directory.mkdir(parents=True, exist_ok=False)
    item = next(item for item in descriptor["versions"] if item["version"] == version)
    actual_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual_version != version:
        raise RuntimeError(f"requested Python {version}, running {actual_version}")
    installation = _shared.installed_wheel_identity(Path(item["wheel"]))
    write_json(directory / "installed-wheel.json", installation)
    if not installation["verified"]:
        raise RuntimeError("installed bytes do not match the supplied wheel")
    model_identity = read_json(descriptor["model_identity"])
    for model, variable in (("mca", "HOTPOT_MCA_MODEL_DIR"), ("cbond", "HOTPOT_CBOND_MODEL_DIR")):
        model_root = Path(environment[variable])
        for relative, record in model_identity[model]["files"].items():
            path = model_root / relative
            if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
                raise RuntimeError(f"{model} model identity differs: {relative}")
    verify = process([sys.executable, "-m", "hotpot", "models", "verify", "--all"],
                     directory, environment, name="models-verify")
    if verify["returncode"] != 0:
        raise RuntimeError("installed model verification failed; see models-verify logs")
    groups = [run_pytest(args, manifest, group, directory / group["id"], environment, installation, collect=True)
              for group in manifest["groups"]]
    packages = {distribution.metadata["Name"]: distribution.version
                for distribution in importlib.metadata.distributions() if distribution.metadata["Name"]}
    state = {
        "status": "completed" if all(group["status"] == "completed" for group in groups) else "error",
        "version": version, "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "hostname": platform.node(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "packages": packages,
        "dependency_lock_sha256": json_digest(packages),
        "environment": {key: environment.get(key) for key in (
            *manifest["policy"]["thread_environment"], "PATH", "PYTHONPATH", "PYTHONSAFEPATH",
            "PYTHONNOUSERSITE", "PYTHONHASHSEED", "LD_PRELOAD", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES",
            "HOTPOT_TEST_WORKDIR", "PYTEST_ADDOPTS",
            "HOTPOT_MCA_MODEL_DIR", "HOTPOT_CBOND_MODEL_DIR", "HOTPOT_MODEL_HOME")},
        "wheel": {"path": str(Path(item["wheel"]).resolve()), "sha256": installation["wheel_sha256"],
                  "binding": artifact(directory / "installed-wheel.json", args.output)},
        "model_identity": {"path": descriptor["model_identity"], "sha256": digest(descriptor["model_identity"])},
        "models_verify": verify, "collection": groups,
    }
    write_json(directory / "result.json", state)
    return state


def attempt(args, manifest, version, environment):
    directory = args.output / version / f"attempt-{args.index:02d}"
    directory.mkdir(parents=True, exist_ok=False)
    installation = read_json(args.output / version / "preflight/installed-wheel.json")
    groups = [run_pytest(args, manifest, group, directory / group["id"], environment, installation)
              for group in manifest["groups"]]
    result = {"version": version, "index": args.index, "warmup": args.index < manifest["policy"]["warmups"],
              "status": "completed" if all(group["status"] == "completed" for group in groups) else "error",
              "seconds": sum(group["seconds"] for group in groups),
              "peak_rss_bytes": max((group["peak_rss_bytes"] for group in groups if group["peak_rss_bytes"] is not None), default=None), "groups": groups,
              "work": [{key: group.get(key) for key in ("group", "requested_nodes", "collected_nodes", "collection_sha256", "junit")}
                       for group in groups]}
    write_json(directory / "result.json", result)
    return result


def child_task(args, manifest, descriptor):
    allocation = next(item["cpu_affinity"] for item in manifest["versions"] if item["version"] == args.worker)
    os.sched_setaffinity(0, allocation)
    _, environment = configured_environment(args, manifest, descriptor, args.worker)
    os.environ.clear()
    os.environ.update(environment)
    directory = args.output / args.worker / ("preflight" if args.index is None else f"attempt-{args.index:02d}")
    try:
        result = preflight(args, manifest, descriptor, args.worker, environment) if args.index is None else attempt(args, manifest, args.worker, environment)
    except Exception as error:
        result = {"status": "error", "version": args.worker, "index": args.index,
                  "warmup": args.index is not None and args.index < manifest["policy"]["warmups"],
                  "seconds": None, "error": repr(error), "traceback": traceback.format_exc(),
                  "missing_evidence": "Worker did not complete its report; all existing group artifacts remain available."}
        write_json(directory / "result.json", result)
    return int(result["status"] != "completed")


def summarize(samples):
    measured = [sample for sample in samples if not sample["warmup"]]
    complete = [sample for sample in measured if sample["status"] == "completed"]
    values = [sample["seconds"] for sample in complete]
    result = {"requested_samples": len(measured), "completed": len(complete),
              "errors": sum(sample["status"] == "error" for sample in measured),
              "not_run": sum(sample["status"] == "not_run" for sample in measured),
              "all_attempt_seconds": sum(sample["seconds"] for sample in measured if sample.get("seconds") is not None),
              "elapsed_unavailable": sum(sample.get("seconds") is None for sample in measured)}
    if values:
        median = statistics.median(values)
        result.update(median_seconds=median, mad_seconds=statistics.median(abs(value - median) for value in values),
                      p95_seconds=percentile(values, 0.95), peak_rss_bytes=max(sample["peak_rss_bytes"] for sample in complete))
    return result


def bind_worker_execution(result, execution):
    result = {**result, "worker_process": execution}
    if execution["returncode"] != 0 or execution.get("process_error"):
        result.update(status="error", worker_error="Worker did not exit successfully; its partial report and process evidence are retained.")
    return result


def run(args, manifest, descriptor):
    expected = [item["version"] for item in manifest["versions"]]
    if sorted(item["version"] for item in descriptor["versions"]) != sorted(expected):
        raise ValueError("environment descriptor must provide each frozen Python version exactly once")
    requested_cpus = [cpu for item in manifest["versions"] for cpu in item["cpu_affinity"]]
    if len(set(requested_cpus)) != len(requested_cpus) or not set(requested_cpus) <= os.sched_getaffinity(0):
        raise ValueError("matrix CPU allocations overlap or are unavailable to the controller")
    if args.output.is_relative_to(args.repository):
        raise ValueError("installed matrix output must be outside the source checkout")
    args.output.mkdir(parents=True, exist_ok=False)
    source = git_identity(args.repository)
    if args.phase == "baseline" and source["commit"] != manifest["baseline_source_commit"]:
        raise ValueError("baseline source checkout is not the frozen selection commit")
    if args.phase == "baseline" and subprocess.call(
        ["git", "-C", str(args.repository), "diff", "--quiet", "HEAD", "--", "hotpot", "tests"]
    ):
        raise ValueError("baseline production or tests differ from the frozen source commit")
    for version in expected:
        support = args.output / version / "test-support"
        support.mkdir(parents=True)
        (support / "tests").symlink_to(args.repository / "tests", target_is_directory=True)
        (support / "_plan20_matrix_probe.py").write_text(PROBE)
        (args.output / version / "matplotlib").mkdir()
        (args.output / version / "tmp").mkdir()
    metadata = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), "phase": args.phase,
                "source": source, "source_repository": str(args.repository),
                "manifest_sha256": digest(args.manifest), "runner_sha256": digest(__file__),
                "shared_helper_sha256": digest(HERE / "run_performance_baseline.py"),
                "probe_sha256": hashlib.sha256(PROBE.encode()).hexdigest(),
                "environment_descriptor_sha256": digest(args.environments),
                "command": [sys.executable, *sys.argv], "load_before": os.getloadavg(),
                "shared_load_note": args.shared_load_note}
    write_json(args.output / "manifest.json", manifest)
    write_json(args.output / "environment-descriptor.json", descriptor)
    write_json(args.output / "metadata.json", metadata)

    def invoke(version, index=None):
        item, environment = configured_environment(args, manifest, descriptor, version)
        label = "preflight" if index is None else f"attempt-{index:02d}"
        command = [item["python"], str(Path(__file__).resolve()), "--repository", str(args.repository),
                   "--environments", str(args.environments), "--manifest", str(args.manifest),
                   "--output", str(args.output), "--phase", args.phase, "--worker", version]
        if index is not None:
            command += ["--index", str(index)]
        directory = args.output / version
        execution = process(command, directory, environment, name=label)
        path = directory / label / "result.json"
        result = read_json(path) if path.exists() else {"status": "error", "version": version, "index": index,
                  "warmup": index is not None and index < manifest["policy"]["warmups"], "seconds": None,
                  "missing_evidence": "Worker exited without a result; launcher logs retain the diagnostic."}
        result = bind_worker_execution(result, execution)
        write_json(directory / f"{label}-execution.json", execution)
        print(f"python={version} {label} status={result['status']}", flush=True)
        return result

    with ThreadPoolExecutor(max_workers=len(expected)) as pool:
        checks = dict(zip(expected, pool.map(invoke, expected)))
        samples = {version: [] for version in expected}
        ready = all(check["status"] == "completed" for check in checks.values())
        for index in range(manifest["policy"]["warmups"] + manifest["policy"]["samples"]):
            if ready:
                results = list(pool.map(lambda version: invoke(version, index), expected))
            else:
                results = [{"version": version, "index": index, "warmup": index < manifest["policy"]["warmups"],
                            "status": "not_run", "seconds": None,
                            "reason": "The complete five-version preflight did not pass; no partial matrix was timed."}
                           for version in expected]
            for version, result in zip(expected, results):
                samples[version].append(result)
    metadata["load_after"] = os.getloadavg()
    report = {"schema": "hotpot-plan20-python-matrix-report/v1", "metadata": metadata, "manifest": manifest,
              "versions": [{"version": version, "preflight": checks[version], "samples": samples[version],
                            "summary": summarize(samples[version])} for version in expected],
              "interpretation": "Per-version technical/performance evidence, separate from the original 71-workload aggregate. Failed or reduced test work is never a speedup."}
    write_json(args.output / "report.json", report)
    lines = ["# Plan20 supported-Python baseline", "", report["interpretation"], "",
             "| Python | Completed / measured | Median seconds | MAD seconds | Errors / not run |",
             "|---|---:|---:|---:|---:|"]
    for item in report["versions"]:
        value = item["summary"]
        lines.append(f"| {item['version']} | {value['completed']} / {value['requested_samples']} | {value.get('median_seconds', '—')} | {value.get('mad_seconds', '—')} | {value['errors']} / {value['not_run']} |")
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    return int(not ready or any(sample["status"] != "completed" for entries in samples.values() for sample in entries))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--environments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=HERE / "python_matrix_manifest.json")
    parser.add_argument("--phase", choices=("baseline", "candidate"), default="baseline")
    parser.add_argument("--shared-load-note", required=False, default="not assessed; comparison remains pending")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--index", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    for key in ("repository", "environments", "output", "manifest"):
        setattr(args, key, getattr(args, key).resolve())
    manifest, descriptor = read_json(args.manifest), read_json(args.environments)
    return child_task(args, manifest, descriptor) if args.worker else run(args, manifest, descriptor)


if __name__ == "__main__":
    raise SystemExit(main())
