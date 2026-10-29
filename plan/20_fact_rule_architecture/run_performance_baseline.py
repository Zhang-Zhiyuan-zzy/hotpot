"""Run frozen Plan20 semantic workloads and preserve every timing attempt.

Examples::

    python run_performance_baseline.py --output /tmp/b20 --only empty_molecule
    python run_performance_baseline.py --output /tmp/test-selection --freeze-tests
    python run_performance_baseline.py --output /tmp/b20 --enable models gpu tests \
        --test-selection /tmp/test-selection/test_nodeids.json --wheel package.whl

Smoke runs are explicitly non-acceptance evidence. Run formal timings only on
the recorded resource allocation, without concurrent task builds or tests.
"""

from __future__ import annotations

import argparse
import base64
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time
import traceback
import tracemalloc
import xml.etree.ElementTree as ET
import zipfile


HERE = Path(__file__).resolve().parent


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify_wheel_installation(wheel, installation_root, installed_record, package_file):
    """Bind installed bytes to the supplied wheel, outside measured workloads."""
    root = Path(installation_root).resolve()
    installed_record = Path(installed_record)
    installed_entries = {row[0]: row[1:] for row in csv.reader(io.StringIO(installed_record.read_text()))}
    files, errors = [], []
    with zipfile.ZipFile(wheel) as archive:
        record_name = next(name for name in archive.namelist() if name.endswith(".dist-info/RECORD"))
        wheel_record = archive.read(record_name)
        if installed_record.resolve() != (root / record_name).resolve():
            errors.append("installed distribution RECORD does not belong to the wheel distribution")
        for name, checksum, size in csv.reader(io.StringIO(wheel_record.decode())):
            if not checksum:
                continue  # RECORD itself is rewritten by the installer.
            installed = (root / name).resolve()
            if not installed.is_relative_to(root):
                errors.append(f"wheel member outside installation: {name}")
                continue
            algorithm, encoded = checksum.split("=", 1)
            if algorithm != "sha256":
                errors.append(f"unsupported wheel RECORD digest: {name}: {algorithm}")
                continue
            expected = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).hex()
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                errors.append(f"wheel bytes do not match wheel RECORD: {name}")
            if installed_entries.get(name) != [checksum, size]:
                errors.append(f"installed RECORD differs from wheel RECORD: {name}")
            actual = digest(installed) if installed.is_file() else None
            if actual != expected:
                errors.append(f"installed bytes do not match wheel: {name}")
            files.append({"wheel_path": name, "installed_path": str(installed), "sha256": expected})
    expected_package = (root / "hotpot/__init__.py").resolve()
    if Path(package_file).resolve() != expected_package:
        errors.append("resolved hotpot import does not belong to the verified installation")
    return {"verified": not errors, "wheel_sha256": digest(wheel), "package_file": str(Path(package_file).resolve()),
            "installed_record_sha256": digest(installed_record), "wheel_record_sha256": hashlib.sha256(wheel_record).hexdigest(),
            "files": files, "errors": errors}


def installed_wheel_identity(wheel):
    if wheel is None:
        return {"verified": False, "errors": ["installed mode requires --wheel for artifact binding"]}
    distribution = importlib.metadata.distribution("hotpot-zzy")
    record = next(path for path in distribution.files if str(path).endswith(".dist-info/RECORD"))
    return verify_wheel_installation(wheel, distribution.locate_file(""), distribution.locate_file(record),
                                     importlib.util.find_spec("hotpot").origin)


def verify_loaded_package_files(installation, paths):
    expected = {entry["installed_path"]: entry["sha256"] for entry in installation["files"]}
    files, errors = [], []
    for filename in sorted(set(paths)):
        path = Path(filename).resolve()
        if path.suffix == ".pyc":
            path = Path(importlib.util.source_from_cache(str(path))).resolve()
        actual = digest(path) if path.is_file() else None
        if str(path) not in expected or actual != expected[str(path)]:
            errors.append(f"loaded package file is not verified wheel content: {path}")
        files.append({"path": str(path), "sha256": actual})
    return {"verified": not errors, "files": files, "errors": errors, "timing_included": False}


def write_json(path, value):
    def finite(item):
        if isinstance(item, float) and not math.isfinite(item):
            return {"nonfinite": str(item)}
        if isinstance(item, dict):
            return {key: finite(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [finite(child) for child in item]
        return item
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(finite(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summary(samples):
    measured = [sample for sample in samples if not sample["warmup"]]
    completed = [sample for sample in measured if sample["status"] == "completed"]
    values = [sample["seconds"] for sample in completed]
    result = {
        "attempts": len(measured), "completed": len(completed),
        "failed": sum(sample["status"] == "error" for sample in measured),
        "no_output": sum(sample["status"] == "no_output" for sample in measured),
        "all_attempt_seconds": sum(sample["seconds"] for sample in measured),
    }
    if values:
        median = statistics.median(values)
        result.update(median_seconds=median, mad_seconds=statistics.median(abs(v - median) for v in values),
                      p95_seconds=percentile(values, 0.95), peak_rss_bytes=max(s["peak_rss_bytes"] for s in completed))
    return result


def work_counters(value):
    """Keep actual work/output counts separate from numerical observations."""
    keys = {"completed_calls", "atoms", "rows", "frames", "molecules", "requested_molecules", "returned_molecules",
            "sites", "site_count", "atom_predictions", "predictions", "rings", "path", "angles", "torsions", "pairs", "hits",
            "calls", "iterations", "iteration_count", "steps", "steps_run", "total_steps", "epochs", "attempt_count",
            "epochs_completed", "selected_segment_epochs_completed", "segment_epochs_completed", "steps_submitted",
            "initialization_steps", "steps_completed", "attempts_completed", "metal_relocation_attempt_count",
            "atom_count", "bond_count", "added_hydrogen_count", "ob_atoms", "rd_atoms",
            "tests", "failures", "errors", "skipped", "shape", "gpu_started"}
    result = {}
    def visit(item, prefix):
        if isinstance(item, dict):
            for key, child in item.items():
                path = f"{prefix}.{key}" if prefix else key
                # A reported unknown step count is evidence, not a zero or an absent field.
                if key in keys and (child is None or isinstance(child, (int, float, list))):
                    result[path] = child
                elif isinstance(child, (dict, list)):
                    visit(child, path)
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{prefix}[{index}]")
    visit(value, "")
    return result


def rss():
    status = Path("/proc/self/status").read_text().splitlines()
    current = next(int(line.split()[1]) * 1024 for line in status if line.startswith("VmRSS:"))
    return {"rss_bytes": current, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}


def gpu_state():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                                text=True, capture_output=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": False, "reason": str(error)}
    return {"available": result.returncode == 0, "returncode": result.returncode,
            "readings": result.stdout.splitlines(), "diagnostic": result.stderr.strip()}


def metadata(args, manifest):
    repo = args.repository
    def git(*parts):
        return subprocess.check_output(["git", "-C", str(repo), *parts], text=True).strip()
    names = sorted({dist.metadata["Name"] for dist in importlib.metadata.distributions() if dist.metadata["Name"]})
    packages = {name: importlib.metadata.version(name) for name in names}
    lock = json.dumps(packages, sort_keys=True).encode()
    models = {}
    for key in ("HOTPOT_MCA_MODEL_DIR", "HOTPOT_CBOND_MODEL_DIR"):
        path = os.environ.get(key)
        models[key] = {"path": path, "exists": bool(path and Path(path).is_dir())}
    cpu_models = sorted({line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")})
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "source_commit": git("rev-parse", "HEAD"),
        "production_tree": git("rev-parse", "HEAD:hotpot"), "worktree_status": git("status", "--porcelain"),
        "manifest_sha256": digest(args.manifest), "runner_sha256": digest(__file__),
        "workloads_sha256": digest(HERE / "_performance_workloads.py"),
        "wheel": {"path": str(args.wheel), "sha256": digest(args.wheel)} if args.wheel else None,
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(), "machine": platform.machine(),
        "hostname": platform.node(), "cpu_models": cpu_models, "cpu_affinity": sorted(os.sched_getaffinity(0)), "load_average_before": os.getloadavg(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "thread_environment": {key: os.environ.get(key) for key in manifest["measurement"]["thread_env"]},
        "packages": packages, "dependency_lock_sha256": hashlib.sha256(lock).hexdigest(),
        "models": models, "model_identity": {"path": str(args.model_identity), "sha256": digest(args.model_identity)} if args.model_identity else None,
        "gpu_before": gpu_state(), "shared_load_note": args.shared_load_note,
        "test_selection_sha256": digest(args.test_selection) if args.test_selection else None,
        "phase": args.phase, "package_mode": args.package_mode, "smoke": args.smoke,
        "installed_wheel": installed_wheel_identity(args.wheel) if args.package_mode == "installed" else None,
    }


def _command_sample(entry, args, sample_dir):
    observation = sample_dir / "observation.json"
    if entry["operation"] == "cold_import":
        code = ("import json,sys,resource; from pathlib import Path; import hotpot; "
                "json.dump({'loaded_backends':sorted(k for k in sys.modules if k.split('.')[0] in ('rdkit','openbabel','onnxruntime') or '_native' in k),"
                "'package_file':str(Path(hotpot.__file__).resolve()),"
                "'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024},open(sys.argv[1],'w'))")
        command = [sys.executable, "-c", code, str(observation)]
    else:
        selection = json.loads(args.test_selection.read_text())
        nodeids = selection["nodeids"]
        if args.phase == "candidate":
            original = "tests/test_ds/test_stage13_generation_efficiency.py::test_shared_asset_cache_rejects_source_change_before_reuse"
            nodeids = [node + "[False]" if node == original else node for node in nodeids]
        targets = [str(args.repository / node.split("::", 1)[0]) + ("::" + node.split("::", 1)[1] if "::" in node else "") for node in nodeids]
        command = [sys.executable, "-m", "pytest", "-q", "--import-mode=importlib", *targets]
        if entry["operation"] == "test_collection":
            command += ["--collect-only"]
        else:
            command += [f"--junitxml={sample_dir / 'junit.xml'}"]
    with (sample_dir / "stdout.log").open("w") as stdout, (sample_dir / "stderr.log").open("w") as stderr:
        started = time.perf_counter()
        process = subprocess.run(command, cwd=args.repository if args.package_mode == "source" else sample_dir,
                                 stdout=stdout, stderr=stderr, env=os.environ.copy())
        seconds = time.perf_counter() - started
    value = json.loads(observation.read_text()) if observation.exists() else {}
    if entry["operation"] == "cold_import" and value and args.package_mode == "installed" and Path(value["package_file"]).is_relative_to(args.repository):
        raise RuntimeError("Installed-package cold import loaded Hotpot from the checkout")
    if entry["operation"].startswith("test_"):
        value["tests"] = len(nodeids)
        xml = sample_dir / "junit.xml"
        if xml.exists():
            suites = list(ET.parse(xml).getroot().iter("testsuite"))
            value.update({key: sum(int(suite.get(key, 0)) for suite in suites) for key in ("tests", "failures", "errors", "skipped")})
    return {"status": "completed" if process.returncode == 0 else "error", "seconds": seconds,
            "returncode": process.returncode, "command": command, "value": value,
            "peak_rss_bytes": value.get("peak_rss_bytes", resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024),
            "work_counters": work_counters(value)}


def worker(args, manifest):
    entry = next(item for item in manifest["workloads"] if item["id"] == args.worker)
    target = args.output
    target.mkdir(parents=True, exist_ok=False)
    if args.package_mode == "source":
        sys.path.insert(0, str(args.repository))
        os.environ["PYTHONPATH"] = str(args.repository)
    policy = {"warmups": 0, "samples": 1} if args.smoke else manifest["policies"][entry["tier"]]
    report = {"id": entry["id"], "entry": entry, "samples": [], "memory_probe": None}
    close = lambda: None
    setup_started = time.perf_counter()
    try:
        if entry["operation"] in {"cold_import", "test_collection", "test_execution"}:
            run = None
        else:
            from _performance_workloads import prepare
            run, close = prepare(entry, args.phase)
            import hotpot
            report["package_file"] = str(Path(hotpot.__file__).resolve())
            if args.package_mode == "installed" and Path(hotpot.__file__).resolve().is_relative_to(args.repository):
                raise RuntimeError("Installed-package run imported Hotpot from the checkout")
        report["setup_seconds"] = time.perf_counter() - setup_started
        for index in range(policy["warmups"] + policy["samples"]):
            sample_dir = target / f"attempt-{index:03d}"
            sample_dir.mkdir()
            gpu_before = gpu_state() if "gpu" in entry.get("requires", []) else None
            started = time.perf_counter()
            try:
                if run is None:
                    sample = _command_sample(entry, args, sample_dir)
                else:
                    value = run()
                    sample = {"status": "completed" if value is not None else "no_output", "seconds": time.perf_counter() - started,
                              "value": value, "work_counters": work_counters(value), **rss()}
            except Exception as error:
                sample = {"status": "error", "seconds": time.perf_counter() - started, "error": repr(error),
                          "traceback": traceback.format_exc(), "work_counters": {}, **rss()}
            sample.update(index=index, warmup=index < policy["warmups"])
            if run is not None and hasattr(run, "runtime_allocation"):
                try:
                    sample["runtime_allocation"] = run.runtime_allocation()
                except Exception as error:
                    sample.update(status="error", allocation_error=repr(error))
            if gpu_before is not None:
                sample.update(gpu_before=gpu_before, gpu_after=gpu_state())
            report["samples"].append(sample)
            write_json(sample_dir / "result.json", sample)
        if run is not None and not args.smoke and entry["tier"] == "micro":
            tracemalloc.start()
            try:
                run()
                current, peak = tracemalloc.get_traced_memory()
                blocks = sum(item.count for item in tracemalloc.take_snapshot().statistics("filename"))
                report["memory_probe"] = {"python_tracked_retained_bytes": current, "python_tracked_peak_bytes": peak,
                                          "python_tracked_blocks": blocks, "timing_included": False, **rss()}
            except Exception as error:
                report["memory_probe"] = {"error": repr(error), "timing_included": False}
            finally:
                tracemalloc.stop()
        if args.installation_identity:
            installation = json.loads(args.installation_identity.read_text())
            loaded = [module.__file__ for name, module in sys.modules.items()
                      if (name == "hotpot" or name.startswith("hotpot.")) and getattr(module, "__file__", None)]
            loaded += [sample["value"]["package_file"] for sample in report["samples"]
                       if sample.get("value", {}).get("package_file")]
            report["loaded_package_identity"] = verify_loaded_package_files(installation, loaded)
            if not report["loaded_package_identity"]["verified"]:
                raise RuntimeError("Loaded Hotpot files do not match the verified wheel")
    except Exception as error:
        report["setup_error"] = {"error": repr(error), "traceback": traceback.format_exc()}
    finally:
        try:
            close()
        except Exception as error:
            report["teardown_error"] = repr(error)
    report["summary"] = summary(report["samples"])
    write_json(target / "result.json", report)
    return int(bool(report.get("setup_error")) or bool(report.get("teardown_error")) or report["summary"]["failed"] > 0 or report["summary"]["no_output"] > 0)


def freeze_tests(args):
    args.output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-m", "pytest", "--collect-only", "-q", "--import-mode=importlib", "tests"]
    process = subprocess.run(command, cwd=args.repository, text=True, capture_output=True)
    (args.output / "collection.stdout.log").write_text(process.stdout)
    (args.output / "collection.stderr.log").write_text(process.stderr)
    if process.returncode:
        return process.returncode
    nodeids = [line.strip() for line in process.stdout.splitlines() if line.startswith("tests/") and "::" in line]
    if not nodeids:
        raise RuntimeError("Pytest collection returned no node IDs")
    write_json(args.output / "test_nodeids.json", {"schema_version": 1, "nodeids": nodeids, "command": command,
               "source_commit": subprocess.check_output(["git", "-C", str(args.repository), "rev-parse", "HEAD"], text=True).strip()})
    print(f"Frozen {len(nodeids)} common test nodes")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=HERE / "performance_manifest.json")
    parser.add_argument("--repository", type=Path, default=HERE.parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("baseline", "candidate"), default="baseline")
    parser.add_argument("--package-mode", choices=("source", "installed"), default="source")
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--enable", nargs="*", choices=("models", "gpu", "tests"), default=[])
    parser.add_argument("--test-selection", type=Path)
    parser.add_argument("--freeze-tests", action="store_true")
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--model-identity", type=Path)
    parser.add_argument("--cpu-affinity", help="Comma-separated CPU IDs; fixed for every subprocess")
    parser.add_argument("--shared-load-note", default="not assessed; timing comparability requires review")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--installation-identity", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.repository = args.repository.resolve()
    args.output = args.output.resolve()
    args.manifest = args.manifest.resolve()
    if args.test_selection:
        args.test_selection = args.test_selection.resolve()
    if args.package_mode == "installed":
        os.environ.pop("PYTHONPATH", None)
        os.environ["HOTPOT_TEST_INSTALLED_PACKAGE"] = "1"
    if args.wheel:
        args.wheel = args.wheel.resolve()
        os.environ["HOTPOT_TEST_WHEEL"] = str(args.wheel)
    manifest = json.loads(args.manifest.read_text())
    for key, value in manifest["measurement"]["thread_env"].items():
        os.environ[key] = value
    if args.cpu_affinity:
        os.sched_setaffinity(0, {int(value) for value in args.cpu_affinity.split(",")})
    if args.worker:
        return worker(args, manifest)
    if args.freeze_tests:
        return freeze_tests(args)
    ids = {entry["id"] for entry in manifest["workloads"]}
    if args.only and set(args.only) - ids:
        parser.error(f"Unknown workload IDs: {set(args.only) - ids}")
    args.output.mkdir(parents=True, exist_ok=False)
    meta = metadata(args, manifest)
    write_json(args.output / "manifest.json", manifest)
    write_json(args.output / "environment.json", meta)
    if args.package_mode == "installed":
        write_json(args.output / "installed-wheel.json", meta["installed_wheel"])
        if not meta["installed_wheel"]["verified"]:
            write_json(args.output / "report.json", {"schema_version": 1, "metadata": meta, "manifest": manifest, "workloads": []})
            print("Installed wheel identity verification failed; see installed-wheel.json")
            return 1
    reports = []
    for entry in manifest["workloads"]:
        if args.only and entry["id"] not in args.only:
            continue
        reason = None
        if entry.get("new_only"):
            reason = "new-only mechanism; no pre-refactor counterpart; candidate binding awaits its owning stage"
        required = set(entry.get("requires", []))
        if required & {"mca_model", "cbond_model"} and "models" not in args.enable:
            reason = "model workload not enabled"
        if "gpu" in required and "gpu" not in args.enable:
            reason = "GPU workload not enabled"
        if "tests" in required and ("tests" not in args.enable or not args.test_selection):
            reason = "frozen common test selection not supplied/enabled"
        if reason:
            reports.append({"id": entry["id"], "entry": entry, "not_run": reason})
            continue
        target = args.output / entry["id"]
        command = [sys.executable, str(Path(__file__).resolve()), "--manifest", str(args.manifest), "--repository", str(args.repository),
                   "--output", str(target), "--worker", entry["id"], "--phase", args.phase, "--package-mode", args.package_mode]
        if args.smoke:
            command.append("--smoke")
        if args.test_selection:
            command += ["--test-selection", str(args.test_selection)]
        if args.package_mode == "installed":
            command += ["--installation-identity", str(args.output / "installed-wheel.json")]
        with (args.output / f"{entry['id']}.stdout.log").open("w") as stdout, (args.output / f"{entry['id']}.stderr.log").open("w") as stderr:
            started = time.perf_counter()
            result = subprocess.run(command, cwd=args.repository if args.package_mode == "source" else args.output, stdout=stdout, stderr=stderr)
            elapsed = time.perf_counter() - started
        report_path = target / "result.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {"id": entry["id"], "entry": entry, "process_error": result.returncode}
        report["worker_process_seconds"] = elapsed
        reports.append(report)
        print(entry["id"], report.get("summary", report.get("process_error")), flush=True)
    meta["gpu_after"] = gpu_state()
    meta["load_average_after"] = os.getloadavg()
    report = {"schema_version": 1, "metadata": meta, "manifest": manifest, "workloads": reports}
    write_json(args.output / "report.json", report)
    rows = ["# Plan20 performance execution", "", f"Phase: {args.phase}; smoke: {args.smoke}. Unexecuted workloads are pending evidence.", "",
            "| Workload | Completed / attempts | Median seconds | Status |", "|---|---:|---:|---|"]
    for item in reports:
        data = item.get("summary", {})
        status = (item.get("not_run") or item.get("setup_error", {}).get("error") or item.get("teardown_error") or
                  ("error" if item.get("process_error") or data.get("failed") else "no_output" if data.get("no_output") else "recorded"))
        rows.append(f"| {item['id']} | {data.get('completed', 0)} / {data.get('attempts', 0)} | {data.get('median_seconds', '—')} | {status} |")
    (args.output / "report.md").write_text("\n".join(rows) + "\n")
    return int(any(item.get("setup_error") or item.get("teardown_error") or item.get("process_error") or
                   item.get("summary", {}).get("failed") or item.get("summary", {}).get("no_output") for item in reports))


if __name__ == "__main__":
    raise SystemExit(main())
