"""Compare Plan20 timing records without crediting reduced or failed work."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import statistics

if __package__:
    from .run_performance_baseline import percentile, write_json
else:
    from run_performance_baseline import percentile, write_json


def geometric_mean(values):
    return math.exp(statistics.mean(math.log(value) for value in values))


def paired_interval(baseline, candidate, policy):
    rng = random.Random(policy["bootstrap_seed"])
    size = len(baseline)
    samples = []
    for _ in range(policy["bootstrap_samples"]):
        indices = rng.choices(range(size), k=size)
        samples.append(statistics.median(candidate[i] for i in indices) / statistics.median(baseline[i] for i in indices))
    tail = (1 - policy["confidence_level"]) / 2
    return [percentile(samples, tail), percentile(samples, 1 - tail)]


def group_interval(rows, policy):
    rng = random.Random(policy["bootstrap_seed"])
    draws = []
    for _ in range(policy["bootstrap_samples"]):
        ratios = []
        for row in rows:
            pairs = row["paired_seconds"]
            selected = rng.choices(pairs, k=len(pairs))
            ratios.append(statistics.median(pair[1] for pair in selected) / statistics.median(pair[0] for pair in selected))
        draws.append(geometric_mean(ratios))
    tail = (1 - policy["confidence_level"]) / 2
    return [percentile(draws, tail), percentile(draws, 1 - tail)]


def selected_gpu_identity(snapshot, indices, visible_devices):
    """Resolve logical device IDs to stable physical identity, excluding load."""
    if not snapshot or not snapshot.get("available"):
        raise ValueError("GPU identity snapshot unavailable")
    devices = {}
    for fields in csv.reader(snapshot.get("readings", []), skipinitialspace=True):
        if len(fields) < 5:
            raise ValueError("GPU identity snapshot is incomplete")
        index, uuid, name, driver, memory = (field.strip() for field in fields[:5])
        devices[index] = {"uuid": uuid, "name": name, "driver_version": driver, "memory_total_mib": memory}
    visible = visible_devices.split(",") if visible_devices is not None else None
    selected = []
    for index in indices:
        try:
            selector = visible[index].strip() if visible is not None else str(index)
            if selector.startswith("GPU-"):
                matches = [value for value in devices.values() if value["uuid"].startswith(selector)]
                if len(matches) != 1:
                    raise ValueError("GPU UUID selection is absent or ambiguous")
                device = matches[0]
            else:
                device = devices[selector]
        except (IndexError, KeyError) as error:
            raise ValueError("allocated GPU absent from identity snapshot") from error
        selected.append({"logical_index": index, **device})
    return selected


def gpu_workload_identity(entry, metadata, workload):
    """Check observed allocation and every before/after snapshot for this task."""
    params = entry["parameters"]
    indices = params.get("gpu_indices", [0])
    samples = [sample for sample in workload.get("samples", []) if not sample["warmup"] and sample["status"] == "completed"]
    if not samples:
        raise ValueError("no completed GPU samples with allocation evidence")
    allocations = []
    for sample in samples:
        value = sample.get("value", {}).get("value", {})
        if entry["operation"] == "mca":
            started = value.get("gpu_started")
            plan = value.get("resource_plan")
            if entry["hardware_mode"] == "1gpu_fallback":
                if started is not False or plan is not None:
                    raise ValueError("below-threshold workload did not retain CPU fallback")
                allocations.append({"effective_backend": "cpu_fallback"})
                continue
            if started is not True or not plan:
                raise ValueError("MCA CUDA startup/allocation evidence missing")
            if plan.get("gpu_indices") != indices or any(plan.get(key) != params[key] for key in ("site_cpu_per_gpu", "infer_cpu_per_gpu")):
                raise ValueError("MCA actual allocation differs from the frozen workload")
            blocks = plan.get("gpu_cpu_assignments", [])
            assigned = [cpu for block in blocks for cpu in block["cpu_ids"]]
            if ([block["gpu_index"] for block in blocks] != indices or len(set(assigned)) != len(assigned)
                    or not set(assigned).issubset(metadata.get("cpu_affinity", []))):
                raise ValueError("MCA actual CPU/GPU assignment is incomplete or inconsistent")
            for block in blocks:
                if (len(block["site_cpu_ids"]) != params["site_cpu_per_gpu"]
                        or len(block["infer_cpu_ids"]) != params["infer_cpu_per_gpu"]
                        or block["cpu_ids"] != block["site_cpu_ids"] + block["infer_cpu_ids"]):
                    raise ValueError("MCA actual CPU allocation differs from the frozen workload")
            allocations.append({"effective_backend": "gpu", "resource_plan": plan})
        elif entry["operation"] == "cbond":
            allocation = sample.get("runtime_allocation", {})
            if value.get("device") != "cuda" or allocation.get("device") != "cuda":
                raise ValueError("CBond actual CUDA allocation evidence missing")
            sessions = allocation.get("sessions", {})
            for name in ("graph", "cbond"):
                session = sessions.get(name, {})
                options = session.get("provider_options", {}).get("CUDAExecutionProvider", {})
                if "CUDAExecutionProvider" not in session.get("providers", []) or str(options.get("device_id")) != "0":
                    raise ValueError("CBond session did not use the frozen CUDA device")
            allocations.append(allocation)
    if any(allocation != allocations[0] for allocation in allocations[1:]):
        raise ValueError("actual GPU allocation changed during the workload")
    if entry["hardware_mode"] == "1gpu_fallback":
        return {"devices": [], "allocation": allocations[0]}
    identity = selected_gpu_identity(metadata.get("gpu_before"), indices, metadata.get("cuda_visible_devices"))
    snapshots = [metadata.get("gpu_after")]
    snapshots += [sample.get(key) for sample in samples for key in ("gpu_before", "gpu_after")]
    for snapshot in snapshots:
        if selected_gpu_identity(snapshot, indices, metadata.get("cuda_visible_devices")) != identity:
            raise ValueError("allocated GPU identity changed during the workload")
    return {"devices": identity, "allocation": allocations[0], "cuda_device_order": metadata.get("cuda_device_order")}


def compare(before, after):
    bm, cm = before["metadata"], after["metadata"]
    policy = before["manifest"]["policies"]
    mismatches = []
    for key in ("manifest_sha256", "python", "platform", "machine", "hostname", "cpu_models", "cpu_affinity", "thread_environment", "dependency_lock_sha256", "package_mode"):
        if bm.get(key) != cm.get(key):
            mismatches.append(key)
    if bm.get("smoke") or cm.get("smoke"):
        mismatches.append("smoke runs are not acceptance evidence")
    for label, metadata in (("baseline", bm), ("candidate", cm)):
        if metadata.get("package_mode") == "installed":
            installation = metadata.get("installed_wheel") or {}
            if not installation.get("verified") or installation.get("wheel_sha256") != (metadata.get("wheel") or {}).get("sha256"):
                mismatches.append(f"{label} installed package is not verified against its wheel")
    bwork = {row["id"]: row for row in before["workloads"]}
    cwork = {row["id"]: row for row in after["workloads"]}
    rows = []
    for entry in before["manifest"]["workloads"]:
        name = entry["id"]
        row = {"id": name, "family": entry["family"], "hardware_mode": entry["hardware_mode"],
               "end_to_end": entry["end_to_end"], "weight": entry["weight"], "aggregate_runtime": entry.get("aggregate_runtime", True),
               "baseline_attempts": bwork.get(name, {}).get("summary", {}), "candidate_attempts": cwork.get(name, {}).get("summary", {})}
        if entry.get("new_only"):
            row.update(status="new_only", comparable=False, reason="No baseline counterpart; excluded from speedup aggregation", candidate=cwork.get(name))
            rows.append(row)
            continue
        reasons = list(mismatches)
        if name not in bwork or name not in cwork:
            reasons.append("workload report missing")
        b = bwork.get(name, {})
        c = cwork.get(name, {})
        if b.get("not_run") or c.get("not_run"):
            reasons.append("workload not executed on both sides")
        if b.get("setup_error") or c.get("setup_error") or b.get("process_error") or c.get("process_error"):
            reasons.append("workload setup/process failed")
        if b.get("teardown_error") or c.get("teardown_error"):
            reasons.append("workload teardown failed")
        if bm.get("package_mode") == "installed" and (not b.get("loaded_package_identity", {}).get("verified") or not c.get("loaded_package_identity", {}).get("verified")):
            reasons.append("loaded package files are not verified wheel content")
        if "gpu" in entry.get("requires", []):
            identities = []
            for label, metadata, workload in (("baseline", bm, b), ("candidate", cm, c)):
                try:
                    identity = gpu_workload_identity(entry, metadata, workload)
                    identities.append(identity)
                    row[f"{label}_gpu_identity"] = identity
                except (ValueError, KeyError, TypeError) as error:
                    reasons.append(f"{label} GPU allocation/identity: {error}")
            if len(identities) == 2 and identities[0] != identities[1]:
                reasons.append("allocated GPU identity or actual resource allocation differs")
        if "tests" in entry.get("requires", []) and bm.get("test_selection_sha256") != cm.get("test_selection_sha256"):
            reasons.append("common test selection differs")
        if {"mca_model", "cbond_model"} & set(entry.get("requires", [])):
            if not bm.get("model_identity") or not cm.get("model_identity") or bm["model_identity"]["sha256"] != cm["model_identity"]["sha256"]:
                reasons.append("verified model identity absent or different")
        bs = {sample["index"]: sample for sample in b.get("samples", []) if not sample["warmup"]}
        cs = {sample["index"]: sample for sample in c.get("samples", []) if not sample["warmup"]}
        pairs = []
        excluded = []
        for index in sorted(set(bs) | set(cs)):
            old, new = bs.get(index), cs.get(index)
            if old is None or new is None:
                excluded.append({"index": index, "reason": "sample absent"})
            elif old["status"] != "completed" or new["status"] != "completed":
                excluded.append({"index": index, "reason": "failure/no output", "baseline_status": old["status"], "candidate_status": new["status"]})
            elif old["work_counters"] != new["work_counters"]:
                excluded.append({"index": index, "reason": "different completed work", "baseline": old["work_counters"], "candidate": new["work_counters"]})
            else:
                pairs.append((old, new))
        row.update(paired_samples=len(pairs), excluded_samples=excluded)
        minimum = policy[entry["tier"]]["samples"]
        if len(pairs) < minimum:
            reasons.append(f"only {len(pairs)} comparable pairs; frozen minimum is {minimum}")
        row["comparable"] = not reasons
        if pairs:
            old = [pair[0]["seconds"] for pair in pairs]
            new = [pair[1]["seconds"] for pair in pairs]
            ratio = statistics.median(new) / statistics.median(old)
            interval = paired_interval(old, new, policy)
            peak_ratio = max(pair[1]["peak_rss_bytes"] for pair in pairs) / max(pair[0]["peak_rss_bytes"] for pair in pairs)
            row.update(runtime_ratio=ratio, paired_bootstrap_95ci=interval, peak_rss_ratio=peak_ratio,
                       paired_indices=[pair[0]["index"] for pair in pairs], paired_seconds=list(zip(old, new)))
            low, high = policy["neutral_band"]
            row["status"] = ("inconclusive" if reasons else "improved" if interval[1] < low else
                             "regression" if interval[0] > high else "parity" if low <= ratio <= high and interval[0] >= low and interval[1] <= high else "inconclusive")
            if not reasons and row["status"] == "inconclusive":
                reasons.append("uncertainty does not establish gain, regression, or equivalence within frozen neutral band")
        else:
            row["status"] = "inconclusive"
        row["reasons"] = reasons
        rows.append(row)
    groups = {}
    for row in rows:
        if row["status"] == "new_only" or not row["aggregate_runtime"]:
            continue
        key = row["family"] + "/" + row["hardware_mode"]
        groups.setdefault(key, []).append(row)
    aggregates = {}
    for key, members in groups.items():
        comparable = [row for row in members if row["comparable"]]
        aggregates[key] = {"members": [row["id"] for row in members], "comparable_members": [row["id"] for row in comparable],
                           "complete": len(comparable) == len(members),
                           "geometric_mean_runtime_ratio": geometric_mean([row["runtime_ratio"] for row in comparable]) if comparable else None,
                           "paired_bootstrap_95ci": group_interval(comparable, policy) if comparable else None}
    end_to_end = [row for row in rows if row["end_to_end"] and row["status"] != "new_only"]
    comparable_e2e = [row for row in end_to_end if row["comparable"]]
    comparable_ratios = [group["geometric_mean_runtime_ratio"] for group in aggregates.values() if group["complete"]]
    return {"schema_version": 1, "baseline_metadata": bm, "candidate_metadata": cm, "policy": policy,
            "environment_mismatches": mismatches, "workloads": rows, "families": aggregates,
            "overall_family_geometric_mean": geometric_mean(comparable_ratios) if comparable_ratios else None,
            "all_families_complete": all(group["complete"] for group in aggregates.values()),
            "end_to_end": {"members": [row["id"] for row in end_to_end], "comparable_members": [row["id"] for row in comparable_e2e],
                           "complete": len(comparable_e2e) == len(end_to_end),
                           "geometric_mean_runtime_ratio": geometric_mean([row["runtime_ratio"] for row in comparable_e2e]) if comparable_e2e else None,
                           "paired_bootstrap_95ci": group_interval(comparable_e2e, policy) if comparable_e2e else None},
            "acceptance": "Requires review of all families, end-to-end, memory, shared-load records and explicit exceptions. No automatic exception approval."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = compare(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()))
    write_json(args.output / "comparison.json", result)
    lines = ["# Plan20 performance comparison", "", "Ratios are candidate / baseline. Failed, absent, changed-work and new-only samples are not speedups.", "",
             "| Workload | Comparable pairs | Runtime ratio | Paired 95% CI | Status |", "|---|---:|---:|---|---|"]
    for row in result["workloads"]:
        lines.append(f"| {row['id']} | {row.get('paired_samples', 0)} | {row.get('runtime_ratio', '—')} | {row.get('paired_bootstrap_95ci', '—')} | {row['status']} |")
    lines += ["", result["acceptance"], "", "Complete accounting, excluded indices/reasons, family and end-to-end aggregates are in comparison.json."]
    (args.output / "comparison.md").write_text("\n".join(lines) + "\n")
    print(args.output / "comparison.json")


if __name__ == "__main__":
    main()
