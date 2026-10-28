"""Compare complete Plan20 scientific records without numerical pass gates."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _scientific_report import read_report, sha256


def _finite(value):
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def _distribution(values: list[float]) -> dict:
    return {
        "count": len(values),
        "minimum": min(values) if values else None,
        "median": float(np.median(values)) if values else None,
        "p95": float(np.percentile(values, 95)) if values else None,
        "maximum": max(values) if values else None,
    }


def _rmsd(left: np.ndarray, right: np.ndarray) -> float:
    if np.array_equal(left, right):
        return 0.0
    left = left - left.mean(axis=0)
    right = right - right.mean(axis=0)
    u, _, vt = np.linalg.svd(left.T @ right)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    residual = left @ (u @ correction @ vt) - right
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


def compare_reports(baseline: Path, candidate: Path) -> dict:
    before, before_root = read_report(baseline)
    after, after_root = read_report(candidate)
    before_cases = {case["case_id"]: case for case in before["cases"]}
    after_cases = {case["case_id"]: case for case in after["cases"]}
    shared = before_cases.keys() & after_cases.keys()
    eligibility_changes = [
        {"case_id": key, "baseline": before_cases[key]["eligibility"],
         "candidate": after_cases[key]["eligibility"]}
        for key in sorted(shared)
        if before_cases[key]["eligibility"]["state"] != after_cases[key]["eligibility"]["state"]
    ]
    before_trials = {(t["case_id"], t["root_seed"]): t for t in before["trials"]}
    after_trials = {(t["case_id"], t["root_seed"]): t for t in after["trials"]}
    context_changes = []
    if before["source"]["sha256"] != after["source"]["sha256"]:
        context_changes.append("source_snapshot_changed")
    for name in ("git", "rules", "runtime", "cbond_model", "report_tools"):
        if before["identity"].get(name) != after["identity"].get(name):
            context_changes.append(f"{name}_identity_changed")
    if before["identity"].get("production_sources", {}).get("sha256") != after["identity"].get("production_sources", {}).get("sha256"):
        context_changes.append("production_source_identity_changed")
    before_native = {name: value["sha256"] for name, value in before["identity"].get("loaded_native_modules", {}).items()}
    after_native = {name: value["sha256"] for name, value in after["identity"].get("loaded_native_modules", {}).items()}
    if before_native != after_native:
        context_changes.append("loaded_native_identity_changed")
    for name in ("settings", "profile", "backend", "metal", "root_seeds"):
        if before["configuration"].get(name) != after["configuration"].get(name):
            context_changes.append(f"{name}_changed")
    pairs = []
    energy_differences, rmsds = [], []
    for key in sorted(before_trials.keys() & after_trials.keys()):
        left, right = before_trials[key], after_trials[key]
        row = {
            "case_id": key[0], "root_seed": key[1],
            "baseline_outcome": left["outcome"], "candidate_outcome": right["outcome"],
            "baseline_effective_seed": left["effective_seed"],
            "candidate_effective_seed": right["effective_seed"],
            "elapsed_seconds": {"baseline": left["elapsed_seconds"], "candidate": right["elapsed_seconds"]},
            "work": {"baseline": left["work"], "candidate": right["work"]},
            "scientific_status": {"baseline": left["scientific_status"], "candidate": right["scientific_status"]},
            "monitor_observations": {
                "baseline": left["observations"].get("production_report", {}).get("quality_report"),
                "candidate": right["observations"].get("production_report", {}).get("quality_report"),
            },
            "paired_energy_difference_kj_mol": None, "aligned_coordinate_rmsd_angstrom": None,
            "comparison_exclusions": [],
        }
        graph_identity = before_cases[key[0]].get("semantic_graph_sha256")
        common_graph = bool(graph_identity) and graph_identity == after_cases[key[0]].get("semantic_graph_sha256")
        row["same_recorded_work"] = (
            left["outcome"] == right["outcome"] == "completed"
            and common_graph and bool(left["work"]) and left["work"] == right["work"]
            and all(before["configuration"].get(name) == after["configuration"].get(name)
                    for name in ("settings", "profile", "backend", "metal"))
        )
        if not row["same_recorded_work"]:
            row["comparison_exclusions"].append("timing_not_equivalent_completed_work")
        if not common_graph:
            row["comparison_exclusions"].append("semantic_graph_changed")
        lframe, rframe = left["selected_frame"], right["selected_frame"]
        if common_graph and lframe and rframe:
            if lframe["atom_identity"] != rframe["atom_identity"] or lframe["shape"] != rframe["shape"]:
                row["comparison_exclusions"].append("coordinate_row_identity_or_shape_changed")
            elif lframe["coordinates_finite"] and rframe["coordinates_finite"]:
                lcoords = np.load(before_root / lframe["coordinates"]["path"], allow_pickle=False)
                rcoords = np.load(after_root / rframe["coordinates"]["path"], allow_pickle=False)
                value = _rmsd(lcoords, rcoords)
                row["aligned_coordinate_rmsd_angstrom"] = value
                rmsds.append(value)
            else:
                row["comparison_exclusions"].append("nonfinite_coordinates")
            le = _finite(left["optimization"].get("best_energy"))
            re = _finite(right["optimization"].get("best_energy"))
            units = {left["optimization"].get("energy_unit"), right["optimization"].get("energy_unit")}
            if lframe["atom_identity"] == rframe["atom_identity"] and le is not None and re is not None and units == {"kJ/mol"}:
                delta = re - le
                row["paired_energy_difference_kj_mol"] = delta
                energy_differences.append(abs(delta))
        else:
            row["comparison_exclusions"].append("missing_comparable_selected_frame")
        changed = (left["outcome"] != right["outcome"]
                   or row["paired_energy_difference_kj_mol"] not in (None, 0.0)
                   or row["aligned_coordinate_rmsd_angstrom"] not in (None, 0.0))
        row["difference_context"] = context_changes.copy() if changed else []
        if changed:
            row["difference_context"].append("uncontrolled_OBBuilder_randomness_possible")
            row["difference_context"].append("causal_attribution_unresolved")
        pairs.append(row)
    return {
        "schema": "hotpot-plan20-scientific-comparison/v1",
        "report_tools": {
            name: sha256(Path(__file__).with_name(name))
            for name in ("compare_scientific_reports.py", "_scientific_report.py")
        },
        "baseline_report_sha256": sha256(baseline / "report.json" if baseline.is_dir() else baseline),
        "candidate_report_sha256": sha256(candidate / "report.json" if candidate.is_dir() else candidate),
        "baseline_summary": before["summary"], "candidate_summary": after["summary"],
        "cohorts": {
            "shared": sorted(shared), "only_baseline": sorted(before_cases.keys() - after_cases.keys()),
            "only_candidate": sorted(after_cases.keys() - before_cases.keys()),
            "baseline_denominator": len(before_cases), "candidate_denominator": len(after_cases),
            "eligibility_changes": eligibility_changes,
        },
        "trial_sets": {
            "paired_count": len(pairs),
            "only_baseline": [list(k) for k in sorted(before_trials.keys() - after_trials.keys())],
            "only_candidate": [list(k) for k in sorted(after_trials.keys() - before_trials.keys())],
            "same_recorded_work_count": sum(p["same_recorded_work"] for p in pairs),
            "exclusion_counts": dict(Counter(r for p in pairs for r in p["comparison_exclusions"])),
        },
        "context_changes": context_changes,
        "distributions": {
            "absolute_energy_difference_kj_mol": _distribution(energy_differences),
            "aligned_coordinate_rmsd_angstrom": _distribution(rmsds),
        },
        "pairs": pairs,
        "technical_execution_errors": before["summary"]["execution_error_count"] + after["summary"]["execution_error_count"],
        "interpretation": "Scientific differences remain observations, not pass/fail. Raw elapsed values do not establish architecture speedup; use the frozen Section12 benchmark policy.",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        comparison = compare_reports(args.baseline, args.candidate)
        args.output.mkdir(parents=True, exist_ok=True)
        output = args.output / "comparison.json"
        if output.exists():
            raise FileExistsError(f"comparison already exists: {output}")
        output.write_text(json.dumps(comparison, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
        rows = ["# Plan20 scientific comparison", "", comparison["interpretation"], "",
                "```json", json.dumps({k: v for k, v in comparison.items() if k != "pairs"}, indent=2), "```", "",
                "| Case | Root seed | Baseline outcome | Candidate outcome | Energy delta (kJ/mol) | Aligned RMSD (Å) |",
                "|---|---:|---|---|---:|---:|"]
        rows += [f"| {p['case_id']} | {p['root_seed']} | {p['baseline_outcome']} | {p['candidate_outcome']} | {p['paired_energy_difference_kj_mol']} | {p['aligned_coordinate_rmsd_angstrom']} |"
                 for p in comparison["pairs"]]
        (args.output / "comparison.md").write_text("\n".join(rows) + "\n")
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"invalid scientific report: {error}", file=sys.stderr)
        return 2
    return int(comparison["technical_execution_errors"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
