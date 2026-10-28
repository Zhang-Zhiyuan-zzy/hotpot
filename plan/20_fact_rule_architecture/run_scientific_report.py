"""Run the existing O0 workflow and report every observation and absence.

The source snapshot and current CBond result define the execution cohort.
Historical population counts and geometry/energy monitor flags are never
scientific acceptance conditions in this tool.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
from importlib.machinery import EXTENSION_SUFFIXES
import json
import multiprocessing as mp
from pathlib import Path
import sys
from time import perf_counter
import traceback


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from rdkit import Chem as _rdkit_chem  # noqa: F401; supported toolkit load order

from _scientific_report import SCHEMA, artifact, sha256, summarize
from tests.benchmarks.coordination_complexes.configuration import (
    CANONICAL_EXTRACTANT_SOURCE, RunProfile, settings_for_profile,
)
from tests.benchmarks.coordination_complexes.io import json_value, load_smiles
from tests.benchmarks.coordination_complexes.validation_cohort import (
    _cohort_atom_ids, _coordination_document, _current_cbond, _graph_document,
    canonical_digest, semantic_case_molecule,
)
from tests.benchmarks.coordination_complexes.validation_evidence import build_acceptance_evidence
from tests.benchmarks.coordination_complexes.validation_routes import execute_provider_route
from tests.benchmarks.coordination_complexes.validation_runner import (
    _atomic_write_json, _execution_trajectory, _json_value,
    _resource_usage_delta, _resource_usage_snapshot, _selected_optimization,
    collect_git_identity, collect_process_resource_identity, collect_runtime_identity,
)


def _error(phase: str, error: Exception) -> dict:
    return {"phase": phase, "type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc()}


def _source_identity(package: Path) -> dict:
    suffixes = {".py", ".pyi", ".c", ".cc", ".cpp", ".h", ".hpp", ".json"}
    files = {
        path.relative_to(package).as_posix(): sha256(path)
        for path in sorted(package.rglob("*"))
        if path.is_file() and path.suffix in suffixes
    }
    return {"root": str(package), "files": files, "sha256": canonical_digest(files)}


@lru_cache(maxsize=None)
def _native_digest(path: str) -> str:
    return sha256(Path(path))


def _loaded_native_identity() -> dict:
    modules = {}
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("hotpot.") and path and path.endswith(tuple(EXTENSION_SUFFIXES)):
            resolved = str(Path(path).resolve())
            modules[name] = {"path": resolved, "sha256": _native_digest(resolved)}
    return modules


def _case(index: int, smiles: str, metal: str, settings) -> dict:
    started = perf_counter()
    record = {
        "index": index, "smiles": smiles,
        "case_id": f"{index:04d}:{hashlib.sha256(smiles.encode()).hexdigest()[:16]}",
    }
    try:
        result = _current_cbond(index, smiles, metal, settings)
        if result is None:
            record["eligibility"] = {
                "state": "ineligible", "reason": "no_coordination_bond_above_current_thresholds"
            }
        else:
            atom_ids = _cohort_atom_ids(result.molecule)
            graph = _graph_document(result.molecule, atom_ids)
            coordination = _coordination_document(
                result.molecule, result.donor_indices, metal, atom_ids
            )
            record.update({
                "status": "cbond", "graph": graph, "coordination": coordination,
                "semantic_graph_sha256": canonical_digest({"graph": graph, "coordination": coordination}),
                "cbond_observations": _json_value({
                    "donor_indices": result.donor_indices,
                    "path_probability": result.path_probability, "steps": result.steps,
                }),
                "eligibility": {"state": "eligible", "reason": "current_cbond_returned_complex"},
            })
    except Exception as error:
        record["eligibility"] = {
            "state": "unknown", "reason": "eligibility_execution_error",
            "error": _error("eligibility", error),
        }
    record["eligibility"]["elapsed_seconds"] = perf_counter() - started
    return record


def _frame_record(trajectory, index: int, branch: str, path: Path, root: Path) -> dict:
    coordinates = trajectory.coordinates(index)
    np.save(path, coordinates, allow_pickle=False)
    return {
        "branch": branch, "frame_index": index,
        "coordinates": artifact(path, root), "shape": list(coordinates.shape),
        "coordinates_finite": bool(np.all(np.isfinite(coordinates))),
        "atom_identity": [[int(a.atom_id), int(a.atomic_number)] for a in trajectory.atoms],
        "frame": _json_value(trajectory.frames[index]),
        "coordinate_unit": "angstrom",
    }


def _archive_frames(archive, directory: Path, root: Path) -> tuple[dict | None, list[dict]]:
    """Retain coordinates irrespective of any geometry/energy monitor flag."""
    if archive is None:
        return None, []
    archive.write(directory / "trajectory", include_sdf=False)
    selected = None
    if archive.main.selected_frame is not None:
        selected = _frame_record(
            archive.main, archive.main.selected_frame.index, "main",
            directory / "selected.npy", root,
        )
    branches = [("main", archive.main)]
    branches += [(f"ligand-{i}", t) for i, t in enumerate(archive.ligand_build_attempts)]
    branches += [(f"preliminary-{i}", t) for i, t in enumerate(archive.preliminary_attempts)]
    last_frames = []
    for name, trajectory in branches:
        frame = next((f for f in reversed(trajectory.frames)
                      if np.all(np.isfinite(trajectory.coordinates(f.index)))), None)
        if frame is not None:
            last_frames.append(_frame_record(
                trajectory, frame.index, name, directory / f"last-finite-{name}.npy", root
            ))
    return selected, last_frames


def _trial_base(case: dict, seed: int) -> dict:
    return {
        "case_id": case["case_id"], "case_index": case["index"],
        "root_seed": seed, "effective_seed": seed + case["index"],
        "seed_derivation": "root_seed + one_based_source_case_index",
        "provider": {
            "name": "OpenBabelConformerAdapter/OBBuilder",
            "requested_seed": seed + case["index"], "effective_seed": None,
            "seed_behavior": "existing OpenBabelConformerAdapter ignores the sampling request seed",
        },
        "eligibility": case["eligibility"]["state"], "attempted": False,
        "outcome": "not_run_unknown", "scientific_status": "unresolved",
        "elapsed_seconds": 0.0, "stages": [], "observations": {},
        "optimization": {}, "work": {}, "selected_frame": None,
        "last_finite_frames": [], "missing_evidence": [], "errors": [], "artifacts": [],
    }


def _run_trial(task: tuple[dict, int, str, str]) -> dict:
    case, seed, profile, output = task
    root = Path(output)
    directory = root / "cases" / f"{case['index']:04d}" / f"seed-{seed}"
    directory.mkdir(parents=True, exist_ok=False)
    record = _trial_base(case, seed)
    started = perf_counter()
    usage = _resource_usage_snapshot()
    archive = None
    phase = "eligibility"
    if case["eligibility"]["state"] != "eligible":
        record["outcome"] = "not_run_" + case["eligibility"]["state"]
        record["missing_evidence"] = [case["eligibility"]["reason"]]
    else:
        record["attempted"] = True
        try:
            phase = "semantic_reconstruction"
            molecule = semantic_case_molecule(case)
            phase = "o0_execution"
            settings = replace(settings_for_profile(RunProfile(profile)), seed=seed)
            execution = execute_provider_route(molecule, "O0", settings, case["index"])
            archive = _execution_trajectory(execution)
            record["stages"].append({"name": phase, "status": "completed"})
            record["observations"]["production_report"] = _json_value(json_value(execution.report))
            optimization = execution.report.optimization
            if optimization is not None:
                record["work"] = {
                    name: getattr(optimization, name) for name in (
                        "epochs_completed", "steps_submitted", "initialization_steps", "steps_completed"
                    )
                }
                record["work"]["build_attempt_count"] = execution.report.build.attempt_count
            phase = "monitor_projection"
            evidence = build_acceptance_evidence(case, execution).to_dict()
            record["observations"]["route_evidence"] = evidence
            record["optimization"] = _selected_optimization(evidence)
            record["optimization"].update({"energy_unit": "kJ/mol", "gradient_unit": "kJ/(mol*angstrom)"})
            record["stages"] += evidence["stage_trace"]["records"]
            record["outcome"] = "completed"
            record["scientific_status"] = "observed"
        except Exception as error:
            record["outcome"] = "execution_error"
            record["errors"].append(_error(phase, error))
            record["stages"].append({"name": phase, "status": "execution_error"})
            if archive is None:
                archive = getattr(error, "trajectory", None)
        try:
            record["selected_frame"], record["last_finite_frames"] = _archive_frames(archive, directory, root)
        except Exception as error:
            record["errors"].append(_error("artifact_write", error))
            record["outcome"] = "execution_error"
            record["scientific_status"] = "unresolved"
        if record["selected_frame"] is None:
            record["missing_evidence"].append("no selected main-workflow frame available")
            if record["outcome"] == "completed":
                record["outcome"] = "no_output"
                record["scientific_status"] = "unresolved"
        if not record["last_finite_frames"]:
            record["missing_evidence"].append("no finite-coordinate frame available in returned or exception archives")
        if "route_evidence" not in record["observations"]:
            record["missing_evidence"].append("full monitor/gradient/recovery projection unavailable; retained trajectory is the available evidence")
    record["elapsed_seconds"] = perf_counter() - started
    record["resources"] = {
        "identity": collect_process_resource_identity(),
        "usage": _resource_usage_delta(usage, _resource_usage_snapshot()),
        "loaded_native_modules": _loaded_native_identity(),
    }
    record["artifacts"] = [artifact(p, root) for p in sorted(directory.rglob("*")) if p.is_file()]
    _atomic_write_json(directory / "report.json", record)
    record["artifacts"].append(artifact(directory / "report.json", root))
    return record


def run_report(output: Path, *, seeds=(42, 43, 44), workers=1, source=CANONICAL_EXTRACTANT_SOURCE,
               profile="standard", metal="Am", limit=None, indices=None, wheel=None) -> dict:
    root = output.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"report output must be empty: {root}")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("root seeds must be nonempty and unique")
    if workers < 1:
        raise ValueError("workers must be positive")
    root.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    snapshot = root / "source.smi"
    snapshot.write_bytes(Path(source).read_bytes())
    full_source = load_smiles(snapshot)
    selected = load_smiles(snapshot, limit=limit, indices=indices)
    if not selected or indices is not None and not set(indices) <= {i for i, _ in full_source}:
        raise ValueError("source selection is empty or contains absent indices")
    settings = settings_for_profile(RunProfile(profile))
    configuration = {
        "backend": "o0", "workflow": "existing O0 complete complex workflow",
        "root_seeds": list(seeds), "profile": profile, "settings": dict(settings.to_manifest()),
        "metal": metal, "workers": workers, "selected_indices": [i for i, _ in selected],
        "seed_derivation": "root_seed + one_based_source_case_index",
    }
    configuration["settings"].pop("seed")
    runtime = collect_runtime_identity()
    import hotpot

    identity = {
        "git": collect_git_identity(), "runtime": runtime,
        "resources": collect_process_resource_identity(),
        "wheel": {"path": str(Path(wheel).resolve()), "sha256": sha256(Path(wheel))} if wheel else None,
        "wheel_identity_note": "source execution; a supplied wheel is provenance only, not proof of wheel import",
        "imported_hotpot_path": str(Path(hotpot.__file__).resolve()),
        "production_sources": _source_identity(Path(hotpot.__file__).resolve().parent),
        "loaded_native_modules": _loaded_native_identity(),
        "report_tools": {
            name: sha256(Path(__file__).with_name(name))
            for name in ("run_scientific_report.py", "_scientific_report.py")
        },
        "command": [sys.executable, *sys.argv],
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    from hotpot.cheminfo.forcefields import acceptance
    from hotpot.cheminfo.AImodels.cbond import apply as cbond_apply
    identity["rules"] = {
        str(Path(module.__file__).relative_to(ROOT)) if Path(module.__file__).is_relative_to(ROOT)
        else module.__name__: sha256(Path(module.__file__))
        for module in (acceptance, cbond_apply, sys.modules[execute_provider_route.__module__])
    }
    _atomic_write_json(root / "manifest.json", {
        "schema": SCHEMA, "configuration": configuration, "identity": identity,
        "source_sha256": sha256(snapshot),
        "completion_record": "report.json",
    })
    cases = []
    for index, smiles in selected:
        case = _case(index, smiles, metal, settings)
        cases.append(case)
        _atomic_write_json(root / "eligibility" / f"{index:04d}.json", case)
        print(f"eligibility case={index} state={case['eligibility']['state']}", flush=True)
    try:
        model = cbond_apply.get_cbond_runtime("cpu")
        identity["cbond_model"] = {
            "directory": str(model.model_dir), "manifest": model.manifest,
            "manifest_sha256": sha256(model.model_dir / "manifest.json"),
            "verified_by_runtime": True,
        }
    except Exception as error:
        identity["cbond_model"] = {"error": _error("model_identity", error)}
    tasks = [(case, seed, profile, str(root)) for seed in seeds for case in cases]
    trials = []
    if workers == 1:
        for task in tasks:
            trial = _run_trial(task)
            trials.append(trial)
            print(f"trial case={trial['case_index']} seed={trial['root_seed']} outcome={trial['outcome']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
            futures = {pool.submit(_run_trial, task): task for task in tasks}
            for future in as_completed(futures):
                case, seed, _, _ = futures[future]
                try:
                    trial = future.result()
                except Exception as error:
                    trial = _trial_base(case, seed)
                    trial.update({"attempted": case["eligibility"]["state"] == "eligible",
                                  "outcome": "execution_error", "elapsed_seconds": None,
                                  "errors": [_error("worker", error)],
                                  "missing_evidence": ["worker did not return a record; elapsed time and any unreturned state are unavailable"]})
                    directory = root / "cases" / f"{case['index']:04d}" / f"seed-{seed}"
                    directory.mkdir(parents=True, exist_ok=True)
                    # A partial worker tree remains available, never silently replaced.
                    trial["artifacts"] = [artifact(p, root) for p in sorted(directory.rglob("*")) if p.is_file()]
                    _atomic_write_json(directory / "worker-error.json", trial)
                    trial["artifacts"].append(artifact(directory / "worker-error.json", root))
                trials.append(trial)
                print(f"trial case={trial['case_index']} seed={trial['root_seed']} outcome={trial['outcome']}", flush=True)
    trials.sort(key=lambda t: (t["root_seed"], t["case_index"]))
    identity["loaded_native_modules"] = _loaded_native_identity()
    report = {
        "schema": SCHEMA, "configuration": configuration, "identity": identity,
        "source": {"original_path": str(Path(source).resolve()), "sha256": sha256(snapshot),
                   "record_count": len(full_source), "snapshot_artifact": artifact(snapshot, root)},
        "cases": cases, "trials": trials, "summary": summarize(cases, trials),
        "wall_seconds": perf_counter() - started,
        "scientific_policy": "Versioned observations; no fixed cohort success count or numerical agreement verdict.",
        "missing_optional_evidence": ["xTB/DFT/QM not requested"],
        "artifacts": [artifact(root / "manifest.json", root)]
        + [artifact(root / "eligibility" / f"{c['index']:04d}.json", root) for c in cases],
    }
    _atomic_write_json(root / "report.json", report)
    (root / "report.md").write_text(
        "# Plan20 O0 scientific execution report\n\n"
        f"Source: `{report['source']['sha256']}`; commit: `{identity['git']['commit']}`.\n\n"
        f"Profile: {profile}; root seeds: {list(seeds)}; actual case seed = root + case index.\n\n"
        "OBBuilder ignores the provider request seed in the existing implementation.\n\n"
        "```json\n" + json.dumps(report["summary"], indent=2) + "\n```\n\n"
        "All cases, trials, monitor values, failures and artifact digests are in `report.json`. "
        "Monitor flags and unresolved observations do not certify scientific correctness.\n"
    )
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("o0",), default="o0")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=CANONICAL_EXTRACTANT_SOURCE)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--profile", choices=("standard", "smoke"), default="standard")
    parser.add_argument("--metal", default="Am")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--indices", nargs="+", type=int)
    parser.add_argument("--wheel", type=Path)
    args = vars(parser.parse_args(argv))
    args.pop("backend")
    try:
        report = run_report(**args)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"scientific report error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report["summary"], sort_keys=True))
    return int(report["summary"]["execution_error_count"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
