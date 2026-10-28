"""Capture untimed, ordered MCA/CBond observations for frozen Plan20 inputs.

Each selected manifest workload runs once, including its declared CBond loop
count. These records never enter timing aggregates or decide scientific truth.
Run after performance measurements finish, with the same resource allocation.
Candidate module paths must be supplied explicitly; no backend is substituted.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

if __package__:
    from .run_performance_baseline import digest, gpu_state, metadata, verify_loaded_package_files, write_json
else:
    from run_performance_baseline import digest, gpu_state, metadata, verify_loaded_package_files, write_json


HERE = Path(__file__).resolve().parent


def _error(phase, error):
    return {"phase": phase, "type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc()}


def _loaded_files():
    return sorted({str(Path(module.__file__).resolve()) for name, module in sys.modules.items()
                   if (name == "hotpot" or name.startswith("hotpot.")) and getattr(module, "__file__", None)})


def _session(session):
    return {"providers": session.get_providers(), "provider_options": session.get_provider_options()}


def _mca(entry, module_name, directory):
    params = entry["parameters"]
    smiles = [params["smiles"][i % len(params["smiles"])] for i in range(params["molecules"])]
    rows = [{"molecule_index": i, "input_smiles": smi, "input_consumed": False,
             "status": "not_returned", "prediction": None} for i, smi in enumerate(smiles)]
    result = {"molecules": rows, "unexpected_outputs": [], "errors": [], "allocation": None,
              "input_index_base": 0, "seed": params["seed"],
              "seed_policy": "Existing predictor conformer_seed; no report-side seed transformation",
              "session_evidence": "GPU worker sessions are not introspected by this report"}
    predictor = None
    entered = False
    phase = "predictor_construction"
    started = time.perf_counter()
    try:
        module = importlib.import_module(module_name)
        if params["backend"] == "cpu":
            predictor = module.MCACpuPredictor(cpu_count=params["cpu_count"], batch_size=64,
                                               conformer_seed=params["seed"])
        else:
            predictor = module.MCAGpuPredictor(
                gpu_indices=params["gpu_indices"], site_cpu_per_gpu=params["site_cpu_per_gpu"],
                infer_cpu_per_gpu=params["infer_cpu_per_gpu"], gpu_batch_size=2048,
                conformer_seed=params["seed"],
            )
        phase = "predictor_enter"
        predictor.__enter__()
        entered = True

        def inputs():
            for row in rows:
                row["input_consumed"] = True
                yield row["input_smiles"]

        phase = "ordered_prediction"
        for index, prediction in enumerate(predictor.iter_predict(inputs())):
            payload = prediction.to_dict()
            sites = {site["atom_index"] for site in payload["sites"]}
            observation = {
                "status": "observed", "prediction": payload,
                "ordered_atom_identity": [[a["atom_index"], a["element"]] for a in payload["atom_predictions"]],
                "ordered_site_identity": [[s["atom_index"], s["element"], s["site_type"]] for s in payload["sites"]],
                "is_Nuc_site": [a["atom_index"] in sites for a in payload["atom_predictions"]],
                "energy_unit": "kJ/mol",
            }
            if index < len(rows):
                rows[index].update(observation)
                write_json(directory / "molecules" / f"{index:06d}.json", rows[index])
            else:
                result["unexpected_outputs"].append({"output_index": index, **observation})
        phase = "output_completeness"
        if result["unexpected_outputs"] or any(row["status"] != "observed" for row in rows):
            raise RuntimeError("Predictor output count differs from the complete input roster")
    except Exception as error:
        result["errors"].append(_error(phase, error))
    finally:
        if predictor is not None:
            plan = getattr(predictor, "resource_plan", None)
            result["allocation"] = {
                "requested_backend": params["backend"], "gpu_started": getattr(predictor, "_gpu_started", False),
                "resource_plan": asdict(plan) if plan is not None else None,
                "cpu_ids": list(getattr(predictor, "cpu_ids", ())),
            }
            if entered:
                try:
                    predictor.__exit__(None, None, None)
                except Exception as error:
                    result["errors"].append(_error("predictor_close", error))
    result["elapsed_seconds"] = time.perf_counter() - started
    for row in rows:
        if row["status"] == "not_returned":
            row["absence_reason"] = "Workflow ended without this output; per-input failure attribution is unavailable"
        write_json(directory / "molecules" / f"{row['molecule_index']:06d}.json", row)
    return result


def _array(directory, name, value):
    import numpy as np
    path = directory / f"{name}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value, allow_pickle=False)
    return {"path": path.name, "sha256": digest(path), "shape": list(value.shape),
            "dtype": str(value.dtype), "all_finite": bool(np.isfinite(value).all())}


def _cbond(entry, module_name, directory):
    import numpy as np
    params = entry["parameters"]
    calls = [{"call_index": i, "artifact_root": f"call-{i:03d}",
              "status": "not_run", "artifacts": {}, "errors": []}
             for i in range(params["loops"])]
    result = {"calls": calls, "errors": [], "sessions": {}, "requested_device": params["device"]}
    atoms = np.asarray([6, 7, 8, 26], dtype=np.int32)
    edges = np.asarray([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=np.int64)
    mask = np.zeros((1, 3), dtype=bool)
    indices = np.asarray([[3, 3], [0, 1]], dtype=np.int64)
    result["inputs"] = {name: _array(directory, name, value) for name, value in
                        (("atomic_numbers", atoms), ("edge_index", edges), ("rings_mask", mask), ("cbond_index", indices))}
    started = time.perf_counter()
    try:
        runtime = importlib.import_module(module_name).CBondRuntime(device=params["device"])
        result["effective_device"] = runtime.device
        result["model_manifest"] = {"directory": str(runtime.model_dir),
                                    "sha256": digest(runtime.model_dir / "manifest.json")}
        result["sessions"] = {name: _session(session) for name, session in
                              (("graph", runtime.graph_session), ("cbond", runtime.cbond_session))}
        for call in calls:
            call_dir = directory / f"call-{call['call_index']:03d}"
            phase = "graph_embedding"
            try:
                xg = runtime.embed_graph(atoms, edges)
                call["artifacts"]["xg"] = _array(call_dir, "xg", xg)
                rings = xg[[0, 1, 2]].reshape(1, 3, 128)
                call["artifacts"]["padded_rings"] = _array(call_dir, "padded_rings", rings)
                phase = "cbond_prediction"
                values = runtime.predict(xg, rings, mask, indices)
                call["artifacts"]["predictions"] = _array(call_dir, "predictions", values)
                call["status"] = "observed"
            except Exception as error:
                call["status"] = "execution_error"
                call["errors"].append(_error(phase, error))
                # The timed workload stops at the first failed call as well.
                break
    except Exception as error:
        result["errors"].append(_error("runtime_construction", error))
    result["elapsed_seconds"] = time.perf_counter() - started
    for call in calls:
        if call["status"] == "not_run":
            call["absence_reason"] = "An earlier construction/call failure prevented this call"
        write_json(directory / f"call-{call['call_index']:03d}" / "report.json", call)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=HERE / "performance_manifest.json")
    parser.add_argument("--repository", type=Path, default=HERE.parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", nargs="+", required=True)
    parser.add_argument("--package-mode", choices=("source", "installed"), default="source")
    parser.add_argument("--phase", choices=("baseline", "candidate"), default="baseline")
    parser.add_argument("--mca-module")
    parser.add_argument("--cbond-module")
    parser.add_argument("--model-identity", type=Path, required=True)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--cpu-affinity")
    parser.add_argument("--shared-load-note", default="Untimed scientific observations; shared load not assessed")
    args = parser.parse_args(argv)
    args.repository, args.output = args.repository.resolve(), args.output.resolve()
    args.manifest, args.model_identity = args.manifest.resolve(), args.model_identity.resolve()
    args.wheel = args.wheel.resolve() if args.wheel else None
    manifest = json.loads(args.manifest.read_text())
    available = {e["id"]: e for e in manifest["workloads"] if e["operation"] in {"mca", "cbond"}}
    if len(set(args.only)) != len(args.only) or set(args.only) - available.keys():
        parser.error("--only must name unique MCA/CBond workloads in the supplied manifest")
    entries = [available[name] for name in args.only]
    if args.phase == "candidate":
        for operation in {entry["operation"] for entry in entries}:
            if getattr(args, f"{operation}_module") is None:
                parser.error(f"candidate observations require explicit --{operation}-module")
    args.mca_module = args.mca_module or "hotpot.cheminfo.AImodels.mca"
    args.cbond_module = args.cbond_module or "hotpot.cheminfo.AImodels.cbond.runtime"
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"observation output must be empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    for key, value in manifest["measurement"]["thread_env"].items():
        os.environ[key] = value
    if args.cpu_affinity:
        os.sched_setaffinity(0, {int(value) for value in args.cpu_affinity.split(",")})
    if args.package_mode == "source":
        sys.path.insert(0, str(args.repository))
        os.environ["PYTHONPATH"] = str(args.repository)
    else:
        os.environ.pop("PYTHONPATH", None)
    args.test_selection, args.smoke = None, False
    identity = metadata(args, manifest)
    identity["identity_helper_sha256"] = identity.pop("runner_sha256")
    identity["capture_tool_sha256"] = digest(__file__)
    identity["command"] = [sys.executable, *sys.argv]
    identity["module_bindings"] = {"mca": args.mca_module, "cbond": args.cbond_module}
    identity["loader_environment"] = {key: os.environ.get(key) for key in ("LD_PRELOAD", "LD_LIBRARY_PATH")}
    installation = identity["installed_wheel"]
    if installation is not None and not installation["verified"]:
        write_json(args.output / "identity-error.json", identity)
        raise RuntimeError("Installed Hotpot bytes/import origin do not match the supplied wheel")
    (args.output / "performance_manifest.json").write_bytes(args.manifest.read_bytes())
    (args.output / "model_identity.json").write_bytes(args.model_identity.read_bytes())
    write_json(args.output / "manifest.json", {"identity": identity, "selected_workloads": args.only})
    from rdkit import Chem as _rdkit_chem  # noqa: F401; supported toolkit load order
    import hotpot
    package_file = Path(hotpot.__file__).resolve()
    if args.package_mode == "source" and package_file != (args.repository / "hotpot/__init__.py").resolve():
        raise RuntimeError("Source observation imported Hotpot outside the selected repository")
    identity["imported_hotpot_path"] = str(package_file)
    reports = []
    for entry in entries:
        directory = args.output / entry["id"]
        directory.mkdir()
        gpu_before = gpu_state()
        result = (_mca(entry, args.mca_module, directory) if entry["operation"] == "mca"
                  else _cbond(entry, args.cbond_module, directory))
        result.update(id=entry["id"], entry=entry, artifact_root=entry["id"],
                      gpu_before=gpu_before, gpu_after=gpu_state())
        result["loaded_package_files"] = [{"path": path, "sha256": digest(path)} for path in _loaded_files()]
        result["loaded_package_scope"] = "Current reporting process only; spawned worker modules are not directly inspected"
        if installation is not None:
            result["installed_package_binding"] = verify_loaded_package_files(installation, _loaded_files())
            if not result["installed_package_binding"]["verified"]:
                result["errors"].append({"phase": "package_identity", "message": "Loaded package bytes differ from wheel"})
        result["artifacts"] = [{"path": p.relative_to(directory).as_posix(), "sha256": digest(p)}
                               for p in sorted(directory.rglob("*")) if p.is_file()]
        write_json(directory / "report.json", result)
        reports.append(result)
        print(entry["id"], "observation recorded", flush=True)
    errors = sum(len(r["errors"]) + sum(len(c["errors"]) for c in r.get("calls", [])) for r in reports)
    report = {"schema": "hotpot-plan20-inference-observations/v1", "identity": identity,
              "workloads": reports, "execution_error_count": errors,
              "policy": "One independent untimed execution per frozen workload. Numerical values/finite flags are observations, not scientific pass criteria; no timing aggregate or backend substitution."}
    write_json(args.output / "report.json", report)
    return int(errors > 0)


if __name__ == "__main__":
    raise SystemExit(main())
