#!/usr/bin/env python3
"""Benchmark eight affinity-isolated eight-CPU MCA predictor instances."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import multiprocessing as mp
import os
import resource
import statistics
import time
import traceback
from pathlib import Path


def _cpu_seconds() -> float:
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime


def _records(predictions):
    return [
        {
            "smiles": prediction.smiles,
            "formal_charge": prediction.formal_charge,
            "atoms": [
                [atom.atom_index, atom.element, float(atom.mca_kj_mol).hex()]
                for atom in prediction.atom_predictions
            ],
            "sites": [
                [
                    site.atom_index,
                    site.site_type,
                    float(site.mca_kj_mol).hex(),
                ]
                for site in prediction.sites
            ],
        }
        for prediction in predictions
    ]


def _digest(records) -> str:
    payload = json.dumps(
        records,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _worker(
    worker_index,
    cpu_ids,
    smiles,
    repeats,
    warmup_count,
    barrier,
    output_queue,
):
    try:
        os.sched_setaffinity(0, set(cpu_ids))
        from hotpot.cheminfo.AImodels.mca import MCACpuPredictor

        options = {"allow_charged": True}
        if "cpu_count" in inspect.signature(MCACpuPredictor).parameters:
            options["cpu_count"] = len(cpu_ids)
        predictor = MCACpuPredictor(**options)
        runs = []
        reference_digest = None
        last_records = None
        with predictor:
            predictor.predict(smiles[:warmup_count])
            barrier.wait()
            for repeat in range(1, repeats + 1):
                barrier.wait()
                cpu_started = _cpu_seconds()
                started = time.perf_counter()
                predictions = predictor.predict(smiles)
                elapsed = time.perf_counter() - started
                cpu_elapsed = _cpu_seconds() - cpu_started
                records = _records(predictions)
                digest = _digest(records)
                if reference_digest is None:
                    reference_digest = digest
                if digest != reference_digest:
                    raise RuntimeError("Predictions changed between repeats")
                runs.append(
                    {
                        "repeat": repeat,
                        "seconds": elapsed,
                        "molecules_per_second": len(smiles) / elapsed,
                        "cpu_seconds": cpu_elapsed,
                    }
                )
                last_records = records
                barrier.wait()
        output_queue.put(
            {
                "worker_index": worker_index,
                "cpu_ids": cpu_ids,
                "runs": runs,
                "prediction_sha256": reference_digest,
                "predictions": last_records,
            }
        )
    except BaseException:
        output_queue.put(
            {
                "worker_index": worker_index,
                "error": traceback.format_exc(),
            }
        )
        barrier.abort()


def _comparison(current_workers, reference_workers):
    differences = []
    site_semantics_equal = True
    output_structure_equal = True
    for current, reference in zip(current_workers, reference_workers):
        current_predictions = current["predictions"]
        reference_predictions = reference["predictions"]
        if len(current_predictions) != len(reference_predictions):
            output_structure_equal = False
            continue
        for actual, expected in zip(current_predictions, reference_predictions):
            if (
                actual["smiles"] != expected["smiles"]
                or actual["formal_charge"] != expected["formal_charge"]
                or len(actual["atoms"]) != len(expected["atoms"])
            ):
                output_structure_equal = False
                continue
            actual_sites = [(site[0], site[1]) for site in actual["sites"]]
            expected_sites = [(site[0], site[1]) for site in expected["sites"]]
            if actual_sites != expected_sites:
                site_semantics_equal = False
            for actual_atom, expected_atom in zip(
                actual["atoms"], expected["atoms"]
            ):
                if actual_atom[:2] != expected_atom[:2]:
                    output_structure_equal = False
                    continue
                differences.append(
                    float.fromhex(actual_atom[2])
                    - float.fromhex(expected_atom[2])
                )
    absolute = [abs(value) for value in differences]
    return {
        "compared_atom_count": len(differences),
        "max_absolute_error_kj_mol": max(absolute, default=0.0),
        "mean_absolute_error_kj_mol": (
            statistics.fmean(absolute) if absolute else 0.0
        ),
        "rmse_kj_mol": (
            math.sqrt(statistics.fmean(value * value for value in differences))
            if differences
            else 0.0
        ),
        "exact_value_count": sum(value == 0.0 for value in differences),
        "output_structure_equal": output_structure_equal,
        "site_semantics_equal": site_semantics_equal,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--instances", type=int, default=8)
    parser.add_argument("--cpus-per-instance", type=int, default=8)
    parser.add_argument("--first-cpu", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-count", type=int, default=8)
    parser.add_argument("--label", required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    smiles = [
        line.split()[0]
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.sample_count]
    cpu_groups = [
        tuple(
            range(
                args.first_cpu + index * args.cpus_per_instance,
                args.first_cpu + (index + 1) * args.cpus_per_instance,
            )
        )
        for index in range(args.instances)
    ]
    available = os.sched_getaffinity(0)
    requested = {cpu for group in cpu_groups for cpu in group}
    if not requested <= available:
        raise RuntimeError("Requested benchmark CPUs are outside process affinity")

    context = mp.get_context("spawn")
    barrier = context.Barrier(args.instances + 1, timeout=180)
    output_queue = context.Queue()
    processes = [
        context.Process(
            target=_worker,
            args=(
                index,
                cpu_ids,
                smiles,
                args.repeats,
                args.warmup_count,
                barrier,
                output_queue,
            ),
        )
        for index, cpu_ids in enumerate(cpu_groups)
    ]
    for process in processes:
        process.start()

    barrier.wait()
    aggregate_runs = []
    for repeat in range(1, args.repeats + 1):
        barrier.wait()
        started = time.perf_counter()
        barrier.wait()
        elapsed = time.perf_counter() - started
        aggregate_runs.append(
            {
                "repeat": repeat,
                "seconds": elapsed,
                "molecules_per_second": (
                    args.instances * len(smiles) / elapsed
                ),
            }
        )

    workers = [output_queue.get(timeout=30) for _ in processes]
    for process in processes:
        process.join(timeout=30)
    errors = [worker for worker in workers if "error" in worker]
    if errors:
        raise RuntimeError(errors[0]["error"])
    workers.sort(key=lambda worker: worker["worker_index"])
    comparison = None
    if args.reference is not None:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        comparison = _comparison(workers, reference["workers"])

    result = {
        "label": args.label,
        "input": str(args.input),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "sample_count_per_instance": len(smiles),
        "instances": args.instances,
        "cpus_per_instance": args.cpus_per_instance,
        "total_cpu_count": len(requested),
        "cpu_groups": cpu_groups,
        "repeats": args.repeats,
        "median_seconds": statistics.median(
            run["seconds"] for run in aggregate_runs
        ),
        "median_molecules_per_second": statistics.median(
            run["molecules_per_second"] for run in aggregate_runs
        ),
        "aggregate_runs": aggregate_runs,
        "worker_prediction_sha256": [
            worker["prediction_sha256"] for worker in workers
        ],
        "comparison_to_reference": comparison,
        "workers": workers,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "workers"}, indent=2))


if __name__ == "__main__":
    main()
