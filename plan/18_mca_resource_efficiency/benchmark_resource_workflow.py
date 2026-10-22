#!/usr/bin/env python3
"""Benchmark the production MCA CPU or GPU workflow with semantic digests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import statistics
import time
from pathlib import Path

import numpy as np

from hotpot.cheminfo.AImodels.mca import MCACpuPredictor, MCAGpuPredictor


def prediction_hashes(predictions):
    values = np.asarray(
        [
            atom.mca_kj_mol
            for prediction in predictions
            for atom in prediction.atom_predictions
        ],
        dtype=np.float64,
    )
    sites = [
        tuple((site.atom_index, site.site_type) for site in prediction.sites)
        for prediction in predictions
    ]
    joint = []
    for molecule_index, prediction in enumerate(predictions):
        sites_by_atom = {site.atom_index: site for site in prediction.sites}
        for atom in prediction.atom_predictions:
            site = sites_by_atom.get(atom.atom_index)
            if site is not None and float(site.mca_kj_mol).hex() != float(
                atom.mca_kj_mol
            ).hex():
                raise RuntimeError("Site MCA does not exactly match atom MCA")
            joint.append(
                {
                    "molecule_index": molecule_index,
                    "atom_index": atom.atom_index,
                    "mca": float(atom.mca_kj_mol).hex(),
                    "is_Nuc_site": site is not None,
                    "site_type": None if site is None else site.site_type,
                    "site_mca": (
                        None if site is None else float(site.mca_kj_mol).hex()
                    ),
                }
            )
    joint_payload = json.dumps(
        joint,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return {
        "atom_count": int(values.size),
        "site_count": sum(len(value) for value in sites),
        "mca_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "sites_sha256": hashlib.sha256(repr(sites).encode()).hexdigest(),
        "joint_sha256": hashlib.sha256(joint_payload).hexdigest(),
    }


def cpu_seconds() -> float:
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-count", type=int)
    parser.add_argument("--cpu-batch-size", type=int, default=64)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--gpu-batch-size", type=int)
    parser.add_argument("--expected-joint-sha256")
    parser.add_argument("--allow-charged", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    smiles = [
        line.split()[0]
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.sample_count]
    initial_affinity = tuple(sorted(os.sched_getaffinity(0)))
    if args.backend == "cpu":
        predictor = MCACpuPredictor(
            batch_size=args.cpu_batch_size,
            cpu_count=args.cpu_count,
            allow_charged=args.allow_charged,
        )
    else:
        predictor = MCAGpuPredictor(
            gpu_indices=(args.gpu_index,),
            gpu_batch_size=args.gpu_batch_size,
            allow_charged=args.allow_charged,
        )

    runs = []
    reference = None
    with predictor:
        for repeat in range(1, args.repeats + 1):
            wall_started = time.perf_counter()
            cpu_started = cpu_seconds()
            predictions = list(predictor.iter_predict(smiles))
            elapsed = time.perf_counter() - wall_started
            elapsed_cpu = cpu_seconds() - cpu_started
            hashes = prediction_hashes(predictions)
            if reference is None:
                reference = hashes
            if hashes != reference:
                raise RuntimeError("Prediction hashes changed between repeats")
            if (
                args.expected_joint_sha256 is not None
                and hashes["joint_sha256"] != args.expected_joint_sha256
            ):
                raise RuntimeError("Joint prediction digest does not match")
            runs.append(
                {
                    "repeat": repeat,
                    "seconds": elapsed,
                    "molecules_per_second": len(predictions) / elapsed,
                    "cpu_seconds": elapsed_cpu,
                    "average_cpu_cores": elapsed_cpu / elapsed,
                    "hashes": hashes,
                }
            )

        if args.backend == "cpu":
            resources = {
                "cpu_count": predictor.cpu_count,
                "cpu_ids": predictor.cpu_ids,
            }
        else:
            resources = {
                "gpu_index": args.gpu_index,
                "gpu_batch_size": predictor.gpu_batch_size,
                "site_cpu_per_gpu": predictor.resource_plan.site_cpu_per_gpu,
                "infer_cpu_per_gpu": predictor.resource_plan.infer_cpu_per_gpu,
                "required_cpu_count": predictor.resource_plan.required_cpu_count,
            }

    if tuple(sorted(os.sched_getaffinity(0))) != initial_affinity:
        raise RuntimeError("MCA predictor did not restore caller CPU affinity")
    result = {
        "backend": args.backend,
        "sample_count": len(smiles),
        "input": str(args.input),
        "initial_cpu_affinity": initial_affinity,
        "resources": resources,
        "median_seconds": statistics.median(run["seconds"] for run in runs),
        "median_molecules_per_second": statistics.median(
            run["molecules_per_second"] for run in runs
        ),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
