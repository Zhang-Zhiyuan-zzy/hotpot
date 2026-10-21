#!/usr/bin/env python3
"""Validate the implemented MCA GPU pipeline against a fixed sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from hotpot.cheminfo.AImodels.mca import MCAGpuPredictor
from hotpot.cheminfo.AImodels.mca.resources import gpu_window_atom_budget


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--gpu-indices", default="0")
    parser.add_argument("--site-cpu-per-gpu", type=int, default=9)
    parser.add_argument("--infer-cpu-per-gpu", type=int, default=7)
    parser.add_argument("--gpu-batch-size", type=int)
    parser.add_argument("--allow-charged", action="store_true")
    parser.add_argument("--expected-mca-sha256")
    parser.add_argument("--expected-sites-sha256")
    parser.add_argument("--expected-joint-sha256")
    parser.add_argument("--require-cross-capacity", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    return parser.parse_args()


def build_validation_hashes(predictions):
    """Build the historical hashes and a strict atom/site relationship hash."""

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
    joint_records = []
    for molecule_index, prediction in enumerate(predictions):
        atom_count = len(prediction.atom_predictions)
        atoms_by_index = {
            atom.atom_index: atom for atom in prediction.atom_predictions
        }
        sites_by_atom = {}
        for site in prediction.sites:
            if not 0 <= site.atom_index < atom_count:
                raise RuntimeError(
                    f"Molecule {molecule_index} has out-of-range site atom index "
                    f"{site.atom_index} for {atom_count} atoms"
                )
            if site.atom_index in sites_by_atom:
                raise RuntimeError(
                    f"Molecule {molecule_index} has duplicate site atom index "
                    f"{site.atom_index}"
                )
            atom = atoms_by_index[site.atom_index]
            if float(site.mca_kj_mol).hex() != float(atom.mca_kj_mol).hex():
                raise RuntimeError(
                    f"Molecule {molecule_index} site MCA at atom "
                    f"{site.atom_index} does not exactly match atom MCA"
                )
            sites_by_atom[site.atom_index] = site

        for atom in prediction.atom_predictions:
            site = sites_by_atom.get(atom.atom_index)
            joint_records.append(
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
        joint_records,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "atom_count": int(values.size),
        "site_count": sum(len(value) for value in sites),
        "mca_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "sites_sha256": hashlib.sha256(repr(sites).encode()).hexdigest(),
        "joint_sha256": hashlib.sha256(joint_payload).hexdigest(),
    }


def admission_capacity_cycle_count(atom_counts, atom_capacity):
    cycles = 0
    resident_atoms = 0
    for atom_count in atom_counts:
        if resident_atoms >= atom_capacity:
            cycles += 1
            resident_atoms = 0
        resident_atoms += atom_count
    return cycles + bool(resident_atoms)


def main() -> None:
    args = parse_args()
    smiles = [
        line.strip().split()[0]
        for line in args.input.open(encoding="utf-8")
        if line.strip()
    ]
    initial_cpu_affinity = tuple(sorted(os.sched_getaffinity(0)))
    constructor_started = time.perf_counter()
    predictor = MCAGpuPredictor(
        gpu_indices=tuple(int(value) for value in args.gpu_indices.split(",")),
        site_cpu_per_gpu=args.site_cpu_per_gpu,
        infer_cpu_per_gpu=args.infer_cpu_per_gpu,
        gpu_batch_size=args.gpu_batch_size,
        allow_charged=args.allow_charged,
    )
    constructor_seconds = time.perf_counter() - constructor_started
    runs = []
    with predictor:
        for repeat in range(args.repeats):
            started = time.perf_counter()
            predictions = list(predictor.iter_predict(smiles))
            elapsed = time.perf_counter() - started
            hashes = build_validation_hashes(predictions)
            if (
                args.expected_mca_sha256 is not None
                and hashes["mca_sha256"] != args.expected_mca_sha256
            ):
                raise RuntimeError(
                    f"MCA digest mismatch: {hashes['mca_sha256']} != "
                    f"{args.expected_mca_sha256}"
                )
            if (
                args.expected_sites_sha256 is not None
                and hashes["sites_sha256"] != args.expected_sites_sha256
            ):
                raise RuntimeError(
                    f"Site digest mismatch: {hashes['sites_sha256']} != "
                    f"{args.expected_sites_sha256}"
                )
            if (
                args.expected_joint_sha256 is not None
                and hashes["joint_sha256"] != args.expected_joint_sha256
            ):
                raise RuntimeError(
                    f"Joint digest mismatch: {hashes['joint_sha256']} != "
                    f"{args.expected_joint_sha256}"
                )
            runs.append(
                {
                    "repeat": repeat + 1,
                    "elapsed_seconds": elapsed,
                    "molecules_per_second": len(predictions) / elapsed,
                    "mca_sha256": hashes["mca_sha256"],
                    "sites_sha256": hashes["sites_sha256"],
                    "joint_sha256": hashes["joint_sha256"],
                }
            )
    if predictor.resource_plan is None:
        raise RuntimeError(
            "Validation sample stayed below the 256-site GPU threshold"
        )
    worker_count = (
        len(predictor.resource_plan.site_worker_assignments)
        + len(predictor.resource_plan.infer_worker_assignments)
    )
    admission_atom_capacity = gpu_window_atom_budget(
        predictor.gpu_batch_size,
        worker_count,
    )
    capacity_cycle_count = admission_capacity_cycle_count(
        (
            len(prediction.atom_predictions)
            for prediction in predictions
        ),
        admission_atom_capacity,
    )
    crosses_capacity = capacity_cycle_count > 1
    if args.require_cross_capacity and not crosses_capacity:
        raise RuntimeError(
            "Sample does not cross the MCA rolling admission capacity: "
            f"{hashes['atom_count']} atoms fit in one credit cycle with "
            f"a {admission_atom_capacity}-atom capacity"
        )
    restored_cpu_affinity = tuple(sorted(os.sched_getaffinity(0)))
    if restored_cpu_affinity != initial_cpu_affinity:
        raise RuntimeError("MCA GPU predictor did not restore caller CPU affinity")
    print(
        json.dumps(
            {
                "sample_count": len(predictions),
                "atom_count": hashes["atom_count"],
                "site_count": hashes["site_count"],
                "gpu_indices": args.gpu_indices,
                "gpu_batch_size": predictor.gpu_batch_size,
                "admission_atom_capacity": admission_atom_capacity,
                "admission_capacity_cycles": capacity_cycle_count,
                "crosses_admission_capacity": crosses_capacity,
                "constructor_seconds": constructor_seconds,
                "cpu_affinity_count": len(initial_cpu_affinity),
                "cpu_affinity_restored": True,
                "runs": runs,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
