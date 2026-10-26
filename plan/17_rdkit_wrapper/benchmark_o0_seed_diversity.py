"""Measure Stage-17 O0 final-structure sensitivity to the configured seed.

The experiment reconstructs the fixed 182-case Plan-17 CBond-positive cohort,
runs the complete owner-selected O0 workflow for each seed base, and compares
the selected structures by stable atom-row identity after rigid Kabsch
alignment.  It deliberately measures the final optimized complex rather than
the Open Babel proposal in isolation.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from tests.benchmarks.coordination_complexes.configuration import STANDARD_SETTINGS
from tests.benchmarks.coordination_complexes.validation_cohort import (
    load_validation_cohort,
    semantic_case_molecule,
)
from tests.benchmarks.coordination_complexes.validation_evidence import evaluate_route
from tests.benchmarks.coordination_complexes.validation_routes import (
    execute_provider_route,
)


_CASES: dict[int, Mapping[str, object]] = {}


def _initialize_worker(cohort_path: str) -> None:
    global _CASES
    cohort = load_validation_cohort(Path(cohort_path))
    _CASES = {
        int(case["index"]): case
        for case in cohort["cases"]
        if case["status"] == "cbond"
    }


def _run_case(task: tuple[int, int]) -> dict[str, object]:
    seed_base, case_index = task
    case = _CASES[case_index]
    started = perf_counter()
    try:
        molecule = semantic_case_molecule(case)
        execution = execute_provider_route(
            molecule,
            "O0",
            replace(STANDARD_SETTINGS, seed=seed_base),
            case_index,
        )
        evidence, acceptance = evaluate_route(case, execution)
        original_atom_count = len(case["graph"]["atoms"])
        coordinates = np.asarray(execution.molecule.coordinates, dtype=np.float64)[
            :original_atom_count
        ]
        selected = next(frame for frame in evidence.frames if frame.role == "selected")
        return {
            "case_index": case_index,
            "seed_base": seed_base,
            "effective_seed": seed_base + case_index,
            "completed": True,
            "accepted": bool(acceptance.passed),
            "diagnostics": list(acceptance.diagnostics),
            "elapsed_seconds": perf_counter() - started,
            "selected_energy_kj_mol": selected.energy.value,
            "coordinates": coordinates.tolist(),
        }
    except Exception as exc:
        return {
            "case_index": case_index,
            "seed_base": seed_base,
            "effective_seed": seed_base + case_index,
            "completed": False,
            "accepted": False,
            "diagnostics": [f"{type(exc).__name__}: {exc}"],
            "elapsed_seconds": perf_counter() - started,
            "selected_energy_kj_mol": None,
            "coordinates": None,
        }


def _kabsch_rmsd(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - np.mean(left, axis=0)
    right_centered = right - np.mean(right, axis=0)
    u, _, vt = np.linalg.svd(left_centered.T @ right_centered)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    residual = left_centered @ rotation - right_centered
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


def _coordination_geometry(
    coordinates: np.ndarray,
    metal_row: int,
    donor_rows: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    metal = coordinates[metal_row]
    vectors = coordinates[np.asarray(donor_rows, dtype=np.int64)] - metal
    distances = np.linalg.norm(vectors, axis=1)
    angles = []
    for first, second in combinations(vectors, 2):
        cosine = np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second))
        angles.append(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    return distances, np.asarray(angles, dtype=np.float64)


def _vector_rms_difference(left: np.ndarray, right: np.ndarray) -> float:
    if not len(left):
        return 0.0
    return float(np.sqrt(np.mean(np.square(left - right))))


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "minimum": min(values) if values else None,
        "p50": _percentile(values, 50.0),
        "p75": _percentile(values, 75.0),
        "p90": _percentile(values, 90.0),
        "p95": _percentile(values, 95.0),
        "maximum": max(values) if values else None,
        "mean": float(np.mean(values)) if values else None,
    }


def _rotatable_bonds(smiles: str) -> int:
    molecule = Chem.MolFromSmiles(smiles)
    return int(rdMolDescriptors.CalcNumRotatableBonds(molecule))


def _summarize(
    cohort: Mapping[str, object],
    records: Sequence[Mapping[str, object]],
    seed_bases: Sequence[int],
) -> dict[str, object]:
    cases = {
        int(case["index"]): case
        for case in cohort["cases"]
        if case["status"] == "cbond"
    }
    by_case: dict[int, list[Mapping[str, object]]] = {index: [] for index in cases}
    for record in records:
        by_case[int(record["case_index"])].append(record)

    case_rows = []
    all_heavy_rmsd = []
    ligand_heavy_rmsd = []
    metal_donor_distance_rmsd = []
    donor_angle_rmsd = []
    energy_absolute_differences = []
    for case_index, case_records in sorted(by_case.items()):
        case = cases[case_index]
        atom_rows = case["graph"]["atoms"]
        metal_row = int(case["coordination"]["metal"]["row"])
        donor_rows = tuple(int(row) for row in case["coordination"]["donor_rows"])
        heavy_rows = np.asarray(
            [row for row, atom in enumerate(atom_rows) if int(atom["atomic_number"]) > 1],
            dtype=np.int64,
        )
        ligand_heavy_rows = heavy_rows[heavy_rows != metal_row]
        accepted = [record for record in case_records if record["accepted"]]
        pairs = []
        for left_record, right_record in combinations(accepted, 2):
            left = np.asarray(left_record["coordinates"], dtype=np.float64)
            right = np.asarray(right_record["coordinates"], dtype=np.float64)
            left_distances, left_angles = _coordination_geometry(
                left, metal_row, donor_rows
            )
            right_distances, right_angles = _coordination_geometry(
                right, metal_row, donor_rows
            )
            energy_difference = abs(
                float(left_record["selected_energy_kj_mol"])
                - float(right_record["selected_energy_kj_mol"])
            )
            pair = {
                "seed_bases": [left_record["seed_base"], right_record["seed_base"]],
                "all_heavy_atom_rmsd_angstrom": _kabsch_rmsd(
                    left[heavy_rows], right[heavy_rows]
                ),
                "ligand_heavy_atom_rmsd_angstrom": _kabsch_rmsd(
                    left[ligand_heavy_rows], right[ligand_heavy_rows]
                ),
                "metal_donor_distance_rmsd_angstrom": _vector_rms_difference(
                    left_distances, right_distances
                ),
                "donor_metal_donor_angle_rmsd_degrees": _vector_rms_difference(
                    left_angles, right_angles
                ),
                "selected_energy_absolute_difference_kj_mol": energy_difference,
            }
            pairs.append(pair)
            all_heavy_rmsd.append(pair["all_heavy_atom_rmsd_angstrom"])
            ligand_heavy_rmsd.append(pair["ligand_heavy_atom_rmsd_angstrom"])
            metal_donor_distance_rmsd.append(
                pair["metal_donor_distance_rmsd_angstrom"]
            )
            donor_angle_rmsd.append(pair["donor_metal_donor_angle_rmsd_degrees"])
            energy_absolute_differences.append(
                pair["selected_energy_absolute_difference_kj_mol"]
            )
        case_rows.append(
            {
                "case_index": case_index,
                "rotatable_bonds": _rotatable_bonds(str(case["smiles"])),
                "completed_count": sum(bool(record["completed"]) for record in case_records),
                "accepted_count": len(accepted),
                "maximum_ligand_heavy_atom_rmsd_angstrom": (
                    max(pair["ligand_heavy_atom_rmsd_angstrom"] for pair in pairs)
                    if pairs
                    else None
                ),
                "pairs": pairs,
            }
        )

    comparable_cases = [
        row for row in case_rows if row["maximum_ligand_heavy_atom_rmsd_angstrom"] is not None
    ]
    thresholds = {}
    for threshold in (0.25, 0.5, 1.0, 2.0):
        count = sum(
            float(row["maximum_ligand_heavy_atom_rmsd_angstrom"]) >= threshold
            for row in comparable_cases
        )
        thresholds[str(threshold)] = {
            "case_count": count,
            "case_fraction": count / len(comparable_cases) if comparable_cases else None,
        }

    seed_outcomes = {}
    for seed_base in seed_bases:
        selected = [record for record in records if record["seed_base"] == seed_base]
        seed_outcomes[str(seed_base)] = {
            "completed": sum(bool(record["completed"]) for record in selected),
            "accepted": sum(bool(record["accepted"]) for record in selected),
            "elapsed_seconds": sum(float(record["elapsed_seconds"]) for record in selected),
            "failed_case_indices": [
                int(record["case_index"]) for record in selected if not record["accepted"]
            ],
        }

    common_accepted = sum(
        row["accepted_count"] == len(seed_bases) for row in case_rows
    )
    return {
        "seed_bases": list(seed_bases),
        "effective_seed_rule": "seed_base + one_based_case_index",
        "case_count": len(cases),
        "seed_outcomes": seed_outcomes,
        "common_accepted_case_count": common_accepted,
        "comparable_case_count": len(comparable_cases),
        "pair_distributions": {
            "all_heavy_atom_rmsd_angstrom": _distribution(all_heavy_rmsd),
            "ligand_heavy_atom_rmsd_angstrom": _distribution(ligand_heavy_rmsd),
            "metal_donor_distance_rmsd_angstrom": _distribution(
                metal_donor_distance_rmsd
            ),
            "donor_metal_donor_angle_rmsd_degrees": _distribution(donor_angle_rmsd),
            "selected_energy_absolute_difference_kj_mol": _distribution(
                energy_absolute_differences
            ),
        },
        "case_thresholds_by_maximum_ligand_heavy_atom_rmsd": thresholds,
        "cases": case_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed-base", type=int, action="append", required=True)
    arguments = parser.parse_args()

    cohort = load_validation_cohort(arguments.cohort)
    case_indices = tuple(int(index) for index in cohort["partition"]["cbond_indices"])
    seed_bases = tuple(arguments.seed_base)
    tasks = tuple((seed_base, index) for seed_base in seed_bases for index in case_indices)
    started = perf_counter()
    records = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=arguments.workers,
        mp_context=context,
        initializer=_initialize_worker,
        initargs=(str(arguments.cohort.resolve()),),
    ) as executor:
        futures = [executor.submit(_run_case, task) for task in tasks]
        for completed_count, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            records.append(record)
            print(
                f"[{completed_count}/{len(tasks)}] case={record['case_index']} "
                f"seed_base={record['seed_base']} accepted={record['accepted']}",
                flush=True,
            )

    records.sort(key=lambda row: (int(row["seed_base"]), int(row["case_index"])))
    document = {
        "schema_version": 1,
        "workflow": "O0 complete complex workflow",
        "wall_seconds": perf_counter() - started,
        "summary": _summarize(cohort, records, seed_bases),
        "runs": records,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in document["summary"].items()
                if key != "cases"
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
