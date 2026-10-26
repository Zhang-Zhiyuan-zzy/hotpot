"""Measure Stage-17 native rdWrapper output diversity across fixed seeds."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from hotpot.cheminfo.AImodels.mca.conformer import ensure_3d_conformer
from hotpot.cheminfo.AImodels.mca.serialization import (
    normalize_molecule,
    rdmol_from_dto,
)
from hotpot.cheminfo.conformers import (
    ConformerSamplingRequest,
    RDKitConformerAdapter,
)
from hotpot.cheminfo.convert import to_hotpot_mol


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


def _coordinates(smiles: str, seed: int, engine: str) -> np.ndarray:
    dto = normalize_molecule(smiles)
    rd_molecule = rdmol_from_dto(dto)
    if engine == "current_rdkit":
        conformer = ensure_3d_conformer(rd_molecule, seed)
        return np.asarray(conformer.GetConformer().GetPositions(), dtype=np.float64)
    hotpot_molecule = to_hotpot_mol(rd_molecule)
    heavy_atom_count = rd_molecule.GetNumAtoms()
    hotpot_molecule.add_hydrogens(
        rm_polar_hs=False,
        rng=np.random.default_rng(seed),
    )
    batch = RDKitConformerAdapter(materialized_hydrogens=True).sample(
        hotpot_molecule,
        ConformerSamplingRequest(conformer_count=1, seed=seed),
    )
    return np.asarray(batch.coordinates[0, :heavy_atom_count], dtype=np.float64)


def _run(task: tuple[int, str, tuple[int, ...], str]) -> dict[str, object]:
    index, smiles, seeds, engine = task
    conformers = {seed: _coordinates(smiles, seed, engine) for seed in seeds}
    pairs = [
        {
            "seeds": [left, right],
            "heavy_atom_rmsd_angstrom": _kabsch_rmsd(
                conformers[left], conformers[right]
            ),
        }
        for left, right in combinations(seeds, 2)
    ]
    molecule = Chem.MolFromSmiles(smiles)
    return {
        "index": index,
        "smiles": smiles,
        "heavy_atom_count": molecule.GetNumHeavyAtoms(),
        "rotatable_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(molecule)),
        "maximum_heavy_atom_rmsd_angstrom": max(
            pair["heavy_atom_rmsd_angstrom"] for pair in pairs
        ),
        "pairs": pairs,
    }


def _distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "minimum": float(np.min(array)),
        "p50": float(np.percentile(array, 50.0)),
        "p75": float(np.percentile(array, 75.0)),
        "p90": float(np.percentile(array, 90.0)),
        "p95": float(np.percentile(array, 95.0)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument(
        "--engine",
        choices=("native_rdwrapper", "current_rdkit"),
        default="native_rdwrapper",
    )
    arguments = parser.parse_args()
    smiles = [
        line.split()[0]
        for line in arguments.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    seeds = tuple(arguments.seed)
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=arguments.workers,
        mp_context=context,
    ) as executor:
        cases = list(
            executor.map(
                _run,
                (
                    (index, value, seeds, arguments.engine)
                    for index, value in enumerate(smiles, 1)
                ),
            )
        )
    pairs = [pair["heavy_atom_rmsd_angstrom"] for case in cases for pair in case["pairs"]]
    thresholds = {}
    for threshold in (0.25, 0.5, 1.0, 2.0):
        count = sum(
            case["maximum_heavy_atom_rmsd_angstrom"] >= threshold for case in cases
        )
        thresholds[str(threshold)] = {
            "case_count": count,
            "case_fraction": count / len(cases),
        }
    document = {
        "schema_version": 1,
        "engine": (
            "native rdWrapper ETKDGv3 without force-field optimization"
            if arguments.engine == "native_rdwrapper"
            else "current Python RDKit ETKDGv3 plus MMFF/UFF optimization"
        ),
        "seeds": list(seeds),
        "case_count": len(cases),
        "pair_distribution": _distribution(pairs),
        "case_thresholds_by_maximum_heavy_atom_rmsd": thresholds,
        "cases": cases,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in document.items() if key != "cases"},
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
