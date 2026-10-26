#!/usr/bin/env python3
"""Stage-17 paired MCA benchmark for current and native conformer engines."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
from rdkit import Chem, rdBase
from scipy.stats import pearsonr, spearmanr

from hotpot.cheminfo import obWrappers, rdWrappers
from hotpot.cheminfo.AImodels.mca import cpu_backend, gpu_backend
from hotpot.cheminfo.AImodels.mca.cpu_backend import MCACpuPredictor
from hotpot.cheminfo.AImodels.mca.gpu_backend import MCAGpuPredictor
from hotpot.cheminfo.AImodels.mca.featurizer import mol_to_unimolv2
from hotpot.cheminfo.AImodels.mca.pipeline import detect_sites
from hotpot.cheminfo.AImodels.mca.serialization import (
    rdmol_from_dto,
)
from hotpot.cheminfo.AImodels.mca.workers import (
    CPUPreprocessResult,
    FeatureResult,
)
from hotpot.cheminfo.conformers import (
    ConformerSamplingRequest,
    RDKitConformerAdapter,
)
from hotpot.cheminfo.convert import to_hotpot_mol
from hotpot.cheminfo.obWrappers.native import _native_module as _ob_native_module


ROUTES = ("current_rdkit", "native_rdwrapper", "native_obwrapper")


def _read_smiles(path: Path, sample_count: int | None) -> list[str]:
    smiles = [
        line.split()[0]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return smiles if sample_count is None else smiles[:sample_count]


def _with_coordinates(mol: Chem.Mol, coordinates: np.ndarray) -> Chem.Mol:
    result = Chem.Mol(mol)
    result.RemoveAllConformers()
    conformer = Chem.Conformer(result.GetNumAtoms())
    conformer.Set3D(True)
    for atom_index, coordinate in enumerate(coordinates):
        conformer.SetAtomPosition(atom_index, coordinate)
    result.AddConformer(conformer, assignId=True)
    return result


def _hydrogenated_hotpot_molecule(molecule, seed: int):
    rdmol = rdmol_from_dto(molecule)
    hotpot_mol = to_hotpot_mol(rdmol)
    heavy_atom_count = rdmol.GetNumAtoms()
    hotpot_mol.add_hydrogens(
        rm_polar_hs=False,
        rng=np.random.default_rng(seed),
    )
    return rdmol, hotpot_mol, heavy_atom_count


def _native_rdkit_features(molecule, max_atoms: int, seed: int):
    rdmol, hotpot_mol, heavy_atom_count = _hydrogenated_hotpot_molecule(
        molecule,
        seed,
    )
    sampled = RDKitConformerAdapter(materialized_hydrogens=True).sample(
        hotpot_mol,
        ConformerSamplingRequest(conformer_count=1, seed=seed),
    )
    conformer = _with_coordinates(
        rdmol,
        sampled.coordinates[0, :heavy_atom_count],
    )
    return mol_to_unimolv2(conformer, max_atoms)


def _native_openbabel_features(molecule, max_atoms: int, seed: int):
    rdmol, hotpot_mol, heavy_atom_count = _hydrogenated_hotpot_molecule(
        molecule,
        seed,
    )
    native = _ob_native_module()
    native.seed_random(seed)
    build_report = obWrappers.build(hotpot_mol)
    if not build_report.succeeded:
        raise RuntimeError("Open Babel could not build initial coordinates")
    try:
        obWrappers.optimize(
            hotpot_mol,
            "MMFF94s",
            epochs=40,
            steps_per_epoch=5,
        )
    except native.ForceFieldSetupError:
        obWrappers.optimize(
            hotpot_mol,
            "UFF",
            epochs=40,
            steps_per_epoch=5,
        )
    conformer = _with_coordinates(
        rdmol,
        np.asarray(hotpot_mol.coordinates[:heavy_atom_count], dtype=np.float64),
    )
    return mol_to_unimolv2(conformer, max_atoms)


def native_rdwrapper_worker(task):
    sites = detect_sites(task.molecule) if task.sites is None else task.sites
    return CPUPreprocessResult(
        molecule_index=task.molecule_index,
        molecule=task.molecule,
        sites=sites,
        feature=_native_rdkit_features(
            task.molecule,
            task.max_atoms,
            task.conformer_seed,
        ),
    )


def native_obwrapper_worker(task):
    sites = detect_sites(task.molecule) if task.sites is None else task.sites
    return CPUPreprocessResult(
        molecule_index=task.molecule_index,
        molecule=task.molecule,
        sites=sites,
        feature=_native_openbabel_features(
            task.molecule,
            task.max_atoms,
            task.conformer_seed,
        ),
    )


def native_rdwrapper_feature_worker(task):
    return FeatureResult(
        molecule_index=task.molecule_index,
        feature=_native_rdkit_features(
            task.molecule,
            task.max_atoms,
            task.conformer_seed,
        ),
    )


def native_obwrapper_feature_worker(task):
    return FeatureResult(
        molecule_index=task.molecule_index,
        feature=_native_openbabel_features(
            task.molecule,
            task.max_atoms,
            task.conformer_seed,
        ),
    )


def _prediction_payload(predictions) -> dict[str, object]:
    molecules = []
    for prediction in predictions:
        sites = tuple(
            (site.atom_index, site.site_type) for site in prediction.sites
        )
        site_indices = {site[0] for site in sites}
        molecules.append(
            {
                "smiles": prediction.smiles,
                "sites": sites,
                "is_Nuc_site": [
                    atom.atom_index in site_indices
                    for atom in prediction.atom_predictions
                ],
                "mca_kj_mol": [
                    atom.mca_kj_mol for atom in prediction.atom_predictions
                ],
            }
        )
    values = np.asarray(
        [value for molecule in molecules for value in molecule["mca_kj_mol"]],
        dtype=np.float64,
    )
    return {
        "molecules": molecules,
        "atom_count": int(values.size),
        "site_count": sum(len(molecule["sites"]) for molecule in molecules),
        "mca_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
    }


def _run_route(arguments) -> None:
    smiles = _read_smiles(arguments.input, arguments.sample_count)
    if arguments.route == "native_rdwrapper":
        cpu_backend.cpu_preprocess_worker = native_rdwrapper_worker
        gpu_backend.build_features_worker = native_rdwrapper_feature_worker
    elif arguments.route == "native_obwrapper":
        cpu_backend.cpu_preprocess_worker = native_obwrapper_worker
        gpu_backend.build_features_worker = native_obwrapper_feature_worker

    constructor_started = time.perf_counter()
    if arguments.backend == "cpu":
        predictor = MCACpuPredictor(
            cpu_count=arguments.cpu_count,
            batch_size=arguments.batch_size,
            conformer_seed=arguments.seed,
            allow_charged=arguments.allow_charged,
        )
    else:
        predictor = MCAGpuPredictor(
            gpu_indices=(arguments.gpu_index,),
            site_cpu_per_gpu=arguments.site_cpu_per_gpu,
            infer_cpu_per_gpu=arguments.infer_cpu_per_gpu,
            gpu_batch_size=arguments.gpu_batch_size,
            conformer_seed=arguments.seed,
            allow_charged=arguments.allow_charged,
        )
    constructor_seconds = time.perf_counter() - constructor_started
    runs = []
    reference = None
    repeat_predictions = []
    with predictor:
        for repeat in range(1, arguments.repeats + 1):
            started = time.perf_counter()
            predictions = list(predictor.iter_predict(smiles))
            seconds = time.perf_counter() - started
            payload = _prediction_payload(predictions)
            if reference is None:
                reference = payload
            repeat_predictions.append(payload)
            runs.append(
                {
                    "repeat": repeat,
                    "seconds": seconds,
                    "molecules_per_second": len(smiles) / seconds,
                    "mca_sha256": payload["mca_sha256"],
                }
            )

    result = {
        "schema_version": 1,
        "route": arguments.route,
        "backend": arguments.backend,
        "input": str(arguments.input),
        "input_sha256": hashlib.sha256(arguments.input.read_bytes()).hexdigest(),
        "sample_count": len(smiles),
        "cpu_count": arguments.cpu_count,
        "batch_size": arguments.batch_size,
        "gpu_index": arguments.gpu_index,
        "gpu_batch_size": arguments.gpu_batch_size,
        "site_cpu_per_gpu": arguments.site_cpu_per_gpu,
        "infer_cpu_per_gpu": arguments.infer_cpu_per_gpu,
        "seed": arguments.seed,
        "allow_charged": arguments.allow_charged,
        "python_rdkit_version": rdBase.rdkitVersion,
        "native_rdwrapper_runtime": vars(rdWrappers.runtime_info()),
        "constructor_seconds": constructor_seconds,
        "median_seconds": statistics.median(run["seconds"] for run in runs),
        "median_molecules_per_second": statistics.median(
            run["molecules_per_second"] for run in runs
        ),
        "runs": runs,
        "repeat_mca_stable": len(
            {payload["mca_sha256"] for payload in repeat_predictions}
        )
        == 1,
        "predictions": reference,
        "repeat_predictions": repeat_predictions,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in {"predictions", "repeat_predictions"}
            },
            indent=2,
        )
    )


def _paired_statistics(reference: dict, candidate: dict) -> dict[str, object]:
    reference_molecules = reference["predictions"]["molecules"]
    candidate_molecules = candidate["predictions"]["molecules"]
    if len(reference_molecules) != len(candidate_molecules):
        raise RuntimeError("Route molecule counts do not match")

    exact_sites = True
    exact_flags = True
    all_reference = []
    all_candidate = []
    site_reference = []
    site_candidate = []
    for reference_molecule, candidate_molecule in zip(
        reference_molecules,
        candidate_molecules,
    ):
        if reference_molecule["smiles"] != candidate_molecule["smiles"]:
            raise RuntimeError("Route molecule ordering does not match")
        exact_sites &= reference_molecule["sites"] == candidate_molecule["sites"]
        exact_flags &= (
            reference_molecule["is_Nuc_site"]
            == candidate_molecule["is_Nuc_site"]
        )
        reference_values = np.asarray(reference_molecule["mca_kj_mol"])
        candidate_values = np.asarray(candidate_molecule["mca_kj_mol"])
        flags = np.asarray(reference_molecule["is_Nuc_site"], dtype=bool)
        all_reference.extend(reference_values)
        all_candidate.extend(candidate_values)
        site_reference.extend(reference_values[flags])
        site_candidate.extend(candidate_values[flags])

    def metrics(reference_values, candidate_values):
        reference_array = np.asarray(reference_values, dtype=np.float64)
        candidate_array = np.asarray(candidate_values, dtype=np.float64)
        difference = candidate_array - reference_array
        return {
            "count": int(reference_array.size),
            "pearson": float(pearsonr(reference_array, candidate_array).statistic),
            "spearman": float(spearmanr(reference_array, candidate_array).statistic),
            "mean_signed_error_kj_mol": float(np.mean(difference)),
            "mae_kj_mol": float(np.mean(np.abs(difference))),
            "median_absolute_error_kj_mol": float(
                np.median(np.abs(difference))
            ),
            "rmse_kj_mol": float(np.sqrt(np.mean(np.square(difference)))),
            "p95_absolute_error_kj_mol": float(
                np.quantile(np.abs(difference), 0.95)
            ),
            "maximum_absolute_error_kj_mol": float(np.max(np.abs(difference))),
        }

    return {
        "candidate_route": candidate["route"],
        "site_records_exact": exact_sites,
        "is_Nuc_site_exact": exact_flags,
        "all_atoms": metrics(all_reference, all_candidate),
        "nucleophilic_sites": metrics(site_reference, site_candidate),
        "median_seconds": candidate["median_seconds"],
        "median_molecules_per_second": candidate["median_molecules_per_second"],
        "throughput_ratio_to_current": (
            candidate["median_molecules_per_second"]
            / reference["median_molecules_per_second"]
        ),
    }


def _analyze(arguments) -> None:
    inputs = [
        json.loads(path.read_text(encoding="utf-8")) for path in arguments.results
    ]
    by_route = {result["route"]: result for result in inputs}
    reference = by_route["current_rdkit"]
    analysis: dict[str, object] = {
        "schema_version": 1,
        "reference_route": "current_rdkit",
        "reference_median_seconds": reference["median_seconds"],
        "reference_median_molecules_per_second": reference[
            "median_molecules_per_second"
        ],
        "comparisons": [],
    }
    comparisons = analysis["comparisons"]
    for route in ROUTES[1:]:
        candidate = by_route[route]
        paired_repeats = []
        for repeat_index, predictions in enumerate(
            candidate.get("repeat_predictions", [candidate["predictions"]]),
            start=1,
        ):
            repeat_candidate = dict(candidate, predictions=predictions)
            comparison = _paired_statistics(reference, repeat_candidate)
            comparison["repeat"] = repeat_index
            paired_repeats.append(comparison)
        repeated_predictions = candidate.get(
            "repeat_predictions",
            [candidate["predictions"]],
        )
        repeat_drift = []
        repeat_reference = dict(candidate, predictions=repeated_predictions[0])
        for repeat_index, predictions in enumerate(
            repeated_predictions[1:],
            start=2,
        ):
            drift_candidate = dict(candidate, predictions=predictions)
            drift = _paired_statistics(repeat_reference, drift_candidate)
            drift["reference_repeat"] = 1
            drift["candidate_repeat"] = repeat_index
            repeat_drift.append(drift)
        comparisons.append(
            {
                "candidate_route": route,
                "repeat_mca_stable": candidate.get("repeat_mca_stable", True),
                "paired_repeats": paired_repeats,
                "repeat_drift": repeat_drift,
            }
        )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(analysis, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(analysis, indent=2))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--route", choices=ROUTES, required=True)
    run_parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    run_parser.add_argument("--input", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--sample-count", type=int)
    run_parser.add_argument("--cpu-count", type=int, default=8)
    run_parser.add_argument("--batch-size", type=int, default=64)
    run_parser.add_argument("--gpu-index", type=int, default=0)
    run_parser.add_argument("--gpu-batch-size", type=int, default=2048)
    run_parser.add_argument("--site-cpu-per-gpu", type=int, default=1)
    run_parser.add_argument("--infer-cpu-per-gpu", type=int, default=3)
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--repeats", type=int, default=3)
    run_parser.add_argument("--allow-charged", action="store_true")
    run_parser.set_defaults(handler=_run_route)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--results", type=Path, nargs=3, required=True)
    analyze_parser.add_argument("--output", type=Path, required=True)
    analyze_parser.set_defaults(handler=_analyze)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
