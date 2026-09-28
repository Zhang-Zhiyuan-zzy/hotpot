"""Generate the small native-backend comparison published in README.md."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
from openbabel import openbabel as ob
from rdkit import Chem
from rdkit.Chem import AllChem

import hotpot as hp
from hotpot.cheminfo import forcefields as ff


README_BENCHMARK_SMILES = (
    "CCO",
    "CC(=O)O",
    "c1ccccc1",
    "c1ccccc1CN",
    "CCN(CC)CC",
    "O=C(O)c1ccccc1O",
    "C1CCCCC1",
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
)


@dataclass(frozen=True)
class BenchmarkResult:
    backend: str
    molecule_count: int
    successful_runs: int
    quality_passes: int
    median_workflow_ms: float
    total_workflow_ms: float


def _rdkit_build(smiles: str, seed: int) -> hp.Molecule:
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = seed
    if AllChem.EmbedMolecule(molecule, parameters) != 0:
        raise RuntimeError("RDKit could not embed the molecule")
    AllChem.UFFOptimizeMolecule(molecule, maxIters=200)
    return hp.to_hotpot_mol(molecule)


def _openbabel_build(smiles: str, seed: int) -> hp.Molecule:
    molecule = hp.read_mol(smiles)
    molecule.add_hydrogens()
    obmol = molecule.to_obmol()
    ob_builder = ob.OBBuilder()
    if not ob_builder.Build(obmol):
        raise RuntimeError("Open Babel could not embed the molecule")
    backend = ob.OBForceField.FindForceField("UFF")
    if backend is None or not backend.Setup(obmol):
        raise RuntimeError("Open Babel could not initialize UFF")
    backend.ConjugateGradients(200)
    backend.GetCoordinates(obmol)
    return hp.to_hotpot_mol(obmol)


def _hotpot_build(smiles: str, seed: int) -> hp.Molecule:
    molecule = hp.read_mol(smiles)
    molecule.build3d(
        forcefield="UFF",
        epochs=2,
        steps_per_epoch=100,
        quality_level="standard",
        seed=seed,
    )
    return molecule


def _measure_backend(
    name: str,
    build: Callable[[str, int], hp.Molecule],
    smiles_values: Iterable[str],
    repeats: int,
) -> BenchmarkResult:
    durations = []
    successful_runs = 0
    quality_passes = 0
    values = tuple(smiles_values)
    for repeat in range(repeats):
        for index, smiles in enumerate(values):
            started = time.perf_counter()
            try:
                molecule = build(smiles, 20260928 + repeat * len(values) + index)
            except Exception:
                durations.append((time.perf_counter() - started) * 1000.0)
                continue
            durations.append((time.perf_counter() - started) * 1000.0)
            successful_runs += 1
            quality_passes += int(
                ff.evaluate_structure_acceptance(
                    molecule,
                    level="standard",
                ).passed
            )
    return BenchmarkResult(
        backend=name,
        molecule_count=len(values) * repeats,
        successful_runs=successful_runs,
        quality_passes=quality_passes,
        median_workflow_ms=statistics.median(durations),
        total_workflow_ms=sum(durations),
    )


def run_benchmark(
    smiles_values: Iterable[str] = README_BENCHMARK_SMILES,
    *,
    repeats: int = 3,
) -> tuple[BenchmarkResult, ...]:
    return (
        _measure_backend("RDKit native", _rdkit_build, smiles_values, repeats),
        _measure_backend("Open Babel native", _openbabel_build, smiles_values, repeats),
        _measure_backend("Hotpot", _hotpot_build, smiles_values, repeats),
    )


def write_results(results: tuple[BenchmarkResult, ...], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    (output_dir / "forcefield_validation.json").write_text(
        json.dumps(rows, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "forcefield_validation.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = [result.backend for result in results]
    median_seconds = [result.median_workflow_ms / 1000.0 for result in results]
    pass_rates = [
        100.0 * result.quality_passes / result.molecule_count
        for result in results
    ]
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    axes[0].bar(labels, median_seconds, color=("#4C78A8", "#F58518", "#54A24B"))
    axes[0].set_ylabel("Median workflow time (s)")
    axes[0].tick_params(axis="x", rotation=18)
    axes[1].bar(labels, pass_rates, color=("#4C78A8", "#F58518", "#54A24B"))
    axes[1].set_ylabel("Hotpot standard quality pass (%)")
    axes[1].set_ylim(0.0, 105.0)
    axes[1].tick_params(axis="x", rotation=18)
    figure.suptitle("UFF build/optimization on the README neutral-molecule set")
    figure.tight_layout()
    figure.savefig(output_dir / "forcefield_validation.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("assets/readme"),
    )
    parser.add_argument("--repeats", type=int, default=3)
    arguments = parser.parse_args()
    write_results(
        run_benchmark(repeats=arguments.repeats),
        arguments.output_dir,
    )


if __name__ == "__main__":
    main()
