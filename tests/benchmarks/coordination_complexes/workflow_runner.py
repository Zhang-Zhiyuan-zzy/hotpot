"""Independent ligand/complex benchmark runner for five fixed backends."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import statistics
import traceback
import warnings
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Optional

import numpy as np

from .backend_comparison import (
    CanonicalCase,
    _CaseFailure,
    _quality_payload,
    _rebuild_complex,
    _record_native_frame,
    _run_openbabel,
    _run_rdkit,
    _write_native_trajectory,
)
from .cohort import BenchmarkCohort, LigandCase, resolve_cohort
from .io import json_value, sha256_file, write_json

if TYPE_CHECKING:
    from hotpot import Molecule
    from hotpot.cheminfo.forcefields import ForceFieldTrajectory


SCHEMA_VERSION = 1
BACKENDS = (
    "rdkit",
    "openbabel",
    "obwrappers",
    "hotpot_optimize_complex",
    "hotpot_auto",
)
FORCEFIELD = "UFF"
OPTIMIZATION_EPOCHS = 100
STEPS_PER_EPOCH = 100
OPTIMIZATION_STEPS = OPTIMIZATION_EPOCHS * STEPS_PER_EPOCH
TRAJECTORY_INTERVAL_STEPS = 100
SEED = 20260921


def _prepare_ligand(case: LigandCase) -> Molecule:
    from hotpot import read_mol
    from hotpot.cheminfo.forcefields.working_copy import (
        _hydrogenated_working_copy,
    )

    ligand = _hydrogenated_working_copy(
        read_mol(case.smiles, fmt="smi"),
        add_hydrogens=True,
        seed=case.seed,
    )
    ligand.coordinates = np.zeros((len(ligand.atoms), 3), dtype=float)
    return ligand


def _new_trajectory(mol: Molecule) -> ForceFieldTrajectory:
    from hotpot.cheminfo import forcefields as ff

    trajectory = ff.ForceFieldTrajectory.from_molecule(
        mol,
        start=ff.TrajectoryStart.LIGAND_BUILD,
    )
    _record_native_frame(
        trajectory,
        mol,
        stage=ff.TrajectoryStage.LIGAND_BUILD,
        event=ff.TrajectoryEvent.INITIAL,
    )
    return trajectory


def _run_rdkit_ligand(
    ligand: Molecule,
    case: LigandCase,
    trajectory: ForceFieldTrajectory,
) -> dict[str, object]:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from hotpot.cheminfo import forcefields as ff

    rd_mol = Chem.MolFromSmiles(case.smiles)
    if rd_mol is None:
        raise _CaseFailure(
            "failed_execution",
            "RDKit could not parse the ligand SMILES",
        )
    rd_mol = Chem.AddHs(rd_mol)
    canonical_heavy = [atom.idx for atom in ligand.atoms if not atom.is_hydrogen]
    rd_heavy = [
        atom.GetIdx() for atom in rd_mol.GetAtoms() if atom.GetAtomicNum() != 1
    ]
    if [ligand.atoms[index].atomic_number for index in canonical_heavy] != [
        rd_mol.GetAtomWithIdx(index).GetAtomicNum() for index in rd_heavy
    ]:
        raise _CaseFailure(
            "failed_execution",
            "RDKit changed the ligand heavy-atom order",
        )
    coordinate_map = np.full(len(ligand.atoms), -1, dtype=np.int64)
    for canonical_index, rd_index in zip(canonical_heavy, rd_heavy):
        coordinate_map[canonical_index] = rd_index
        canonical_hydrogens = sorted(
            atom.idx
            for atom in ligand.atoms[canonical_index].neighbours
            if atom.is_hydrogen
        )
        rd_hydrogens = sorted(
            atom.GetIdx()
            for atom in rd_mol.GetAtomWithIdx(rd_index).GetNeighbors()
            if atom.GetAtomicNum() == 1
        )
        if len(canonical_hydrogens) != len(rd_hydrogens):
            raise _CaseFailure(
                "failed_execution",
                f"RDKit hydrogen count differs at atom {canonical_index}",
            )
        coordinate_map[canonical_hydrogens] = rd_hydrogens
    if np.any(coordinate_map < 0):
        raise _CaseFailure(
            "failed_execution",
            "RDKit-to-Hotpot ligand atom mapping is incomplete",
        )
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = case.seed
    parameters.useRandomCoords = True
    parameters.ignoreSmoothingFailures = True

    build_started = perf_counter()
    embed_status = int(AllChem.EmbedMolecule(rd_mol, parameters))
    build_seconds = perf_counter() - build_started
    if embed_status != 0:
        raise _CaseFailure(
            "failed_execution",
            f"RDKit ETKDGv3 returned status {embed_status}",
            {"build_seconds": build_seconds},
        )
    ligand.coordinates = np.asarray(
        rd_mol.GetConformer().GetPositions(),
        dtype=float,
    )[coordinate_map]
    _record_native_frame(
        trajectory,
        ligand,
        stage=ff.TrajectoryStage.LIGAND_BUILD,
        event=ff.TrajectoryEvent.BUILD_COMPLETE,
    )

    if AllChem.MMFFHasAllMoleculeParams(rd_mol):
        properties = AllChem.MMFFGetMoleculeProperties(rd_mol, mmffVariant="MMFF94")
        forcefield = AllChem.MMFFGetMoleculeForceField(rd_mol, properties)
        forcefield_name = "MMFF94"
    else:
        forcefield = AllChem.UFFGetMoleculeForceField(rd_mol)
        forcefield_name = "UFF"
    if forcefield is None:
        raise _CaseFailure(
            "failed_execution",
            "RDKit could not construct an MMFF or UFF force field",
            {"build_seconds": build_seconds},
        )

    optimize_started = perf_counter()
    forcefield.Initialize()
    optimize_status = 1
    snapshot_count = 0
    for snapshot_count in range(1, OPTIMIZATION_EPOCHS + 1):
        optimize_status = int(forcefield.Minimize(maxIts=TRAJECTORY_INTERVAL_STEPS))
        ligand.coordinates = np.asarray(
            rd_mol.GetConformer().GetPositions(),
            dtype=float,
        )[coordinate_map]
        _record_native_frame(
            trajectory,
            ligand,
            stage=ff.TrajectoryStage.FINAL_OPTIMIZATION,
            event=ff.TrajectoryEvent.EPOCH_COMPLETE,
            energy_kj_mol=float(forcefield.CalcEnergy()) * 4.184,
            step=snapshot_count,
        )
        if optimize_status <= 0:
            break
    optimization_seconds = perf_counter() - optimize_started
    if optimize_status < 0:
        raise _CaseFailure(
            "failed_execution",
            f"RDKit {forcefield_name} returned status {optimize_status}",
            {
                "build_seconds": build_seconds,
                "optimization_seconds": optimization_seconds,
            },
        )
    return {
        "build_seconds": build_seconds,
        "optimization_seconds": optimization_seconds,
        "forcefield_name": forcefield_name,
        "converged": optimize_status == 0,
        "optimization_snapshot_count": snapshot_count,
    }


def _run_openbabel_ligand(
    ligand: Molecule,
    case: LigandCase,
    trajectory: ForceFieldTrajectory,
) -> dict[str, object]:
    import os

    from openbabel import openbabel as ob

    from hotpot.cheminfo import forcefields as ff
    from hotpot.cheminfo.obconvert import extract_obmol_coordinates

    obmol = ligand.to_obmol()
    previous_seed = os.environ.get("OB_RANDOM_SEED")
    os.environ["OB_RANDOM_SEED"] = str(case.seed)
    try:
        build_started = perf_counter()
        built = bool(ob.OBBuilder().Build(obmol))
        build_seconds = perf_counter() - build_started
    finally:
        if previous_seed is None:
            os.environ.pop("OB_RANDOM_SEED", None)
        else:
            os.environ["OB_RANDOM_SEED"] = previous_seed
    if not built:
        raise _CaseFailure(
            "failed_execution",
            "Open Babel OBBuilder returned false",
            {"build_seconds": build_seconds},
        )
    ligand.coordinates = extract_obmol_coordinates(obmol)
    _record_native_frame(
        trajectory,
        ligand,
        stage=ff.TrajectoryStage.LIGAND_BUILD,
        event=ff.TrajectoryEvent.BUILD_COMPLETE,
    )

    forcefield = ob.OBForceField.FindForceField(FORCEFIELD)
    if forcefield is None or not forcefield.Setup(obmol):
        raise _CaseFailure(
            "failed_execution",
            "Open Babel UFF setup failed",
            {"build_seconds": build_seconds},
        )
    optimize_started = perf_counter()
    forcefield.ConjugateGradientsInitialize(OPTIMIZATION_STEPS)
    snapshot_count = 0
    for snapshot_count in range(1, OPTIMIZATION_EPOCHS + 1):
        continuing = bool(
            forcefield.ConjugateGradientsTakeNSteps(TRAJECTORY_INTERVAL_STEPS)
        )
        forcefield.GetCoordinates(obmol)
        ligand.coordinates = extract_obmol_coordinates(obmol)
        _record_native_frame(
            trajectory,
            ligand,
            stage=ff.TrajectoryStage.FINAL_OPTIMIZATION,
            event=ff.TrajectoryEvent.EPOCH_COMPLETE,
            energy_kj_mol=float(forcefield.Energy()),
            step=snapshot_count,
        )
        if not continuing:
            break
    optimization_seconds = perf_counter() - optimize_started
    return {
        "build_seconds": build_seconds,
        "optimization_seconds": optimization_seconds,
        "forcefield_name": FORCEFIELD,
        "converged": None,
        "optimization_snapshot_count": snapshot_count,
    }


def _run_native_target(
    backend: str,
    target: str,
    mol: Molecule,
    case: object,
    trajectory: ForceFieldTrajectory,
) -> dict[str, object]:
    compute_started = perf_counter()
    if target == "complex":
        runner = _run_rdkit if backend == "rdkit" else _run_openbabel
    else:
        runner = (
            _run_rdkit_ligand
            if backend == "rdkit"
            else _run_openbabel_ligand
        )
    result = runner(mol, case, trajectory)
    result["compute_seconds"] = perf_counter() - compute_started
    return result


def _record_obwrappers_trajectory(
    mol: Molecule,
    trajectory: ForceFieldTrajectory,
    optimization_report: object,
) -> None:
    from hotpot.cheminfo import forcefields as ff

    frame_indices = []
    for frame in optimization_report.frames:
        mol.coordinates = frame.coordinates
        recorded = trajectory.record_molecule(
            mol,
            stage=ff.TrajectoryStage.FINAL_OPTIMIZATION,
            event=ff.TrajectoryEvent.EPOCH_COMPLETE,
            energy_kj_mol=frame.energy,
            step=frame.epoch_index,
        )
        frame_indices.append(recorded.index)
    mol.coordinates = optimization_report.coordinates
    if frame_indices:
        if optimization_report.selected_frame_index >= 0:
            trajectory.select(frame_indices[optimization_report.selected_frame_index])
        trajectory.set_terminal(frame_indices[-1])
    else:
        frame = trajectory.record_molecule(
            mol,
            stage=ff.TrajectoryStage.FINAL_OPTIMIZATION,
            event=ff.TrajectoryEvent.TERMINAL,
            energy_kj_mol=optimization_report.final_energy,
        )
        trajectory.select(frame.index)
        trajectory.set_terminal(frame.index)


def _run_obwrappers_target(
    mol: Molecule,
    trajectory: ForceFieldTrajectory,
) -> dict[str, object]:
    from hotpot.cheminfo import forcefields as ff
    from hotpot.cheminfo import obWrappers

    build_started = perf_counter()
    build_report = obWrappers.build(mol)
    build_seconds = perf_counter() - build_started
    if not build_report.succeeded:
        raise _CaseFailure(
            "failed_execution",
            "obWrappers.build returned an unsuccessful report",
            {"build_seconds": build_seconds},
        )
    _record_native_frame(
        trajectory,
        mol,
        stage=ff.TrajectoryStage.LIGAND_BUILD,
        event=ff.TrajectoryEvent.BUILD_COMPLETE,
    )
    optimize_started = perf_counter()
    optimization_report = obWrappers.optimize(
        mol,
        FORCEFIELD,
        epochs=OPTIMIZATION_EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        retain_frames=True,
        retain_epoch_history=True,
        convergence_level=ff.ConvergenceLevel.FAST,
    )
    optimization_seconds = perf_counter() - optimize_started
    _record_obwrappers_trajectory(mol, trajectory, optimization_report)
    return {
        "build_seconds": build_seconds,
        "optimization_seconds": optimization_seconds,
        "compute_seconds": build_seconds + optimization_seconds,
        "forcefield_name": FORCEFIELD,
        "converged": bool(optimization_report.converged),
        "termination_reason": optimization_report.termination_reason,
        "build_rules": json_value(build_report.rules),
        "optimization_rules": json_value(optimization_report.rules),
    }


def _run_hotpot_target(
    backend: str,
    target: str,
    mol: Molecule,
    case_dir: Path,
    seed: int,
) -> dict[str, object]:
    from hotpot.cheminfo import forcefields as ff

    build_trajectory_path = case_dir / "trajectory" / "build"
    optimize_trajectory_path = case_dir / "trajectory" / "optimize"
    build_started = perf_counter()
    if target == "ligand":
        build_report = ff.build3d(
            mol,
            add_hydrogens=False,
            seed=seed,
        )
    else:
        build_report = ff.build_complex3d(
            mol,
            FORCEFIELD,
            add_hydrogens=False,
            seed=seed,
            save_movie=True,
            trajectory_start=ff.TrajectoryStart.LIGAND_BUILD,
        )
    build_seconds = perf_counter() - build_started
    mol.write(case_dir / "built.mol2", overwrite=True, write_single=True)

    optimize_started = perf_counter()
    if backend == "hotpot_auto":
        optimization_report = ff.auto_optimize(
            mol,
            FORCEFIELD,
            epochs=OPTIMIZATION_EPOCHS,
            steps_per_epoch=STEPS_PER_EPOCH,
            add_hydrogens=False,
            quality_level="standard",
            seed=seed,
            convergence_level=ff.ConvergenceLevel.FAST,
            save_movie=True,
            trajectory_start=(
                ff.TrajectoryStart.FINAL_OPTIMIZATION
                if target == "ligand"
                else ff.TrajectoryStart.COMPLEX_UNTANGLING
            ),
        )
    elif target == "ligand":
        optimization_report = ff.optimize(
            mol,
            FORCEFIELD,
            epochs=OPTIMIZATION_EPOCHS,
            steps_per_epoch=STEPS_PER_EPOCH,
            add_hydrogens=False,
            quality_level="off",
            seed=seed,
            convergence_level=ff.ConvergenceLevel.FAST,
            save_movie=True,
            trajectory_start=ff.TrajectoryStart.FINAL_OPTIMIZATION,
        )
    else:
        optimization_report = ff.optimize_complex(
            mol,
            FORCEFIELD,
            epochs=OPTIMIZATION_EPOCHS,
            steps_per_epoch=STEPS_PER_EPOCH,
            add_hydrogens=False,
            quality_level="off",
            seed=seed,
            convergence_level=ff.ConvergenceLevel.FAST,
            save_movie=True,
            trajectory_start=ff.TrajectoryStart.COMPLEX_UNTANGLING,
        )
    optimization_seconds = perf_counter() - optimize_started
    if target == "complex" and build_report.trajectory is not None:
        build_report.trajectory.write(build_trajectory_path)
    if optimization_report.trajectory is not None:
        optimization_report.trajectory.write(optimize_trajectory_path)
    result = {
        "build_seconds": build_seconds,
        "optimization_seconds": optimization_seconds,
        "compute_seconds": build_seconds + optimization_seconds,
        "forcefield_name": FORCEFIELD,
        "converged": bool(optimization_report.converged),
        "termination_reason": optimization_report.termination_reason,
        "build_report": json_value(build_report),
        "optimization_report": json_value(optimization_report),
        "routing_report": json_value(optimization_report.routing_report),
        "trajectory": {
            "build": (
                str(build_trajectory_path.relative_to(case_dir))
                if target == "complex"
                else None
            ),
            "optimize": str(optimize_trajectory_path.relative_to(case_dir)),
        },
    }
    if backend == "hotpot_auto":
        quality_report = optimization_report.quality_report
        if quality_report is None:
            raise RuntimeError("auto_optimize omitted its quality report")
        result.update(
            validation=_quality_payload(quality_report),
            quality_passed=bool(quality_report.passed),
        )
    return result


def _write_structure(case_dir: Path, mol: Molecule) -> dict[str, str]:
    mol2_path = case_dir / "optimized.mol2"
    sdf_path = case_dir / "optimized.sdf"
    mol.write(mol2_path, overwrite=True, write_single=True)
    mol.write(sdf_path, overwrite=True, write_single=True)
    return {
        "mol2": str(mol2_path.name),
        "sdf": str(sdf_path.name),
    }


def _run_target(
    backend: str,
    target: str,
    ligand_case: LigandCase,
    complex_case: Optional[CanonicalCase],
    case_dir: Path,
) -> dict[str, object]:
    from hotpot.cheminfo import forcefields as ff

    target_dir = case_dir / target
    target_dir.mkdir(parents=True, exist_ok=True)
    if target == "complex" and complex_case is None:
        return {"status": "not_eligible"}

    record: dict[str, object] = {
        "status": "running",
        "compute_seconds": None,
        "quality_passed": False,
        "validation": None,
        "structure": None,
        "trajectory": None,
    }
    if complex_case is not None and target == "complex":
        record.update(
            metal=complex_case.metal,
            donor_indices=list(complex_case.donor_indices),
        )
    mol = None
    trajectory = None
    try:
        mol = (
            _prepare_ligand(ligand_case)
            if target == "ligand"
            else _rebuild_complex(complex_case)
        )
        topology_reference = ff.capture_topology(
            mol,
            allow_added_hydrogens=False,
        )
        with warnings.catch_warnings(record=True) as emitted:
            warnings.simplefilter("always")
            if backend in ("rdkit", "openbabel"):
                trajectory = _new_trajectory(mol)
                result = _run_native_target(
                    backend,
                    target,
                    mol,
                    complex_case if target == "complex" else ligand_case,
                    trajectory,
                )
            elif backend == "obwrappers":
                trajectory = _new_trajectory(mol)
                result = _run_obwrappers_target(mol, trajectory)
            else:
                result = _run_hotpot_target(
                    backend,
                    target,
                    mol,
                    target_dir,
                    ligand_case.seed,
                )
        record.update(result)
        record["warnings"] = [str(item.message) for item in emitted]
        finite_coordinates = bool(np.all(np.isfinite(mol.coordinates)))
        record["finite_final_coordinates"] = finite_coordinates
        if finite_coordinates:
            if record["validation"] is None:
                validation = ff.evaluate_structure_acceptance(
                    mol,
                    level="standard",
                    topology_reference=topology_reference,
                )
                record["validation"] = _quality_payload(validation)
                record["quality_passed"] = bool(validation.passed)
            record["status"] = (
                "passed" if record["quality_passed"] else "failed_quality"
            )
            record["structure"] = _write_structure(target_dir, mol)
        else:
            record["status"] = "failed_quality"
    except _CaseFailure as error:
        record.update(error.evidence)
        record.update(
            status="failed_execution",
            error_type=type(error).__name__,
            error_message=str(error),
        )
    except Exception as error:  # noqa: BLE001 - each benchmark case is evidence.
        record.update(
            status="failed_execution",
            error_type=type(error).__name__,
            error_message=str(error),
            traceback=traceback.format_exc(),
        )

    if record.get("compute_seconds") is None:
        build_seconds = float(record.get("build_seconds", 0.0))
        optimization_seconds = float(record.get("optimization_seconds", 0.0))
        if build_seconds or optimization_seconds:
            record["compute_seconds"] = build_seconds + optimization_seconds
    if trajectory is not None and len(trajectory):
        try:
            archive, sdf_written = _write_native_trajectory(
                trajectory,
                target_dir / "trajectory",
            )
            record["trajectory"] = {
                "path": "trajectory",
                "frame_count": len(archive.main),
                "sdf_written": sdf_written,
            }
        except Exception as error:  # noqa: BLE001 - preserve the case report.
            record["trajectory_error"] = f"{type(error).__name__}: {error}"
    return record


def run_case(
    backend: str,
    ligand_case_data: Mapping[str, object],
    complex_case_data: Optional[Mapping[str, object]],
    output_root_text: str,
    resume: bool = False,
) -> dict[str, object]:
    """Run both targets for one corpus row and write one case report."""
    ligand_case = LigandCase(
        index=int(ligand_case_data["index"]),
        smiles=str(ligand_case_data["smiles"]),
        seed=int(ligand_case_data["seed"]),
    )
    complex_case = (
        None
        if complex_case_data is None
        else CanonicalCase.from_dict(complex_case_data)
    )
    case_dir = Path(output_root_text) / "cases" / f"{ligand_case.index:04d}"
    report_path = case_dir / "report.json"
    if resume and report_path.is_file():
        return json.loads(report_path.read_text(encoding="utf-8"))
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "input.smi").write_text(ligand_case.smiles + "\n", encoding="utf-8")
    record = {
        "schema_version": SCHEMA_VERSION,
        "index": ligand_case.index,
        "smiles": ligand_case.smiles,
        "workflow": backend,
        "targets": {
            "ligand": _run_target(
                backend,
                "ligand",
                ligand_case,
                complex_case,
                case_dir,
            ),
            "complex": _run_target(
                backend,
                "complex",
                ligand_case,
                complex_case,
                case_dir,
            ),
        },
    }
    write_json(report_path, record)
    return record


def _target_summary(
    records: Sequence[Mapping[str, object]],
    target: str,
) -> dict[str, object]:
    target_records = [
        record["targets"][target]
        for record in records
        if record["targets"][target]["status"] != "not_eligible"
    ]
    durations = [
        float(record["compute_seconds"])
        for record in target_records
        if record["status"] in ("passed", "failed_quality")
        and record.get("compute_seconds") is not None
    ]
    pass_count = sum(bool(record.get("quality_passed")) for record in target_records)
    denominator = len(target_records)
    selected_routes = Counter(
        str(record["routing_report"]["selected_route"])
        for record in target_records
        if record.get("routing_report") is not None
    )
    return {
        "denominator": denominator,
        "report_count": denominator,
        "pass_count": pass_count,
        "pass_rate": pass_count / denominator,
        "median_compute_seconds": statistics.median(durations) if durations else None,
        "aggregate_compute_seconds": sum(durations),
        "timed_count": len(durations),
        "status_counts": dict(
            Counter(str(record["status"]) for record in target_records)
        ),
        "selected_route_counts": dict(selected_routes),
    }


def _write_summary(
    backend: str,
    output_root: Path,
    cohort: BenchmarkCohort,
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    observed_indices = {int(record["index"]) for record in records}
    expected_indices = {case.index for case in cohort.ligand_cases}
    if observed_indices != expected_indices:
        raise ValueError(
            "result reports differ from the input corpus: "
            f"missing={sorted(expected_indices - observed_indices)}, "
            f"unexpected={sorted(observed_indices - expected_indices)}"
        )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "workflow": backend,
        "input_count": len(cohort.ligand_cases),
        "complex_eligible_count": len(cohort.complex_cases),
        "manifest_sha256": sha256_file(output_root / "manifest.json"),
        "targets": {
            "ligand": _target_summary(records, "ligand"),
            "complex": _target_summary(records, "complex"),
        },
    }
    write_json(output_root / "summary.json", summary)
    return summary


def _manifest_payload(backend: str, cohort: BenchmarkCohort) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "workflow": backend,
        "input": {
            "path": str(cohort.input_path),
            "sha256": cohort.input_sha256,
            "sample_count": len(cohort.ligand_cases),
        },
        "cohort": {
            "path": str(cohort.cohort_path),
            "sha256": cohort.cohort_sha256,
            "sample_count": len(cohort.complex_cases),
            "indices": list(cohort.complex_indices),
        },
        "settings": {
            "forcefield": FORCEFIELD,
            "optimization_epochs": OPTIMIZATION_EPOCHS,
            "steps_per_epoch": STEPS_PER_EPOCH,
            "optimization_steps": OPTIMIZATION_STEPS,
            "seed": SEED,
            "quality_level": "standard",
            "convergence_level": (
                "FAST"
                if backend in {"obwrappers", "hotpot_optimize_complex", "hotpot_auto"}
                else None
            ),
            "timing_scope": "build and optimize calls only",
        },
    }


def _write_or_check_manifest(
    backend: str,
    cohort: BenchmarkCohort,
    output_root: Path,
    *,
    resume: bool,
) -> None:
    manifest_path = output_root / "manifest.json"
    payload = _manifest_payload(backend, cohort)
    if manifest_path.is_file():
        if not resume:
            raise FileExistsError(
                f"{manifest_path} exists; use --resume or choose a new output"
            )
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("--resume configuration differs from manifest.json")
        return
    write_json(manifest_path, payload)


def run_benchmark(
    backend: str,
    cohort: BenchmarkCohort,
    output_root: Path,
    *,
    workers: int = 16,
    resume: bool = False,
) -> dict[str, object]:
    """Run one fixed backend across all 187 ligand and 181 complex targets."""
    if backend not in BACKENDS:
        raise ValueError(f"unsupported benchmark backend {backend!r}")
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _write_or_check_manifest(backend, cohort, output_root, resume=resume)

    tasks = tuple(
        (
            backend,
            case.to_dict(),
            (
                asdict(cohort.complex_cases[case.index])
                if case.index in cohort.complex_cases
                else None
            ),
            str(output_root),
            resume,
        )
        for case in cohort.ligand_cases
    )
    records = []
    if workers == 1:
        for completed, task in enumerate(tasks, start=1):
            record = run_case(*task)
            records.append(record)
            print(
                f"[{backend} {completed}/{len(tasks)}] "
                f"case={int(record['index']):04d} "
                f"ligand={record['targets']['ligand']['status']} "
                f"complex={record['targets']['complex']['status']}",
                flush=True,
            )
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = {pool.submit(run_case, *task): task[1]["index"] for task in tasks}
            for completed, future in enumerate(as_completed(futures), start=1):
                record = future.result()
                records.append(record)
                print(
                    f"[{backend} {completed}/{len(tasks)}] "
                    f"case={int(record['index']):04d} "
                    f"ligand={record['targets']['ligand']['status']} "
                    f"complex={record['targets']['complex']['status']}",
                    flush=True,
                )
    return _write_summary(backend, output_root, cohort, records)


def build_parser(backend: str) -> argparse.ArgumentParser:
    """Create the common CLI while fixing the selected backend."""
    if backend not in BACKENDS:
        raise ValueError(f"unsupported benchmark backend {backend!r}")
    parser = argparse.ArgumentParser(
        description=(
            f"Run the {backend} benchmark for 187 ligands and the frozen "
            "181-complex CBond cohort."
        )
    )
    parser.set_defaults(backend=backend)
    parser.add_argument("--input", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--reference", type=Path)
    source.add_argument("--cohort", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    return parser


def main_for_backend(
    backend: str,
    argv: Optional[Sequence[str]] = None,
) -> dict[str, object]:
    """Resolve CLI inputs and execute one backend without cross-dispatch."""
    arguments = build_parser(backend).parse_args(argv)
    cohort = resolve_cohort(
        arguments.input,
        arguments.output,
        reference_root=arguments.reference,
        cohort_path=arguments.cohort,
        seed=SEED,
    )
    return run_benchmark(
        backend,
        cohort,
        arguments.output,
        workers=arguments.workers,
        resume=arguments.resume,
    )


__all__ = (
    "BACKENDS",
    "build_parser",
    "main_for_backend",
    "run_benchmark",
    "run_case",
)
