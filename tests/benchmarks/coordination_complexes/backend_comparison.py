"""Compare native RDKit/Open Babel complex building against a Hotpot run.

The Hotpot run supplies only the chemical contract: the original ligand
SMILES, CBond donor indices, and the selected trajectory topology.  Native
backends never consume Hotpot coordinates or serialized ``cbond.smi`` files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
import platform
import statistics
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Mapping, Optional, Sequence, TYPE_CHECKING

import numpy as np

from .io import json_value, load_case_reports, sha256_file, write_json
from .pipeline import _trajectory_payload


BACKENDS = ("rdkit", "openbabel")
SCHEMA_VERSION = 1
OPTIMIZATION_STEPS = 10_000
TRAJECTORY_INTERVAL_STEPS = 100


if TYPE_CHECKING:
    from hotpot.cheminfo.forcefields import (
        ForceFieldTrajectory,
        TrajectoryEvent,
        TrajectoryStage,
    )


@dataclass(frozen=True)
class CanonicalCase:
    """Coordinate-free complex topology reconstructed from CBond evidence."""

    index: int
    smiles: str
    metal: str
    donor_indices: tuple[int, ...]
    seed: int
    complex_smiles: str
    atom_count: int
    bond_count: int
    topology_sha256: str
    hotpot_status: str
    hotpot_validation: Mapping[str, object]
    hotpot_total_seconds: Optional[float]
    hotpot_converged: Optional[bool] = None

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "CanonicalCase":
        return cls(
            index=int(value["index"]),
            smiles=str(value["smiles"]),
            metal=str(value["metal"]),
            donor_indices=tuple(int(index) for index in value["donor_indices"]),
            seed=int(value["seed"]),
            complex_smiles=str(value["complex_smiles"]),
            atom_count=int(value["atom_count"]),
            bond_count=int(value["bond_count"]),
            topology_sha256=str(value["topology_sha256"]),
            hotpot_status=str(value["hotpot_status"]),
            hotpot_validation=dict(value["hotpot_validation"]),
            hotpot_total_seconds=(
                None
                if value.get("hotpot_total_seconds") is None
                else float(value["hotpot_total_seconds"])
            ),
            hotpot_converged=(
                None
                if value.get("hotpot_converged") is None
                else bool(value["hotpot_converged"])
            ),
        )


class _CaseFailure(RuntimeError):
    """A classified backend outcome that belongs in benchmark evidence."""

    def __init__(
        self,
        status: str,
        message: str,
        evidence: Optional[Mapping[str, object]] = None,
    ):
        super().__init__(message)
        self.status = status
        self.evidence = dict(evidence or {})


def _normalized_topology(mol: object) -> dict[str, object]:
    atoms = [
        {
            "atomic_number": int(atom.atomic_number),
            "formal_charge": int(atom.formal_charge),
        }
        for atom in mol.atoms
    ]
    bonds = sorted(
        (
            min(int(bond.atom1.idx), int(bond.atom2.idx)),
            max(int(bond.atom1.idx), int(bond.atom2.idx)),
            float(bond.bond_order),
            bond.bond_kind.value,
        )
        for bond in mol.bonds
    )
    return {"atoms": atoms, "bonds": bonds}


def _trajectory_topology(trajectory: Mapping[str, object]) -> dict[str, object]:
    selected_index = int(trajectory["selected_index"])
    topology_revision = int(trajectory["frames"][selected_index]["topology_revision"])
    return {
        "atoms": [
            {
                "atomic_number": int(atom["atomic_number"]),
                "formal_charge": int(atom["formal_charge"]),
            }
            for atom in trajectory["atoms"]
        ],
        "bonds": sorted(
            (
                min(int(bond["atom_indices"][0]), int(bond["atom_indices"][1])),
                max(int(bond["atom_indices"][0]), int(bond["atom_indices"][1])),
                float(bond["bond_order"]),
                str(bond["bond_kind"]),
            )
            for bond in trajectory["topologies"][topology_revision]
        ),
    }


def _topology_sha256(topology: Mapping[str, object]) -> str:
    canonical = json.dumps(
        topology,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _rebuild_complex(case: CanonicalCase):
    """Reproduce the production explicit-H CBond topology at zero geometry."""
    from hotpot import Atom, read_mol
    from hotpot.cheminfo.forcefields.working_copy import (
        _hydrogenated_working_copy,
    )

    complex_mol = read_mol(case.smiles, fmt="smi")
    complex_mol.add_hydrogens()
    complex_mol.force_remove_polar_hydrogens()
    metal = complex_mol.add_atom(Atom(symbol=case.metal))
    for donor_index in case.donor_indices:
        complex_mol.add_bond(metal, complex_mol.atoms[donor_index])
    complex_mol = _hydrogenated_working_copy(
        complex_mol,
        add_hydrogens=True,
        seed=case.seed,
    )
    complex_mol.coordinates = np.zeros((len(complex_mol.atoms), 3), dtype=float)
    topology = _normalized_topology(complex_mol)
    if len(complex_mol.atoms) != case.atom_count:
        raise _CaseFailure(
            "canonical_topology_mismatch",
            f"atom count {len(complex_mol.atoms)} != {case.atom_count}",
        )
    if len(complex_mol.bonds) != case.bond_count:
        raise _CaseFailure(
            "canonical_topology_mismatch",
            f"bond count {len(complex_mol.bonds)} != {case.bond_count}",
        )
    if _topology_sha256(topology) != case.topology_sha256:
        raise _CaseFailure(
            "canonical_topology_mismatch",
            "reconstructed topology differs from the Hotpot selected-frame topology",
        )
    return complex_mol


def export_canonical_manifest(
    reference_root: Path,
    output_path: Path,
    *,
    metal: str = "Eu",
    expected_count: Optional[int] = None,
) -> dict[str, object]:
    """Export compact, coordinate-free cases from one completed Hotpot run."""
    reference_manifest_path = reference_root / "manifest.json"
    reference_manifest = json.loads(
        reference_manifest_path.read_text(encoding="utf-8")
    )
    reference_summary_path = reference_root / "summary.json"
    reference_summary = (
        json.loads(reference_summary_path.read_text(encoding="utf-8"))
        if reference_summary_path.is_file()
        else {}
    )
    if (
        expected_count is None
        and reference_summary.get("cbond_success_count") is not None
    ):
        expected_count = int(reference_summary["cbond_success_count"])
    cases = []
    for report in load_case_reports(reference_root):
        cbond = report.get("cbond")
        if not cbond:
            continue
        trajectory_path = (
            reference_root
            / "cases"
            / f"{int(report['index']):04d}"
            / "trajectory"
            / "main"
            / "trajectory.json"
        )
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        expected_topology = _trajectory_topology(trajectory)
        seed = int(report["settings"]["seed"]) + int(report["index"])
        provisional = CanonicalCase(
            index=int(report["index"]),
            smiles=str(report["smiles"]),
            metal=metal,
            donor_indices=tuple(int(i) for i in cbond["donor_indices"]),
            seed=seed,
            complex_smiles=str(cbond["complex_smiles"]),
            atom_count=len(expected_topology["atoms"]),
            bond_count=len(expected_topology["bonds"]),
            topology_sha256=_topology_sha256(expected_topology),
            hotpot_status=str(report["status"]),
            hotpot_validation=dict(report.get("validation") or {}),
            hotpot_total_seconds=(
                None
                if report.get("total_seconds") is None
                else float(report["total_seconds"])
            ),
            hotpot_converged=(
                None
                if (report.get("optimization") or {}).get("converged") is None
                else bool(report["optimization"]["converged"])
            ),
        )
        _rebuild_complex(provisional)
        cases.append(asdict(provisional))
    if expected_count is not None and len(cases) != expected_count:
        raise ValueError(
            f"expected {expected_count} CBond-successful cases, found {len(cases)}"
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "reference_root": str(reference_root.resolve()),
            "manifest_sha256": sha256_file(reference_manifest_path),
            "git": reference_manifest.get("git"),
            "input_sha256": (
                reference_manifest.get("scientific_configuration", {}).get(
                    "input_sha256"
                )
            ),
            "coordinate_source": None,
            "topology_reference": "selected trajectory frame",
        },
        "protocol": {
            "metal": metal,
            "input": "original ligand SMILES + CBond donor indices",
            "hydrogens": "Hotpot production hidden-metal-bond completion",
            "coordinate_initialization": "backend-native; no Hotpot coordinates",
        },
        "cases": cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, payload)
    return payload


def load_canonical_cases(path: Path) -> tuple[CanonicalCase, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload["schema_version"]) != SCHEMA_VERSION:
        raise ValueError(f"unsupported canonical manifest schema: {payload['schema_version']}")
    return tuple(CanonicalCase.from_dict(case) for case in payload["cases"])


def _quality_payload(report: object) -> dict[str, object]:
    return {
        "level": report.level,
        "passed": bool(report.passed),
        "checks": json_value(report.checks),
        "failures": json_value(report.failures),
        "warnings": json_value(report.warnings),
        "metrics": json_value(report.metrics),
    }


def _record_native_frame(
    trajectory: "ForceFieldTrajectory",
    complex_mol: object,
    *,
    stage: "TrajectoryStage",
    event: "TrajectoryEvent",
    energy_kj_mol: Optional[float] = None,
    step: Optional[int] = None,
) -> None:
    """Record one native-backend checkpoint and select it when finite."""
    frame = trajectory.record_molecule(
        complex_mol,
        stage=stage,
        event=event,
        energy_kj_mol=energy_kj_mol,
        step=step,
    )
    if np.all(np.isfinite(complex_mol.coordinates)):
        trajectory.select(frame.index)


def _write_native_trajectory(
    trajectory: "ForceFieldTrajectory",
    output_path: Path,
) -> tuple[object, bool]:
    """Persist every retained frame and return its archive plus SDF status."""
    from hotpot.cheminfo import forcefields as ff

    trajectory.set_terminal(len(trajectory) - 1)
    archive = ff.ForceFieldTrajectoryArchive(main=trajectory)
    all_coordinates_finite = all(
        np.all(np.isfinite(trajectory.coordinates(frame.index)))
        for frame in trajectory.frames
    )
    archive.write(output_path, include_sdf=all_coordinates_finite)
    return archive, all_coordinates_finite


def _rdkit_coordinate_map(complex_mol: object, rd_mol: object) -> np.ndarray:
    """Map RDKit's AddHs ordering back to the canonical Hotpot atom table."""
    canonical_heavy = [
        atom.idx
        for atom in complex_mol.atoms
        if not atom.is_hydrogen and not atom.is_metal
    ]
    rd_heavy = [
        atom.GetIdx()
        for atom in rd_mol.GetAtoms()
        if atom.GetAtomicNum() not in (1, 63)
    ]
    if [complex_mol.atoms[i].atomic_number for i in canonical_heavy] != [
        rd_mol.GetAtomWithIdx(i).GetAtomicNum() for i in rd_heavy
    ]:
        raise _CaseFailure(
            "backend_topology_mismatch",
            "RDKit changed the ligand heavy-atom order",
        )
    mapping = np.full(len(complex_mol.atoms), -1, dtype=np.int64)
    for canonical_index, rd_index in zip(canonical_heavy, rd_heavy):
        mapping[canonical_index] = rd_index

    canonical_metal = [atom.idx for atom in complex_mol.atoms if atom.is_metal]
    rd_metal = [
        atom.GetIdx() for atom in rd_mol.GetAtoms() if atom.GetAtomicNum() == 63
    ]
    if len(canonical_metal) != 1 or len(rd_metal) != 1:
        raise _CaseFailure(
            "backend_topology_mismatch",
            "the comparison requires exactly one Eu centre",
        )
    mapping[canonical_metal[0]] = rd_metal[0]

    for canonical_index, rd_index in zip(canonical_heavy, rd_heavy):
        canonical_hydrogens = sorted(
            atom.idx
            for atom in complex_mol.atoms[canonical_index].neighbours
            if atom.is_hydrogen
        )
        rd_hydrogens = sorted(
            atom.GetIdx()
            for atom in rd_mol.GetAtomWithIdx(rd_index).GetNeighbors()
            if atom.GetAtomicNum() == 1
        )
        if len(canonical_hydrogens) != len(rd_hydrogens):
            raise _CaseFailure(
                "backend_topology_mismatch",
                f"hydrogen count differs at canonical atom {canonical_index}",
            )
        mapping[canonical_hydrogens] = rd_hydrogens
    if np.any(mapping < 0) or len(set(int(i) for i in mapping)) != len(mapping):
        raise _CaseFailure(
            "backend_topology_mismatch",
            "RDKit-to-canonical atom mapping is incomplete",
        )
    return mapping


def _run_rdkit(
    complex_mol: object,
    case: CanonicalCase,
    trajectory: "ForceFieldTrajectory",
) -> dict[str, object]:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from hotpot.cheminfo import forcefields as ff

    ligand = Chem.MolFromSmiles(case.smiles)
    if ligand is None:
        raise _CaseFailure("build_failed", "RDKit could not parse the ligand SMILES")
    rd_mol = Chem.AddHs(ligand)
    editable = Chem.RWMol(rd_mol)
    metal = Chem.Atom(case.metal)
    metal.SetFormalCharge(0)
    metal_index = editable.AddAtom(metal)
    for donor_index in case.donor_indices:
        editable.AddBond(donor_index, metal_index, Chem.BondType.DATIVE)
    rd_mol = editable.GetMol()
    rd_mol.UpdatePropertyCache(strict=False)
    coordinate_map = _rdkit_coordinate_map(complex_mol, rd_mol)

    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = case.seed
    parameters.useRandomCoords = True
    parameters.ignoreSmoothingFailures = True
    build_started = perf_counter()
    embed_status = int(AllChem.EmbedMolecule(rd_mol, parameters))
    build_seconds = perf_counter() - build_started
    if embed_status != 0:
        raise _CaseFailure(
            "build_failed",
            f"RDKit ETKDGv3 returned status {embed_status}",
            {
                "build_seconds": build_seconds,
                "embedding_status": embed_status,
            },
        )
    conformer = rd_mol.GetConformer()
    rd_coordinates = np.asarray(conformer.GetPositions(), dtype=float)
    coordinates = rd_coordinates[coordinate_map]
    if not np.all(np.isfinite(coordinates)):
        complex_mol.coordinates = coordinates
        _record_native_frame(
            trajectory,
            complex_mol,
            stage=ff.TrajectoryStage.LIGAND_BUILD,
            event=ff.TrajectoryEvent.BUILD_COMPLETE,
        )
        raise _CaseFailure(
            "nonfinite_coordinates",
            "RDKit ETKDGv3 produced non-finite coordinates",
        )
    complex_mol.coordinates = coordinates
    _record_native_frame(
        trajectory,
        complex_mol,
        stage=ff.TrajectoryStage.LIGAND_BUILD,
        event=ff.TrajectoryEvent.BUILD_COMPLETE,
    )
    result: dict[str, object] = {
        "build_succeeded": True,
        "build_seconds": build_seconds,
        "finite_build_coordinates": True,
        "backend_atom_count": int(rd_mol.GetNumAtoms()),
        "backend_bond_count": int(rd_mol.GetNumBonds()),
        "coordination_bond_encoding": "RDKit DATIVE (donor -> Eu)",
        "embedding": {
            "algorithm": "ETKDGv3",
            "use_random_coordinates": True,
            "ignore_smoothing_failures": True,
        },
    }

    mmff_complete = bool(AllChem.MMFFHasAllMoleculeParams(rd_mol))
    uff_complete = bool(AllChem.UFFHasAllMoleculeParams(rd_mol))
    if mmff_complete:
        properties = AllChem.MMFFGetMoleculeProperties(
            rd_mol,
            mmffVariant="MMFF94",
        )
        forcefield = AllChem.MMFFGetMoleculeForceField(rd_mol, properties)
        forcefield_name = "MMFF94"
        parameterization = "full"
    else:
        forcefield = AllChem.UFFGetMoleculeForceField(rd_mol)
        forcefield_name = "UFF"
        parameterization = "full" if uff_complete else "partial"
    if forcefield is None:
        result.update(
            forcefield_supported=False,
            forcefield_setup_succeeded=False,
            forcefield_fully_parameterized=False,
            forcefield_parameterization="unavailable",
            optimization_succeeded=False,
        )
        raise _CaseFailure(
            "forcefield_unsupported",
            "RDKit could not construct an MMFF or UFF force field",
            result,
        )
    result.update(
        forcefield_supported=True,
        forcefield_setup_succeeded=True,
        forcefield_name=forcefield_name,
        forcefield_parameterization=parameterization,
        forcefield_fully_parameterized=parameterization == "full",
        mmff_has_all_parameters=mmff_complete,
        uff_has_all_parameters=uff_complete,
    )
    optimize_started = perf_counter()
    forcefield.Initialize()
    optimize_status = 1
    snapshot_count = 0
    for snapshot_count in range(1, OPTIMIZATION_STEPS // TRAJECTORY_INTERVAL_STEPS + 1):
        optimize_status = int(
            forcefield.Minimize(maxIts=TRAJECTORY_INTERVAL_STEPS)
        )
        coordinates = np.asarray(
            rd_mol.GetConformer().GetPositions(), dtype=float
        )[coordinate_map]
        complex_mol.coordinates = coordinates
        energy_kj_mol = (
            float(forcefield.CalcEnergy()) * 4.184
            if np.all(np.isfinite(coordinates))
            else None
        )
        _record_native_frame(
            trajectory,
            complex_mol,
            stage=ff.TrajectoryStage.FINAL_OPTIMIZATION,
            event=ff.TrajectoryEvent.EPOCH_COMPLETE,
            energy_kj_mol=energy_kj_mol,
            step=snapshot_count,
        )
        if optimize_status <= 0:
            break
    result["optimization_seconds"] = perf_counter() - optimize_started
    result["trajectory_interval_steps"] = TRAJECTORY_INTERVAL_STEPS
    result["optimization_snapshot_count"] = snapshot_count
    if optimize_status < 0:
        raise _CaseFailure(
            "optimization_failed",
            f"RDKit {forcefield_name} returned status {optimize_status}",
            result,
        )
    coordinates = np.asarray(rd_mol.GetConformer().GetPositions(), dtype=float)[
        coordinate_map
    ]
    if not np.all(np.isfinite(coordinates)):
        raise _CaseFailure(
            "nonfinite_coordinates",
            f"RDKit {forcefield_name} produced non-finite coordinates",
            result,
        )
    complex_mol.coordinates = coordinates
    result.update(
        optimization_succeeded=True,
        optimizer_status=optimize_status,
        converged=optimize_status == 0,
        final_energy=float(forcefield.CalcEnergy()),
        energy_unit="kcal/mol",
    )
    return result


def _run_openbabel(
    complex_mol: object,
    case: CanonicalCase,
    trajectory: "ForceFieldTrajectory",
) -> dict[str, object]:
    from openbabel import openbabel as ob
    from hotpot.cheminfo import forcefields as ff
    from hotpot.cheminfo.obconvert import extract_obmol_coordinates

    obmol = complex_mol.to_obmol()
    expected_atoms = obmol.NumAtoms()
    expected_bonds = obmol.NumBonds()
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
            "build_failed",
            "Open Babel OBBuilder returned false",
            {"build_seconds": build_seconds},
        )
    if obmol.NumAtoms() != expected_atoms or obmol.NumBonds() != expected_bonds:
        raise _CaseFailure(
            "backend_topology_mismatch",
            "Open Babel OBBuilder changed the canonical atom or bond count",
        )
    coordinates = extract_obmol_coordinates(obmol)
    if not np.all(np.isfinite(coordinates)):
        complex_mol.coordinates = coordinates
        _record_native_frame(
            trajectory,
            complex_mol,
            stage=ff.TrajectoryStage.LIGAND_BUILD,
            event=ff.TrajectoryEvent.BUILD_COMPLETE,
        )
        raise _CaseFailure(
            "nonfinite_coordinates",
            "Open Babel OBBuilder produced non-finite coordinates",
        )
    complex_mol.coordinates = coordinates
    _record_native_frame(
        trajectory,
        complex_mol,
        stage=ff.TrajectoryStage.LIGAND_BUILD,
        event=ff.TrajectoryEvent.BUILD_COMPLETE,
    )
    result: dict[str, object] = {
        "build_succeeded": True,
        "build_seconds": build_seconds,
        "finite_build_coordinates": True,
        "backend_atom_count": int(obmol.NumAtoms()),
        "backend_bond_count": int(obmol.NumBonds()),
        "coordination_bond_encoding": "Open Babel single bond",
    }

    forcefield = ob.OBForceField.FindForceField("UFF")
    if forcefield is None or not forcefield.Setup(obmol):
        result.update(
            forcefield_supported=False,
            forcefield_setup_succeeded=False,
            forcefield_fully_parameterized=False,
            forcefield_parameterization="unavailable",
            optimization_succeeded=False,
        )
        raise _CaseFailure(
            "forcefield_unsupported",
            "Open Babel UFF could not parameterize the complete complex",
            result,
        )
    result["forcefield_supported"] = True
    result["forcefield_setup_succeeded"] = True
    result["forcefield_name"] = "UFF"
    result["forcefield_parameterization"] = "full"
    result["forcefield_fully_parameterized"] = True
    optimize_started = perf_counter()
    forcefield.ConjugateGradientsInitialize(OPTIMIZATION_STEPS)
    snapshot_count = 0
    for snapshot_count in range(1, OPTIMIZATION_STEPS // TRAJECTORY_INTERVAL_STEPS + 1):
        continue_optimization = bool(
            forcefield.ConjugateGradientsTakeNSteps(TRAJECTORY_INTERVAL_STEPS)
        )
        forcefield.GetCoordinates(obmol)
        coordinates = extract_obmol_coordinates(obmol)
        complex_mol.coordinates = coordinates
        energy_kj_mol = (
            float(forcefield.Energy())
            if np.all(np.isfinite(coordinates))
            else None
        )
        _record_native_frame(
            trajectory,
            complex_mol,
            stage=ff.TrajectoryStage.FINAL_OPTIMIZATION,
            event=ff.TrajectoryEvent.EPOCH_COMPLETE,
            energy_kj_mol=energy_kj_mol,
            step=snapshot_count,
        )
        if not continue_optimization:
            break
    result["optimization_seconds"] = perf_counter() - optimize_started
    result["trajectory_interval_steps"] = TRAJECTORY_INTERVAL_STEPS
    result["optimization_snapshot_count"] = snapshot_count
    if not np.all(np.isfinite(coordinates)):
        raise _CaseFailure(
            "nonfinite_coordinates",
            "Open Babel UFF produced non-finite coordinates",
            result,
        )
    complex_mol.coordinates = coordinates
    result.update(
        optimization_succeeded=True,
        converged=None,
        final_energy=float(forcefield.Energy()),
        energy_unit=str(forcefield.GetUnit()),
    )
    return result


_BACKEND_RUNNERS = {
    "rdkit": _run_rdkit,
    "openbabel": _run_openbabel,
}


def run_backend_case(
    case_data: Mapping[str, object],
    backend: str,
    output_root_text: str,
    resume: bool = False,
) -> dict[str, object]:
    """Run one native backend case and persist its complete classification."""
    from hotpot.cheminfo import forcefields as ff

    case = CanonicalCase.from_dict(case_data)
    report_path = (
        Path(output_root_text)
        / "cases"
        / backend
        / f"{case.index:04d}"
        / "report.json"
    )
    if resume and report_path.is_file():
        return json.loads(report_path.read_text(encoding="utf-8"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, object] = {
        "index": case.index,
        "smiles": case.smiles,
        "metal": case.metal,
        "donor_indices": case.donor_indices,
        "backend": backend,
        "status": "running",
        "phase": "canonical_topology",
        "build_succeeded": False,
        "forcefield_supported": None,
        "forcefield_fully_parameterized": None,
        "optimization_succeeded": False,
        "finite_final_coordinates": False,
        "topology_preserved": False,
        "quality_passed": False,
    }
    started = perf_counter()
    trajectory = None
    try:
        complex_mol = _rebuild_complex(case)
        trajectory = ff.ForceFieldTrajectory.from_molecule(
            complex_mol,
            start=ff.TrajectoryStart.LIGAND_BUILD,
        )
        _record_native_frame(
            trajectory,
            complex_mol,
            stage=ff.TrajectoryStage.LIGAND_BUILD,
            event=ff.TrajectoryEvent.INITIAL,
        )
        topology_reference = ff.capture_topology(
            complex_mol,
            allow_added_hydrogens=False,
        )
        record["phase"] = "build_optimize"
        backend_result: dict[str, object] = {}
        try:
            backend_result = _BACKEND_RUNNERS[backend](
                complex_mol,
                case,
                trajectory,
            )
        except _CaseFailure as error:
            record.update(error.evidence)
            record.update(
                status=error.status,
                error_type=type(error).__name__,
                error_message=str(error),
            )
        else:
            record.update(backend_result)
            record["finite_final_coordinates"] = bool(
                np.all(np.isfinite(complex_mol.coordinates))
            )
            record["phase"] = "validation"
            validation = ff.evaluate_structure_acceptance(
                complex_mol,
                level="standard",
                topology_reference=topology_reference,
            )
            record["validation"] = _quality_payload(validation)
            topology_check = next(
                check for check in validation.checks if check.name == "topology"
            )
            record["topology_preserved"] = bool(topology_check.passed)
            record["quality_passed"] = bool(validation.passed)
            record["status"] = "passed" if validation.passed else "failed_quality"
            record["phase"] = "complete"
            complex_mol.write(
                report_path.parent / "optimized.mol2",
                overwrite=True,
                write_single=True,
            )
            complex_mol.write(
                report_path.parent / "optimized.sdf",
                overwrite=True,
                write_single=True,
            )
    except _CaseFailure as error:
        record.update(
            status=error.status,
            error_type=type(error).__name__,
            error_message=str(error),
        )
    except Exception as error:
        record.update(
            status="failed_internal",
            error_type=type(error).__name__,
            error_message=str(error),
            traceback=traceback.format_exc(),
        )
    if trajectory is not None:
        try:
            archive, sdf_written = _write_native_trajectory(
                trajectory,
                report_path.parent / "trajectory",
            )
            record["trajectory"] = _trajectory_payload(archive)
            record["trajectory"]["sdf_written"] = sdf_written
            record["trajectory"]["interval_steps"] = TRAJECTORY_INTERVAL_STEPS
        except Exception as error:
            record.update(
                status="failed_internal",
                phase="trajectory_write",
                error_type=type(error).__name__,
                error_message=str(error),
                traceback=traceback.format_exc(),
            )
    record["total_seconds"] = perf_counter() - started
    write_json(report_path, record)
    return record


def _hotpot_records(cases: Sequence[CanonicalCase]) -> list[dict[str, object]]:
    records = []
    for case in cases:
        checks = case.hotpot_validation.get("checks", ())
        finite = next(
            (bool(check["passed"]) for check in checks if check["name"] == "finite_coordinates"),
            False,
        )
        topology = next(
            (bool(check["passed"]) for check in checks if check["name"] == "topology"),
            False,
        )
        records.append(
            {
                "index": case.index,
                "backend": "hotpot",
                "status": case.hotpot_status,
                "build_succeeded": True,
                "forcefield_supported": True,
                "forcefield_fully_parameterized": True,
                "optimization_succeeded": bool(case.hotpot_validation),
                "converged": case.hotpot_converged,
                "finite_final_coordinates": finite,
                "topology_preserved": topology,
                "quality_passed": bool(case.hotpot_validation.get("passed", False)),
                "total_seconds": case.hotpot_total_seconds,
            }
        )
    return records


def _summary_row(backend: str, records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    durations = [
        float(record["total_seconds"])
        for record in records
        if record.get("total_seconds") is not None
    ]
    count = len(records)
    build_count = sum(bool(record.get("build_succeeded")) for record in records)
    supported_count = sum(bool(record.get("forcefield_supported")) for record in records)
    fully_parameterized_count = sum(
        bool(record.get("forcefield_fully_parameterized")) for record in records
    )
    optimized_count = sum(bool(record.get("optimization_succeeded")) for record in records)
    finite_count = sum(bool(record.get("finite_final_coordinates")) for record in records)
    topology_count = sum(bool(record.get("topology_preserved")) for record in records)
    quality_count = sum(bool(record.get("quality_passed")) for record in records)
    convergence_values = [
        bool(record["converged"])
        for record in records
        if record.get("converged") is not None
    ]
    trajectories = [
        record["trajectory"]
        for record in records
        if record.get("trajectory") is not None
    ]
    return {
        "backend": backend,
        "sample_count": count,
        "build_success_count": build_count,
        "build_success_rate": build_count / count,
        "forcefield_supported_count": supported_count,
        "forcefield_supported_rate": supported_count / count,
        "forcefield_fully_parameterized_count": fully_parameterized_count,
        "forcefield_fully_parameterized_rate": fully_parameterized_count / count,
        "optimization_success_count": optimized_count,
        "optimization_success_rate": optimized_count / count,
        "convergence_reported_count": len(convergence_values),
        "converged_count": sum(convergence_values),
        "trajectory_archive_count": len(trajectories),
        "trajectory_frame_count": sum(
            int(trajectory["main_frame_count"])
            for trajectory in trajectories
        ),
        "trajectory_sdf_count": sum(
            trajectory.get("sdf_written") is True
            for trajectory in trajectories
        ),
        "finite_coordinate_count": finite_count,
        "topology_preserved_count": topology_count,
        "quality_pass_count": quality_count,
        "quality_pass_rate": quality_count / count,
        "median_case_seconds": statistics.median(durations) if durations else None,
        "aggregate_case_seconds": sum(durations),
        "status_counts": dict(Counter(str(record["status"]) for record in records)),
    }


def write_comparison_plot(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    """Plot build, optimization, and final common-gate success rates."""
    import matplotlib.pyplot as plt

    display_names = {
        "hotpot": "Hotpot",
        "rdkit": "RDKit",
        "openbabel": "Open Babel",
    }
    labels = [display_names[str(row["backend"])] for row in rows]
    x = np.arange(len(labels), dtype=float)
    width = 0.24
    figure, axis = plt.subplots(figsize=(9.2, 4.8))
    series = (
        ("3D build", "build_success_rate", "#4C78A8"),
        ("Finite optimized output", "optimization_success_rate", "#F58518"),
        ("Standard geometry gate", "quality_pass_rate", "#54A24B"),
    )
    for offset, (label, field, color) in zip((-width, 0.0, width), series):
        values = [100.0 * float(row[field]) for row in rows]
        bars = axis.bar(x + offset, values, width, label=label, color=color)
        axis.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=8)
    axis.set_xticks(x, labels)
    axis.set_ylabel(
        f"Success rate across {int(rows[0]['sample_count'])} CBond complexes (%)"
    )
    axis.set_ylim(0.0, 110.0)
    axis.set_title("Build–optimization outcomes for Eu–ligand complexes")
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        frameon=False,
        ncol=3,
    )
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    figure.savefig(path, dpi=200)
    plt.close(figure)


def aggregate_comparison(
    output_root: Path,
    cases: Sequence[CanonicalCase],
    backends: Sequence[str],
) -> dict[str, object]:
    """Write per-case and aggregate three-backend evidence."""
    canonical_manifest_path = output_root / "canonical_cases.json"
    canonical_manifest = json.loads(
        canonical_manifest_path.read_text(encoding="utf-8")
    )
    canonical_source = canonical_manifest["source"]
    by_backend: dict[str, list[dict[str, object]]] = {
        "hotpot": _hotpot_records(cases)
    }
    expected_indices = {case.index for case in cases}
    for backend in backends:
        by_backend[backend] = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((output_root / "cases" / backend).glob("*/report.json"))
        ]
        observed_indices = {int(record["index"]) for record in by_backend[backend]}
        if observed_indices != expected_indices:
            missing = sorted(expected_indices - observed_indices)
            unexpected = sorted(observed_indices - expected_indices)
            raise ValueError(
                f"{backend} result cohort differs from the canonical cohort; "
                f"missing={missing}, unexpected={unexpected}"
            )
    rows = [
        _summary_row(backend, by_backend[backend])
        for backend in ("hotpot", *backends)
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol": {
            "sample_count": len(cases),
            "input": (
                "identical coordinate-free explicit-H atom table and Eu-CBond "
                "connectivity; backend-native metal-bond encoding"
            ),
            "rdkit": (
                "ETKDGv3 (random coordinates, smoothing failures allowed); "
                "full MMFF, otherwise full or explicitly labelled partial UFF; "
                f"minimization in {TRAJECTORY_INTERVAL_STEPS}-step chunks"
            ),
            "openbabel": (
                "OBBuilder; incrementally stepped UFF conjugate gradients, "
                f"{OPTIMIZATION_STEPS} steps maximum; the upstream builder does "
                "not guarantee deterministic seeding"
            ),
            "trajectory": (
                "zero-coordinate canonical input, completed native build, and "
                f"optimization checkpoints every {TRAJECTORY_INTERVAL_STEPS} "
                "requested steps; native builder internals are not exposed"
            ),
            "hotpot": "frozen reference result",
            "validation": "Hotpot standard structure-acceptance gate",
            "optimization_success_definition": (
                "optimizer returned finite coordinates; this does not imply "
                "convergence to a backend stopping criterion"
            ),
        },
        "provenance": {
            "canonical_manifest_sha256": sha256_file(canonical_manifest_path),
            "hotpot_reference_manifest_sha256": canonical_source["manifest_sha256"],
            "hotpot_reference_git": canonical_source.get("git"),
            "input_sha256": canonical_source.get("input_sha256"),
            "python": platform.python_version(),
            "hotpot-zzy": importlib.metadata.version("hotpot-zzy"),
            "rdkit": importlib.metadata.version("rdkit"),
            "openbabel": importlib.metadata.version("openbabel"),
        },
        "results": rows,
    }
    write_json(output_root / "comparison.json", payload)
    flat_fields = tuple(key for key in rows[0] if key != "status_counts")
    with (output_root / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=flat_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in flat_fields} for row in rows)
    case_fields = (
        "index",
        "backend",
        "status",
        "build_succeeded",
        "forcefield_supported",
        "forcefield_fully_parameterized",
        "forcefield_parameterization",
        "forcefield_name",
        "optimization_succeeded",
        "converged",
        "finite_final_coordinates",
        "topology_preserved",
        "quality_passed",
        "build_seconds",
        "optimization_seconds",
        "total_seconds",
        "error_type",
        "error_message",
        "failed_checks",
        "trajectory_frame_count",
        "trajectory_sdf_written",
    )
    with (output_root / "case_results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=case_fields)
        writer.writeheader()
        for backend in ("hotpot", *backends):
            for record in sorted(by_backend[backend], key=lambda item: int(item["index"])):
                row = {field: record.get(field) for field in case_fields}
                row["failed_checks"] = ";".join(
                    str(check["name"])
                    for check in (record.get("validation") or {}).get("failures", ())
                )
                trajectory = record.get("trajectory") or {}
                row["trajectory_frame_count"] = trajectory.get(
                    "main_frame_count"
                )
                row["trajectory_sdf_written"] = trajectory.get("sdf_written")
                writer.writerow(row)
    write_comparison_plot(rows, output_root / "comparison.png")
    return payload


def run_comparison(
    reference_root: Path,
    output_root: Path,
    *,
    backends: Sequence[str] = BACKENDS,
    workers: int = 16,
    resume: bool = False,
) -> dict[str, object]:
    """Run selected native backends over the frozen Hotpot CBond cohort."""
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "canonical_cases.json"
    if not manifest_path.is_file():
        export_canonical_manifest(reference_root, manifest_path)
    cases = load_canonical_cases(manifest_path)
    context = mp.get_context("spawn")
    for backend in backends:
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            futures = {
                pool.submit(
                    run_backend_case,
                    asdict(case),
                    backend,
                    str(output_root),
                    resume,
                ): case.index
                for case in cases
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                record = future.result()
                print(
                    f"[{backend} {completed}/{len(cases)}] "
                    f"case={int(record['index']):04d} "
                    f"status={record['status']} "
                    f"seconds={float(record['total_seconds']):.3f}",
                    flush=True,
                )
    return aggregate_comparison(output_root, cases, backends)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=BACKENDS,
        default=BACKENDS,
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    arguments = build_parser().parse_args(argv)
    run_comparison(
        arguments.reference.resolve(),
        arguments.output.resolve(),
        backends=tuple(arguments.backends),
        workers=arguments.workers,
        resume=arguments.resume,
    )


if __name__ == "__main__":
    main()
