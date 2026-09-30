"""Contract tests for the opt-in coordination-backend comparison."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from hotpot import Atom, read_mol
from hotpot.cheminfo.forcefields.working_copy import _hydrogenated_working_copy

from .backend_comparison import (
    CanonicalCase,
    _normalized_topology,
    _rebuild_complex,
    _summary_row,
    _topology_sha256,
    build_parser,
    export_canonical_manifest,
    load_canonical_cases,
    run_backend_case,
)


def _synthetic_complex():
    molecule = read_mol("N", fmt="smi")
    molecule.add_hydrogens()
    molecule.force_remove_polar_hydrogens()
    metal = molecule.add_atom(Atom(symbol="Eu"))
    molecule.add_bond(metal, molecule.atoms[0])
    return _hydrogenated_working_copy(molecule, add_hydrogens=True, seed=43)


def _write_reference(root: Path) -> None:
    complex_mol = _synthetic_complex()
    topology = _normalized_topology(complex_mol)
    case_dir = root / "cases" / "0001"
    trajectory_dir = case_dir / "trajectory" / "main"
    trajectory_dir.mkdir(parents=True)
    (root / "manifest.json").write_text("{}\n", encoding="utf-8")
    report = {
        "index": 1,
        "smiles": "N",
        "status": "passed",
        "settings": {"seed": 42},
        "total_seconds": 1.25,
        "cbond": {
            "donor_indices": [0],
            "complex_smiles": complex_mol.smiles,
        },
        "validation": {
            "passed": True,
            "checks": [
                {"name": "finite_coordinates", "passed": True},
                {"name": "topology", "passed": True},
            ],
        },
    }
    (case_dir / "report.json").write_text(
        json.dumps(report) + "\n",
        encoding="utf-8",
    )
    trajectory = {
        "atoms": [
            {
                "index": index,
                "atom_id": int(atom.id),
                "atomic_number": int(atom.atomic_number),
                "formal_charge": int(atom.formal_charge),
                "symbol": atom.symbol,
            }
            for index, atom in enumerate(complex_mol.atoms)
        ],
        "frames": [{"index": 0, "topology_revision": 0}],
        "selected_index": 0,
        "topologies": [[
            {
                "atom_indices": [first, second],
                "bond_order": order,
                "bond_kind": kind,
            }
            for first, second, order, kind in topology["bonds"]
        ]],
    }
    (trajectory_dir / "trajectory.json").write_text(
        json.dumps(trajectory) + "\n",
        encoding="utf-8",
    )


def test_canonical_manifest_rebuilds_selected_topology_without_coordinates(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    output = tmp_path / "canonical.json"
    _write_reference(reference)

    payload = export_canonical_manifest(
        reference,
        output,
        expected_count=1,
    )
    case = load_canonical_cases(output)[0]
    reconstructed = _rebuild_complex(case)

    assert payload["source"]["coordinate_source"] is None
    assert case.atom_count == len(reconstructed.atoms)
    assert case.bond_count == len(reconstructed.bonds)
    assert case.topology_sha256 == _topology_sha256(
        _normalized_topology(reconstructed)
    )
    np.testing.assert_array_equal(
        reconstructed.coordinates,
        np.zeros((case.atom_count, 3)),
    )


def test_rdkit_labels_partial_forcefield_without_hiding_successful_embedding(
    tmp_path: Path,
) -> None:
    complex_mol = _synthetic_complex()
    case = CanonicalCase(
        index=1,
        smiles="N",
        metal="Eu",
        donor_indices=(0,),
        seed=43,
        complex_smiles=complex_mol.smiles,
        atom_count=len(complex_mol.atoms),
        bond_count=len(complex_mol.bonds),
        topology_sha256=_topology_sha256(_normalized_topology(complex_mol)),
        hotpot_status="passed",
        hotpot_validation={"passed": True, "checks": ()},
        hotpot_total_seconds=1.0,
    )

    record = run_backend_case(asdict(case), "rdkit", str(tmp_path))

    assert record["build_succeeded"] is True
    assert record["finite_build_coordinates"] is True
    assert record["forcefield_supported"] is True
    assert record["forcefield_parameterization"] == "partial"
    assert record["forcefield_fully_parameterized"] is False
    assert record["optimization_succeeded"] is True
    assert (
        tmp_path / "cases" / "rdkit" / "0001" / "report.json"
    ).is_file()


def test_summary_rates_use_the_complete_fixed_cohort() -> None:
    row = _summary_row(
        "native",
        (
            {
                "status": "passed",
                "build_succeeded": True,
                "forcefield_supported": True,
                "forcefield_fully_parameterized": True,
                "optimization_succeeded": True,
                "finite_final_coordinates": True,
                "topology_preserved": True,
                "quality_passed": True,
                "total_seconds": 2.0,
            },
            {
                "status": "forcefield_unsupported",
                "build_succeeded": True,
                "forcefield_supported": False,
                "forcefield_fully_parameterized": False,
                "optimization_succeeded": False,
                "finite_final_coordinates": False,
                "topology_preserved": False,
                "quality_passed": False,
                "total_seconds": 1.0,
            },
        ),
    )

    assert row["sample_count"] == 2
    assert row["build_success_rate"] == 1.0
    assert row["optimization_success_rate"] == 0.5
    assert row["forcefield_fully_parameterized_rate"] == 0.5
    assert row["quality_pass_rate"] == 0.5
    assert row["median_case_seconds"] == 1.5


def test_cli_accepts_explicit_backend_subset(tmp_path: Path) -> None:
    arguments = build_parser().parse_args(
        (
            "--reference",
            str(tmp_path / "reference"),
            "--output",
            str(tmp_path / "output"),
            "--backends",
            "openbabel",
            "--workers",
            "16",
        )
    )

    assert arguments.backends == ["openbabel"]
    assert arguments.workers == 16
