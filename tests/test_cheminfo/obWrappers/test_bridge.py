"""Integration checks for the direct native Open Babel backend."""

from __future__ import annotations

from math import isfinite
from pathlib import Path

import numpy as np
import pytest

from hotpot import read_mol
from hotpot.cheminfo.core import Molecule
from hotpot.cheminfo.obWrappers import RuleStage, build, optimize
from hotpot.cheminfo.obWrappers.forcefield import _single_optimize


EXTRACTANT_FILE = (
    Path(__file__).resolve().parents[3]
    / "molecules"
    / "extractant"
    / "extractants.smi"
)
PHOSPHORUS_CASES = frozenset(
    {38, 39, 40, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 140}
)


def _topology(mol: Molecule) -> tuple[object, ...]:
    atoms = tuple(
        (atom.atomic_number, atom.formal_charge)
        for atom in mol.atoms
    )
    bonds = tuple(
        sorted(
            (
                min(bond.atom1.idx, bond.atom2.idx),
                max(bond.atom1.idx, bond.atom2.idx),
                bond.bond_order,
                bond.bond_kind,
            )
            for bond in mol.bonds
        )
    )
    return atoms, bonds


def _angle_sine(
    coordinates: np.ndarray,
    first: int,
    center: int,
    second: int,
) -> float:
    first_vector = coordinates[first] - coordinates[center]
    second_vector = coordinates[second] - coordinates[center]
    return float(
        np.linalg.norm(np.cross(first_vector, second_vector))
        / (np.linalg.norm(first_vector) * np.linalg.norm(second_vector))
    )


def test_nonmatching_build_updates_hotpot_molecule_without_rule_application():
    mol = read_mol("CCO")

    report = build(mol)

    assert report.succeeded
    assert not report.rules.applied
    assert report.rules.stage is RuleStage.PRE_BUILD
    assert mol.coordinates.shape == (3, 3)
    assert np.all(np.isfinite(mol.coordinates))


def test_native_boundary_rejects_lossy_dative_bond_conversion():
    mol = read_mol("N.[Eu+3]", fmt="smi")
    mol.add_bond(0, 1, bond_order=1.0, bond_kind="dative")

    with pytest.raises(
        ValueError,
        match="cannot represent dative bonds losslessly",
    ):
        build(mol)


def test_phosphorus_build_rule_preserves_topology_and_removes_linear_geometry():
    mol = read_mol(
        "CCOP(=O)(OCC)c1ccc2ccc3ccc(P(=O)(OCC)OCC)nc3c2n1"
    )
    initial_topology = _topology(mol)

    report = build(mol)

    assert report.succeeded
    assert len(report.rules.applications) == 2
    assert {
        application.descriptor.rule_id
        for application in report.rules.applications
    } == {"tetracoordinate_pentavalent_phosphorus_build"}
    assert _topology(mol) == initial_topology
    assert np.all(np.isfinite(mol.coordinates))
    for phosphorus in (
        atom for atom in mol.atoms if atom.atomic_number == 15
    ):
        neighbours = tuple(atom.idx for atom in phosphorus.neighbours)
        assert min(
            _angle_sine(mol.coordinates, first, phosphorus.idx, second)
            for offset, first in enumerate(neighbours)
            for second in neighbours[offset + 1 :]
        ) > 1.0e-3


def test_all_extractant_phosphorus_centers_build_and_optimize_finitely():
    smiles_by_case = {
        index: smiles
        for index, smiles in enumerate(
            EXTRACTANT_FILE.read_text().splitlines(),
            start=1,
        )
        if index in PHOSPHORUS_CASES
    }
    assert smiles_by_case.keys() == PHOSPHORUS_CASES
    application_count = 0

    for smiles in smiles_by_case.values():
        mol = read_mol(smiles)
        mol.add_hydrogens()
        build_report = build(mol)
        application_count += len(build_report.rules.applications)
        optimization = _single_optimize(mol, "UFF", 1)

        assert build_report.succeeded
        assert isfinite(optimization.energy)
        assert np.all(np.isfinite(mol.coordinates))

    assert application_count == 28


def test_optimizer_returns_selected_and_terminal_native_frames():
    mol = read_mol("CCO")
    assert build(mol).succeeded

    report = optimize(
        mol,
        "UFF",
        epochs=3,
        steps_per_epoch=5,
        retain_frames=True,
        retain_epoch_history=True,
    )

    assert report.energy_unit == "kJ/mol"
    assert report.epochs_completed == len(report.frames)
    assert len(report.epoch_energies) == report.epochs_completed
    assert np.array_equal(mol.coordinates, report.coordinates)
    assert report.terminal_coordinates.shape == mol.coordinates.shape
    assert tuple(frame.epoch_index for frame in report.frames) == tuple(
        range(report.epochs_completed)
    )
    assert all(np.all(np.isfinite(frame.coordinates)) for frame in report.frames)


@pytest.mark.parametrize("forcefield", ["MMFF94", "MMFF94s", "GAFF"])
def test_non_uff_optimization_does_not_apply_uff_guard(forcefield):
    mol = read_mol("CCO")
    assert build(mol).succeeded

    report = optimize(
        mol,
        forcefield,
        epochs=1,
        steps_per_epoch=2,
    )

    assert not report.rules.applied
    assert isfinite(report.best_energy)
