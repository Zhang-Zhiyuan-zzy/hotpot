"""Real Open Babel integration checks for the rule-aware wrappers."""

from __future__ import annotations

from math import isfinite
from pathlib import Path

import numpy as np
import pytest
from openbabel import openbabel as ob

from hotpot.cheminfo.obWrappers import (
    RuleStage,
    build,
    prepare_optimization,
    validate_forcefield_state,
)


EXTRACTANT_FILE = (
    Path(__file__).resolve().parents[3]
    / "molecules"
    / "extractant"
    / "extractants.smi"
)
PHOSPHORUS_CASES = frozenset(
    {38, 39, 40, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 140}
)


def _read_smiles(smiles: str) -> ob.OBMol:
    conversion = ob.OBConversion()
    assert conversion.SetInFormat("smi")
    obmol = ob.OBMol()
    assert conversion.ReadString(obmol, smiles)
    return obmol


def _topology(obmol: ob.OBMol) -> tuple[object, ...]:
    atoms = tuple(
        (
            atom.GetAtomicNum(),
            atom.GetFormalCharge(),
        )
        for atom in ob.OBMolAtomIter(obmol)
    )
    bonds = tuple(
        sorted(
            (
                min(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                max(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                bond.GetBondOrder(),
                bond.IsAromatic(),
            )
            for bond in ob.OBMolBondIter(obmol)
        )
    )
    return atoms, bonds


def _coordinates(obmol: ob.OBMol) -> np.ndarray:
    return np.asarray(
        [
            (atom.GetX(), atom.GetY(), atom.GetZ())
            for atom in ob.OBMolAtomIter(obmol)
        ],
        dtype=float,
    )


def _set_coordinates(obmol: ob.OBMol, coordinates: np.ndarray) -> None:
    for atom, coordinate in zip(ob.OBMolAtomIter(obmol), coordinates):
        atom.SetVector(*coordinate)
    obmol.SetDimension(3)


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


class _CountingBuilder:
    def __init__(self) -> None:
        self.calls = 0
        self.native = ob.OBBuilder()

    def Build(self, obmol: ob.OBMol) -> bool:
        self.calls += 1
        return bool(self.native.Build(obmol))


def test_nonmatching_build_delegates_to_native_builder_once():
    obmol = _read_smiles("CCO")
    builder = _CountingBuilder()

    report = build(obmol, builder=builder)

    assert report.succeeded
    assert not report.rules.applied
    assert report.rules.stage is RuleStage.PRE_BUILD
    assert builder.calls == 1


def test_phosphorus_build_rule_is_temporary_and_removes_linear_geometry():
    obmol = _read_smiles(
        "CCOP(=O)(OCC)c1ccc2ccc3ccc(P(=O)(OCC)OCC)nc3c2n1"
    )
    phosphorus_indices = tuple(
        atom.GetIdx()
        for atom in ob.OBMolAtomIter(obmol)
        if atom.GetAtomicNum() == 15
    )
    initial_topology = _topology(obmol)
    initial_hybridizations = tuple(
        obmol.GetAtom(index).GetHyb() for index in phosphorus_indices
    )

    report = build(obmol)

    assert report.succeeded
    assert len(report.rules.applications) == 2
    assert {
        application.descriptor.rule_id
        for application in report.rules.applications
    } == {"tetracoordinate_pentavalent_phosphorus_build"}
    assert tuple(
        obmol.GetAtom(index).GetHyb() for index in phosphorus_indices
    ) == initial_hybridizations
    assert _topology(obmol) == initial_topology
    coordinates = _coordinates(obmol)
    assert np.all(np.isfinite(coordinates))
    for phosphorus_index in phosphorus_indices:
        phosphorus = obmol.GetAtom(phosphorus_index)
        center = phosphorus.GetIdx() - 1
        neighbors = tuple(
            bond.GetNbrAtom(phosphorus).GetIdx() - 1
            for bond in ob.OBAtomBondIter(phosphorus)
        )
        assert min(
            _angle_sine(coordinates, first, center, second)
            for offset, first in enumerate(neighbors)
            for second in neighbors[offset + 1 :]
        ) > 1.0e-3


def test_all_extractant_phosphorus_centers_build_with_finite_uff_gradients():
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
        obmol = _read_smiles(smiles)
        obmol.AddHydrogens()
        report = build(obmol)
        application_count += len(report.rules.applications)
        backend = ob.OBForceField.FindForceField("UFF")
        assert report.succeeded
        assert backend.Setup(obmol)
        state = validate_forcefield_state(backend, obmol)
        assert state.passed
        assert isfinite(state.energy)

    assert application_count == 28


def test_degenerate_nonlinear_torsion_is_repaired_deterministically():
    initial_coordinates = np.asarray(
        [
            (-2.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        ]
    )
    repaired_coordinates = []

    for _ in range(2):
        obmol = _read_smiles("CCC(C)C")
        _set_coordinates(obmol, initial_coordinates)
        report = prepare_optimization(obmol, "UFF")
        coordinates = _coordinates(obmol)
        repaired_coordinates.append(coordinates)

        assert report.applied
        assert {
            application.descriptor.rule_id
            for application in report.rules.applications
        } == {"degenerate_nonlinear_torsion"}
        assert _angle_sine(coordinates, 1, 2, 3) > 1.0e-6
        assert np.allclose(
            np.linalg.norm(
                coordinates[[0, 1, 3, 4]] - coordinates[2],
                axis=1,
            ),
            np.linalg.norm(
                initial_coordinates[[0, 1, 3, 4]]
                - initial_coordinates[2],
                axis=1,
            ),
        )

    assert np.array_equal(repaired_coordinates[0], repaired_coordinates[1])


@pytest.mark.parametrize("forcefield", ["MMFF94", "MMFF94s", "GAFF"])
def test_non_uff_preparation_is_a_strict_noop(forcefield):
    obmol = _read_smiles("CCC(C)C")
    initial = _coordinates(obmol)

    report = prepare_optimization(obmol, forcefield)

    assert not report.applied
    assert np.array_equal(_coordinates(obmol), initial)
