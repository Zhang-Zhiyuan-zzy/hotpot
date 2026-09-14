import numpy as np
from openbabel import openbabel as ob, pybel

from hotpot.cheminfo import ob2chem
from hotpot.cheminfo.core_utils import read_mol


def test_ob2chem_compatibility_contracts(monkeypatch):
    source = pybel.readstring("smi", "c1ccccc1O")
    atoms, bonds, index_to_row = ob2chem.to_arrays(source.OBMol)

    assert atoms.shape == (7, 9)
    assert bonds.shape == (7, 4)
    np.testing.assert_array_equal(
        bonds[:, 3],
        [int(bond.IsAromatic()) for bond in ob.OBMolBondIter(source.OBMol)],
    )
    assert index_to_row == {index: index - 1 for index in range(1, 8)}
    assert ob2chem.ob_dump(source.OBMol, "smi") == source.write("smi")
    assert next(ob2chem.read("CCO", fmt="smi")).NumAtoms() == 3

    molecule = read_mol("CCO", fmt="smi")
    converted, row_to_index = ob2chem.to_obmol(molecule)
    assert converted.NumAtoms() == 3
    assert converted.NumBonds() == 2
    assert row_to_index == {0: 1, 1: 2, 2: 3}

    expected = object()
    calls = []

    def canonical(candidate):
        calls.append(candidate)
        return expected

    monkeypatch.setattr(ob2chem, "mol2obmol", canonical)

    assert ob2chem.to_obmol(molecule) is expected
    assert calls == [molecule]
