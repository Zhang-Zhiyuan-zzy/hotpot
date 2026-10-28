import numpy as np
from openbabel import openbabel as ob, pybel

from hotpot import Molecule
from hotpot.cheminfo import obconvert
from hotpot.cheminfo.core_utils import read_mol


def test_ob2chem_compatibility_contracts(monkeypatch):
    source = pybel.readstring("smi", "c1ccccc1O")
    atoms, bonds, index_to_row = obconvert.to_arrays(source.OBMol)

    assert atoms.shape == (7, 9)
    assert bonds.shape == (7, 3)
    imported = obconvert.obmol2mol(source.OBMol, Molecule())
    np.testing.assert_array_equal(
        [bond.bond_source_metadata["is_aromatic"] for bond in imported.bonds],
        [int(bond.IsAromatic()) for bond in ob.OBMolBondIter(source.OBMol)],
    )
    assert index_to_row == {index: index - 1 for index in range(1, 8)}
    assert obconvert.write_obmol_to_string(source.OBMol, "smi") == source.write("smi")
    assert len(read_mol("CCO", fmt="smi").atoms) == 3

    molecule = read_mol("CCO", fmt="smi")
    converted, row_to_index = obconvert.mol2obmol(molecule)
    assert converted.NumAtoms() == 3
    assert converted.NumBonds() == 2
    assert row_to_index == {0: 1, 1: 2, 2: 3}

    expected = object()
    calls = []

    def canonical(candidate):
        calls.append(candidate)
        return expected, {}

    monkeypatch.setattr(obconvert, "mol2obmol", canonical)

    fresh = Molecule()
    assert fresh.to_obmol() is expected
    assert calls == [fresh]
