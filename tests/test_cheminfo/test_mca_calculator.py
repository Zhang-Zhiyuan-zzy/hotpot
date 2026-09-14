import pytest

from hotpot import read_mol
from hotpot.calculator import mca
from hotpot.cheminfo.calculator import _get_mca_predictor


def test_mca_is_attached_to_atoms_as_a_read_only_property():
    mol = read_mol("C1CCCCN1")

    with pytest.raises(AttributeError, match="hotpot.calculator.mca"):
        _ = mol.atoms[0].mca
    with pytest.raises(AttributeError, match="hotpot.calculator.mca"):
        _ = mol.mca_sites

    prediction = mca(mol, device="cpu")
    atom_values = {
        atom.atom_index: atom.mca_kj_mol
        for atom in prediction.atom_predictions
    }

    assert mol.properties["mca"] is prediction
    assert len(atom_values) == len(mol.atoms) == 6
    assert all(isinstance(atom.mca, float) for atom in mol.atoms)
    assert [atom.mca for atom in mol.atoms] == [
        atom_values[atom.idx] for atom in mol.atoms
    ]
    assert len(mol.mca_sites) == 1
    site, value = next(iter(mol.mca_sites.items()))
    assert site.symbol == "N"
    assert value == pytest.approx(503.25, abs=1e-6)

    returned_sites = mol.mca_sites
    returned_sites.clear()
    assert len(mol.mca_sites) == 1

    with pytest.raises(AttributeError):
        mol.atoms[-1].mca = 0.0
    with pytest.raises(AttributeError):
        mol.mca_sites = {}


def test_hotpot_and_smiles_inputs_produce_identical_atom_predictions():
    mol = read_mol("C1CCCCN1")
    hotpot_prediction = mca(mol, device="cpu")
    smiles_prediction = _get_mca_predictor("cpu", False).predict("C1CCCCN1")

    assert [atom.mca_kj_mol for atom in hotpot_prediction.atom_predictions] == pytest.approx(
        [atom.mca_kj_mol for atom in smiles_prediction.atom_predictions],
        abs=1e-6,
    )
    assert [site.mca_kj_mol for site in hotpot_prediction.sites] == pytest.approx(
        [site.mca_kj_mol for site in smiles_prediction.sites],
        abs=1e-6,
    )
