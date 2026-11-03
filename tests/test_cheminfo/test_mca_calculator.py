import json

import pytest

from hotpot import read_mol
from hotpot.cheminfo.AImodels.mca import predict_mca_cpu
from hotpot.cheminfo.calculator import mca


pytestmark = pytest.mark.model_artifact


def test_mca_is_attached_to_atoms_as_a_read_only_property(tmp_path):
    mol = read_mol("C1CCCCN1")

    with pytest.raises(AttributeError, match="hotpot.cheminfo.calculator.mca"):
        _ = mol.atoms[0].mca
    with pytest.raises(AttributeError, match="hotpot.cheminfo.calculator.mca"):
        _ = mol.mca_sites

    prediction = mca(mol, backend="cpu")
    (tmp_path / "attached_prediction.json").write_text(
        json.dumps({"case": "attached", "prediction": prediction.to_dict()}, indent=2) + "\n"
    )
    atom_values = {
        atom.atom_index: atom.mca_kj_mol
        for atom in prediction.atom_predictions
    }

    assert mol._mca_result.payload is prediction
    assert "mca" not in mol.properties
    assert "_mca_sites" not in mol.__dict__
    assert all("_mca" not in atom.__dict__ for atom in mol.atoms)
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


def test_hotpot_and_smiles_inputs_produce_identical_atom_predictions(tmp_path):
    mol = read_mol("C1CCCCN1")
    hotpot_prediction = mca(mol, backend="cpu")
    smiles_prediction = predict_mca_cpu("C1CCCCN1")
    (tmp_path / "equivalence_predictions.json").write_text(
        json.dumps({
            "hotpot": hotpot_prediction.to_dict(),
            "smiles": smiles_prediction.to_dict(),
        }, indent=2) + "\n"
    )

    assert [atom.mca_kj_mol for atom in hotpot_prediction.atom_predictions] == pytest.approx(
        [atom.mca_kj_mol for atom in smiles_prediction.atom_predictions],
        abs=1e-6,
    )
    assert [site.mca_kj_mol for site in hotpot_prediction.sites] == pytest.approx(
        [site.mca_kj_mol for site in smiles_prediction.sites],
        abs=1e-6,
    )


def test_molecule_mca_sites_supports_empty_and_multiple_results(tmp_path):
    no_sites = read_mol("C")
    multiple_sites = read_mol("CC(=O)C")

    no_sites_prediction = mca(no_sites, backend="cpu")
    multiple_sites_prediction = mca(multiple_sites, backend="cpu")
    (tmp_path / "site_count_predictions.json").write_text(
        json.dumps({
            "empty_sites": no_sites_prediction.to_dict(),
            "multiple_sites": multiple_sites_prediction.to_dict(),
        }, indent=2) + "\n"
    )

    assert no_sites.mca_sites == {}
    assert len(multiple_sites.mca_sites) == 2
    assert {atom.symbol for atom in multiple_sites.mca_sites} == {"C", "O"}
