import pytest

import hotpot as hp
from hotpot.cheminfo.AImodels.mca.site_detection import NUCLEOPHILE_RULES


pytestmark = pytest.mark.smarts_core


POSITIVE_TARGETS = {
    "Ether": "COC",
    "Ketone": "CC(=O)C",
    "Amide": "CC(=O)N",
    "Enolate": "C=C[O-]",
    "Aldehyde": "CC=O",
    "Imine": "CC=N",
    "Nitranion": "[NH-]C",
    "Carbanion": "[CH2-]C",
    "Nitronate": "C=[N+]([O-])[O-]",
    "Ester": "CC(=O)OC",
    "Carboxylic acid": "CC(=O)O",
    "Amine": "CCN",
    "Cyanoalkyl/nitrile anion": "C=C=[N-]",
    "Nitrile": "CC#N",
    "Isonitrile": "[C-]#[N+]C",
    "Phenol": "Oc1ccccc1",
    "Silyl_ether": "CO[Si](C)(C)C",
    "Pyridine_like_nitrogen": "n1ccccc1",
    "anion_with_charge_minus1": "[O-]",
    "double_bond": "C=C",
    "double_bond_neighbouratom_with_charge_plus1": "C=[N+](C)C",
    "triple_bond": "C#C",
    "triple_bond_neighbouratom_with_charge_plus1": "C#[S+]",
    "atom_with_lone_pair": "CS",
}


@pytest.mark.parametrize(("name", "smarts"), NUCLEOPHILE_RULES)
def test_every_production_mca_rule_has_a_reviewed_positive_and_mapped_anchor(
    name, smarts
):
    query = hp.Substructure.from_smarts(smarts)
    anchor_indices = [atom.idx for atom in query.query_atoms if atom.map_number == 1]
    hits = hp.Searcher(query).search(hp.read_mol(POSITIVE_TARGETS[name], "smi"))

    assert anchor_indices == [0]
    assert hits, name
    assert any(hit.mapped_atom_indices(0) for hit in hits), name


@pytest.mark.parametrize(("name", "smarts"), NUCLEOPHILE_RULES)
def test_every_production_mca_rule_rejects_methane_control(name, smarts):
    query = hp.Substructure.from_smarts(smarts)

    assert not hp.Searcher(query).search(hp.read_mol("C", "smi")), name
