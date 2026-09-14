from rdkit import Chem
from rdkit.Chem import AllChem

from mca.featurizer import mol_to_unimolv2


def test_feature_shapes():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCN"))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    feature = mol_to_unimolv2(mol)
    assert feature["atom_feat"].shape == (3, 8)
    assert feature["edge_feat"].shape == (3, 3, 3)
    assert feature["attn_bias"].shape == (4, 4)
