import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from mca.featurizer import _shortest_paths, mol_to_unimolv2


def test_feature_shapes():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCN"))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    feature = mol_to_unimolv2(mol)
    assert feature["atom_feat"].shape == (3, 8)
    assert feature["edge_feat"].shape == (3, 3, 3)
    assert feature["attn_bias"].shape == (4, 4)


def test_dense_shortest_paths_preserve_weighted_and_disconnected_semantics():
    adjacency = np.zeros((5, 5), dtype=np.int32)
    adjacency[0, 1] = 2
    adjacency[1, 2] = 3
    adjacency[0, 2] = 9

    distances = _shortest_paths(adjacency)

    np.testing.assert_array_equal(
        distances,
        np.array(
            [
                [0, 2, 5, 510, 510],
                [510, 0, 3, 510, 510],
                [510, 510, 0, 510, 510],
                [510, 510, 510, 0, 510],
                [510, 510, 510, 510, 0],
            ],
            dtype=np.int32,
        ),
    )
    assert distances.dtype == np.int32


def test_sparse_shortest_paths_preserve_clipping_and_disconnected_semantics():
    atom_count = 512
    adjacency = np.zeros((atom_count, atom_count), dtype=np.int32)
    indices = np.arange(atom_count - 2)
    adjacency[indices, indices + 1] = 1
    adjacency[indices + 1, indices] = 1

    distances = _shortest_paths(adjacency)

    connected = np.minimum(
        np.abs(
            np.arange(atom_count - 1)[:, None]
            - np.arange(atom_count - 1)[None, :]
        ),
        510,
    ).astype(np.int32)
    expected = np.full((atom_count, atom_count), 510, dtype=np.int32)
    expected[:-1, :-1] = connected
    expected[-1, -1] = 0

    np.testing.assert_array_equal(distances, expected)
    assert distances.dtype == np.int32


def test_sparse_cutover_is_exactly_equal_to_the_dense_algorithm():
    atom_count = 256
    adjacency = np.zeros((atom_count, atom_count), dtype=np.int32)
    for index in range(atom_count - 2):
        adjacency[index, index + 1] = 1 + index % 3
    adjacency[0, 64] = 4
    adjacency[64, 128] = 5
    adjacency[128, 253] = 6

    expected = np.where(adjacency == 0, 510, adjacency).astype(np.int32)
    np.fill_diagonal(expected, 0)
    for intermediate in range(atom_count):
        expected = np.minimum(
            expected,
            expected[:, intermediate, None] + expected[None, intermediate, :],
        )
    expected[expected >= 510] = 510

    np.testing.assert_array_equal(_shortest_paths(adjacency), expected)
