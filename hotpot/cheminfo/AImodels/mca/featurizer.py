"""Dependency-light Uni-Mol v2 graph featurization and padding.

The feature definitions are derived from DP Technology's MIT-licensed
``unimol_tools.data.conformer`` implementation. See THIRD_PARTY_LICENSES.md.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem


ALLOWABLE = {
    "atomic_num": list(range(1, 119)) + ["misc"],
    "chirality": [
        "CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW",
        "CHI_TRIGONALBIPYRAMIDAL", "CHI_OCTAHEDRAL", "CHI_SQUAREPLANAR", "CHI_OTHER",
    ],
    "degree": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc"],
    "formal_charge": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, "misc"],
    "num_h": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
    "radical_e": [0, 1, 2, 3, 4, "misc"],
    "hybridization": ["SP", "SP2", "SP3", "SP3D", "SP3D2", "misc"],
    "aromatic": [False, True],
    "in_ring": [False, True],
    "bond_type": ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "misc"],
    "bond_stereo": ["STEREONONE", "STEREOZ", "STEREOE", "STEREOCIS", "STEREOTRANS", "STEREOANY"],
    "conjugated": [False, True],
}


def _safe_index(values, value):
    return values.index(value) if value in values else len(values) - 1


def _atom_features(atom):
    return [
        _safe_index(ALLOWABLE["atomic_num"], atom.GetAtomicNum()),
        ALLOWABLE["chirality"].index(str(atom.GetChiralTag())),
        _safe_index(ALLOWABLE["degree"], atom.GetTotalDegree()),
        _safe_index(ALLOWABLE["formal_charge"], atom.GetFormalCharge()),
        _safe_index(ALLOWABLE["num_h"], atom.GetTotalNumHs()),
        _safe_index(ALLOWABLE["radical_e"], atom.GetNumRadicalElectrons()),
        _safe_index(ALLOWABLE["hybridization"], str(atom.GetHybridization())),
        ALLOWABLE["aromatic"].index(atom.GetIsAromatic()),
        ALLOWABLE["in_ring"].index(atom.IsInRing()),
    ]


def _bond_features(bond):
    return [
        _safe_index(ALLOWABLE["bond_type"], str(bond.GetBondType())),
        ALLOWABLE["bond_stereo"].index(str(bond.GetStereo())),
        ALLOWABLE["conjugated"].index(bond.GetIsConjugated()),
    ]


def _single_embedding(values, sizes):
    embedded = values.copy()
    offset = 1
    for column, size in enumerate(sizes):
        if np.any(embedded[..., column] >= size):
            raise ValueError(f"Feature column {column} exceeds embedding size {size}")
        embedded[..., column] += offset
        offset += size
    return embedded


def _shortest_paths(adjacency):
    distances = np.where(adjacency == 0, 510, adjacency).astype(np.int32)
    np.fill_diagonal(distances, 0)
    for intermediate in range(distances.shape[0]):
        distances = np.minimum(
            distances,
            distances[:, intermediate, None] + distances[None, intermediate, :],
        )
    distances[distances >= 510] = 510
    return distances


def mol_to_unimolv2(mol: Chem.Mol, max_atoms: int = 512):
    mol = Chem.RemoveAllHs(Chem.Mol(mol))
    atom_count = mol.GetNumAtoms()
    if atom_count > max_atoms:
        raise ValueError(f"Molecule has {atom_count} heavy atoms; maximum is {max_atoms}")
    if mol.GetNumConformers() == 0:
        raise ValueError("A 3D conformer is required before featurization")

    node_attributes = np.asarray([_atom_features(atom) for atom in mol.GetAtoms()], dtype=np.int32)
    edge_pairs = []
    edge_attributes = []
    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feature = _bond_features(bond)
        edge_pairs.extend(((begin, end), (end, begin)))
        edge_attributes.extend((feature, feature))

    adjacency = np.zeros((atom_count, atom_count), dtype=np.int32)
    edge_feat = np.zeros((atom_count, atom_count, 3), dtype=np.int32)
    if edge_pairs:
        edge_index = np.asarray(edge_pairs, dtype=np.int32).T
        edges = np.asarray(edge_attributes, dtype=np.int32)
        adjacency[edge_index[0], edge_index[1]] = 1
        edge_feat[edge_index[0], edge_index[1]] = _single_embedding(edges, [16, 16, 16]) + 1

    atom_feat = _single_embedding(node_attributes[:, 1:], [16] * 8) + 2
    edge_feat += 2
    degree = adjacency.sum(axis=-1) + 2
    shortest_path = _shortest_paths(adjacency) + 1
    atoms = atom_feat[..., 0]
    pair_type = np.stack(
        (np.repeat(atoms[:, None], atom_count, axis=1), np.repeat(atoms[None, :], atom_count, axis=0)),
        axis=-1,
    )
    pair_type = _single_embedding(pair_type, [128, 128])
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    periodic_table = Chem.GetPeriodicTable()
    return {
        "atom_feat": atom_feat,
        "atom_mask": np.ones(atom_count, dtype=np.int64),
        "edge_feat": edge_feat,
        "shortest_path": shortest_path,
        "degree": degree,
        "pair_type": pair_type,
        "attn_bias": np.zeros((atom_count + 1, atom_count + 1), dtype=np.float32),
        "src_tokens": np.asarray(
            [periodic_table.GetAtomicNumber(atom.GetSymbol()) for atom in mol.GetAtoms()],
            dtype=np.int64,
        ),
        "src_coord": coordinates,
    }


def collate_site_rows(features, molecule_indices, atom_indices):
    batch_size = len(atom_indices)
    max_atoms = max(features[index]["atom_feat"].shape[0] for index in molecule_indices)
    arrays = {
        "atom_feat": np.zeros((batch_size, max_atoms, 8), dtype=np.int32),
        "atom_mask": np.zeros((batch_size, max_atoms), dtype=np.int64),
        "edge_feat": np.zeros((batch_size, max_atoms, max_atoms, 3), dtype=np.int32),
        "shortest_path": np.zeros((batch_size, max_atoms, max_atoms), dtype=np.int32),
        "degree": np.zeros((batch_size, max_atoms), dtype=np.int32),
        "pair_type": np.zeros((batch_size, max_atoms, max_atoms, 2), dtype=np.int32),
        "attn_bias": np.zeros((batch_size, max_atoms + 1, max_atoms + 1), dtype=np.float32),
        "src_tokens": np.zeros((batch_size, max_atoms), dtype=np.int64),
        "src_coord": np.zeros((batch_size, max_atoms, 3), dtype=np.float32),
        "atom_index": np.asarray(atom_indices, dtype=np.int64),
    }
    for row, molecule_index in enumerate(molecule_indices):
        feature = features[molecule_index]
        atom_count = feature["atom_feat"].shape[0]
        arrays["atom_feat"][row, :atom_count] = feature["atom_feat"]
        arrays["atom_mask"][row, :atom_count] = feature["atom_mask"]
        arrays["edge_feat"][row, :atom_count, :atom_count] = feature["edge_feat"]
        arrays["shortest_path"][row, :atom_count, :atom_count] = feature["shortest_path"]
        arrays["degree"][row, :atom_count] = feature["degree"]
        arrays["pair_type"][row, :atom_count, :atom_count] = feature["pair_type"]
        arrays["attn_bias"][row, : atom_count + 1, : atom_count + 1] = feature["attn_bias"]
        arrays["src_tokens"][row, :atom_count] = feature["src_tokens"]
        arrays["src_coord"][row, :atom_count] = feature["src_coord"]
    return arrays
