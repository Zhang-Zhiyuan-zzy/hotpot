from typing import Optional, Iterable, Any
from itertools import product
from operator import attrgetter

import numpy as np

from ..core import Molecule, Atom, AtomPair


def direct_edge_to_indirect(attr_or_index: np.ndarray) -> np.ndarray:
    return np.concatenate([attr_or_index, np.flip(attr_or_index, axis=0)], axis=1)


_default_catom_elements = {'O', 'N', 'S', 'P', 'Si', 'B'}
def extract_potentials_cbonds(
        mol: Molecule,
        catoms_symbols: Optional[Iterable[str]] = None,
        data: dict[str, np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """ Just a numpy version of hotpot.plugins.PyG.data.utils.extract_potentials_cbonds """
    if not catoms_symbols:
        catoms_symbols = _default_catom_elements
    else:
        catoms_symbols = set(catoms_symbols)

    cbond_index = []
    is_cbond = []
    all_potentials = [a for a in mol.atoms if a.symbol in catoms_symbols]
    for metal, pca in product(mol.metals, all_potentials):
        cbond_index.append([metal.idx, pca.idx])
        try:
            _ = mol.bond(metal.idx, pca.idx)
            is_cbond.append(1)
        except KeyError:
            is_cbond.append(0)

    assert len(is_cbond) == len(cbond_index)
    is_cbond = np.array(is_cbond, dtype=np.int32)
    cbond_index = np.array(cbond_index, dtype=np.int64).T

    if isinstance(data, dict):
        data.update({'cbond_index': cbond_index, 'is_cbond': is_cbond})
    else:
        data = {'cbond_index': cbond_index, 'is_cbond': is_cbond}

    return data

def extract_atom_attrs(mol: Molecule, data: dict[str, Any] = None, atomic_number_only: bool = False) -> dict:
    if atomic_number_only:
        x_names = ['atomic_number']
        x = np.array([a.attrs[0] for a in mol.atoms], dtype=np.int32)
    else:
        x_names = Atom._attrs_enumerator[:15]
        additional_attr_names = ('is_metal',)
        x_names = x_names + additional_attr_names
        additional_attr_getter = attrgetter(*additional_attr_names)
        x = np.array([a.attrs[:15].tolist() + [additional_attr_getter(a)] for a in mol.atoms], dtype=np.float64).reshape(-1, 16)

    if isinstance(data, dict):
        data.update({'x': x, 'x_names': x_names})
    else:
        data = {'x': x, 'x_names': x_names}

    return data


def extract_bond_attrs(mol: Molecule, edge_attr_names: Iterable[str] = None, data: dict[str, Any] = None) -> dict:
    if (link_matrix := mol.link_matrix).ndim == 2:
        edge_index = direct_edge_to_indirect(link_matrix.T)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)

    if edge_attr_names:
        bond_attr_getter = attrgetter(*edge_attr_names)
        edge_attr = direct_edge_to_indirect(np.array([(bond_attr_getter(b)) for b in mol.bonds], dtype=np.float64))
        edge_attr_dict = {'edge_attr': edge_attr, 'edge_attr_names': edge_attr_names}
    else:
        edge_attr_dict = {}

    if isinstance(data, dict):
        data.update({'edge_index': edge_index})
    else:
        data = {'edge_index': edge_index}
    data.update(edge_attr_dict)

    return data


def extract_atom_pairs(mol: Molecule, data: dict[str, Any] = None) -> dict:
    atom_pairs = mol.atom_pairs
    atom_pairs.update_pairs()
    if (idx_matrix := atom_pairs.idx_matrix).ndim == 2:
        pair_index = idx_matrix
    else:
        pair_index = np.empty((2, 0), dtype=np.int64)

    pair_attr_names = AtomPair.attr_names
    pair_attr = np.array([p.attrs for k, p in atom_pairs.items()], dtype=np.float64).reshape(-1, len(pair_attr_names))

    if isinstance(data, dict):
        data.update({'pair_index': pair_index, 'pair_attr_names': pair_attr_names, 'pair_attr': pair_attr})
    else:
        data = {'pair_index': pair_index, 'pair_attr_names': pair_attr_names, 'pair_attr': pair_attr}

    return data


def extract_ring_attrs(mol: Molecule, rings_attr_names: Iterable[str] = None, data: dict[str, Any] = None) -> dict:
    rings = mol.ligand_rings

    if rings:
        rings_node_index = [r.atoms_indices for r in rings]
        rings_node_nums = [len(rni) for rni in rings_node_index]

        rings_node_index = np.array(sum(rings_node_index, start=[]), dtype=np.int64)
        mol_rings_nums = np.array([len(rings_node_nums)], dtype=np.int64)
        rings_node_nums = np.array(rings_node_nums, dtype=np.int32)
        mol_rings_node_nums = np.array([rings_node_nums.sum()], dtype=np.int32)

    else:
        mol_rings_nums = np.array([0], dtype=np.int64)
        rings_node_index = np.array([], dtype=np.int64)
        rings_node_nums = np.array([], dtype=np.int32)
        mol_rings_node_nums = np.array([], dtype=np.int32)

    rings_struct_dict = {
        'mol_rings_nums': mol_rings_nums,
        'rings_node_index': rings_node_index,
        'rings_node_nums': rings_node_nums,
        'mol_rings_node_nums': mol_rings_node_nums,
    }

    if rings_attr_names:
        rings_attr_names = list(rings_attr_names)
        ring_attr_getter = attrgetter(*rings_attr_names)
        rings_attr = np.array(
            [ring_attr_getter(r) for r in rings], dtype=np.float64
        ).reshape(len(rings), len(rings_attr_names))
        rings_attr_dict = {'rings_attr': rings_attr, 'rings_attr_names': rings_attr_names}
    else:
        rings_attr_dict = {}


    if isinstance(data, dict):
        data.update(rings_struct_dict)
    else:
        data = rings_struct_dict
    data.update(rings_attr_dict)

    return data



