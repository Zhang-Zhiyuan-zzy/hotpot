import os.path as osp
from glob import glob
from typing import Iterable
from operator import attrgetter

from tqdm import tqdm
import numpy as np
import torch

from hotpot.cheminfo.core import Molecule, Atom, AtomPair


__all__ = [
    "direct_edge_to_indirect",
    "extract_atom_attrs",
    "extract_bond_attrs",
    "extract_atom_pairs",
    "extract_ring_attrs",
    "merge_individual_data_to_block"
]

def direct_edge_to_indirect(attr_or_index: torch.Tensor, is_index=True) -> torch.Tensor:
    """"""
    if is_index:
        return torch.cat([attr_or_index, attr_or_index.flip(0)], dim=1)
    else:
        return torch.cat([attr_or_index, attr_or_index.flip(0)], dim=0)


def extract_atom_attrs(mol: Molecule) -> (torch.Tensor, list):
    x_names = Atom._attrs_enumerator[:15]
    additional_attr_names = ('is_metal',)
    x_names = x_names + additional_attr_names
    additional_attr_getter = attrgetter(*additional_attr_names)
    x = torch.from_numpy(np.array([a.attrs[:15].tolist() + [additional_attr_getter(a)] for a in mol.atoms])).float()

    return x, x_names

def extract_bond_attrs(mol: Molecule, edge_attr_names: Iterable[str]) -> (torch.Tensor, torch.Tensor):
    bond_attr_getter = attrgetter(*edge_attr_names)
    if (link_matrix := mol.link_matrix).ndim == 2:
        edge_index = direct_edge_to_indirect(torch.tensor(link_matrix).T).long()
    else:
        edge_index = torch.empty(0, dtype=torch.long)
    edge_attr = direct_edge_to_indirect(torch.from_numpy(np.array([(bond_attr_getter(b)) for b in mol.bonds])), is_index=False).float()

    return edge_index, edge_attr

def extract_atom_pairs(mol: Molecule) -> (torch.Tensor, torch.Tensor, list):
    atom_pairs = mol.atom_pairs
    atom_pairs.update_pairs()
    if (idx_metrix := atom_pairs.idx_matrix).ndim == 2:
        pair_index = torch.tensor(atom_pairs.idx_matrix).T.long()
    else:
        pair_index = torch.empty(0, dtype=torch.long)
    pair_attr = torch.tensor([p.attrs for k, p in atom_pairs.items()]).float()
    pair_attr_names = AtomPair.attr_names

    return pair_index, pair_attr, pair_attr_names

def extract_ring_attrs(mol: Molecule, ring_attr_names: Iterable[str]) -> (torch.Tensor, torch.Tensor):
    rings = mol.ligand_rings
    ring_attr_getter = attrgetter(*ring_attr_names)
    rings_node_index = [r.atoms_indices for r in rings]
    rings_node_nums = [len(rni) for rni in rings_node_index]
    if rings_node_index:
        mol_rings_nums = torch.tensor([len(rings_node_nums)], dtype=torch.long)
        rings_node_index = torch.tensor(sum(rings_node_index, start=[]), dtype=torch.long)
        rings_node_nums = torch.tensor(rings_node_nums, dtype=torch.int)
        mol_rings_node_nums = torch.tensor([rings_node_nums.sum()], dtype=torch.int)
        rings_attr = torch.from_numpy(np.array([ring_attr_getter(r) for r in rings])).float()
    else:
        mol_rings_nums = torch.tensor([0], dtype=torch.long)
        rings_node_index = torch.tensor([], dtype=torch.long)
        rings_node_nums = torch.tensor([], dtype=torch.int)
        mol_rings_node_nums = torch.tensor([], dtype=torch.int)
        rings_attr = torch.tensor([], dtype=torch.float)

    return mol_rings_nums, rings_node_index, rings_node_nums, mol_rings_node_nums, rings_attr


def merge_individual_data_to_block(indiv_data_dir, merged_data_dir, bundle_size: int = 200000):
    list_data = []
    total = 0
    for i, p in enumerate(tqdm(glob(osp.join(indiv_data_dir, "*.pt")), 'Merging data'), 1):
        if torch.__version__ >= '2.6':
            list_data.append(torch.load(p, weights_only=False))
        else:
            list_data.append(torch.load(p))

        if i % bundle_size == 0:
            torch.save(list_data, osp.join(merged_data_dir, f"{i}.pt"))
            list_data = []
            total += len(list_data)

    if list_data:
        total += len(list_data)
        torch.save(list_data, osp.join(merged_data_dir, f"{total}.pt"))

