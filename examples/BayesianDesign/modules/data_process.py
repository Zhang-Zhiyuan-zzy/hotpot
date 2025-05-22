# -*- coding: utf-8 -*-
"""
===========================================================
 Project   : hotpot
 File      : data_process
 Created   : 2025/5/15 19:47
 Author    : Zhiyuan Zhang
 Python    : Python 3.9
-----------------------------------------------------------
 Description
 Convert raw data with various format to PyG Data object and save.

 The raw data are default stored in proj/raws_ds/directory.
 ----------------------------------------------------------
 
===========================================================
"""
import os
import os.path as osp
from typing import Iterable, Any, Union, Optional
from itertools import chain, combinations

from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
from torch_geometric.data import Data

import hotpot as hp
from hotpot.plugins.PyG.data.utils import *

__all__ = [
    'process_SclogK',
    'ccdc_struct_to_data'
]

_cols = [
    'W', 'Tech.', 'SMILES', 'Metal', 'Medium', 'Solvent', 't', 'I-str', 'pH', 'P/bar',
    'Density_medium (kg/m3)', 'Molar Mass_medium (g/mol)', 'Melting Point_medium (K)'
]

def split_metal(metal_info: str):
    if '+' in metal_info:
        sign = '+'
    elif '-' in metal_info:
        sign = '-'
    else:
        raise ValueError('Invalid metal_info')

    metal, charge = metal_info.split('+')
    metal = metal.strip()
    charge = int(charge.strip())

    if sign == '-':
        charge = -charge

    return metal, charge

_exclude_ions = {'UO2', 'VO', 'NpO2', 'PuO2', 'PuO', 'PoO', 'MoO2', 'AmO2', 'PaO2', 'TcO', 'ZrO', 'Hg2'}
def process_SclogK(path_raw: str, data_dir: str, store_metal_cluster: bool = False):
    df = pd.read_excel(path_raw, )
    df = df[_cols]

    metal_clusters = set()
    for i, row in tqdm(df.iterrows(), 'Processing SclogK dataset'):
        smi = row['SMILES'].strip()
        mol = next(hp.MolReader(smi, fmt='smi'))

        metal_sym, charge = split_metal(row['Metal'])

        try:
            mol.create_atom(
                symbol=metal_sym,
                formal_charge=charge,
            )
        except ValueError:
            metal_clusters.add(metal_sym)
            continue

        x, x_names = extract_atom_attrs(mol)

        edge_attr_names = ('bond_order', 'is_aromatic', 'is_metal_ligand_bond')
        edge_index, edge_attr = extract_bond_attrs(mol, edge_attr_names)
        pair_index, pair_attr, pair_attr_names = extract_atom_pairs(mol)

        ring_attr_names = ('is_aromatic', 'has_metal')
        mol_ring_nums, ring_node_index, ring_node_nums, mol_ring_node_nums, ring_attr = extract_ring_attrs(mol, ring_attr_names)

        y_names = ['W', 't', 'I-str', 'pH', 'P/bar',
                   'Density_medium (kg/m3)', 'Molar Mass_medium (g/mol)', 'Melting Point_medium (K)']
        y = torch.from_numpy(np.float_(row[y_names].values.flatten()))

        other_info_names = ['W', 'Tech.', 'Metal', 'Medium', 'Solvent']
        other_info = row[other_info_names].values.flatten()

        data = Data(
            x=x,
            x_names=x_names,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_attr_names=edge_attr_names,
            pair_index=pair_index,
            pair_attr=pair_attr,
            pair_attr_names=pair_attr_names,
            y=y,
            y_names=y_names,
            identifier=str(i),
            mol_rings_nums=mol_ring_nums,
            rings_node_index=ring_node_index,
            rings_node_nums=ring_node_nums,
            mol_rings_node_nums=mol_ring_node_nums,
            rings_attr=ring_attr,
            rings_attr_names=ring_attr_names,
            smiles=smi,
            other_info=other_info,
            other_info_names=other_info_names,
        )
        torch.save(data, osp.join(data_dir, f"{data.identifier}.pt"))

    if store_metal_cluster and metal_clusters:
        print(metal_clusters)
        metal_clusters = pd.Series(list(metal_clusters))
        with pd.ExcelWriter(path_raw, mode='a') as writer:
            metal_clusters.to_excel(writer, sheet_name='metal_clusters')

def ccdc_struct_to_data(struct_dir: str, data_dir: str):
    list_files = os.listdir(struct_dir)

    for file in tqdm(list_files):
        mol = next(hp.MolReader(os.path.join(struct_dir, file)))

        x, x_names = extract_atom_attrs(mol)

        edge_attr_names = ('bond_order', 'is_aromatic', 'is_metal_ligand_bond')
        edge_index, edge_attr = extract_bond_attrs(mol, edge_attr_names)
        pair_index, pair_attr, pair_attr_names = extract_atom_pairs(mol)

        ring_attr_names = ('is_aromatic', 'has_metal')
        mol_ring_nums, ring_node_index, ring_node_nums, mol_ring_node_nums, ring_attr = extract_ring_attrs(mol, ring_attr_names)

        y = None
        y_names = None

        data = Data(
            x=x,
            x_names=x_names,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_attr_names=edge_attr_names,
            pair_index=pair_index,
            pair_attr=pair_attr,
            pair_attr_names=pair_attr_names,
            y=y,
            y_names=y_names,
            identifier=file,
            mol_rings_nums=mol_ring_nums,
            rings_node_index=ring_node_index,
            rings_node_nums=ring_node_nums,
            mol_rings_node_nums=mol_ring_node_nums,
            rings_attr=ring_attr,
            rings_attr_names=ring_attr_names,
        )

        torch.save(data, osp.join(data_dir, f"{data.identifier}.pt"))


def _full_combinations(
        x: Iterable[Any],
        get_list: bool = True,
        include_empty: bool = False,
        include_self: bool = True,
) -> Union[list[Any], chain]:
    start = 0 if include_empty else 1
    end = len(x) +1 if include_self else len(x)
    if get_list:
        return list(chain.from_iterable(combinations(x, r) for r in range(start, end)))
    return chain.from_iterable(combinations(x, r) for r in range(start, end))

def _convert_hp_mol_to_pyg_data(
        mol: hp.Molecule,
        y=None, y_names=None,
        identifier: Optional[str] = None,
        **attrs
):
    # Organize the PyG Data
    x, x_names = extract_atom_attrs(mol)

    edge_attr_names = ('bond_order', 'is_aromatic', 'is_metal_ligand_bond')
    edge_index, edge_attr = extract_bond_attrs(mol, edge_attr_names)
    pair_index, pair_attr, pair_attr_names = extract_atom_pairs(mol)

    ring_attr_names = ('is_aromatic', 'has_metal')
    mol_ring_nums, ring_node_index, ring_node_nums, mol_ring_node_nums, ring_attr = extract_ring_attrs(mol, ring_attr_names)

    identifier = mol.identifier if not isinstance(identifier, str) else identifier

    return Data(
        x=x,
        x_names=x_names,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_attr_names=edge_attr_names,
        pair_index=pair_index,
        pair_attr=pair_attr,
        pair_attr_names=pair_attr_names,
        y=y,
        y_names=y_names,
        identifier=identifier,
        mol_rings_nums=mol_ring_nums,
        rings_node_index=ring_node_index,
        rings_node_nums=ring_node_nums,
        mol_rings_node_nums=mol_ring_node_nums,
        rings_attr=ring_attr,
        rings_attr_names=ring_attr_names,
        **attrs
    )


catoms_elements = {'C', 'O', 'N', 'S', 'P', 'Si'}
def convert_ml_pairs_to_cbond_broken_data(path_struct: str, data_dir: str):
    struct_name = osp.splitext(osp.basename(path_struct))[0]

    mol = next(hp.MolReader(path_struct))
    metal = mol.metals[0]
    metal_idx = metal.idx
    list_catoms_index = [a.idx for a in metal.neighbours]
    full_catom_options = {a.idx for a in mol.atoms if a.symbol in catoms_elements}

    # This block to choose combinations of coordination bonds from the raw M-L pairs.
    # The chosen bonds in a combination will be retained in the M-L pairs and the others
    # will be broken.
    full_catoms_combinations = _full_combinations(list_catoms_index)  # Full combination of catom indices
    smiles_set = set()  # Recoding smiles to exclude redundant pairs with same 2d graph as the former
    for i, chosen_catoms in enumerate(full_catoms_combinations):
        clone = mol.copy()

        # Specify which cbond to be broken
        broken_cbond_catoms = set(list_catoms_index) - set(chosen_catoms)
        broken_cbond = [clone.bond(metal.idx, a_idx) for a_idx in broken_cbond_catoms]
        clone.remove_bonds(broken_cbond)  # Breaking the unchosen cbond

        # Check the smiles
        if clone.smiles in smiles_set:
            continue
        else:
            smiles_set.add(clone.smiles)

        # Specify the cbond options in the processed M-L pair and if they are true cbonds
        catom_options = full_catom_options - set(chosen_catoms)  # Exclude retrained cbond from possible cbond options
        cbond_options = [[metal_idx, ca_idx] for ca_idx in catom_options]
        # If above `cbond_options` are true of cbond
        is_true_cbond = [float(cb[1] in list_catoms_index) for cb in cbond_options]

        # Assign the pair identifier
        clone.identifier = f'{struct_name}_C{len(list_catoms_index)}_{len(chosen_catoms)}_{i}'

        data = _convert_hp_mol_to_pyg_data(
            clone, identifier=clone.identifier,
            cbond_index=torch.tensor(cbond_options, dtype=torch.long).mT if cbond_options else torch.tensor(cbond_options, dtype=torch.long),
            is_cbond=torch.tensor(is_true_cbond, dtype=torch.int)
        )
        torch.save(data, osp.join(data_dir, f"{data.identifier}.pt"))
