"""
@File Name:        utils
@Project:          
@Author:           Zhiyuan Zhang
@Created On:       2025/11/16 21:36
@Project:          Hotpot
"""
from typing import Optional

import pebble
from pebble import ProcessExpired

import torch
from torch_geometric.data import Data

import hotpot as hp
from hotpot.plugins.PyG.data.utils import (
    extract_atom_attrs,
    extract_bond_attrs,
    extract_ring_attrs,
    extract_atom_pairs,
    graph_extraction
)

med_info_names = ['names', 'CASs', 'formulas', 'Cid', 'smiless']
med_attr_names = [
    'MW', 'similarity_variables', 'Vmg_STPs', 'rhog_STPs',
    'rhog_STPs_mass', 'similarity_variable', 'Cpg', 'Cpgm', 'Cpl',
    'Cplm', 'Cps', 'Cpsm', 'Cvg', 'Cvgm', 'isentropic_exponent', 'JTg',
    'kl', 'rhog', 'SGg', 'Density_medium (kg/m3)',
    'Molar Mass_medium (g/mol)', 'Tm_medium (K)'
]
sol_info_names = ['names', 'CASs', 'formulas', 'Cid', 'smiless']
sol_attr_names = [
    'Hvap_298s', 'Hvap_298s_mass', 'Hvap_Tbs', 'Hvap_Tbs_mass', 'MW', 'omegas',
    'Parachors', 'Pcs', 'Psat_298s', 'Pts', 'rhol_STPs', 'rhol_STPs_mass',
    'similarity_variables', 'StielPolars', 'Tbs', 'Tcs', 'Tms', 'Tts',
    'Van_der_Waals_areas', 'Van_der_Waals_volumes', 'Vml_STPs', 'Vml_Tms',
    'UNIFAC_Rs', 'UNIFAC_Qs', 'rhos_Tms', 'Vms_Tms', 'rhos_Tms_mass',
    'Vml_60Fs', 'rhol_60Fs', 'rhol_60Fs_mass', 'rhog_STPs_mass',
    'sigma_STPs', 'sigma_Tms', 'sigma_Tbs'
]


def convert_hp_mol_to_pyg_data(
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


def empty_medium(prefix: str = 'med') -> tuple[list, torch.Tensor, dict]:
    """ Generate an empty medium data """
    med_info = ['Inf.Dilute', '0000-00-0', '', 0, '']
    med_attr = torch.zeros(len(med_attr_names), dtype=torch.float).reshape((1, -1))

    med_graph: dict = graph_extraction(None, prefix, with_batch=True)

    return med_info, med_attr, med_graph


def empty_solvent(prefix: str = 'sol2') -> tuple[list, torch.Tensor, dict]:
    sol2_info = ['Empty', '0000-00-0', '', 0, '']
    sol2_attr = torch.zeros(len(sol_attr_names), dtype=torch.float).reshape((1, -1))
    sol2_graph: dict = graph_extraction(None, prefix, with_batch=True)
    return sol2_info, sol2_attr, sol2_graph


def extract_graph_from_smiles(smiles: str, prefix: str) -> dict:
    medium = next(hp.MolReader(smiles, fmt='smi'))
    medium.add_hydrogens()
    med_graph: dict = graph_extraction(medium, prefix, with_batch=True)
    return med_graph

def extract_medium(ref_df, data_id, prefix: str) -> tuple[list, torch.Tensor, dict]:
    info = ref_df.loc[data_id, med_info_names]
    attr = torch.tensor(ref_df.loc[data_id, med_attr_names].tolist(), dtype=torch.float).reshape((1, -1))

    smiles = info['smiless'].strip()
    graph: dict = extract_graph_from_smiles(smiles, prefix)

    info = info.tolist()
    return info, attr, graph


def extract_solvent(ref_df, data_id, prefix: str):
    info = ref_df.loc[data_id, sol_info_names]
    attr = torch.tensor(ref_df.loc[data_id, sol_attr_names].tolist(), dtype=torch.float).reshape((1, -1))

    smiles = info['smiless'].strip()
    graph: dict = extract_graph_from_smiles(smiles, prefix)

    info = info.tolist()
    return info, attr, graph
