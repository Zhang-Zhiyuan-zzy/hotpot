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
from hotpot.utils.mp import mp_run
from hotpot.cheminfo.core import AtomPair
from hotpot.plugins.PyG.data.utils import *
from hotpot.plugins.ComplexFormer.data_process.sc_logk.data import ExtractionData

__all__ = [
    'process_SclogK',
    'ccdc_struct_to_data'
]

_cols = [
    'W', 'Tech.', 'SMILES', 'Metal', 'Medium', 'Solvent', 't', 'I-str', 'pH', 'P/bar',
    'Density_medium (kg/m3)', 'Molar Mass_medium (g/mol)', 'Melting Point_medium (K)',
    'Value'
]

_edge_attr_names = ('bond_order', 'is_aromatic', 'is_metal_ligand_bond')
_rings_attr_names = ('is_aromatic', 'has_metal')

_num_atom_pair_attr = len(AtomPair.attr_names)
def _make_empty_graph(prefix: str = ''):
    return {
        f'{prefix}x': torch.empty((0, 16), dtype=torch.float),
        f'{prefix}x_names': [],
        f'{prefix}edge_index': torch.empty((2, 0), dtype=torch.long),
        f'{prefix}edge_attr': torch.empty((0, len(_edge_attr_names)), dtype=torch.float),
        f'{prefix}edge_attr_names': [],
        f'{prefix}pair_index': torch.empty((2, 0), dtype=torch.long),
        f'{prefix}pair_attr': torch.empty((0, _num_atom_pair_attr), dtype=torch.float),
        f'{prefix}pair_attr_names': [],
        f'{prefix}mol_rings_nums': torch.zeros(1, dtype=torch.int),
        f'{prefix}rings_node_index': torch.empty(0, dtype=torch.long),
        f'{prefix}rings_node_nums': torch.empty(0, dtype=torch.int),
        f'{prefix}mol_rings_node_nums': torch.zeros(1, dtype=torch.int),
        f'{prefix}rings_attr': torch.empty((0, len(_rings_attr_names)), dtype=torch.float),
        f'{prefix}rings_attr_names': [],
    }


def _graph_extraction(mol: hp.Molecule = None, prefix: str = '', with_batch: bool = False) -> dict:
    if prefix and not prefix.endswith('_'):
        prefix += '_'

    if mol is None:
        graph_data = make_empty_graph(prefix)

    else:
        x, x_names = extract_atom_attrs(mol)

        edge_index, edge_attr = extract_bond_attrs(mol, _edge_attr_names)
        pair_index, pair_attr, pair_attr_names = extract_atom_pairs(mol)

        mol_rings_nums, rings_node_index, rings_node_nums, mol_rings_node_nums, rings_attr = (
            extract_ring_attrs(mol, _rings_attr_names))

        graph_data = {
            f'{prefix}x': x,
            f'{prefix}x_names': x_names,
            f'{prefix}edge_index': edge_index,
            f'{prefix}edge_attr': edge_attr,
            f'{prefix}edge_attr_names': _edge_attr_names,
            f'{prefix}pair_index': pair_index,
            f'{prefix}pair_attr': pair_attr,
            f'{prefix}pair_attr_names': pair_attr_names,
            f'{prefix}mol_rings_nums': mol_rings_nums,
            f'{prefix}rings_node_index': rings_node_index,
            f'{prefix}rings_node_nums': rings_node_nums,
            f'{prefix}mol_rings_node_nums': mol_rings_node_nums,
            f'{prefix}rings_attr': rings_attr,
            f'{prefix}rings_attr_names': _rings_attr_names,
        }

    if with_batch:
        graph_data.update({f'{prefix}batch': torch.zeros(len(graph_data[f'{prefix}x']), dtype=torch.long)})

    return graph_data


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
sc_cols = [
    'W', 'number', 'Tech.', 'SMILES', 'Metal', 'Medium', 'Med_cid', 'Sol1',
    'Sol1_cid', 'Sol2', 'Sol2_cid', 'Sol1Ratio', 'Sol2Ratio', 'RatioMetric',
    't', 'I-str', 'pH', 'P/bar', 'logK1'
]
med_info_names = ['names', 'CASs', 'formulas', 'smiless', 'Cid']
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


# TODO: Test #########
def _process_single_SclogK(
        i: int,
        row: pd.Series,
        sol: pd.DataFrame,
        med: pd.DataFrame,
        data_dir: str
):
    smi = row['SMILES'].strip()
    mol = next(hp.MolReader(smi, fmt='smi'))

    metal_sym, charge = split_metal(row['Metal'])

    try:
        mol.create_atom(
            symbol=metal_sym,
            formal_charge=charge,
        )
    except ValueError:
        return metal_sym

    mol.add_hydrogens()
    graph_data: dict = graph_extraction(mol)

    # Compile solvent info
    sol_attr_length = len(sol_attr_names)
    sol1name, sol1id = row[['Sol1', 'Sol1_cid']]
    sol2name, sol2id = row[['Sol2', 'Sol2_cid']]

    sol1_info = sol.loc[sol1id, sol_info_names]
    sol1_attr = torch.tensor(
        np.float64(sol.loc[sol1id, sol_attr_names].values),
        dtype=torch.float
    ).reshape((1, -1))  # attr Tensor [[0.8541, 1.675, ...]], dim=2
    try:
        sol1_smi = sol1_info['smiless'].strip()
    except Exception as e:
        print(sol1_info['smiless'])
        raise e

    solvent1 = hp.read_mol(sol1_smi, fmt='smi')
    solvent1.add_hydrogens()
    sol1_graph: dict = graph_extraction(solvent1, 'sol1', with_batch=True)

    sol1_info = sol1_info.tolist()

    if not np.isnan(sol2id):
        sol2_info = sol.loc[sol2id, sol_info_names]
        sol2_attr = torch.tensor(
            np.float64(sol.loc[sol2id, sol_attr_names].values),
            dtype=torch.float
        ).reshape((1, -1))

        sol2_smi = sol2_info['smiless'].strip()
        solvent2 = hp.read_mol(sol2_smi, fmt='smi')
        solvent2.add_hydrogens()
        sol2_graph: dict = graph_extraction(solvent2, 'sol2', with_batch=True)

        sol2_info = sol2_info.tolist()

    else:
        sol2_info = []
        sol2_attr = torch.zeros(sol_attr_length, dtype=torch.float).reshape((1, -1))
        sol2_graph: dict = graph_extraction(None, 'sol2', with_batch=True)

    if sol2_info:
        sol_ratio = torch.tensor(row[['Sol1Ratio', 'Sol2Ratio']].tolist(), dtype=torch.float).reshape((1, -1))
        assert not sol_ratio.isnan().any().tolist(), f"Found NaN {sol_ratio} in {i} and number={row['number']}"
        sol_ratio = sol_ratio / torch.sum(sol_ratio)
        sol_ratio_metric = row['RatioMetric']
        if pd.isna(sol_ratio_metric):
            sol_ratio_metric = torch.tensor([0], dtype=torch.int8)
        elif sol_ratio_metric == 'Vol':
            sol_ratio_metric = torch.tensor([1], dtype=torch.int8)
        elif sol_ratio_metric == 'Wgt':
            sol_ratio_metric = torch.tensor([2], dtype=torch.int8)
        elif sol_ratio_metric == 'Mol':
            sol_ratio_metric = torch.tensor([2], dtype=torch.int8)
            sol1_Mw = sol.loc[sol1id, 'MW']
            sol2_Mw = sol.loc[sol2id, 'MW']

            sol_ratio = sol_ratio.flatten()
            weighted_ratio = torch.tensor([sol_ratio[0] * sol1_Mw, sol_ratio[1] * sol2_Mw], dtype=torch.float)
            sum_weight = sol_ratio[0] * sol1_Mw + sol_ratio[1] * sol2_Mw
            sol_ratio = (weighted_ratio / sum_weight).reshape((1, -1))

    else:
        sol_ratio = torch.tensor([[1, 0]], dtype=torch.float)
        sol_ratio_metric = torch.tensor([-1], dtype=torch.int8)

    assert not sol_ratio.isnan().any().tolist()

    # Compile Medium Info
    med_name, med_id = row[['Medium', 'Med_cid']]
    if med_id == 0:
        med_info = ['Inf.Dilute', '0000-00-0', '', 0, '']
        med_attr = torch.zeros(len(med_attr_names), dtype=torch.float).reshape((1, -1))

        med_graph: dict = graph_extraction(None, 'med', with_batch=True)

    else:
        med_info = med.loc[med_id, med_info_names]
        med_attr = torch.tensor(med.loc[med_id, med_attr_names].tolist(), dtype=torch.float).reshape((1, -1))

        med_smi = med_info['smiless'].strip()
        medium = next(hp.MolReader(med_smi, fmt='smi'))
        medium.add_hydrogens()
        med_graph: dict = graph_extraction(medium, 'med', with_batch=True)

        med_info = med_info.tolist()

    mol_level_info_names = ['t', 'I-str', 'pH', 'P/bar']
    mol_level_info = torch.from_numpy(np.float64(row[mol_level_info_names].values.flatten()))

    y_names = ['logK1']
    y = torch.tensor(row[y_names].tolist(), dtype=torch.float).reshape(1, -1)

    other_info_names = ['W', 'Tech.', 'Metal', 'Medium', 'Med_cid', 'Sol1', 'Sol1_cid', 'Sol2', 'Sol2_cid']
    other_info = row[other_info_names].tolist()

    data = ExtractionData(
        sol1_info=sol1_info,
        sol1_attr=sol1_attr,
        sol2_info=sol2_info,
        sol2_attr=sol2_attr,
        sol_info_names=sol_info_names,
        sol_attr_names=sol_attr_names,
        sol_ratio=sol_ratio,
        sol_ratio_metric=sol_ratio_metric,
        med_info=med_info,
        med_attr=med_attr,
        med_info_names=med_info_names,
        med_attr_names=med_attr_names,
        mol_level_info=mol_level_info,
        mol_level_info_names=mol_level_info_names,
        y=y,
        y_names=y_names,
        identifier=str(i),
        smiles=smi,
        other_info=other_info,
        other_info_names=other_info_names,
        **graph_data,
        **sol1_graph,
        **sol2_graph,
        **med_graph
    )
    torch.save(data, osp.join(data_dir, f"{data.identifier}.pt"))
    return None


def mp_process_SclogK(path_raw: str, data_dir: str, store_metal_cluster: bool = False):
    df = pd.read_excel(osp.join(path_raw, 'Sc.xlsx'))
    med = pd.read_excel(osp.join(path_raw, 'MedProp.xlsx'), sheet_name='clean')
    sol = pd.read_excel(osp.join(path_raw, 'SolProp.xlsx'), sheet_name='clean')

    med.index = med['Cid'].tolist()
    sol.index = sol['Cid'].tolist()

    args = [
        (i, row, sol, med, data_dir)
        for i, row in tqdm(df.iterrows(), 'Propering Argumnets', total=len(df))
    ]

    results = mp_run(_process_single_SclogK, args, error_to_None=False)
    results.remove(None)
    metal_clusters = set(results)

    if store_metal_cluster and metal_clusters:
        print(metal_clusters)
        metal_clusters = pd.Series(list(metal_clusters))
        with pd.ExcelWriter(path_raw, mode='a') as writer:
            metal_clusters.to_excel(writer, sheet_name='metal_clusters')
# TODO: ###############################################################

def process_SclogK(path_raw: str, data_dir: str, store_metal_cluster: bool = False, link_cbond: bool = False):
    df = pd.read_excel(osp.join(path_raw, 'Sc.xlsx'))
    med = pd.read_excel(osp.join(path_raw, 'MedProp.xlsx'), sheet_name='clean')
    sol = pd.read_excel(osp.join(path_raw, 'SolProp.xlsx'), sheet_name='clean')

    med.index = med['Cid'].tolist()
    sol.index = sol['Cid'].tolist()

    metal_clusters = set()
    nonmetal = []
    for i, row in tqdm(df.iterrows(), 'Processing SclogK dataset', total=len(df)):
        smi = row['SMILES'].strip()
        mol = next(hp.MolReader(smi, fmt='smi'))

        metal_sym, charge = split_metal(row['Metal'])

        try:
            metal = mol.create_atom(
                symbol=metal_sym,
                formal_charge=charge,
            )
        except ValueError:
            metal_clusters.add(metal_sym)
            continue

        if not metal.is_metal:
            nonmetal.append(metal_sym)
            continue

        mol.add_hydrogens()
        if link_cbond:
            mol.auto_pair_metal(metal)

        graph_data: dict = graph_extraction(mol)

        # Compile solvent info
        sol_attr_length = len(sol_attr_names)
        sol1name, sol1id = row[['Sol1', 'Sol1_cid']]
        sol2name, sol2id = row[['Sol2', 'Sol2_cid']]

        sol1_info = sol.loc[sol1id, sol_info_names]
        sol1_attr = torch.tensor(
            np.float64(sol.loc[sol1id, sol_attr_names].values),
            dtype=torch.float
        ).reshape((1, -1))  # attr Tensor [[0.8541, 1.675, ...]], dim=2
        try:
            sol1_smi = sol1_info['smiless'].strip()
        except Exception as e:
            print(sol1_info['smiless'])
            raise e

        solvent1 = hp.read_mol(sol1_smi, fmt='smi')
        solvent1.add_hydrogens()
        sol1_graph: dict = graph_extraction(solvent1, 'sol1', with_batch=True)

        sol1_info = sol1_info.tolist()

        if not np.isnan(sol2id):
            sol2_info = sol.loc[sol2id, sol_info_names]
            sol2_attr = torch.tensor(
                np.float64(sol.loc[sol2id, sol_attr_names].values),
                dtype=torch.float
            ).reshape((1, -1))

            sol2_smi = sol2_info['smiless'].strip()
            solvent2 = hp.read_mol(sol2_smi, fmt='smi')
            solvent2.add_hydrogens()
            sol2_graph: dict = graph_extraction(solvent2, 'sol2', with_batch=True)

            sol2_info = sol2_info.tolist()

        else:
            sol2_info = []
            sol2_attr = torch.zeros(sol_attr_length, dtype=torch.float).reshape((1, -1))
            sol2_graph: dict = graph_extraction(None, 'sol2', with_batch=True)

        if sol2_info:
            sol_ratio = torch.tensor(row[['Sol1Ratio', 'Sol2Ratio']].tolist(), dtype=torch.float).reshape((1, -1))
            assert not sol_ratio.isnan().any().tolist(), f"Found NaN {sol_ratio} in {i} and number={row['number']}"
            sol_ratio = sol_ratio / torch.sum(sol_ratio)
            sol_ratio_metric = row['RatioMetric']
            if pd.isna(sol_ratio_metric):
                sol_ratio_metric = torch.tensor([0], dtype=torch.int8)
            elif sol_ratio_metric == 'Vol':
                sol_ratio_metric = torch.tensor([1], dtype=torch.int8)
            elif sol_ratio_metric == 'Wgt':
                sol_ratio_metric = torch.tensor([2], dtype=torch.int8)
            elif sol_ratio_metric == 'Mol':
                sol_ratio_metric = torch.tensor([2], dtype=torch.int8)
                sol1_Mw = sol.loc[sol1id, 'MW']
                sol2_Mw = sol.loc[sol2id, 'MW']

                sol_ratio = sol_ratio.flatten()
                weighted_ratio = torch.tensor([sol_ratio[0] * sol1_Mw, sol_ratio[1] * sol2_Mw], dtype=torch.float)
                sum_weight = sol_ratio[0] * sol1_Mw + sol_ratio[1] * sol2_Mw
                sol_ratio = (weighted_ratio / sum_weight).reshape((1, -1))

        else:
            sol_ratio = torch.tensor([[1, 0]], dtype=torch.float)
            sol_ratio_metric = torch.tensor([-1], dtype=torch.int8)

        assert not sol_ratio.isnan().any().tolist()

        # Compile Medium Info
        med_name, med_id = row[['Medium', 'Med_cid']]
        if med_id == 0:
            med_info = ['Inf.Dilute', '0000-00-0', '', 0, '']
            med_attr = torch.zeros(len(med_attr_names), dtype=torch.float).reshape((1, -1))

            med_graph: dict = graph_extraction(None, 'med', with_batch=True)

        else:
            med_info = med.loc[med_id, med_info_names]
            med_attr = torch.tensor(med.loc[med_id, med_attr_names].tolist(), dtype=torch.float).reshape((1, -1))

            med_smi = med_info['smiless'].strip()
            medium = next(hp.MolReader(med_smi, fmt='smi'))
            medium.add_hydrogens()
            med_graph: dict = graph_extraction(medium, 'med', with_batch=True)

            med_info = med_info.tolist()

        mol_level_info_names = ['t', 'I-str', 'pH', 'P/bar']
        mol_level_info = torch.from_numpy(np.float64(row[mol_level_info_names].values.flatten()))

        y_names = ['logK1']
        y = torch.tensor(row[y_names].tolist(), dtype=torch.float).reshape(1, -1)

        other_info_names = ['W', 'Tech.', 'Metal', 'Medium', 'Med_cid', 'Sol1', 'Sol1_cid', 'Sol2', 'Sol2_cid']
        other_info = row[other_info_names].tolist()

        data = ExtractionData(
            # x=x,
            # x_names=x_names,
            # edge_index=edge_index,
            # edge_attr=edge_attr,
            # edge_attr_names=edge_attr_names,
            # pair_index=pair_index,
            # pair_attr=pair_attr,
            # pair_attr_names=pair_attr_names,
            sol1_info=sol1_info,
            sol1_attr=sol1_attr,
            sol2_info=sol2_info,
            sol2_attr=sol2_attr,
            sol_info_names=sol_info_names,
            sol_attr_names=sol_attr_names,
            sol_ratio=sol_ratio,
            sol_ratio_metric=sol_ratio_metric,
            med_info=med_info,
            med_attr=med_attr,
            med_info_names=med_info_names,
            med_attr_names=med_attr_names,
            mol_level_info=mol_level_info,
            mol_level_info_names=mol_level_info_names,
            y=y,
            y_names=y_names,
            identifier=str(i),
            # mol_rings_nums=mol_ring_nums,
            # rings_node_index=ring_node_index,
            # rings_node_nums=ring_node_nums,
            # mol_rings_node_nums=mol_ring_node_nums,
            # rings_attr=ring_attr,
            # rings_attr_names=ring_attr_names,
            smiles=smi,
            pair_smiles=mol.smiles,
            other_info=other_info,
            other_info_names=other_info_names,
            **graph_data,
            **sol1_graph,
            **sol2_graph,
            **med_graph
        )
        torch.save(data, osp.join(data_dir, f"{data.identifier}.pt"))

    if store_metal_cluster and metal_clusters:
        print(metal_clusters)
        metal_clusters = pd.Series(list(metal_clusters))
        with pd.ExcelWriter(path_raw, mode='a') as writer:
            metal_clusters.to_excel(writer, sheet_name='metal_clusters')

    print(f"non-metal elements: {set(nonmetal)}")
    print(f'number of non-metal elements: {len(nonmetal)}')

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


default_catoms_elements = {'O', 'N', 'S', 'P', 'Si', 'B'}
def convert_ml_pairs_to_cbond_broken_data(path_struct: str, data_dir: str, catoms_elements=None) -> Optional[set[str]]:
    struct_name = osp.splitext(osp.basename(path_struct))[0]

    if not catoms_elements:
        catoms_elements = default_catoms_elements

    mol = next(hp.MolReader(path_struct))
    metal = mol.metals[0]
    metal_idx = metal.idx

    metal_neighbours = [a for a in metal.neighbours]

    # Exclude pairs with extra metal-neigh atoms outside the catoms_elements
    metal_neigh_symbol = {a.symbol for a in metal.neighbours}
    if extra_natom := metal_neigh_symbol.difference(catoms_elements):
        # print(f"Exclude {struct_name} with extra catom {extra_natom} outside of {catoms_elements}")
        return extra_natom

    list_catoms_index = [a.idx for a in metal_neighbours]
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
