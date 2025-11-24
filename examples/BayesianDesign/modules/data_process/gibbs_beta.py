"""
@File Name:        gibbs_beta
@Project:          
@Author:           Zhiyuan Zhang
@Created On:       2025/11/16 21:59
@Project:          Hotpot
"""
import os.path as osp
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch


import hotpot as hp
from hotpot.cheminfo.MolProps import get_medium_dataframe, get_solvent_dataframe
from hotpot.plugins.PyG.data.utils import graph_extraction
from hotpot.plugins.ComplexFormer.data_process.sc_logk.data import ExtractionData
from .utils import (
    extract_medium,
    extract_solvent,
    empty_solvent,
    empty_medium,
    sol_info_names,
    sol_attr_names,
    med_info_names,
    med_attr_names
)


def process_gibbs_beta(
        path_raw: str, data_dir: str,
        link_cbond: bool = False
):
    df = pd.read_excel(path_raw)
    med = get_medium_dataframe()
    sol = get_solvent_dataframe()

    med.index = med['CASs'].tolist()
    sol.index = sol['CASs'].tolist()

    metal_clusters = set()
    for i, row in tqdm(df.iterrows(), 'Processing SclogK dataset', total=len(df)):

        # Compile Metal-ligand pair
        smi = row['SMILES'].strip()
        mol = hp.read_mol(smi, fmt='smi')

        metal_sym = row['Metal'].strip()
        charge = int(row['Charges'])

        try:
            metal = mol.create_atom(
                symbol=metal_sym,
                formal_charge=charge,
            )
        except ValueError:
            metal_clusters.add(metal_sym)
            continue

        mol.add_hydrogens()
        if link_cbond:
            mol.auto_pair_metal(metal)

        graph_data: dict = graph_extraction(mol)

        ############################################
        # Compile solvents
        sol1name, sol1id = row[['Solvent', 'Sol.CAS']]

        sol1_info, sol1_attr, sol1_graph = extract_solvent(sol, sol1id, 'sol1')
        sol2_info, sol2_attr, sol2_graph = empty_solvent(prefix='sol2')

        sol_ratio = torch.tensor([[1, 0]], dtype=torch.float)
        sol_ratio_metric = torch.tensor([-1], dtype=torch.int8)

        #############################################
        # Compile mediums
        med_name, med_id = row[['Media', 'Med.Cas']]
        if not isinstance(med_id, str):
            med_info, med_attr, med_graph = empty_medium()
        else:
            med_info, med_attr, med_graph = extract_medium(med, med_id, 'med')

        ##############################################
        mol_level_info_names = ['Med.Conc.(M)', 'groups']
        mol_level_info = torch.from_numpy(np.float64(row[mol_level_info_names].values.flatten()))

        y_names = ['logK']
        y = torch.tensor(row[y_names].tolist(), dtype=torch.float).reshape(1, -1)

        other_info_names = ['Name', 'Metal', 'Charges', 'Media', 'Med.Cas', 'Solvent', 'Sol.CAS']
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
            pair_smiles=mol.smiles,
            other_info=other_info,
            other_info_names=other_info_names,
            **graph_data,
            **sol1_graph,
            **sol2_graph,
            **med_graph
        )
        torch.save(data, osp.join(data_dir, f"{data.identifier}.pt"))
