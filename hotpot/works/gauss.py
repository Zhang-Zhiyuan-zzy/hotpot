"""
python v3.9.0
@Project: hotpot
@File   : gauss
@Auther : Zhiyuan Zhang
@Data   : 2024/12/25
@Time   : 15:25
"""
from typing import Optional
from os.path import join as opj
from glob import glob

import numpy as np
import pandas as pd

from hotpot.plugins.qm.gaussian import GaussOut, export_results


def export_M_L_pair_calc_results(
        pair_log_dir,
        ligand_log_dir,
        metal_log_dir,
        nproc: Optional[int] = None,
        timeout: Optional[float] = None
):
    pair_results, pairs = export_results(*glob(opj(pair_log_dir, '*.log')), nproc=nproc, timeout=timeout)
    ligand_results, ligands = export_results(*glob(opj(ligand_log_dir, '*.log')), nproc=nproc, timeout=timeout)
    metal_results, _ = export_results(*glob(opj(metal_log_dir, '*.log')), retrieve_mol=False, nproc=nproc, timeout=timeout)

    # Get pair and ligand intersection
    index = np.intersect1d(pair_results.index.array, ligand_results.index.array)
    pair_results = pair_results.loc[index, :]
    ligand_results = ligand_results.loc[index, :]

    pair_metal = [pairs[i].metals[0].symbol for i in index]
    metal_results = metal_results.loc[pair_metal, :]
    metal_results.index = index

    delta = pair_results - ligand_results - metal_results

    pair_results.columns = [f"Pair_{c}" for c in pair_results.columns]
    ligand_results.columns = [f"Ligand_{c}" for c in ligand_results.columns]
    metal_results.columns = [f"Metal_{c}" for c in metal_results.columns]
    delta.columns = [f"Delta_{c}" for c in delta.columns]

    pairs_smiles = [pairs[i].smiles for i in index]
    ligands_smiles = [ligands[i].smiles for i in index]
    info = pd.DataFrame(
        [pairs_smiles, ligands_smiles, pair_metal],
        index=['Pair', 'Ligand', 'Metal'],
        columns=index).T

    return pd.concat([info, pair_results, ligand_results, metal_results, delta], axis=1)
