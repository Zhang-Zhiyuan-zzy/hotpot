# -*- coding: utf-8 -*-
"""
===========================================================
 Project   : hotpot
 File      : 1_data_convert
 Created   : 2025/5/16 14:36
 Author    : zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 
===========================================================
"""
import os
from glob import glob
import os.path as osp
import multiprocessing as mp
from tqdm import tqdm

from hotpot.utils.mp import Pool
from machines_config import *
from modules.data_process import process_SclogK, ccdc_struct_to_data, convert_ml_pairs_to_cbond_broken_data

_atom_features_list = (
    "ionization_energies",
    "density",
    "melting_point",
    "boiling_point",
    ...
)


def convert_mono_complex_to_data():
    struct_dir = osp.join(project_root, 'raws_ds', 'mono_complexes')
    data_dir = osp.join(project_root, 'ds_repo', 'mono_complexes')
    ccdc_struct_to_data(struct_dir, data_dir)


def func(*args):
    struct_path, data_path = args
    return convert_ml_pairs_to_cbond_broken_data(struct_path, data_path)


def convert_pairs_to_data():
    struct_dir = osp.join(project_root, 'raws_ds', 'all_mono', 'reduced_mono_ml_pair')
    data_dir = osp.join(project_root, 'ds_repo', 'all_mono_pair')

    struct_files = glob(osp.join(struct_dir, '*.mol2'))
    args = [(p, data_dir) for p in struct_files]

    # with mp.Pool(processes=mp.cpu_count()) as pool:
    #     results = pool.map(func, args)

    pool = Pool(nproc=os.cpu_count(), desc="Converting mono pairs to data", timeout=1800)
    results = pool.run(func, args)

    results_with_extra_atoms = [r for r in results if r is not None]
    print(f"Total pairs: {len(results)}, Excluded pairs: {results_with_extra_atoms}")

    return results_with_extra_atoms


def _convert_pairs_to_data():
    struct_dir = osp.join(project_root, 'raws_ds', 'g16pairs')
    data_dir = osp.join(project_root, 'ds_repo', 'g16pairs')

    struct_files = glob(osp.join(struct_dir, '*.mol2'))

    for struct_path in tqdm(struct_files, desc="Converting mono pairs to data"):
        convert_ml_pairs_to_cbond_broken_data(struct_path, data_dir)


if __name__ == '__main__':
    # convert_mono_complex_to_data()
    res = convert_pairs_to_data()
