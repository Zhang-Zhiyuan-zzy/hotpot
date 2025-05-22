# -*- coding: utf-8 -*-
"""
===========================================================
 Project   : hotpot
 File      : 0_mining_csd_complexes
 Created   : 2025/5/16 9:54
 Author    : zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 
===========================================================
"""
import shutil
import os.path as osp

import json

from tqdm import tqdm

from machines_config import *
from modules.data_mining import (
    mining_mono_metal_complexes,
    statistics_complexes,
    filter_high_cn_complexes,
    filter_metallocene_complexes,
    extract_ml_pair_from_complexes,
    stat_redundant_pairs
)

# Data mining
def mining_complexes():
    mono_complexes_dir = osp.join(project_root, 'raws_ds', 'mono_complexes')
    mining_mono_metal_complexes(mono_complexes_dir)
##################

# Data statistics
def statistic_complexes_info():
    mono_complexes_dir = osp.join(project_root, 'raws_ds', 'mono_complexes')
    statistics_results_dir = osp.join(project_root, 'results', 'ds_statistics', 'complexes_statistics.json')
    statistics_complexes(mono_complexes_dir, statistics_results_dir)

def statistic_mono_info():
    mono_complexes_dir = osp.join(project_root, 'raws_ds', 'mono')
    statistics_results_dir = osp.join(project_root, 'results', 'ds_statistics', 'mono_statistics.json')
    statistics_complexes(mono_complexes_dir, statistics_results_dir)
##################

# Data cleaning
def copy_high_low_cn_complexes():
    mono_complexes_dir = osp.join(project_root, 'raws_ds', 'mono_complexes')
    statistics_results_dir = osp.join(project_root, 'raws_ds', 'hic')
    filter_high_cn_complexes(mono_complexes_dir, statistics_results_dir)

def copy_organometallic():
    mono_complexes_dir = osp.join(project_root, 'raws_ds', 'mono_complexes')
    statistics_results_dir = osp.join(project_root, 'raws_ds', 'orgmetal')
    filter_metallocene_complexes(mono_complexes_dir, statistics_results_dir)

def extract_ml_pair():
    mono_complexes_dir = osp.join(project_root, 'raws_ds', 'mono_complexes')
    ml_pair_dir = osp.join(project_root, 'raws_ds', 'mono_ml_pair')
    statistics_results_dir = osp.join(project_root, 'results', 'ds_statistics')
    extract_ml_pair_from_complexes(mono_complexes_dir, ml_pair_dir, statistics_results_dir)

def _stat_redundant_pairs():
    ml_pair_dir = osp.join(project_root, 'raws_ds', 'mono_ml_pair')
    statistics_results_path = osp.join(project_root, 'results', 'ds_statistics', 'redundant_pairs.json')

    stat_redundant_pairs(ml_pair_dir, statistics_results_path)

def reduce_redundant_pair():
    stat_result_path = osp.join(project_root, 'results', 'ds_statistics', 'redundant_pairs.json')
    ml_pair_dir = osp.join(project_root, 'raws_ds', 'mono_ml_pair')
    des_dir = osp.join(project_root, 'raws_ds', 'reduced_mono_ml_pair')

    with open(stat_result_path, 'r', encoding='utf-8') as f:
        stat_result = json.load(f)

    for file_name in tqdm(stat_result, "Copy reduced pair"):
        shutil.copy(osp.join(ml_pair_dir, f"{file_name}.mol2"), osp.join(des_dir, f"{file_name}.mol2"))


###################


if __name__ == '__main__':
    # statistic_complexes_info()
    # statistic_mono_info()
    # copy_high_low_cn_complexes()
    # copy_organometallic()
    # extract_ml_pair()
    # _stat_redundant_pairs()
    reduce_redundant_pair()
