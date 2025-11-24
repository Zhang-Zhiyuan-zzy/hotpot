"""
@File Name:        ccdc
@Project:          
@Author:           Zhiyuan Zhang
@Created On:       2025/11/16 21:42
@Project:          Hotpot
"""
import os
import os.path as osp

import pebble
from pebble import ProcessExpired
from tqdm import tqdm

import torch

import hotpot as hp
from .utils import convert_hp_mol_to_pyg_data

__all__ = ['structs_to_PyG_data']


def structs_to_PyG_data(struct_dir: str, data_dir: str):
    list_files = os.listdir(struct_dir)

    for file in tqdm(list_files):
        mol = next(hp.MolReader(os.path.join(struct_dir, file)))
        data = convert_hp_mol_to_pyg_data(mol, identifier=file)
        torch.save(data, osp.join(data_dir, f"{data.identifier}.pt"))
