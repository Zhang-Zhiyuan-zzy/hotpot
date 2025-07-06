# -*- coding: utf-8 -*-
"""
===========================================================
 Project   : hotpot
 File      : get_data
 Created   : 2025/5/15 21:12
 Author    : zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 
===========================================================
"""
import os
import os.path as osp
from glob import glob

from tqdm import tqdm
import torch
from torch_geometric.loader import DataLoader

from examples.BayesianDesign.machines_config import (
    project_root
)

from modules.data_process import process_SclogK


def run_SclogK_process():
    SclogK_data_dir = osp.join(project_root, 'datasets', 'SclogK')
    if not osp.exists(SclogK_data_dir):
        os.mkdir(SclogK_data_dir)

    process_SclogK(
        osp.join(project_root, 'raws_ds', 'ScData'),
        SclogK_data_dir,
        # store_metal_cluster=True
    )

def test_loading_ScData():
    SclogK_data_dir = osp.join(project_root, 'datasets', 'SclogK')

    pt_files = glob(osp.join(SclogK_data_dir, '*.pt'))[:4000]
    data = [torch.load(p, weights_only=False) for p in pt_files]
    loader = DataLoader(data, batch_size=64, shuffle=True)

    for batch in loader:
        print(batch)


if __name__ == '__main__':
    run_SclogK_process()
    # test_loading_ScData()
    # ScData_cook()

    # SclogK_data_dir = osp.join(project_root, 'datasets', 'SclogK')
    # data = [torch.load(f, weights_only=False) for f in glob(osp.join(SclogK_data_dir, '*.pt'))[:1024]]
    # loader = DataLoader(data, batch_size=64)
    #
    # for batch in loader:
    #     print(batch)

