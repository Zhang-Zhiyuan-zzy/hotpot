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
from examples.BayesianDesign.machines_config import (
    project_root
)

from modules.data_process import process_SclogK


def run_SclogK_process():
    SclogK_data_dir = osp.join(project_root, 'datasets', 'SclogK')
    if not osp.exists(SclogK_data_dir):
        os.mkdir(SclogK_data_dir)

    process_SclogK(
        osp.join(project_root, 'raws_ds', 'SClogK1.xlsx'),
        SclogK_data_dir,
        # store_metal_cluster=True
    )

if __name__ == '__main__':
    run_SclogK_process()
