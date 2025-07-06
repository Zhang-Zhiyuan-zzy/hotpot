# -*- coding: utf-8 -*-
"""
===========================================================
 Project   : hotpot
 File      : repair_dataset
 Created   : 2025/7/6 13:46
 Author    : zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 
===========================================================
"""
import os.path as osp
from glob import glob
from typing import Callable

import torch
from torch_geometric.data import Data

from examples.BayesianDesign.machines_config import (
    project_root
)

from hotpot.utils.mp import mp_run


def repair_dataset(which: str, func: Callable[[Data], Data]):
    data_dir = osp.join(project_root, 'datasets', which)
    list_f = map(lambda f: (f,), glob(osp.join(data_dir, '*.pt')))
    mp_run(_mp_func(func), args=list_f)


def _mp_func(func: Callable[[Data], Data]) -> Callable[[str], None]:
    def running_func(f: str):
        try:
            data = torch.load(f, weights_only=False)
            data = func(data)
            torch.save(data, f)

        except Exception as e:
            print(f"An error occurred while processing file {f}: {str(e)}")

    return running_func

attr_nums = {
    'rings_attr': 2,
    'pair_attr': 2,
    'edge_attr': 3,
}
def _reshape_empty_attrs(data: Data):
    for key in data.keys():
        if key.endswith('_attr') and data[key].numel() == 0:
            data[key] = data[key].reshape(0, attr_nums[key])

    return data


if __name__ == '__main__':
    repair_dataset('mono', _reshape_empty_attrs)
