"""
@File Name:        00_cook_raw_data
@Project:          
@Author:           Zhiyuan Zhang
@Created On:       2025/11/14 17:43
@Project:          Hotpot
"""
import os.path as osp
from examples.BayesianDesign.modules import data_process
from examples.BayesianDesign.machines_config import (
    project_root
)


def cook_gibbs_beta():
    data_process.process_gibbs_beta(
        path_raw=osp.join(project_root, 'raw_ds', 'logβ-ΔG.xlsx'),
        data_dir=osp.join(project_root, 'datasets', 'GibbsBeta'),
        # link_cbond=True,
    )


if __name__ == '__main__':
    cook_gibbs_beta()

