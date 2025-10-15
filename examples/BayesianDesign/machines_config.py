# -*- coding: utf-8 -*-
"""
===========================================================
 Project   : hotpot
 File      : machines
 Created   : 2025/5/15 21:08
 Author    : zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 
===========================================================
"""
import sys
import os.path as osp
import socket

import torch

machine_name = socket.gethostname()
torch.set_default_dtype(torch.bfloat16)
if torch.cuda.is_available():
    if machine_name == '4090':
        device = torch.device("cuda:0")
    else:
        device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


# Initialize paths.
print(machine_name)
if machine_name == '4090':
    project_root = '/home/zzy/proj'
    sys.path.append(osp.join(project_root, 'hotpot'))
elif machine_name == 'DESKTOP-G9D9UUB':  # 221 PC
    project_root = '/mnt/d/zhang/OneDrive/Papers/BayesDesign/results'
    sys.path.append(osp.join(project_root, 'hotpot'))
elif machine_name == 'LAPTOP-K2H04HI4':
    project_root = '/mnt/d/zhang/OneDrive/Papers/BayesDesign/results'
    sys.path.append(osp.join(project_root, 'hotpot'))
elif machine_name == 'docker':
    project_root = '/app/proj'
    sys.path.append(osp.join(project_root, 'hotpot'))
elif machine_name == '3090':
    project_root = '/home/zz1/docker/proj'
    sys.path.append(osp.join(project_root, 'hotpot'))

# Running in Super
elif str.split(__file__, '/')[1:4] == ['data', 'run01', 'scz0s3z']:
    print('In Super')
    project_root = '/HOME/scz0s3z/run/proj/'
    sys.path.append(osp.join(project_root, 'hotpot-zzy'))
elif str.split(__file__, '/')[1:4] == ['data', 'user', 'hd54396']:
    print('In zksl Super')
    project_root = '/data/user/hd54396/proj'
    sys.path.append(osp.join(project_root, 'hotpot'))
else:
    raise ValueError(__file__)

models_dir = osp.join(project_root, 'models')
# dataset save paths
if str.split(__file__, '/')[1:4] == ['data', 'run01', 'scz0s3z']:
    print('in /dev/shm')
    dir_datasets = osp.join('/dev', 'shm', 'datasets')
else:
    dir_datasets = osp.join(project_root, 'datasets')


__all__ = [
    'machine_name',
    'project_root',
    'dir_datasets',
    'models_dir',
    'device',
]
