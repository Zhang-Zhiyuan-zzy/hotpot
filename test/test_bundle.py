"""
python v3.7.9
@Project: hotpot
@File   : test_bundle.py
@Author : Zhiyuan Zhang
@Date   : 2023/3/19
@Time   : 21:25
"""
import glob
from src.bundle import MolBundle
import numpy as np


if __name__ == '__main__':
    dir_log1 = '/home/zz1/proj/gauss/new/log/Cs-VOHSAM-5'
    dir_log2 = '/home/zz1/proj/gauss/new/log/Cs-'
    dir_logs = '/home/zz1/proj/gauss/new/log/*'

    list_mb = []
    for dir_log in glob.glob(dir_logs):
        print(dir_log)
        mb = MolBundle.read_from_dir('g16log', dir_log)
        list_mb.append(mb)

    mbs = np.array(list_mb).sum()
    # var = mb.atom_num
    # v = mb.atomic_numbers

    # sum_mol = mb.merge_conformers()
