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
    dir_log = '/home/zz1/proj/gauss/new/log/Cs-DIXXOZ-18'
    md = MolBundle.read_from_dir('g16log', dir_log, match_pattern='*.log')
