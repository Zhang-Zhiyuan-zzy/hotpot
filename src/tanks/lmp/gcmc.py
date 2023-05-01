"""
python v3.7.9
@Project: hotpot
@File   : gcmc.py
@Author : Zhiyuan Zhang
@Date   : 2023/4/30
@Time   : 22:19
"""
import os
from typing import *
from src.tanks.lmp.base import HpLammps


class GCMC(HpLammps):
    """ Performing the Grand-Canonical Monte Carlo (GCMC) simulation """
    def __call__(self, guests: Dict[str, float], facucity):
        """"""
