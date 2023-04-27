"""
python v3.7.9
@Project: hotpot
@File   : lmp.py
@Author : Zhiyuan Zhang
@Date   : 2023/4/24
@Time   : 3:43
"""
import json
from lammps import lammps
import src


class Lammps:
    """
    A wrapper to run LAMMPS tasks
    """
    _defaults: dict = json.load(open('../data/lmp_default.json'))
    _dir_tmp: str = src.pkg_root + '/tmp'

    def __init__(self, main):
        """"""
        self.main: ci.Molecule = main


import src.cheminfo as ci
