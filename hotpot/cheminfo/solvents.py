# -*- coding: utf-8 -*-
"""
===========================================================
 Project   : hotpot
 File      : solvents
 Created   : 2025/6/12 11:04
 Author    : zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 Loading and processing solvents data and properties
===========================================================
"""
import os.path as osp
from glob import glob
import pandas as pd

from hotpot import read_mol

_sol_properties_file = osp.abspath(osp.join(osp.dirname(__file__), 'ChemData', 'SolventsProperties.xlsx'))
_sol_structures_dirs = osp.abspath(osp.join(osp.dirname(__file__), 'ChemData', 'solvent'))


__all__ = ['solvents_repo']

class SolventsRepo:
    """"""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SolventsRepo, cls).__new__(cls)

        return cls._instance

    def __init__(self):
        self.sol_struct = {
            osp.splitext(osp.split(p)[-1])[0]: read_mol(p, 'mol2')
            for p in glob(osp.join(_sol_structures_dirs, '*.mol2'))
        }
        self.sol_proper = pd.read_excel(_sol_properties_file)


solvents_repo = SolventsRepo()


if __name__ == '__main__':
    proper_names = solvents_repo.sol_proper['Name'].values.tolist()
    for name, mol in solvents_repo.sol_struct.items():
        print(name, mol.smiles, name in proper_names)

    print(solvents_repo.sol_proper)
