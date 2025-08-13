# -*- coding: utf-8 -*-
"""
===========================================================
 Python    : v3.9.0
 Project   : hotpot
 File      : draw_svg
 Created   : 2025/8/11 16:50
 Author    : Zhiyuan Zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 
===========================================================
"""
import os
import os.path as osp

import glob
from hotpot import read_mol
from hotpot.cheminfo.draw import draw_single_mol


def generate_svg():
    mol_dir = '/mnt/d/zhang/OneDrive/Liu/nuclear medical patent/Molecule/mol_files'
    list_smiles = []
    for i, file_path in enumerate(glob.glob(os.path.join(mol_dir, '*.mol'))):
        file_name = osp.splitext(osp.split(file_path)[-1])[0]
        mol = read_mol(file_path)
        try:
            draw_single_mol(mol, osp.join(mol_dir, '..', 'svg', file_name + '.svg'))
        except ValueError:
            list_smiles.append(mol.smiles)


if __name__ == "__main__":
    generate_svg()

