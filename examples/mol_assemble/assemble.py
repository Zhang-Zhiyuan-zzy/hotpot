# -*- coding: utf-8 -*-
"""
===========================================================
 Python    : v3.9.0
 Project   : hotpot
 File      : assemble
 Created   : 2025/7/19 18:07
 Author    : Zhiyuan Zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 
===========================================================
"""
from itertools import combinations

import hotpot as hp



def make_phen_mol():
    phen_smi = 'n1cccc2c1c3c(cc2)cccn3'
    branches = [
        'C(=O)N',
        'c1ncccc1',
        'P(=O)(O)O',
        'C(=O)O',
        'c1[nH]ccc1',
        'c1sccc1',
        'S(=O)(O)O'
    ]

    phen = hp.read_mol(phen_smi)

    mols = []
    for b1, b2 in combinations(branches, 2):
        m = hp.atom_link_atom_action(phen.copy(), [1], hp.read_mol(b1), [0])
        m = hp.atom_link_atom_action(m.copy(), [12], hp.read_mol(b2), [0])
        mols.append(m)

    factory = hp.AssembleFactory.load_default_assembler(catch_path=f'/mnt/d/zhang/OneDrive/Desktop/frame/smi.txt')
    results = factory.make(mols)

    return results

if __name__ == '__main__':
    results = make_phen_mol()

