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
from hotpot.cheminfo.draw import draw_grid



def make_phen_mol():
    phen_smi = 'n1cccc2c1c3c(cc2)cccn3'
    branches = [
        'C(=O)N',
        'c1ncccc1',
        'P(=O)(O)O',
        'C(=O)O',
        'c1[nH]ccc1',
        'c1sccc1',
        'S(=O)(=O)O'
    ]

    phen = hp.read_mol(phen_smi)

    # Assemble base Frameworks
    mols = []
    for b1, b2 in combinations(branches, 2):
        m = hp.atom_link_atom_action(phen.copy(), [1], hp.read_mol(b1), [0])
        m = hp.atom_link_atom_action(m.copy(), [12], hp.read_mol(b2), [0])
        mols.append(m)

    # with open('/home/zz1/datasets/PhenMols/smi.txt') as f:
    #     mols = f.read().splitlines()

    factory = hp.AssembleFactory.init_with_default_assembler(
        catch_path='/home/zz1/datasets/PhenMols/smi.txt',
        max_results=30000000,
        iter_step=5
    )
    # factory.assembler = [a for a in factory.assembler if isinstance(a, hp.AtomReplace)]
    results = factory.mp_make(mols, batch_size=5)

    return results

if __name__ == '__main__':
    import random

    results = list(make_phen_mol())
    # chosen_res = random.choices(results, k=133)
    # draw_grid(chosen_res, save_path='/mnt/d/zhang/OneDrive/Desktop/mol_image.png')
