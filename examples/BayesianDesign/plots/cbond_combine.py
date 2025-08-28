# -*- coding: utf-8 -*-
"""
===========================================================
 Python    : v3.9.0
 Project   : hotpot
 File      : cbond_combine
 Created   : 2025/8/24 14:04
 Author    : Zhiyuan Zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 
===========================================================
"""
from typing import Iterable, Any, Union
from itertools import combinations, chain

import hotpot as hp
from hotpot.cheminfo import draw


def _full_combinations(
        x: Iterable[Any],
        get_list: bool = True,
        include_empty: bool = False,
        include_self: bool = True,
) -> Union[list[Any], chain]:
    start = 0 if include_empty else 1
    end = len(x) +1 if include_self else len(x)
    if get_list:
        return list(chain.from_iterable(combinations(x, r) for r in range(start, end)))
    return chain.from_iterable(combinations(x, r) for r in range(start, end))


def main():
    daphen = 'CN(C)C1=O[243Am]23[N]4=C1C=CC5=C4C([N]2=C(C(N(C)C)=O3)C=C6)=C6C=C5'
    mol = hp.read_mol(daphen)

    metal = mol.metals[0]
    metal_neighbours = [a for a in metal.neighbours]
    list_catoms_index = [a.idx for a in metal_neighbours]
    full_catoms_combinations = _full_combinations(list_catoms_index)  # Full combination of catom indices

    smiles_set = set()  # Recoding smiles to exclude redundant pairs with same 2d graph as the former
    for i, chosen_catoms in enumerate(full_catoms_combinations):
        clone = mol.copy()

        # Specify which cbond to be broken
        broken_cbond_catoms = set(list_catoms_index) - set(chosen_catoms)
        broken_cbond = [clone.bond(metal.idx, a_idx) for a_idx in broken_cbond_catoms]
        clone.remove_bonds(broken_cbond)  # Breaking the unchosen cbond

        # Check the smiles
        if clone.smiles in smiles_set:
            continue
        else:
            smiles_set.add(clone.smiles)

        # Assign the pair identifier
        clone.identifier = f'C{len(list_catoms_index)}_{len(chosen_catoms)}_{i}'

    draw.draw_grid(list(smiles_set), save_path='/mnt/d/zhang/OneDrive/Desktop/comb_mol.svg', n_cols=3)


if __name__ == '__main__':
    main()
