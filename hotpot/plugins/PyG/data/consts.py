# -*- coding: utf-8 -*-
"""
===========================================================
 Project   : hotpot
 File      : consts
 Created   : 2025/7/7 13:32
 Author    : zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 
===========================================================
"""
from hotpot.cheminfo.core import AtomPair


__all__ = [
    'edge_attr_names',
    'rings_attr_names',
    'pair_attr_names',
    'num_atom_pair_attr',
]

edge_attr_names = ('bond_order', 'is_aromatic', 'is_metal_ligand_bond')
rings_attr_names = ('is_aromatic', 'has_metal')
pair_attr_names = tuple(AtomPair.attr_names)
num_atom_pair_attr = len(AtomPair.attr_names)