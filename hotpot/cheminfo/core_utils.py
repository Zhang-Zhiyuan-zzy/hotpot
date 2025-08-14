# -*- coding: utf-8 -*-
"""
===========================================================
 Python    : v3.9.0
 Project   : hotpot
 File      : core_utils
 Created   : 2025/8/14 20:23
 Author    : Zhiyuan Zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 
===========================================================
"""
from ._io import MolReader
from .core import Molecule


def read_mol(src, fmt=None, **kwargs):
    return next(MolReader(src, fmt, **kwargs))


def atom_idx_pair_to_bond_idx(mol: Molecule, *atom_idx_pair: tuple[int, int]) -> list[int]:
    bond_atom_idx_pair = [frozenset((b.a1idx, b.a2idx)) for b in mol.bonds]
    atom_idx_pair = [frozenset(a_pair) for a_pair in atom_idx_pair]
    return [bond_atom_idx_pair.index(a_pair) for a_pair in atom_idx_pair]