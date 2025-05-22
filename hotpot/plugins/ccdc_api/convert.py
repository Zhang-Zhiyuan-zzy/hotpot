# -*- coding: utf-8 -*-
"""
===========================================================
 Project   : hotpot
 File      : convert
 Created   : 2025/5/17 22:19
 Author    : zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 
===========================================================
"""
import numpy as np
from ccdc.molecule import Molecule as CMol, Atom as CAtom, Bond as CBond
from hotpot import Molecule as Mol
from hotpot.cheminfo.elements import elements

ccdc_all_bond_type = CBond.BondType.all_bond_types()

def _c_atom_is_aromatic(c_atom: CAtom) -> bool:
    return any(ring.is_aromatic for ring in c_atom.rings)

def _c_bond_is_aromatic(c_bond: CBond) -> bool:
    return any(ring.is_aromatic for ring in c_bond.rings)

def _add_mol_bonds_from_ccdc_mol(mol, c_mol, label_to_idx):
    for c_bond in c_mol.bonds:
        bond_order = ccdc_all_bond_type[str(c_bond.bond_type)]
        ca1, ca2 = c_bond.atoms
        a1idx = label_to_idx[ca1.label]
        a2idx = label_to_idx[ca2.label]
        # is_aromatic = _c_bond_is_aromatic(c_bond)
        mol.add_bond(a1idx, a2idx, bond_order=bond_order)

def mol_from_ccdc_to_hp(c_mol: CMol):
    c_mol.normalise_labels()
    label_to_idx = {a.label: i for i, a in enumerate(c_mol.atoms)}

    mol = Mol()
    for c_atom in c_mol.atom:
        n, s, p, d, f, g = elements.electron_configs[c_atom.atomic_number]
        x, y, z = c_atom.coordinates

        mol._create_atom_from_array(
            attrs_array=np.array([
                c_atom.atomic_number,
                n, s, p, d, f, g,  # electron configure
                c_atom.formal_charge,
                c_atom.partial_charge,
                float(_c_atom_is_aromatic(c_atom)),
                x, y, z,
                0,
                0,
                0,
                0, 0, 0,
                ], dtype=np.float64)
            )

    _add_mol_bonds_from_ccdc_mol(mol, c_mol, label_to_idx)
    mol._update_graph()
