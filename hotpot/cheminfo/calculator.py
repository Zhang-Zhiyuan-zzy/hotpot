# -*- coding: utf-8 -*-
"""
===========================================================
 Project   : hotpot
 File      : calculator
 Created   : 2025/5/19 11:07
 Author    : zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
  The collection of `Calculators` to determine Crystal, Molecule, Ring, Bond, Atom
 attributes
 
===========================================================
"""
from .core import Molecule


_inorg_frag_charges = {
    "C12=C3C4=C5C6=C7C3=C3C8=C1C1=C9C%10=C2C4=C2C4=C%10C%10=C%11C%12=C4C4=C%13C%14=C(C5=C24)C6=C2C4=C7C3=C3C5=C8C1=C1C(=C9%10)C6=C%11C7=C8C9=C6C1=C5C1=C9C5=C(C4=C31)C2=C%14C(=C85)C%13=C%127": 0,
    'O[Te](F)(F)(F)(F)F': -1,

}


class Calculator:
    """ The base class of calculator """
    pass


def _calc_mol_charge(mol: Molecule, pH: float = 7.4) -> int:
    if mol.is_organic or mol.is_full_halogenated:
        if len(mol.hydrogens) == 0:
            return 0
        else:
            obmol = mol.to_obmol()
            obmol.AddHydrogens(False, True, pH)
            return len(mol.atoms) - obmol.NumAtoms()

    elif len(mol.atoms) == 1:
        atom = mol.atoms[0]
        if atom.formal_charge == 0:
            return mol.atoms[0].get_formal_charge()
        else:
            return atom.formal_charge

    elif len(mol.metals) >= 1:
        clone = mol.copy()
        clone.hide_metal_ligand_bonds()
        return sum(_calc_mol_charge(c, pH=pH) for c in clone.components)

    # If the molecule is inorganic fragment
    elif mol.smiles in _inorg_frag_charges:
        return _inorg_frag_charges[mol.smiles]

    else:
        raise ValueError(f'Unknown molecule fragment: {mol.smiles}')


class MolChargeCalculator(Calculator):
    def __call__(self, mol, pH: float = 7.4) -> float:
        return _calc_mol_charge(mol, pH=pH)
