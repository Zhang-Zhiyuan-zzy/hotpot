"""Legacy molecular-charge calculator."""

from ..core import Molecule
from .base import Calculator

__all__ = ["MolChargeCalculator"]


_INORGANIC_FRAGMENT_CHARGES = {
    "C12=C3C4=C5C6=C7C3=C3C8=C1C1=C9C%10=C2C4=C2C4=C%10C%10=C%11C%12=C4C4=C%13C%14=C(C5=C24)C6=C2C4=C7C3=C3C5=C8C1=C1C(=C9%10)C6=C%11C7=C8C9=C6C1=C5C1=C9C5=C(C4=C31)C2=C%14C(=C85)C%13=C%127": 0,
    "O[Te](F)(F)(F)(F)F": -1,
}


def _calc_mol_charge(mol: Molecule, pH: float = 7.4) -> int:
    if mol.is_organic or mol.is_full_halogenated:
        if len(mol.hydrogens) == 0:
            return 0
        obmol = mol.to_obmol()
        obmol.AddHydrogens(False, True, pH)
        return len(mol.atoms) - obmol.NumAtoms()

    if len(mol.atoms) == 1:
        atom = mol.atoms[0]
        if atom.formal_charge == 0:
            return mol.atoms[0].get_formal_charge()
        return atom.formal_charge

    if len(mol.metals) >= 1:
        clone_mol = mol.copy()
        clone_mol.hide_metal_ligand_bonds()
        return sum(_calc_mol_charge(component, pH=pH) for component in clone_mol.components)

    if mol.smiles in _INORGANIC_FRAGMENT_CHARGES:
        return _INORGANIC_FRAGMENT_CHARGES[mol.smiles]

    raise ValueError(f"Unknown molecule fragment: {mol.smiles}")


class MolChargeCalculator(Calculator):
    """Estimate a molecule's integer charge with the legacy rules."""

    def __call__(self, mol: Molecule, pH: float = 7.4) -> int:
        return _calc_mol_charge(mol, pH=pH)

