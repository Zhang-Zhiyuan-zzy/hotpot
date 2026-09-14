"""Deterministic, fail-fast RDKit conformer generation."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem


def ensure_3d_conformer(mol: Chem.Mol, seed: int = 42) -> Chem.Mol:
    if mol.GetNumConformers() > 0 and mol.GetConformer().Is3D():
        return Chem.RemoveAllHs(Chem.Mol(mol))

    with_hydrogens = Chem.AddHs(Chem.Mol(mol))
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    status = AllChem.EmbedMolecule(with_hydrogens, params)
    if status != 0:
        raise RuntimeError("RDKit ETKDG failed to generate a 3D conformer")

    if AllChem.MMFFHasAllMoleculeParams(with_hydrogens):
        optimize_status = AllChem.MMFFOptimizeMolecule(with_hydrogens)
    else:
        optimize_status = AllChem.UFFOptimizeMolecule(with_hydrogens)
    if optimize_status < 0:
        raise RuntimeError("RDKit force-field optimization failed")
    return Chem.RemoveAllHs(with_hydrogens)

