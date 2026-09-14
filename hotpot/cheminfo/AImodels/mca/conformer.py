"""Deterministic, fail-fast RDKit conformer generation."""

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def _has_usable_3d_geometry(mol: Chem.Mol) -> bool:
    if mol.GetNumConformers() == 0:
        return False
    conformer = mol.GetConformer()
    if not conformer.Is3D():
        return False
    positions = np.asarray(conformer.GetPositions(), dtype=np.float64)
    if not np.isfinite(positions).all():
        return False
    if len(positions) < 2:
        return True
    return bool(np.max(np.linalg.norm(positions - positions[0], axis=1)) > 1e-6)


def ensure_3d_conformer(mol: Chem.Mol, seed: int = 42) -> Chem.Mol:
    if _has_usable_3d_geometry(mol):
        return Chem.RemoveAllHs(Chem.Mol(mol))

    clean = Chem.Mol(mol)
    clean.RemoveAllConformers()
    with_hydrogens = Chem.AddHs(clean)
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
