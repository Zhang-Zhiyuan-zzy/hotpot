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
    # Very large, highly flexible molecules can exhaust ETKDG's initial-coordinate
    # attempts.  Keep this established path unchanged.  A possible recovery is to
    # retry only after failure, using a freshly rebuilt hydrogenated molecule, the
    # same seed, and ``useRandomCoords=True``, followed by the same MMFF/UFF step.
    # This is not prediction-neutral: an ai-14 A/B run on 256 stratified complex
    # molecules (6,486 atoms; 1,915 sites) measured warm throughput of 13.518 vs
    # 13.574 mol/s, while site MCA changed by MAE 2.654, RMSE 4.072, P95 8.25,
    # and maximum 37.625 kJ/mol.  Global site-rank Spearman was 0.999115,
    # within-molecule mean Spearman was 0.98623, top-site retention was 95.63%,
    # and the maximum global rank displacement was 171.5 of 1,915 positions.
    # Site MCA spans were 528.0 vs 527.25 kJ/mol; the paired-difference span was
    # 62.5 kJ/mol.  Site identities and ``is_Nuc_site`` were unchanged.  Therefore
    # RandomCoords should be considered only as a failure fallback, not a default.
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
