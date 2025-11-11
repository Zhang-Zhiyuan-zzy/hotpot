"""
@File Name:        rdworks
@Project:          
@Author:           Zhiyuan Zhang
@Created On:       2025/11/10 19:36
@Project:          Hotpot

Functions to tackle ChemInfo by rdkit module
"""
from typing import Literal
from functools import reduce
from operator import or_

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import cDataStructs as cDS


_fp_generators = {
    'Morgen': AllChem.GetMorganGenerator,
    'RdKit': AllChem.GetRDKitFPGenerator,
    'Torison': AllChem.GetTopologicalTorsionGenerator,
    'AtomPair': AllChem.GetAtomPairGenerator,

}

def load_fp(
        list_mols: list[Chem.Mol],
        generator: Literal['Morgen', 'RdKit', 'AtomPair', 'Torison'] = 'Morgen',
        gen_kw: dict = None,
        sparse: bool = True,
        counts: bool = True,
):
    if gen_kw is None:
        gen_kw = {}
    if generator not in _fp_generators:
        raise ValueError(f'invalid generator {generator}, please choose from {_fp_generators.keys()}')
    else:
        fpgen = _fp_generators[generator](**gen_kw)

    if counts:
        if sparse:
            gen_fn = fpgen.GetSparseCountFingerprint
        else:
            gen_fn = fpgen.GetCountFingerprint
    else:
        if sparse:
            gen_fn = fpgen.GetSparseFingerprint
        else:
            gen_fn = fpgen.GetFingerprint

    if generator not in  ['Morgen', 'RdKit']:
        list_fp = [gen_fn(m) for m in list_mols]
        list_bit = []
    else:
        list_fp = []
        list_bit = []
        if generator == 'Morgen':
            collector = 'CollectBitInfoMap'
            getter = 'GetBitInfoMap'
        elif generator == 'RdKit':
            collector = 'CollectBitPaths'
            getter = 'GetBitPaths'
        else:
            raise ValueError(f'invalid generator {generator}')

        for mol in list_mols:
            ao = AllChem.AdditionalOutput()
            getattr(ao, collector)()
            fp = gen_fn(mol, additionalOutput=ao)
            bit = getattr(ao, getter)()

            list_fp.append(fp)
            list_bit.append(bit)

    return list_mols, list_fp, list_bit

def merge_fps(list_fp: list[cDS.ULongSparseIntVect]):
    """ Merge all fingerprints to one merged fp, by or operation """
    return reduce(or_, list_fp)

def fps_to_array(fps: list, to_bit: bool = False):
    """
    Convert list of RdKit fingerprints to numpy array.
    This function is useful, when you prepare to pass the fps to a machine learning model.
    """
    mfps = list(merge_fps(fps).GetNonzeroElements())
    arr = np.zeros((len(fps), len(mfps)))

    for i, fp in enumerate(fps):
        for f, counts in fp.GetNonzeroElements().items():
            arr[i, mfps.index(f)] = counts

    if to_bit:
        arr = arr.astype(bool).astype(int)

    return arr, mfps

def assign_fp_values_atoms(
    mol: Chem.Mol,
    scores_values: np.ndarray,  # [E, C]
    list_fp: list[int],       # 长度 E，对应 shap 行顺序
    fp_map: dict[int, list[tuple[int, int]]],  # hash → [(center atom, radius), ...]
    mode: str = "sum",        # "sum" or "mean"
    scale_to_unit: bool = True  # 是否缩放到 [-1, 1]
):
    """
    Distribute fingerprint-level SHAP values to each atom in the molecule,
    with a choice of summation or averaging mode.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Target molecule.
    scores_values : array [E, C]
        SHAP value matrix for each fingerprint (bit).
    list_fp : list[int]
        List of fingerprint hash values, aligned with shap_values.
    fp_map : dict
        Mapping from hash → list of (center atom index, radius) pairs.
    mode : str
        'sum' means summing SHAP values of all fingerprints involving an atom;
        'mean' means averaging (smoothing when an atom hits multiple fingerprints).
    scale_to_unit : bool
        Whether to scale the result to the range [-1, 1].

    Returns
    -------
    atom_sv : np.ndarray [N_atoms, C]
        Per-atom SHAP contributions (one column per class).
    """
    if mode not in {"sum", "mean"}:
        raise ValueError(f"Invalid mode '{mode}', must be 'sum' or 'mean'.")

    assert scores_values.ndim == 1, "The score value should be a flatten array."

    n_atoms = mol.GetNumAtoms()
    atom_sv = np.zeros(n_atoms, dtype=float)

    # 每个原子对应的指纹集合
    atom_to_fps = {i: [] for i in range(n_atoms)}

    for fp_hash, centers in fp_map.items():
        for center_idx, radius in centers:
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, center_idx)
            if not env:
                continue
            sub_atoms = set()
            for bidx in env:
                bond = mol.GetBondWithIdx(bidx)
                sub_atoms.add(bond.GetBeginAtomIdx())
                sub_atoms.add(bond.GetEndAtomIdx())
            for aidx in sub_atoms:
                atom_to_fps[aidx].append(fp_hash)

    fp_index_map = {fp: i for i, fp in enumerate(list_fp)}  # 哈希→行索引

    # 遍历每个原子累积 shap
    for atom_idx, fps in atom_to_fps.items():
        if not fps:
            continue
        contribs = []
        for fp_hash in fps:
            if fp_hash not in fp_index_map:
                continue
            fp_row = fp_index_map[fp_hash]
            contribs.append(scores_values[fp_row])
        if not contribs:
            continue
        contribs = np.stack(contribs)
        if mode == "sum":
            atom_sv[atom_idx] = contribs.sum(axis=0)
        else:  # mean mode
            atom_sv[atom_idx] = contribs.mean(axis=0)

    if scale_to_unit:
        abs_max = np.max(np.abs(atom_sv))
        if abs_max > 0:
            atom_sv = atom_sv / abs_max  # 保持符号, 缩放范围 [-1,1]

    return atom_sv
