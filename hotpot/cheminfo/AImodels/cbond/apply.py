# -*- coding: utf-8 -*-
"""
===========================================================
 Python    : v3.9.0
 Project   : hotpot
 File      : cbond
 Created   : 2025/9/5 23:40
 Author    : Zhiyuan Zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 
===========================================================
"""
import bisect
import logging
import warnings
import os.path as osp
from copy import copy
from typing import Any, Union

import numpy as np
import onnxruntime as ort

from ...core import Molecule, Atom
from .. import data_extract as de


_file_dir = osp.dirname(__file__)

# Basic arguments
_allow_rings_nums = (2, 4, 8, 12, 16, 32)
_allow_rings_size = (6, 8, 12, 16, 32, 64)
MAX_RINGS_NUMS = max(_allow_rings_nums)
MAX_RINGS_SIZE = max(_allow_rings_size)

providers = ort.get_available_providers()
# print(f"Available providers: {providers}")

cbond_session_stat = {}

so = ort.SessionOptions()
so.log_severity_level = 3
warnings.filterwarnings("ignore")
model_graph_partition = ort.InferenceSession(
    osp.join(_file_dir, 'onnx', "opset21_graph.onnx"), so,
    providers=providers,
)

_cbond_models: dict[tuple[int, int], Any] = {}
def get_cbond_model(rings_nums: int, rings_size: int) -> (ort.InferenceSession, int, int):
    try:
        rings_nums = _allow_rings_nums[bisect.bisect_left(_allow_rings_nums, rings_nums)]
        rings_size = _allow_rings_size[bisect.bisect_left(_allow_rings_size, rings_size)]
    except IndexError:
        raise ValueError(
            f'The molecule rings_nums larger than the allowed maximum {rings_nums} > {MAX_RINGS_NUMS}'
            f'Or, molecule rings_size larger than the allowed maximum {rings_size} > {MAX_RINGS_SIZE}'
        )

    cbond_session_stat[(rings_nums, rings_size)] = cbond_session_stat.get((rings_nums, rings_size), 0) + 1
    model = _cbond_models.get((rings_nums, rings_size), None)
    if model:
        return model, rings_nums, rings_size
    else:
        logging.info(f'[blue]Loading new InferenceSession with ({rings_nums}-{rings_size})')
        model = _cbond_models[(rings_nums, rings_size)] = ort.InferenceSession(
            osp.join(_file_dir, 'onnx', f"opset21_cbond({rings_nums}-{rings_size}).onnx"), so,
            providers=providers,
        )
        return model, rings_nums, rings_size


def extract_cbond_inputs(mol: Molecule) -> dict[str, Any]:
    _data = de.extract_atom_attrs(mol, {}, atomic_number_only=True)
    _data = de.extract_bond_attrs(mol, data=_data)
    _data = de.extract_ring_attrs(mol, data=_data)
    _data = de.extract_potentials_cbonds(mol, data=_data)
    return _data


def get_graph_cbond_inputs(_data: dict[str, Any]):
    return {
        'x': _data['x'],
        'edge_index': _data['edge_index']
    }

def padding_rings(xg, rings_node_index, rings_node_nums, rings_nums, rings_size):
    xr = xg[rings_node_index]
    indices = np.arange(rings_size)
    padded_rings_num = np.expand_dims(np.pad(rings_node_nums, (0, rings_nums - len(rings_node_nums))), axis=-1)
    rings_mask = indices >= padded_rings_num

    # split and padding
    batch_size, length = rings_mask.shape

    padded_X = np.zeros(
        (batch_size, length, xr.shape[-1]),
        dtype=xr.dtype
    )
    padded_X[~rings_mask] = xr
    return padded_X, rings_mask


def get_cbond_inputs_model(_data: dict[str, Any], xg):
    rings_node_index = _data['rings_node_index']
    rings_node_nums = _data['rings_node_nums']

    if (rings_nums := len(rings_node_nums)) > 0:
        model, rings_nums, rings_size = get_cbond_model(rings_nums, max(rings_node_nums))
    else:
        model, rings_nums, rings_size = get_cbond_model(0, 0)

    padded_X, rings_mask = padding_rings(xg, rings_node_index, rings_node_nums, rings_nums, rings_size)
    return model, padded_X, rings_mask


def pred_xg(mol_data: dict[str, Any]):
    return model_graph_partition.run(['xg'], get_graph_cbond_inputs(mol_data))[0]

def pred_cb_value(model, xg, padded_Xr, rings_mask, cbond_index):
    return model.run(
        ['cbond'],
        {'xg': xg, 'padded_Xr': padded_Xr, 'rings_mask': rings_mask, 'cbond_index': cbond_index},
    )[0]

def cbond_prediction(mol_data: dict[str, Any]):
    xg = pred_xg(mol_data)
    cb_model, padded_Xr, rings_mask = get_cbond_inputs_model(mol_data, xg)

    cbond_index = mol_data['cbond_index']
    try:
        cbond = pred_cb_value(cb_model, xg, padded_Xr, rings_mask, cbond_index)
    except ort.capi.onnxruntime_pybind11_state.InvalidArgument as e:
        print(f'xg shape {xg.shape}')
        print(f'padded_Xr shape {padded_Xr.shape}')
        print(f'rings_mask shape: {rings_mask.shape}')
        print(f'cbond_index shape {cbond_index.shape}')
        raise e

    return cbond, cbond_index, mol_data['is_cbond']


def signmod_with_offset(x, offset: float = 0.):
    return 1. / (1 + np.exp(-(x - offset)))



def init_metal_ligand_pair(mol: Molecule, metal: Union[int, str, Atom]):
    if isinstance(metal, str):
        metal = Atom(symbol=metal)
    elif isinstance(metal, int):
        metal = Atom(atomic_number=metal)
    elif isinstance(metal, Atom):
        metal = metal
    else:
        raise TypeError('metal should be the atomic_number(int), ato-mic_symbol(str) or an Atom object')

    assert metal.is_metal, f'{metal.symbol} is not a metal'
    mol.add_hydrogens()
    mol.force_remove_polar_hydrogens()
    if not metal in mol.atoms:
        assert len(mol.metals) == 0, "Only support identification of coordination pattern between a single metal and a ligand"
        metal = mol.add_atom(metal)

    return mol, metal



def auto_build_cbond(
        mol: Molecule,
        metal: Union[int, str, Atom],
        threshold: float = 0.,
        greedy: bool = True,
        sum_prob: bool = True,
):
    mol, metal = init_metal_ligand_pair(mol, metal)

    metal_idx = metal.idx

    mol_data = extract_cbond_inputs(mol)
    if mol_data['cbond_index'].size == 0:
        raise AttributeError(f'{mol} with zero cbond index!, cbond_index : {mol_data["cbond_index"].size} : {mol_data["cbond_index"].shape}')

    ligand_edge_index = mol_data['edge_index']

    pred_cb, cb_index, _ = cbond_prediction(mol_data)

    pred_cb = pred_cb.flatten()
    max_value = np.max(pred_cb)
    _exist_cbond = set(na.idx for na in metal.neighbours)
    has_cbond = copy(_exist_cbond)
    has_cbond.add(metal_idx)
    cbond_values = []
    while max_value > threshold:
        sort_idx = np.argsort(pred_cb)  # Sort the probability value from LOW to HIGH

        target_idx = sort_idx[-1]
        target_value = pred_cb[target_idx]  # The probability value of target CBond
        assert target_value == max_value, f'The target value is not equal to max value, {target_value} != {max_value}'
        ca_index = int(cb_index[1, target_idx])

        # If the selected bond has been in the cbond set
        if ca_index in has_cbond:
            if not greedy:
                break

            # Adjust the `target_idx` and `ca_index` for the first not in has_cbond set
            logging.info(f"{ca_index} has in the cbond set {has_cbond}")
            i = 0
            for i in range(2, len(sort_idx) + 1):
                # Locate the target CBond when found a CBond not in the `has_cbond`
                if int(cb_index[1, sort_idx[-i]]) not in has_cbond:
                    target_idx = sort_idx[-i]
                    target_value = pred_cb[target_idx]
                    ca_index = int(cb_index[1, target_idx])
                    break
                logging.info(f"{int(cb_index[1, sort_idx[-i]])} has in the cbond set {has_cbond}")

            # Exit bond link if the target bond score less than the threshold
            if pred_cb[target_idx] <= threshold or i == len(sort_idx):
                break

        # Show logging
        logging.debug(f"target_idx: {target_idx}; ca_index: {ca_index}； c_atom: {mol.atoms[ca_index]}")
        logging.debug(pred_cb)

        has_cbond.add(ca_index)
        cbond_values.append(signmod_with_offset(target_value, threshold))
        # mol.add_bond(-1, ca_index)
        cbond_edges = np.array([
            [metal_idx] * len(has_cbond) + list(has_cbond),  # upper nodes
            list(has_cbond) + [metal_idx] * len(has_cbond),  # lower nodes
        ])
        mol_data['edge_index'] = np.concatenate((ligand_edge_index, cbond_edges), axis=1)

        pred_cb, cb_index, _ = cbond_prediction(mol_data)
        pred_cb = pred_cb.flatten()
        max_value = np.max(pred_cb)

    # Add cbonds
    for ca_index in (has_cbond - _exist_cbond):
        mol.add_bond(metal_idx, ca_index)

    if sum_prob:
        return mol, np.prod(cbond_values)
    else:
        return mol, cbond_values


def build_one_cbond(mol: Molecule, metal: Union[int, str], threshold: float = 0., get_all: bool = False):
    mol, metal = init_metal_ligand_pair(mol.copy(), metal)

    mol_data = extract_cbond_inputs(mol)
    if mol_data['cbond_index'].size == 0:
        raise AttributeError(f'{mol} with zero cbond index!, cbond_index : {mol_data["cbond_index"].size} : {mol_data["cbond_index"].shape}')

    pred_cb, cb_index, _ = cbond_prediction(mol_data)
    pred_cb = pred_cb.flatten()

    if np.max(pred_cb) < threshold:
        logging.info(f"Not found any suitable cbond!")
        return None, None

    if not get_all:
        ca_index = int(cb_index[1, np.argmax(pred_cb)])
        mol.add_bond(metal.idx, ca_index)

        return mol, [np.max(pred_cb)]

    else:
        cbo_index = pred_cb > threshold
        cb_indices = cb_index[1][cbo_index].tolist()
        prob_cbs = pred_cb[cbo_index].reshape(-1, 1).tolist()

        mols = []
        for cb_idx in cb_indices:
            clone = mol.copy()
            clone.add_bond(metal.idx, cb_idx)
            mols.append(clone)

        return mols, prob_cbs


def build_all_possible_cbond(
        mol: Molecule,
        m: Union[int, str],
        threshold: float = 0.,
        greedy: bool = True,
        normalize_prob: bool = True,
):
    pairs, pairs_prob = build_one_cbond(mol, m, threshold, get_all=True)

    # Notation
    pairs: list[Molecule]
    pairs_prob: list[list[float]]
    assert len(pairs) == len(pairs_prob)

    _max_cb_length = 0
    linked_pairs = []
    linked_pairs_prob = []
    for pair, probs in zip(pairs, pairs_prob):
        m = pair.metals[0]
        linked_pair, linked_prob = auto_build_cbond(pair, m, threshold, greedy=greedy, sum_prob=False)
        _max_cb_length = max(_max_cb_length, len(linked_prob))
        linked_pairs.append(linked_pair)
        linked_pairs_prob.append(probs + linked_prob)

    linked_pairs_prob = [np.prod(linked_prob) for linked_prob in linked_pairs_prob]
    if normalize_prob:
        sum_prob = sum(linked_pairs_prob)
        linked_pairs_prob = [linked_prob / sum_prob for linked_prob in linked_pairs_prob]
    return linked_pairs, linked_pairs_prob

