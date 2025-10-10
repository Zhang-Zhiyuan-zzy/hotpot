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
import logging
import os.path as osp
from typing import Any, Union

import numpy as np
import onnxruntime as ort

from ...core import Molecule, Atom
from .. import data_extract as de


_file_dir = osp.dirname(__file__)

providers = ort.get_available_providers()
print(f"Available providers: {providers}")

model_graph_partition = ort.InferenceSession(
    osp.join(_file_dir, 'onnx', "opset21_graph.onnx"),
    providers=providers,
)
model_cbond_partition = ort.InferenceSession(
    osp.join(_file_dir, 'onnx', "opset21_cbond.onnx"),
    providers=providers,
)


def extract_cbond_inputs(mol: Molecule) -> dict[str, Any]:
    _data = de.extract_atom_attrs(mol, {}, atomic_number_only=True)
    _data = de.extract_bond_attrs(mol, data=_data)
    _data = de.extract_ring_attrs(mol, data=_data)
    _data = de.extract_potentials_cbonds(mol, data=_data)
    return _data


def get_graph_cbond_inputs(_data: dict[str, Any]):
    graph_inputs = {
        'x': _data['x'].flatten(),
        'edge_index': _data['edge_index']
    }
    return graph_inputs


def get_cbond_inputs(_data: dict[str, Any], xg):
    rings_node_index = _data['rings_node_index']
    rings_node_nums = _data['rings_node_nums']
    return padding_rings(xg, rings_node_index, rings_node_nums)


def cbond_prediction(mol: Molecule):
    mol_data = extract_cbond_inputs(mol)
    graph_inputs = get_graph_cbond_inputs(mol_data)

    list_xg = model_graph_partition.run(['xg'], graph_inputs)
    padded_Xr, rings_mask = get_cbond_inputs(mol_data, list_xg[0])

    cbond_inputs = {
        'xg': list_xg[0],
        'padded_Xr': padded_Xr,
        'rings_mask': rings_mask,
        'cbond_index': mol_data['cbond_index'],
    }

    list_cbond = model_cbond_partition.run(['cbond'], cbond_inputs)
    return list_cbond[0], mol_data['cbond_index'], mol_data['is_cbond']


def auto_build_cbond(mol: Molecule, metal: Union[int, str], threshold: float = 0.):
    if isinstance(metal, str):
        metal = Atom(symbol=metal)
    elif isinstance(metal, int):
        metal = Atom(atomic_number=metal)

    assert metal.is_metal
    mol.add_atom(metal)

    pred_cb, cb_index, _ = cbond_prediction(mol)

    max_value = np.max(pred_cb)
    has_cbond = set()
    while max_value > threshold:
        max_idx = np.argmax(pred_cb.flatten())
        ca_index = int(cb_index[1, max_idx])

        logging.debug(f"max_idx: {max_idx}; ca_index: {ca_index}； c_atom: {mol.atoms[ca_index]}")
        logging.debug(pred_cb.flatten())

        if ca_index in has_cbond:
            break

        has_cbond.add(ca_index)
        mol.add_bond(-1, ca_index)

        pred_cb, cb_index, _ = cbond_prediction(mol)
        max_value = np.max(pred_cb)

    return mol


def padding_rings(xg, rings_node_index, rings_node_nums):
    xr = xg[rings_node_index]
    indices = np.arange(64)
    padded_rings_num = np.expand_dims(np.pad(rings_node_nums, (0, 128 - len(rings_node_nums))), axis=-1)
    rings_mask = indices >= padded_rings_num

    # split and padding
    batch_size, length = rings_mask.shape

    padded_X = np.zeros(
        (batch_size, length, xr.shape[-1]),
        dtype=xr.dtype
    )
    padded_X[~rings_mask] = xr
    return padded_X, rings_mask
