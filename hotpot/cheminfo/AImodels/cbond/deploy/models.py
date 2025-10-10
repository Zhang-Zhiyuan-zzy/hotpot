# -*- coding: utf-8 -*-
"""
===========================================================
 Python    : v3.9.0
 Project   : hotpot
 File      : infer_models
 Created   : 2025/9/5 15:11
 Author    : Zhiyuan Zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 
===========================================================
"""
import os.path as osp
import glob
from operator import attrgetter
from typing import Union, Optional, Iterable, Literal

import torch
import torch.nn.functional as F
from torch.export.dynamic_shapes import Dim

import torch_geometric as pyg

from hotpot.plugins.ComplexFormer import data
from hotpot.cheminfo.AImodels.cbond.deploy.cbond_infer_model import InferGraph, CBondInfer

from hotpot.plugins.ComplexFormer.run import run_tools as rt
from hotpot.utils.configs import setup_logging



def _convert_float_precision(
        args: list[torch.Tensor],
        float_precision: Literal['fp16', 'fp32', 'fp64', 'bf16']
) -> list[torch.Tensor]:
    _args = []
    for arg in args:
        if not torch.is_floating_point(arg):
            _args.append(arg)
        elif float_precision == 'fp16':
            _args.append(arg.to(torch.float16))
        elif float_precision == 'fp32':
            _args.append(arg.to(torch.float32))
        elif float_precision == 'fp64':
            _args.append(arg.to(torch.float64))
        elif float_precision == 'bf16':
            _args.append(arg.to(torch.bfloat32))
        else:
            raise NotImplementedError(f'Unsupported floating point precision {args}')
    return _args

def unflatten_data(
        _data: pyg.data.Data,
        float_precision: Literal['fp16', 'fp32', 'fp64', 'bf16'] = None,
):
    # Configure dynamic shape
    s0 = Dim('s0')
    s1 = Dim('s1')
    s2 = Dim('s2')
    s3 = Dim('s3')
    s4 = Dim('s4')

    dynamic_shape = {
        'x': {0: s0},
        'edge_index': {0: 2, 1: s1,},
        'rings_node_index': {0: s2,},
        'rings_node_nums': {0: s3},
        'cbond_index': {0: 2, 1: s4}
    }

    arg_names = ['x', 'edge_index', 'rings_node_index', 'rings_node_nums', 'cbond_index']

    args = list(attrgetter(*arg_names)(_data))
    args[0] = args[0][:, 0].int()

    if isinstance(float_precision, str):
        args = _convert_float_precision(args, float_precision)
    return tuple(args), arg_names, dynamic_shape


def deploy(
        work_dir: str,
        export_path: str,
        checkpoint_path: Union[str, int],

        # DataModule Arguments
        dir_datasets: str,

        # torch.onnx.export Arguments
        opset_version: Optional[int] = None,

        # Options
        float_precision: Literal['fp16', 'fp32', 'fp64', 'bf16'] = 'fp32',
        strict_core_load: bool = False,

        extract_predictors: Optional[str] = None,
        debug: bool = False,
):
    setup_logging(debug=debug)
    torch.backends.mha.set_fastpath_enabled(False)
    ##################### Base Args ##########################

    list_files = glob.glob(osp.join(dir_datasets, '*.pt'))
    example_data = data.torch_load_data(list_files[0])

    model = CBondInfer()
    infer_graph = InferGraph()

    ckpt = rt.load_ckpt(work_dir, checkpoint_path)
    rt.load_model_state_dict(model, ckpt, extract_predictors, strict_core_load)
    infer_graph.node_processor.load_state_dict(model.core.node_processor.state_dict())

    args, items, dynamic_shape = unflatten_data(example_data, float_precision)

    model = model.to(torch.float32).eval()
    infer_graph = infer_graph.to(torch.float32).eval()

    xg = infer_graph(*args[:2])
    torch.onnx.export(
        infer_graph,
        args[:2],
        osp.join(export_path,f'opset{opset_version}_graph' + '.onnx'),
        input_names=items[:2],
        output_names=['xg'],
        opset_version=opset_version,
        dynamo=True,  # force legacy path
        dynamic_axes={'x': {0: 'node_num'}, 'edge_index': {1: 'edge_num'}},
        report=True,
        external_data=False,
    )

    kw_to_extractor = {
        'xg': xg,
        'rings_node_index': args[2],
        'rings_node_nums': args[3],
    }
    padded_Xr, rings_mask = model.extract_X_rings(**kw_to_extractor)

    cbond_infer_args = [xg, padded_Xr, rings_mask, args[4]]
    cbond_infer_item = ['xg', 'padded_Xr', 'rings_mask', 'cbond_index']

    # Configure dynamic shapes
    s0 = Dim('s0')
    s1 = Dim('s1')

    dynamic_shape = {
        'xg': {0: s0},
        'padded_Xr': {0: 128, 1: 64, 2: 128},
        'rings_mask': {0: 128, 1: 64},
        'cbond_index': {0: 2, 1: s1},
    }

    cbond = model(*cbond_infer_args)
    torch.onnx.export(
        model,
        tuple(cbond_infer_args),
        osp.join(export_path,f'opset{opset_version}_cbond' + '.onnx'),
        input_names=cbond_infer_item,
        output_names=['cbond'],
        opset_version=opset_version,
        dynamo=True,  # force legacy path
        dynamic_shapes=dynamic_shape,
        report=True,
        external_data=False,
    )
    print(cbond)
