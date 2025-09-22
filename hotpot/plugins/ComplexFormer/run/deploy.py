# -*- coding: utf-8 -*-
"""
===========================================================
 Python    : v3.9.0
 Project   : hotpot
 File      : deploy
 Created   : 2025/9/2 18:37
 Author    : Zhiyuan Zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 
===========================================================
"""
from typing import Union, Iterable, Optional, Sequence, Type, Callable, Literal
from operator import attrgetter

import torch
import onnx
from torch import nn
from torch.optim import Optimizer

import torch_geometric as pyg

from hotpot.utils.configs import setup_logging
from .forward import ForwardBlock
from .. import (
    models as M,
    types as tp,
    tools,
    tasks,
    data,
    configs,
    module,
    callbacks as cbs,
    run_tools as rt
)


class InferModule(ForwardBlock):
    def forward(self, inputs: Union[pyg.data.Data, pyg.data.Batch]):
        return self.f(inputs)[0]

def load_infer(
        work_dir: str,
        core: nn.Module,
        task_kwargs: Union[dict, list[dict]],
        task: Union[tasks.SingleTask, tasks.MultiTask, tasks.MultiDataTask],
        checkpoint_path: str,
) -> InferModule:
    if isinstance(task_kwargs, list):
        assert isinstance(task, tasks.MultiDataTask)
        predictor = {}
        for kw in task_kwargs:
            predictor.update(kw['predictor'])
    else:
        predictor = task_kwargs['predictor']

    model = InferModule(core, predictor, task)
    ckpt = rt.load_ckpt(work_dir, checkpoint_path)
    rt.load_model_state_dict(model, ckpt)
    model.eval()
    return model

def export_infer_model(
        work_dir: str,
        export_path: str,
        core: nn.Module,
        task_kwargs: Union[dict, list[dict]],
        task: Union[tasks.SingleTask, tasks.MultiTask, tasks.MultiDataTask],
        checkpoint_path: str,
        input_data: Union[pyg.data.Data, pyg.data.Batch],
):
    model = load_infer(work_dir, core, task_kwargs, task, checkpoint_path)

    torch.onnx.export(
        model,
        (input_data,),
        # export_path,
        input_names=['inputs'],
        output_names=['outputs'],
        dynamo=True,
    )

    onnx_model = onnx.load(export_path)
    onnx.checker.check_model(onnx_model)
    return onnx_model

def deploy(
        # Global information Arguments
        work_name: str,
        work_dir: str,
        export_path: str,
        core: M.CoreBase,
        hypers: Union[dict, tools.Hypers],

        # DataModule Arguments
        dir_datasets: str,
        dataset_names: Union[str, Sequence[str]] = None,
        exclude_datasets: Union[str, Sequence[str]] = None,
        shuffle_dataset: bool = True,
        dataModule_seed: int = 315,
        data_split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),

        # Flow control Arguments
        stages: Optional[Union[tp.Stages, Iterable[tp.Stages]]] = None,

        # Arguments of checkpoints
        checkpoint_path: Union[str, int] = None,

        # Optimizer configuration
        optimizer: Optional[Type[Optimizer]] = None,
        constant_lr: bool = False,
        lr_scheduler: Optional[Callable] = None,
        lr_scheduler_frequency: int = 2,
        lr_scheduler_kwargs: Optional[dict] = None,
        loss_weight_calculator: Optional[Union[Callable, bool]] = None,
        loss_weight_method: Literal['inverse-count', 'cross-entropy', 'sqrt-invert_count'] = 'inverse-count',

        # Inputs specification
        xyz_perturb_sigma: Optional[float] = None,
        batch_preprocessor: Optional[Union[tp.BatchPreProcessor, list[tp.BatchPreProcessor]]] = None,
        inputs_preprocessor: Optional[Union[Callable, list[Callable]]] = None,
        x_masker: Optional[Union[str, Callable]] = None,
        mask_need_task: Optional[list[str]] = None,

        # Dataset level Arguments
        with_xyz: Optional[Union[bool, Iterable[bool]]] = None,
        with_sol: Optional[Union[bool, Iterable[bool]]] = None,
        with_med: Optional[Union[bool, Iterable[bool]]] = None,
        with_env: Optional[Union[bool, Iterable[bool]]] = None,

        # Task level Arguments
        task_names: Optional[Union[list[str], list[list[str]]]] = None,
        target_getter: tp.TargetGetterInput = None,
        feature_extractor: Optional[tp.FeatureExtractorInput] = None,
        predictor: Optional[tp.PredictorInput] = None,
        loss_fn: Optional[tp.LossFnInput] = None,
        primary_metrics: Optional[tp.MetricType] = None,
        other_metrics: Optional[Union[tp.OtherMetricConfig, list[tp.OtherMetricConfig]]] = None,
        extractor_attr_getter: Optional[Union[Callable, dict[str, Callable], list[dict, Callable]]] = None,
        onehot_types: Optional[Union[int, dict[str, int], list[dict[str, int]]]] = None,
        loss_fn_wrap_tasks: Optional[Union[bool, str, set[str], list[bool]]] = None,

        # Environmental configuration and device
        devices: Optional[int] = None,
        float32_matmul_precision='medium',
        **kwargs,
):
    setup_logging(debug=True)

    ##################### Base Args ##########################
    torch.set_float32_matmul_precision(float32_matmul_precision)
    inputs_getter = attrgetter(
        'x', 'edge_index', 'edge_attr', 'rings_node_index',
        'rings_node_nums', 'mol_rings_nums', 'batch', 'ptr')

    # TODO: Adjust the input args to more efficiency.
    dataModule = data.DataModule(
        dir_datasets,
        dataset_names,
        exclude_datasets,
        seed=dataModule_seed,
        ratios=data_split_ratios,
        debug=True,
        batch_size=hypers.batch_size,
        shuffle=shuffle_dataset,
        devices=devices,
        num_replicas=devices,
        test_only=('test' in stages and 'train' not in stages),
    )

    task, task_kwargs = rt.config_task(
        batch_preprocessor, constant_lr, core, dataModule, extractor_attr_getter,
        feature_extractor, hypers, inputs_getter, inputs_preprocessor, kwargs, loss_fn,
        loss_fn_wrap_tasks, loss_weight_calculator, loss_weight_method, lr_scheduler,
        lr_scheduler_frequency, lr_scheduler_kwargs, mask_need_task, onehot_types,
        optimizer, other_metrics, predictor, primary_metrics, target_getter, task_names,
        with_med, with_sol, with_xyz, work_name, x_masker, xyz_perturb_sigma
    )

    input_data = dataModule.first_data  # TODO: debug
    onnx_model = export_infer_model(work_dir, export_path, core, task_kwargs, task, checkpoint_path, input_data)
    return onnx_model