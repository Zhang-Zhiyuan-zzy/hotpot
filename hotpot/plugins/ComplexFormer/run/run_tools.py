# -*- coding: utf-8 -*-
"""
===========================================================
 Python    : v3.9.0
 Project   : hotpot
 File      : run_tools
 Created   : 2025/9/2 20:06
 Author    : Zhiyuan Zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 
===========================================================
"""
import os
import glob
import os.path as osp
from typing import Optional, Union

import torch
from torch import nn

from hotpot.utils import fmt_print

from . import (
    models as M,
    types as tp,
    tools,
    tasks,
    configs,
    module,
    callbacks as cbs,
)

def _get_ckpt_files(work_dir):
    # Use glob to find all .ckpt files in the specified directory
    ckpt_files = glob.glob(osp.join(work_dir, '**', '*.ckpt'), recursive=True)
    if not ckpt_files:
        raise RuntimeError(f"No checkpoints found in {work_dir}")

    # Sort the files by creation time
    ckpt_files.sort(key=os.path.getctime)

    return ckpt_files

def load_ckpt(work_dir, which: Optional[Union[int, str]] = -1):
    if isinstance(which, int):
        ckpt_files = _get_ckpt_files(work_dir)
        ckpt_file = ckpt_files[which]
    elif isinstance(which, str):
        if osp.exists(which):
            ckpt_file = which
        else:
            raise FileNotFoundError(f"Checkpoint file {which} does not exist")
    else:
        raise NotImplementedError

    fmt_print.dark_green(f"Loading checkpoint from {ckpt_file}")
    return torch.load(ckpt_file)

def load_model_state_dict(model, ckpt):
    if not isinstance(model.predictors, nn.ModuleDict):
        model.load_state_dict(ckpt['state_dict'])
        fmt_print.dark_green('load model')
    else:
        # Load core module
        core_dict = {'.'.join(k.split('.')[1:]): v for k, v in ckpt['state_dict'].items() if k.startswith('core.')}
        model.core.load_state_dict(core_dict)
        fmt_print.dark_green('load core')

        predictor_dict = {}
        for key, values in ckpt['state_dict'].items():
            if key.startswith('predictors.'):
                p_dict = predictor_dict.setdefault(key.split('.')[1], {})
                p_dict['.'.join(key.split('.')[2:])] = values

        # Load predictors
        for p_name, p_module in model.predictors.items():
            if p_name in predictor_dict:
                p_module.load_state_dict(predictor_dict[p_name])
                fmt_print.dark_green(f'load predictor[{p_name}]')
            else:
                fmt_print.bold_magenta(f"Warning: predictor['{p_name}'] not found in checkpoint, skipped!!")


def config_task(batch_preprocessor, constant_lr, core, dataModule, extractor_attr_getter, feature_extractor, hypers,
                inputs_getter, inputs_preprocessor, kwargs, loss_fn, loss_fn_wrap_tasks, loss_weight_calculator,
                loss_weight_method, lr_scheduler, lr_scheduler_frequency, lr_scheduler_kwargs, mask_need_task,
                onehot_types, optimizer, other_metrics, predictor, primary_metrics, target_getter, task_names, with_med,
                with_sol, with_xyz, work_name, x_masker, xyz_perturb_sigma
):
    task_type = tasks.specify_task_types(dataModule.is_multi_datasets, target_getter)
    task_kwargs = configs.config(
        work_name=work_name,
        task_names=task_names,
        task_type=task_type,
        dataModule=dataModule,
        inputs_getter=inputs_getter,
        core=core,
        predictor=predictor,
        feature_extractor=feature_extractor,
        target_getter=target_getter,
        loss_fn=loss_fn,
        primary_metrics=primary_metrics,
        other_metrics=other_metrics,
        hypers=hypers,
        batch_preprocessor=batch_preprocessor,
        inputs_preprocessor=inputs_preprocessor,
        with_xyz=with_xyz,
        with_sol=with_sol,
        with_med=with_med,
        xyz_perturb_sigma=xyz_perturb_sigma,
        extractor_attr_getter=extractor_attr_getter,
        loss_weight_calculator=loss_weight_calculator,
        loss_weight_method=loss_weight_method,
        loss_fn_wrap_tasks=loss_fn_wrap_tasks,
        onehot_types=onehot_types,
        x_masker=x_masker,
        mask_need_task=mask_need_task,
        optimizer=optimizer,
        constant_lr=constant_lr,
        lr_scheduler=lr_scheduler,
        lr_scheduler_frequency=lr_scheduler_frequency,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        **kwargs,
    )
    # Initialize Task object
    if task_type is tasks.MultiDataTask:
        assert isinstance(task_kwargs, list)
        task = task_type(list_kwargs=task_kwargs)
    else:
        assert isinstance(task_kwargs, dict)
        task = task_type(**task_kwargs)
    return task, task_kwargs