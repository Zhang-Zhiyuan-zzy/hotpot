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
import logging
import os
import glob
import copy
import os.path as osp
import datetime
from typing import Optional, Union
from dataclasses import asdict

import torch
from sympy import hyper
from torch import nn

import lightning as L
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch import strategies

import hotpot.plugins.opti.params_space
from hotpot.utils import fmt_print

from hotpot.plugins.ComplexFormer import (
    tasks,
    configs,
    models as M,
    callbacks as cbs,
    types as tp,
)
from . import train, datacls
from hotpot.plugins.opti import ParamSets


############################################
# Init Module
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
    return torch.load(ckpt_file, map_location=torch.device('cpu'))

def _extract_state_dict(ckpt):
    core_dict = {'.'.join(k.split('.')[1:]): v for k, v in ckpt['state_dict'].items() if k.startswith('core.')}

    predictor_dict = {}
    for key, values in ckpt['state_dict'].items():
        if key.startswith('predictors.'):
            p_dict = predictor_dict.setdefault(key.split('.')[1], {})
            p_dict['.'.join(key.split('.')[2:])] = values

    return core_dict, predictor_dict

def load_model_state_dict(
        model: nn.Module, ckpt: dict,
        extract_predictor: Optional[str] = None,
        strict_core_load: bool = True,
):
    if isinstance(model.predictors, nn.ModuleDict):
        # Load core run
        core_dict, predictor_dict = _extract_state_dict(ckpt)
        model.core.load_state_dict(core_dict, strict=strict_core_load)
        logging.info('[bold #006400]Load Core[\]')

        # Load predictors
        for p_name, p_module in model.predictors.items():
            if p_name in predictor_dict:
                p_module.load_state_dict(predictor_dict[p_name])
                logging.info(f'[bold #006400]load predictor[{p_name}][\]')
            else:
                fmt_print.bold_magenta(f"Warning: predictor['{p_name}'] not found in checkpoint, skipped!!")

    else:
        if extract_predictor is None:
            model.load_state_dict(ckpt['state_dict'], strict=strict_core_load)
            logging.info('[bold #006400]load model[\]')

        elif isinstance(extract_predictor, str):
            core_dict, predictor_dict = _extract_state_dict(ckpt)
            assert extract_predictor in predictor_dict, f"Your specified predictor does not exist, with names {predictor_dict.keys()}"
            model.core.load_state_dict(core_dict, strict=strict_core_load)

            # The predictors is a nn.Module, instead of nn.ModuleDict
            model.predictors.load_state_dict(predictor_dict[extract_predictor])
            logging.info(f'[bold #006400]Load core and specific [{extract_predictor}] predictor[\]')


def init_core(hypers: hotpot.plugins.opti.params_space.ParamSets) -> M.CoreBase:
    return M.Core(
        vec_dim=hypers.VEC_DIM,
        emb_type=hypers.EMB_TYPE,
        x_label_nums=hypers.ATOM_TYPES,
        ring_layers=hypers.RING_LAYERS,
        ring_nheads=hypers.RING_HEADS,
        ring_encoder_kw={'dim_feedforward': hypers.DIM_FEEDFORWARD},
        mol_layers=hypers.MOL_LAYERS,
        mol_nheads=hypers.MOL_HEADS,
        mol_encoder_kw={'dim_feedforward': hypers.DIM_FEEDFORWARD},
        graph_layer=hypers.GRAPH_LAYERS,
        med_props_nums=22,
        sol_props_nums=34,
        with_sol_encoder=True,
        with_med_encoder=True,
    )

def init_model(
        core,
        task_kwargs: Union[dict, list[dict]],
        task: Union[tasks.SingleTask, tasks.MultiTask, tasks.MultiDataTask],
        optim_configure: configs.OptimizerConfigure,
):
    if isinstance(task_kwargs, list):
        assert isinstance(task, tasks.MultiDataTask)
        predictor = {}
        for kw in task_kwargs:
            predictor.update(kw['predictor'])
    else:
        predictor = task_kwargs['predictor']

    return train.LightPretrain(core, predictor, task, optim_configure)

def determine_work_name(task_kwargs: Union[dict, list]):
    if isinstance(task_kwargs, list):
        work_name = f'MDTask({len(task_kwargs)})'
    elif isinstance(task_kwargs, dict):
        if isinstance(task_kwargs['task_name'], str):
            work_name = task_kwargs['task_name']
        elif isinstance(task_kwargs['task_name'], (list, tuple)):
            work_name = f'MultiTask({len(task_kwargs["task_name"])})'
        else:
            raise ValueError(f'task_name must be str or Sequence, not {type(task_kwargs["task_name"])}')
    else:
        raise ValueError(f'task_kwargs must be a dict or list, not {type(task_kwargs)}')
    return work_name

def init_model_dir(work_dir, task_kwargs: Union[dict, list] = None, work_name: Optional[str] = None, prefix: str = ''):
    if work_name is None:
        if task_kwargs is not None:
            work_name = determine_work_name(task_kwargs)
        else:
            raise ValueError(f'work_name and task_kwargs should be given at least one!')

    model_dir = str(osp.join(work_dir, work_name))
    logs_dir = osp.join(model_dir, "logs")

    logger = pl_loggers.TensorBoardLogger(
        save_dir=logs_dir,
        version=f'{prefix}{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    )

    fmt_print.bold_dark_green(f'ModelDir: {model_dir}')
    fmt_print.bold_dark_green(f'LogsDir: {logs_dir}')

    return model_dir, logger
#################################################################

##################################################################
# Callable config helper
def _train_callbacks(
        early_stop_step, early_stopping, minimize_metric,
        optim_configure, show_pbar, use_debugger,
        **kwargs
):
    callbacks = []
    # Configure EarlyStop
    if isinstance(early_stopping, int) and early_stopping > 0:
        early_stop_callback = EarlyStopping(
            monitor=optim_configure.primary_monitor,  # Invoke and align the monitor with optimizer
            mode='min' if minimize_metric else 'max',
            patience=early_stop_step,
        )
        callbacks.append(early_stop_callback)

    # Progress bar
    if show_pbar:
        progress_bar = cbs.Pbar()
        callbacks.append(progress_bar)

    if use_debugger:
        callbacks.append(cbs.Debugger())
    if not callbacks:
        callbacks = None
    return callbacks

def _test_callbacks(**kwargs):
    """ NotImplemented """
    return []

def config_callbacks(stages: list[tp.Stages], **kwargs):
    callbacks = []
    if 'train' in stages:
        callbacks.extend(_train_callbacks(**kwargs))

    if 'test' in stages:
        callbacks.extend(_test_callbacks(**kwargs))

    return callbacks
############################################################

##################################################################
# Tasks defining
def config_task(hypers, batch_preprocessor, constant_lr, dataModule, extractor_attr_getter, feature_extractor,
                inputs_getter, inputs_preprocessor, loss_fn, loss_fn_wrap_tasks, loss_weight_calculator,
                loss_weight_method, lr_scheduler, lr_scheduler_frequency, lr_scheduler_kwargs, mask_need_task,
                onehot_types, optimizer, other_metrics, predictor, primary_metrics, target_getter, task_names, with_med,
                with_sol, with_xyz, work_name, x_masker, xyz_perturb_sigma, xyz_perturb_mode, show_pbar, kwargs,
):
    core = init_core(hypers)
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
        xyz_perturb_mode=xyz_perturb_mode,
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

    # Add global configuration
    task.show_pbar = show_pbar
    task.hypers = hypers  # Save Hyper object callback in the end of train or test stage

    return core, task, task_kwargs

########################################################################################

def prepare_pl_trainer_module(
        work_dir, model_dir, hypers, optim_kw,
        task, task_kwargs,
        core, checkpoint_path,
        stages, cbk_kw, logger, epochs,
        precision, devices, profiler,
        overfit_test
) -> tuple[L.Trainer, L.LightningModule]:
    # Configure optimizer and lr_scheduler
    optim_configure = configs.OptimizerConfigure(
        task=task, lr=hypers.lr, weight_decay=hypers.weight_decay,
        **optim_kw
    )

    # Initialize model
    pl_module = init_model(core, task_kwargs, task, optim_configure)

    # Automatically loading Checkpoint
    if isinstance(checkpoint_path, (int, str, os.PathLike)):
        ckpt = load_ckpt(work_dir, checkpoint_path)
        load_model_state_dict(pl_module, ckpt)

    ################### Callback configuration #########################
    callbacks = config_callbacks(
        stages,
        optim_configure=optim_configure,
        **cbk_kw
    )
    ################## End of the Callbacks configure ###################

    ######################## Run ############################
    # Compile the model
    torch.compile(pl_module)

    trainer = L.Trainer(
        default_root_dir=model_dir,
        logger=logger,
        max_epochs=epochs,
        callbacks=callbacks,
        precision=precision,
        accelerator='cuda',
        devices=devices,
        strategy=strategies.DDPStrategy(find_unused_parameters=True, timeout=datetime.timedelta(seconds=6000)),
        use_distributed_sampler=False,
        profiler = profiler,
        overfit_batches=1.0 if overfit_test else 0.0,
    )

    return trainer, pl_module


def reload_model(log_dir, config_args: datacls.ConfigArgs, lr: float = None):
    config_kw = asdict(config_args)
    config_kw['hypers'] = hypers = ParamSets.from_json(osp.join(log_dir, 'hparams.json'))

    core, task, task_kwargs = config_task(**config_kw)

    # Reload the optimizer
    optim_args = datacls.build_from_kwargs(datacls.OptimConfig, config_kw)
    if isinstance(lr, float):
        optim_args.lr = lr
        optim_args.weight_decay = lr * 0.04
    else:
        optim_args.lr = hypers.lr
        optim_args.weight_decay = hypers.weight_decay

    optim_configure = configs.OptimizerConfigure(task=task, **asdict(optim_args))

    # Initialize model
    pl_module = init_model(core, task_kwargs, task, optim_configure)

    try:
        ckpt_path = glob.glob(osp.join(log_dir, 'checkpoints', '*.ckpt'))[0]
    except IndexError:
        raise RuntimeError(f"No checkpoints found in {log_dir}")

    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    load_model_state_dict(pl_module, ckpt)
    torch.compile(pl_module)

    return pl_module

