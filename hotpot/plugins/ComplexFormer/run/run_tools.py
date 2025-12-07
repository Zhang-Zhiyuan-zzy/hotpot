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
import os.path as osp
import datetime
from typing import Optional, Union

import torch
from torch import nn

import lightning as L
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch import strategies

from hotpot.utils import fmt_print

from hotpot.plugins.ComplexFormer import (
    tasks,
    models as M,
    optim_config,
)
from . import train, datacls
from .datacls import RunArgs


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


def init_model(
        core,
        task_kwargs: Union[dict, list[dict]],
        task: Union[tasks.SingleTask, tasks.MultiTask, tasks.MultiDataTask],
        optim_configure: optim_config.OptimizerConfigure,
):
    if isinstance(task_kwargs, list):
        assert isinstance(task, tasks.MultiDataTask)
        predictor = {}
        for kw in task_kwargs:
            predictor.update(kw['predictor'])
    else:
        predictor = task_kwargs['predictor']

    return train.LightPretrain(core, predictor, task, optim_configure)

def init_model_dir(work_dir, work_name, prefix: str = ''):
    model_dir = str(osp.join(work_dir, work_name))
    logs_dir = osp.join(model_dir, "logs")

    logger = pl_loggers.TensorBoardLogger(
        save_dir=logs_dir,
        version=f'{prefix}{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    )

    fmt_print.bold_dark_green(f'ModelDir: {model_dir}')
    fmt_print.bold_dark_green(f'LogsDir: {logs_dir}')

    return model_dir, logger

#####################################################################################
def prepare_trainer_pl_module(
        run_args: RunArgs,
        model_dir, task, core, logger,
        predictors: dict[str, M.Predictor]
):
    optim_configure = datacls.merge_dataclass(optim_config.OptimizerConfigure, run_args, task=task)
    pl_module = train.LightPretrain(core, predictors, task, optim_configure)

    # Automatically loading Checkpoint
    if isinstance(run_args.checkpoint_path, (int, str, os.PathLike)):
        ckpt = load_ckpt(run_args.work_dir, run_args.checkpoint_path)
        load_model_state_dict(pl_module, ckpt)

    ################### Callback configuration #########################
    callback_config = datacls.merge_dataclass(
        datacls.CallbackConfig, run_args,
        optim_configure=optim_configure
    )
    callbacks = callback_config.build()
    ######################## Trainer Init ############################
    # Compile the model
    torch.compile(pl_module)

    trainer = L.Trainer(
        default_root_dir=model_dir,
        logger=logger,
        max_epochs=run_args.epochs,
        callbacks=callbacks,
        precision=run_args.precision,
        accelerator=run_args.accelerator,
        devices=run_args.devices,
        strategy=strategies.DDPStrategy(find_unused_parameters=True, timeout=datetime.timedelta(seconds=6000)),
        use_distributed_sampler=False,
        profiler = run_args.profiler,
        overfit_batches=1.0 if run_args.overfit_test else 0.0,
    )

    return trainer, pl_module

