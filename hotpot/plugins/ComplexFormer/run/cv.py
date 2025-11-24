"""
@File Name:        cv
@Project:          
@Author:           Zhiyuan Zhang
@Created On:       2025/11/17 21:59
@Project:          Hotpot
"""
import copy
import datetime
from typing import *
from dataclasses import asdict

import torch
from torch.utils.data import default_collate

import lightning as pl
from lightning.pytorch import strategies

from peft import LoraConfig

from hotpot.utils.configs import setup_logging
from . import run_tools as rt
from hotpot.plugins.opti import ParamSets

from ..data.data_module import DataModule
from .train import LightPretrain
from . import datacls
from .. import (
    callbacks as cbs,
    models as M
)
from hotpot.plugins.ComplexFormer import check


def cross_validation(
        trainer: pl.Trainer,
        pl_module: LightPretrain,
        datamodule: DataModule,
        n_splits: int = 5,
        lora_cfg: LoraConfig = None
):
    clone_predictor = copy.deepcopy(pl_module.predictors)

    cv_datasets = datamodule.cross_val_split(cv=n_splits)

    list_pred, list_target = [], []
    for i, (train_dataset, test_dataset) in enumerate(cv_datasets):
        pl_module.predictors = copy.deepcopy(clone_predictor)
        pl_module.freeze_()
        lora_kw = asdict(lora_cfg) if isinstance(lora_cfg, LoraConfig) else {}
        pl_module.apply_lora_to_predictors(**lora_kw)

        # check.check_gradient_values(pl_module)

        # min(datamodule.batch_size, len(train_dataset))
        # min(datamodule.batch_size, len(test_dataset))
        train_loader = datamodule.get_loader(train_dataset, len(train_dataset), datamodule.shuffle)
        pred_loader = datamodule.get_loader(test_dataset, 8, datamodule.shuffle)

        trainer.fit(pl_module, train_dataloaders=train_loader, val_dataloaders=pred_loader)
        pred, target = default_collate(trainer.predict(pl_module, pred_loader))

        list_pred.append(pred)
        list_target.append(target)

    return default_collate(list_pred), default_collate(list_target)


def run_cv(
        # Global information Arguments
        work_name: str,
        work_dir: str,
        log_dir: str,

        config_args: datacls.ConfigArgs,

        # DataModule Arguments
        dir_datasets: str,
        dataset_names: Union[str, Sequence[str]] = None,
        shuffle_dataset: bool = True,
        dataModule_seed: int = 315,
        lr: float = 1e-5,
        n_splits: int = 5,

        # Training loop control
        epochs: int = 20,
        batch_size: int = 512,

        # Environmental configuration and device
        devices: Optional[int] = None,
        precision='bf16-mixed',
        float32_matmul_precision='medium',
        profiler="simple",
        debug: bool = False,
        **kwargs,
):
    setup_logging(debug=debug)
    if debug:
        epochs = 10
    ##################### Base Args ##########################
    torch.set_float32_matmul_precision(float32_matmul_precision)

    # Devices
    if devices is None:
        devices = 1

    assert isinstance(devices, (int, list, tuple)) or devices is None
    ###########################################################
    config_args.dataModule = dataModule = DataModule(
        dir_datasets,
        dataset_names,
        seed=dataModule_seed,
        batch_num=debug,
        batch_size=batch_size,
        shuffle=shuffle_dataset,
        devices=devices,
        num_replicas=devices,
    )

    pl_module = rt.reload_model(log_dir, config_args, lr=lr)
    model_dir, logger = rt.init_model_dir(work_dir, work_name=work_name, prefix='CV')
    callbacks = [cbs.Pbar()] if not debug else []

    trainer = pl.Trainer(
        default_root_dir=model_dir,
        logger=logger,
        max_epochs=epochs,
        callbacks=callbacks,
        precision=precision,
        accelerator='cuda',
        devices=devices,
        strategy=strategies.DDPStrategy(find_unused_parameters=True, timeout=datetime.timedelta(seconds=6000)),
        use_distributed_sampler=False,
        # profiler = profiler
    )

    cross_validation(trainer, pl_module, dataModule, n_splits=n_splits)
