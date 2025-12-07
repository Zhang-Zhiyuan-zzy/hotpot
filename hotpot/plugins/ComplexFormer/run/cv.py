"""
@File Name:        cv
@Project:          
@Author:           Zhiyuan Zhang
@Created On:       2025/11/17 21:59
@Project:          Hotpot
"""
import os
import glob
import copy
import logging
import os.path as osp
from dataclasses import asdict

from torch.utils.data import default_collate

import lightning as pl

from peft import LoraConfig

from hotpot.plugins.opti import ParamSets
from hotpot.plugins.ComplexFormer import check
from ..config_task import config_task
from ..data.data_module import DataModule
from . import datacls
from . import run_tools as rt
from .train import LightPretrain


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

        check.check_gradient_values(pl_module)

        train_loader = datamodule.get_loader(train_dataset, len(train_dataset), datamodule.shuffle)
        pred_loader = datamodule.get_loader(test_dataset, 8, datamodule.shuffle)

        trainer.fit(pl_module, train_dataloaders=train_loader, val_dataloaders=pred_loader)
        pred, target = default_collate(trainer.predict(pl_module, pred_loader))

        list_pred.append(pred)
        list_target.append(target)

    return default_collate(list_pred), default_collate(list_target)


def run_cv(
        log_dir: os.PathLike,
        run_args: datacls.RunArgs,
        n_splits: int = 5,
):
    run_args.checkpoint_path = glob.glob(osp.join(log_dir, 'checkpoints', '*.ckpt'))[0]
    run_args.hypers = ParamSets.from_json(osp.join(log_dir, 'hparams.json'))

    data_module = DataModule(datacls.merge_dataclass(datacls.DataModuleArgs, run_args))
    cfg_args = datacls.merge_dataclass(
        datacls.ConfigArgs,
        run_args,
        dataModule=data_module,
    )

    logging.info(f"Running cross validation")
    core, task, predictors = config_task(cfg_args)
    model_dir, logger = rt.init_model_dir(run_args.work_dir, run_args.work_name)
    trainer, pl_module = rt.prepare_trainer_pl_module(run_args, model_dir, task, core, logger, predictors)

    cross_validation(trainer, pl_module, data_module, n_splits=n_splits)
