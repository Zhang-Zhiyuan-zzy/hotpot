import copy
import glob
import os
import shutil
import logging
import os.path as osp
from typing import *
import warnings
import traceback
from dataclasses import replace

from sklearn.exceptions import UndefinedMetricWarning

import numpy as np
import optuna

from lightning.pytorch import loggers as pl_loggers

from hotpot.utils.configs import setup_logging
from .. import (
    tasks,
    config_task as cfg_task,
)
from . import (
    run_tools as rt,
    datacls,
)
from hotpot.plugins.ComplexFormer.data import DataModule
from hotpot.plugins.opti import ParamSpace, ParamSets

# Contract
INPUT_X_ATTR = ('atomic_number', 'n', 's', 'p', 'd', 'f', 'g', 'x', 'y', 'z')
COORD_X_ATTR = ('x', 'y', 'z')


# Handle the third-party warnings and errors
warnings.filterwarnings('error', category=UndefinedMetricWarning)
def _custom_warning_handler(message, category, filename, lineno, file=None, line=None):
    """ Custom warning handler which raises an exception. """
    # Get the traceback
    tb = traceback.format_stack()

    # Raise an error with details about the warning and its location
    raise RuntimeWarning(f"{message} in {filename} at line {lineno}\n\n\nTraceback:\n{''.join(tb)}")


def _manually_save_checkpoint(trainer):
    ckpt_dir = osp.join(trainer.logger.log_dir, 'checkpoints')
    if not osp.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
        trainer.save_checkpoint(osp.join(ckpt_dir, 'test_autosave.ckpt'))
    elif not os.listdir(ckpt_dir):
        trainer.save_checkpoint(osp.join(ckpt_dir, 'test_autosave.ckpt'))

def _calc_target_metrics(target_metrics, task):
    # Calculate the target metrics
    if target_metrics is None:
        _target_metrics = None
    elif isinstance(target_metrics, str):
        _target_metrics = task.test_primary_metrics[target_metrics]
    elif isinstance(target_metrics, list):
        metrics_items = [task.test_primary_metrics[tm] for tm in target_metrics]
        _target_metrics = sum(metrics_items) / len(metrics_items)
    elif isinstance(target_metrics, Callable):
        _target_metrics = target_metrics(task.test_primary_metrics)
    else:
        raise TypeError('target_metrics must be a str, list of str, or a callable[[dict], float]')
    return _target_metrics

def _add_metric_to_logdir(logger, _target_metrics):
    if _target_metrics is not None and not np.isnan(_target_metrics):
        logs_dir = logger.log_dir + f'_{round(_target_metrics, 3)}'
        shutil.move(logger.log_dir, logs_dir)
    else:
        logs_dir = logger.log_dir
    return logs_dir


############## Perform helpers #####################
def _perform(
        run_args: datacls.RunArgs,
        config_args: datacls.ConfigArgs
) -> tuple[Optional[float], str]:
    core, task, predictors = cfg_task.config_task(config_args)

    # Initialize work directory
    if run_args.save_model:
        model_dir, logger = rt.init_model_dir(run_args.work_dir, run_args.work_name)
    else:
        model_dir, logger = None, None

    trainer, pl_module = rt.prepare_trainer_pl_module(run_args, model_dir, task, core, logger, predictors)
    if 'train' in run_args.stages:
        trainer.fit(pl_module, datamodule=config_args.dataModule)

    _target_metrics = None
    if 'test' in run_args.stages:
        trainer.test(pl_module, datamodule=config_args.dataModule)
        _manually_save_checkpoint(trainer)
        _target_metrics = _calc_target_metrics(run_args.target_metrics, task)

    logs_dir = _add_metric_to_logdir(logger, _target_metrics)

    # Return Optional[test metrics] and log_dir
    return _target_metrics, logs_dir

def _run_optimization(run_args: datacls.RunArgs, cfg_args: datacls.ConfigArgs):
    hparams_space = copy.copy(run_args.hypers)
    def objective(trial: optuna.Trial):
        hyper = ParamSets(hparams_space.copy_to_optuna_trial(trial))
        cfg_args.hypers = hyper

        try:
            metric, logdir = _perform(run_args, cfg_args)
        except (tasks.NaNMetricError, RuntimeError):
            return -10000.
        if np.isnan(metric):
            return -10000.

        trial.set_user_attr("logdir", logdir)

        return metric

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.GPSampler())
    study.optimize(objective, n_trials=run_args.num_trials)
    return study


def _run_external_test(
        log_dir: str,
        run_args: datacls.RunArgs,
        cfg_args: datacls.ConfigArgs
):
    hypers = ParamSets.from_json(osp.join(log_dir, 'hparams.json'))
    ckpt_path = glob.glob(osp.join(log_dir, 'checkpoints', '*.ckpt'))[0]
    model_dir = str(osp.join(run_args.work_dir, run_args.work_name))
    logger = pl_loggers.TensorBoardLogger(log_dir, version=f'external')

    datamodule = DataModule(datacls.merge_dataclass(
        DataModule, run_args,
        dataset_names=run_args.external_datasets,
        ratios=(0., 0., 1.),
        test_only=True
    ))

    run_args = replace(run_args, checkpoint_path=ckpt_path, overfit_test=False)
    cfg_args = replace(cfg_args, dataModule=datamodule, hypers=hypers)

    core, task, predictors = cfg_task.config_task(cfg_args)
    trainer, pl_module = rt.prepare_trainer_pl_module(run_args, model_dir, task, core, logger, predictors)
    trainer.test(pl_module, datamodule=datamodule)


def run(run_args: datacls.RunArgs):
    """ The high-level API for pretraining the ComplexFormer. """
    setup_logging(debug=run_args.debug)

    # Set the warnings to be converted into errors
    if not run_args.warning_allowed:
        warnings.showwarning = _custom_warning_handler
    ###########################################################
    data_module_args = datacls.merge_dataclass(datacls.DataModuleArgs, run_args)
    dataModule = DataModule(data_module_args)

    cfg_args = datacls.merge_dataclass(
        datacls.ConfigArgs, run_args,
        dataModule=dataModule,
    )

    if isinstance(run_args.hypers, ParamSets):
        logging.info(f"Single hyper-parameters running!")
        _, log_dir = _perform(run_args, cfg_args)

    elif isinstance(run_args.hypers, ParamSpace):
        logging.info(f"Multiple hyper-parameters optimization!")
        if run_args.target_metrics is None:
            raise ValueError("target_metrics must be given in the hyper-parameters optimization!")

        res_study = _run_optimization(run_args, cfg_args)
        log_dir = res_study.best_trial.user_attrs.get("logdir")

    else:
        raise TypeError(f'Unknown hyper-type: {type(run_args.hypers)}')

    # External test
    if isinstance(run_args.external_datasets, str):
        _run_external_test(log_dir, run_args, cfg_args)

