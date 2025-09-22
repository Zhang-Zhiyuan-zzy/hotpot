import os
import os.path as osp
import sys
import glob
import logging
from typing import *
import datetime
import warnings
import traceback
from operator import attrgetter

from rich.logging import RichHandler
from rich.console import Console

import torch
import torch.nn as nn
from torch.optim import Optimizer

import lightning as L
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch import strategies

from hotpot.utils import fmt_print
from hotpot.utils.configs import setup_logging
from . import (
    models as M,
    types as tp,
    tools,
    tasks,
    configs,
    module,
    callbacks as cbs,
    run_tools as rt
)
from .data import DataModule

# Contract
INPUT_X_ATTR = ('atomic_number', 'n', 's', 'p', 'd', 'f', 'g', 'x', 'y', 'z')
COORD_X_ATTR = ('x', 'y', 'z')


# Handle the third-party warnings and errors
def _custom_warning_handler(message, category, filename, lineno, file=None, line=None):
    """ Custom warning handler which raises an exception. """
    # Get the traceback
    tb = traceback.format_stack()

    # Raise an error with details about the warning and its location
    raise RuntimeWarning(f"{message} in {filename} at line {lineno}\n\n\nTraceback:\n{''.join(tb)}")

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

    return module.LightPretrain(core, predictor, task, optim_configure)

def init_model_dir(work_dir, task_kwargs: Union[dict, list]):

    if isinstance(task_kwargs, list):
        task_name = f'MDTask({len(task_kwargs)})'
    elif isinstance(task_kwargs, dict):
        if isinstance(task_kwargs['task_name'], str):
            task_name = task_kwargs['task_name']
        elif isinstance(task_kwargs['task_name'], (list, tuple)):
            task_name = f'MultiTask({len(task_kwargs["task_name"])})'
        else:
            raise ValueError(f'task_name must be str or Sequence, not {type(task_kwargs["task_name"])}')
    else:
        raise ValueError(f'task_kwargs must be a dict or list, not {type(task_kwargs)}')

    model_dir = str(osp.join(work_dir, task_name))
    logs_dir = osp.join(model_dir, "logs")

    logger = pl_loggers.TensorBoardLogger(save_dir=logs_dir)

    fmt_print.bold_dark_green(f'ModelDir: {model_dir}')
    fmt_print.bold_dark_green(f'LogsDir: {logs_dir}')

    return model_dir, logger


def _train_callbacks(
        early_stop_step, early_stopping, minimize_metric,
        model, optim_configure, show_pbar, use_debugger,
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
        model.show_pbar = True
    else:
        model.show_pbar = False
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

def run(
        # Global information Arguments
        work_name: str,
        work_dir: str,
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
        eval_each_step: Optional[int] = 1,

        # Training loop control
        epochs: int = 100,
        early_stopping: bool = True,
        early_stop_step: int = 10,
        freeze_core: Optional[bool] = None,
        keep_grad_state: bool = False,

        # Arguments of checkpoints
        checkpoint_path: Union[str, int] = None,
        load_core_only: bool = True,

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
        minimize_metric: bool = False,
        onehot_types: Optional[Union[int, dict[str, int], list[dict[str, int]]]] = None,
        loss_fn_wrap_tasks: Optional[Union[bool, str, set[str], list[bool]]] = None,

        # Postprocessing arguments
        save_model: bool = True,

        # Environmental configuration and device
        devices: Optional[int] = None,
        precision='bf16-mixed',
        float32_matmul_precision='medium',
        profiler="simple",
        show_pbar: bool = True,
        debug: bool = False,
        use_debugger: bool = False,
        warning_allowed: bool = True,
        **kwargs,
):
    """
    The high-level API for pretraining the ComplexFormer.
    Args:
        # Global information Arguments
        work_name(str): The name of the work being trained. While this argument allows any string,
            a standardized nomenclature is recommended, where ...
        work_dir(str): The directory where the trained models and inspected info will be saved.
        core(nn.Module): The general Encoder block, i.e. ComplexFormer.

        # Flow control Arguments
        stages: Which stages will be performed during invoking the interface.
        eval_each_step: How many epochs to evaluate the model.

        # Training loop control
        epochs: The Maximum of epochs to train. Defaults to 100.
        early_stopping: Whether early stopping is enabled. Defaults to True.
        early_stop_step: How many steps when the model's performance is not improved to perform the early stopping.
        freeze_core: Whether to freeze the core model in the first epoch, defaults to None. If None, the core
            module will be frozen in the first epoch if the core module is loaded from checkpoint and the
            predictor is fresh.
        keep_grad_state: Whether to keep the gradient state (requires_grad = True or False) to be solid,
            Defaults to False. If True, the gradient state will not be adjusted automatically.

        # DataModule Arguments
        dir_datasets(str): The directory where the datasets will be saved. The datasets are organized as:
            - dir_datasets
                - dataset_name1
                    - data1.pt
                    - data2.pt
                    - ...
                - dataset_name2
                - ...
        hypers: Hyperparameters for optimizer, dataloader, and others except for model
        dataset_names: Which datasets to use. The names must exactly match the fold name under dir_datasets.
            defaults to None, use all datasets under dir_datasets.
        exclude_datasets: A mutual option with `dataset_names`. When given a None to `dataset_name`, this
            argument is used to exclude specific datasets.
        shuffle_dataset: Whether to shuffle the dataset.
        dataModule_seed: seed for dataloader.
        data_split_ratios: ratio of train, validation and test splits.

        # Optimizer configuration
        optimizer: The type of optimizer to use. If None, the Adam optimizer will be used.
        constant_lr: Whether to use constant learning rate. Defaults to False. If False, a lr_scheduler
            will be used to adjust the learning rate.
        lr_scheduler: The type of learning rate scheduler to use. Defaults to None. If None, a ExponentialLR
            scheduler with `gamma=0.95` will be used. If the lr_schedular is specified, 'lr_schedular_kwargs
            should pass its required arguments`.
        lr_scheduler_frequency:
        lr_scheduler_kwargs: Keyword arguments passed to `lr_scheduler`.

        # Arguments of checkpoints
        checkpoint_path(str|int): the checkpoint file path if given a str. Otherwise, when an int(i)
            is given, the ith model under the work_dir will be loaded.
        load_core_only: Whether to load only the core model, if True, the predictor parameter will be
            ignored. Defaults to True.

        # Inputs specification
        batch_preprocessor:
        inputs_preprocessor:
        xyz_perturb_sigma: Add Gaussian noise to the coordinates based on the sigma value specified by
            this parameter to achieve random perturbation.
        x_masker:
        mask_need_task:

        # Dataset level Arguments
        with_xyz: Whether to load xyz to ComplexFormer. Defaults to True.
        with_sol: Whether to extract solvents information from Dataset.
        with_med: Whether to extract medium information from Dataset.
        with_env: Whether to extract environment information from Dataset.

        # Task level Arguments
        task_names: define task tags for each given datasets. If a single dataset is there, a list of names[str]
            should be passed in; otherwise, if a multiple datasets are there, a list of lists of names[str] should
            be passed. If the `task_names` is not specified, the task_names can be speculated from other given
            required information.
        feature_extractor: Which feature extractor to use. Defaults to None.
        predictor: Which predictor to use. A nn.Module object or `onehot`, `num`, `binary`, or `xyz`
        target_getter(Callable|str): A callable to extract target values from batch.
        loss_fn: loss function
        primary_metrics: The primary metric to control the training processing.
        other_metrics: Other metric to measure the model performance, but not impact the training process.
        minimize_metric:
        loss_weight_calculator: A function to calculate the weights for each category, Applied for onehot labels.
        loss_weight_method: How to calculate the coefficients ki before the sum of loss Σ(ki*loi)
        onehot_types: specify how many types for each onehot predictor. The arguments can pass a single integer
            for the single task training. For (single dataset) multitask works, a dict as {`onehot_task_name`: int}
            should be given. For multi-datasets multitask works, a list of dict as {`onehot_task_name`: int} should
            be given, where the order of the dict should align the orders of corresponding datasets.
        loss_fn_wrap_tasks: Whether to add a metric wrapper to the loss func.

        # Postprocessing arguments
        save_model: Whether to save the model. Defaults to True.

        # Environmental configuration and device
        devices: Number of GPU devices to use. Defaults to None.
        precision: The default precision used by PyTorch. Defaults to bf16-mixed.
        float32_matmul_precision: Sets the internal precision of float32 matrix multiplications. just a link to
            torch.set_float32_matmul_precision(precision)
        profiler: The same one passing into pytorch lightning Trainer, Defaults to 'sample'
        show_pbar: Whether to show the progress bar. Defaults to True.
        debug: turn on the debug mode. Defaults to False.
        use_debugger: Whether to use a debugger. Defaults to False.
        warning_allowed: If false, the warning massage will raise an Error.

    Keyword Args:
        # For hotpot.plugin.ComplexFormer.config._wrap_loss_fn_with_metric
        lofn_wrap_tasks: Optional[Union[str, Sequence[str]]] = None,
        lofn_wrap_exclude_tasks: Optional[Union[str, Sequence[str]]] = None
        lofn_wrap_metric_names: Optional[Union[str, dict[str, str]]] = None
        lofn_wrap_metric_weights: Optional[Union[float, dict[str, float]]] = None

    Returns:
        None
    """
    setup_logging(debug=debug)
    if debug:
        epochs = 10

    # Set the warnings to be converted into errors
    if not warning_allowed:
        warnings.showwarning = _custom_warning_handler

    if stages is None:
        stages = ['train', 'test']
    elif isinstance(stages, str) and stages in get_args(tp.Stages):
        stages = [stages]
    elif isinstance(stages, Container):
        stages = list(stages)
        assert all(stage in get_args(tp.Stages) for stage in stages)
    else:
        raise ValueError(f"Unknown stages type: {type(stages)}, choose from {get_args(tp.Stages)}")

    ##################### Base Args ##########################
    torch.set_float32_matmul_precision(float32_matmul_precision)
    inputs_getter = attrgetter(
        'x', 'edge_index', 'edge_attr', 'rings_node_index',
        'rings_node_nums', 'mol_rings_nums', 'batch', 'ptr')

    # Devices
    if devices is None:
        devices = 1
    assert isinstance(devices, (int, list, tuple)) or devices is None
    ###########################################################
    dataModule = DataModule(
        dir_datasets,
        dataset_names,
        exclude_datasets,
        seed=dataModule_seed,
        ratios=data_split_ratios,
        debug=debug,
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

    # Configure optimizer and lr_scheduler
    optim_configure = configs.OptimizerConfigure(
        task=task,
        hypers=hypers,
        optimizer=optimizer,
        constant_lr=constant_lr,
        lr_scheduler=lr_scheduler,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        lr_scheduler_frequency=lr_scheduler_frequency,
    )

    # Initialize model
    model = init_model(core, task_kwargs, task, optim_configure)

    # Automatically loading Checkpoint
    if isinstance(checkpoint_path, (int, str, os.PathLike)):
        ckpt = rt.load_ckpt(work_dir, checkpoint_path)
        rt.load_model_state_dict(model, ckpt)

    # Initialize work directory
    if save_model:
        model_dir, logger = init_model_dir(work_dir, task_kwargs)
    else:
        model_dir, logger = None, None

    ################### Callback configuration #########################
    callbacks = config_callbacks(
        stages,
        early_stop_step=early_stop_step,
        early_stopping=early_stopping,
        minimize_metric=minimize_metric,
        model=model,
        optim_configure=optim_configure,
        show_pbar=show_pbar,
        use_debugger=use_debugger,
    )
    ################## End of the Callbacks configure ###################

    ######################## Run ############################
    # Compile the model
    torch.compile(model)

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
        profiler = profiler
    )

    if 'train' in stages:
        trainer.fit(model, datamodule=dataModule)

    if 'test' in stages:
        trainer.test(model, datamodule=dataModule)
