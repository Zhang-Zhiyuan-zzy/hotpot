import os
import os.path as osp
import glob
import logging
from typing import *
import datetime
import warnings
import traceback
from operator import attrgetter

import torch
import torch.nn as nn
from torch.optim import Optimizer

import lightning as L
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch import strategies

from hotpot.utils import fmt_print
from . import (
    models as M,
    types as tp,
    tools,
    tasks,
    configs,
    train,
    callbacks as cbs,
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

    return train.LightPretrain(core, predictor, task, optim_configure)

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
        need_test: bool = True,
        eval_each_step: Optional[int] = 1,

        # Training loop control
        epochs: int = 100,
        early_stopping: bool = True,
        early_stop_step: int = 5,
        freeze_core: Optional[bool] = None,
        keep_grad_state: bool = False,

        # Arguments of checkpoints
        checkpoint_path: Union[str, int] = None,
        load_core_only: bool = True,

        # Optimizer configuration
        optimizer: Optional[Type[Optimizer]] = None,
        constant_lr: bool = False,
        lr_scheduler: Optional[Callable] = None,
        lr_scheduler_frequency: int = 1,
        lr_scheduler_kwargs: Optional[dict] = None,
        loss_weight_calculator: Optional[Union[Callable, bool]] = None,
        loss_weight_method: Literal['inverse-count', 'cross-entropy', 'sqrt-invert_count'] = 'inverse-count',

        # Inputs specification
        xyz_perturb_sigma: Optional[float] = None,
        batch_preprocessor: Optional[Union[tp.BatchPreProcessor, list[tp.BatchPreProcessor]]] = None,
        inputs_preprocessor: Optional[Union[Callable, list[Callable]]] = None,
        x_masker: Optional[Union[str, Callable]] = None,
        mask_need_task: Optional[list[str]] = None,

        # Model architecture control
        mol_info_dim: Optional[int] = None,

        # Task-specific Arguments
        task_names: Optional[Union[list[str], list[list[str]]]] = None,
        target_getter: tp.TargetGetterInput = None,
        feature_extractor: Optional[tp.FeatureExtractorInput] = None,
        predictor: Optional[tp.PredictorInput] = None,
        loss_fn: Optional[tp.LossFnInput] = None,
        primary_metric: Optional[tp.MetricType] = None,
        other_metric: Optional[Union[tp.MetricType, Iterable[tp.MetricType], dict[str, Callable]]] = None,
        extractor_attr_getter: Optional[Union[Callable, dict[str, Callable], list[dict, Callable]]] = None,
        minimize_metric: bool = False,
        onehot_types: Optional[Union[int, dict[str, int], list[dict[str, int]]]] = None,
        with_xyz: Union[bool, Iterable[bool]] = None,
        with_sol: Optional[int] = None,
        with_med: Optional[int] = None,

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
        need_test: Whether to perform test process

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

        # Task-specific Arguments
        task_names: define task tags for each given datasets. If a single dataset is there, a list of names[str]
            should be passed in; otherwise, if a multiple datasets are there, a list of lists of names[str] should
            be passed. If the `task_names` is not specified, the task_names can be speculated from other given
            required information.
        feature_extractor: Which feature extractor to use. Defaults to None.
        predictor: Which predictor to use. A nn.Module object or `onehot`, `num`, `binary`, or `xyz`
        target_getter(Callable|str): A callable to extract target values from batch.
        loss_fn: loss function
        primary_metric: The primary metric to control the training processing.
        other_metric: Other metric to measure the model performance, but not impact the training process.
        minimize_metric:
        loss_weight_calculator: A function to calculate the weights for each category, Applied for onehot labels.
        loss_weight_method: How to calculate the coefficients ki before the sum of loss Σ(ki*loi)
        eval_each_step: How many epochs to evaluate the model.
        onehot_types: specify how many types for each onehot predictor. The arguments can pass a single integer
            for the single task training. For (single dataset) multitask works, a dict as {`onehot_task_name`: int}
            should be given. For multi-datasets multitask works, a list of dict as {`onehot_task_name`: int} should
            be given, where the order of the dict should align the orders of corresponding datasets.
        with_xyz: Whether to load xyz to ComplexFormer. Defaults to True.
        with_sol: Whether to allow ComplexFormer to encode solvent information.
        with_med: Whether to allow ComplexFormer to encode medium information.

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
        sol_graph_inputs(Iterable[str])
        med_graph_inputs(Iterable[str])

    Returns:
        None
    """
    if debug:
        logging.basicConfig(level=logging.DEBUG)
        epochs = 6

    # Set the warnings to be converted into errors
    if not warning_allowed:
        warnings.showwarning = _custom_warning_handler

    ##################### Base Args ##########################
    torch.set_float32_matmul_precision(float32_matmul_precision)
    inputs_getter = attrgetter(
        'x', 'edge_index', 'edge_attr', 'rings_node_index',
        'rings_node_nums', 'mol_rings_nums', 'batch', 'ptr')

    # Devices
    if not isinstance(devices, int):
        devices = torch.cuda.device_count()
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
    )

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
        primary_metric=primary_metric,
        other_metric=other_metric,
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
        ckpt = load_ckpt(work_dir, checkpoint_path)
        load_model_state_dict(model, ckpt)

    # Compile the model
    torch.compile(model)

    # Initialize work directory
    if save_model:
        model_dir, logger = init_model_dir(work_dir, task_kwargs)
    else:
        model_dir, logger = None, None

    # Callback item configuration
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
    ################## End of the Callbacks configure ###################

    ######################## Run ############################
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

    trainer.fit(model, datamodule=dataModule)

    if need_test:
        trainer.test(model, datamodule=dataModule)
