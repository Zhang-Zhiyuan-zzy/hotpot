"""
@File Name:        optim
@Project:          
@Author:           Zhiyuan Zhang
@Created On:       2025/11/27 20:17
@Project:          Hotpot
"""
from typing import *
from dataclasses import dataclass, field

import torch
from torch.optim import Optimizer, Adam
import torch.optim.lr_scheduler as lrs

import lightning as L

from . import tasks
from hotpot.plugins.opti.params_space import ParamSets


@dataclass
class OptimizerConfigure:
    """Configures optimizer and scheduler for LightningModules."""

    # -- Hyperparameters --
    hypers: ParamSets

    # -- Components --
    # Default to None so we can handle the fallback logic in __post_init__
    optimizer: Optional[Type[Optimizer]] = None
    lr_scheduler: Optional[Type[torch.optim.lr_scheduler.LRScheduler]] = None

    # -- Scheduler Config --
    constant_lr: bool = False
    lr_scheduler_frequency: int = 2
    # use default_factory for mutable defaults
    lr_scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)

    # -- Context / Logic --
    primary_monitor: Optional[str] = None
    task: Optional[Any] = None  # Type hint: tasks.BaseTask

    required_hypers: list[str] = ('lr', 'weight_decay')

    @staticmethod
    def parse_primary_monitor_from_task_type(task):
        if isinstance(task, tasks.SingleTask):
            return task.primary_metric
        elif isinstance(task, (tasks.MultiTask, tasks.MultiDataTask)):
            return 'smtrc'
        else:
            raise NotImplementedError(f"task type is not defined: {type(task)}")

    def __post_init__(self):
        """Validates inputs and infers defaults after initialization."""
        # 1. Resolve Optimizer (Default to Adam)
        if self.optimizer is None or not issubclass(self.optimizer, Optimizer):
            self.optimizer = Adam

        # 2. Resolve Scheduler kwargs
        # Ensure it's a dict even if None was explicitly passed
        if self.lr_scheduler_kwargs is None:
            self.lr_scheduler_kwargs = {}

        # 3. Resolve Primary Monitor
        # If explicit monitor is given, do nothing (it's already set).
        # If not, try to infer from task.
        if self.primary_monitor is None:
            if self.task is not None:
                # assumes isinstance(self.task, tasks.BaseTask) check if needed
                self.primary_monitor = self.parse_primary_monitor_from_task_type(self.task)
            else:
                raise NotImplementedError(f"Unknown monitor type: {self.primary_monitor}")

    def _raise_missing_required_hypers(self, name):
        raise ValueError(f"Missing required hyperparameter: {name} for Optimizer")

    def __call__(self, pl_module: L.LightningModule) -> Dict[str, Any]:
        """Creates the optimizer and scheduler configuration dict."""
        for hyper_name in self.required_hypers:
            if not hasattr(self.hypers, hyper_name):
                self._raise_missing_required_hypers(hyper_name)

        # Initialize Optimizer
        optimizer = self.optimizer(
            pl_module.parameters(),
            lr=self.hypers.lr,
            weight_decay=self.hypers.weight_decay
        )

        if self.constant_lr:
            return {"optimizer": optimizer}

        # Initialize Scheduler
        if self.lr_scheduler:
            scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_kwargs)
        else:
            # Default fallback scheduler
            scheduler = lrs.ReduceLROnPlateau(optimizer, **self.lr_scheduler_kwargs)

        opti_config = {
            'optimizer': optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.primary_monitor,
                "frequency": self.lr_scheduler_frequency,
            },
        }

        # strictly speaking, print inside utility classes is bad practice (use logging),
        # but kept to match original behavior.
        print(opti_config)

        return opti_config