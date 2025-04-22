import logging
from typing import Union, Optional, Iterable, Any, Callable

import torch
import torch.nn as nn
from lightning.pytorch.core.optimizer import LightningOptimizer
from torch import Tensor
from torch.optim import Optimizer
from torch_geometric.data import Batch


import lightning as L

from .tasks import Task
from .configs import OptimizerConfigure

class LightPretrain(L.LightningModule):
    def __init__(
            self,
            core: Union[nn.Module, str],
            predictors: Union[nn.Module, dict[str, nn.Module]],
            tasks: Union[Task, Iterable[Task]],
            optim_configure: OptimizerConfigure
    ):
        super().__init__()
        self.core = core
        if isinstance(predictors, nn.Module):
            self.predictors = predictors
        elif isinstance(predictors, dict):
            self.predictors = nn.ModuleDict(predictors)
        else:
            raise NotImplementedError('predictors must be a nn.Module or dict of nn.Module')
        self.tasks = tasks
        self.optim_configure = optim_configure

        self.train_metrics = {}
        self.val_metrics = {}
        self.pred_inspect = None

    # Forward process
    def f(self, batch):
        # Regularize dtype of Tensors in batch
        self.tasks.batch_dtype_preprocessor(batch)
        inputs = self.tasks.inputs_getter(self.tasks.batch_preprocessor(batch))
        xyz = self.tasks.get_xyz(inputs)
        inputs = self.tasks.inputs_preprocessor(inputs)

        # Mask inputs
        inputs, masked_idx = self.tasks.x_masker(inputs)

        # Forward pass through core
        core_output = self.core(*inputs, xyz=xyz)

        # Extract features
        feature = self.tasks.peel_unmaksed_obj(
            self.tasks.feature_extractor(*core_output, batch),
            masked_idx
        )

        # Make predictor
        pred = self.tasks.predict(self.predictors, feature)
        return pred, masked_idx

    # Get target
    def get_target(
            self,
            batch: Batch,
            masked_idx: Optional[torch.Tensor] = None,
            **kwargs
    ):

        target = self.tasks.label2oh_conversion(
            self.tasks.peel_unmaksed_obj(
                self.tasks.target_getter(batch),
                masked_idx))

        # Calc loss weights
        loss_weight = self.tasks.loss_weight_calculator(target)
        return target, loss_weight

    def training_step(self, batch, batch_idx):
        # Forward
        pred, masked_idx = self.f(batch)

        # Retrieve target and loss_weight for categorical task
        target, loss_weight = self.get_target(batch, masked_idx)

        # Calculation loss value
        loss = self.tasks.loss_fn(pred, target, loss_weight)

        # Log the loss and accuracy
        self.tasks.log_on_train_batch_end(self, loss, pred, target)

        return loss

    def on_after_backward(self) -> None:
        """ Preparing for DDP strategy """
        if isinstance(self.predictors, dict):
            for predictor in self.predictors.values():
                for name, param in predictor.named_parameters():
                    if param.requires_grad and param.grad is None:
                        logging.debug(f'param {name} has no gradient')
                        param.grad = torch.zeros_like(param)

    def validation_step(self, batch, batch_idx):
        pred, masked_idx = self.f(batch)
        target, loss_weight = self.get_target(batch, masked_idx)
        self.tasks.add_val_pred_target(pred, target)

    def on_validation_epoch_end(self) -> None:
        self.tasks.eval_on_val_end(self)

    def configure_optimizers(self):
        return self.optim_configure(self)
