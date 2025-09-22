import os
import logging
from typing import Union, Optional, Iterable

import torch
import torch.nn as nn

from torch_geometric.data import Batch


import lightning as L

from hotpot.plugins.ComplexFormer.tasks import Task
from hotpot.plugins.ComplexFormer.configs import OptimizerConfigure
from hotpot.utils import fmt_print

from .forward import ForwardBlock

class LightPretrain(L.LightningModule, ForwardBlock):
    def __init__(
            self,
            core: Union[nn.Module, str],
            predictors: Union[nn.Module, dict[str, nn.Module]],
            tasks: Union[Task, Iterable[Task]],
            optim_configure: OptimizerConfigure
    ):
        super().__init__()
        super(ForwardBlock, self).__init__(core, predictors, tasks)
        # self.core = core
        # if isinstance(predictors, nn.Module):
        #     self.predictors = predictors
        # elif isinstance(predictors, dict):
        #     self.predictors = nn.ModuleDict(predictors)
        # else:
        #     raise NotImplementedError('predictors must be a nn.Module or dict of nn.Module')
        # self.tasks = tasks
        self.optim_configure = optim_configure

        self.train_metrics = {}
        self.val_metrics = {}
        self.test_metrics = {}
        self.pred_inspect = None

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
        # logging.debug(f'loss_weight: {list(loss_weight.keys())}')
        return target, loss_weight

    def _show_gpu_info(self):
        """ Retrieve the GPU devices information """
        dev = self.device  # e.g., cuda:0 (local index)
        rank = self.trainer.global_rank
        local_rank = self.trainer.local_rank

        # CUDA local index inside the visible set
        cuda_local = torch.cuda.current_device() if torch.cuda.is_available() else None
        name = torch.cuda.get_device_name(cuda_local) if cuda_local is not None else "CPU"

        # Map local index -> physical GPU index if CUDA_VISIBLE_DEVICES is set
        visible = os.getenv("CUDA_VISIBLE_DEVICES")
        if visible and cuda_local is not None:
            visible_ids = [int(x) for x in visible.split(",")]
            physical = visible_ids[cuda_local]
        else:
            physical = cuda_local

        # Print from every process so you see all GPUs in DDP
        fmt_print.dark_green(f"[Lightning] global_rank={rank} local_rank={local_rank} "
              f"device={dev} cuda_local={cuda_local} physical={physical} name={name}")

    def on_fit_start(self):
        self._show_gpu_info()

    def on_test_start(self) -> None:
        self._show_gpu_info()

    def training_step(self, batch, batch_idx):
        # Forward
        pred, masked_idx = self.f(batch, batch_idx)

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
        pred, masked_idx = self.f(batch, batch_idx)
        target, loss_weight = self.get_target(batch, masked_idx)
        self.tasks.add_val_pred_target(pred, target)

    def on_validation_epoch_end(self) -> None:
        self.tasks.eval_on_val_end(self)

    def configure_optimizers(self):
        return self.optim_configure(self)

    def test_step(self, batch, batch_idx):
        pred, masked_idx = self.f(batch, batch_idx)
        target, loss_weight = self.get_target(batch, masked_idx)
        self.tasks.add_test_pred_target(pred, target)

    def on_test_epoch_end(self) -> None:
        self.tasks.store_save_metrics_table(self)
        self.tasks.log_plots(self)
