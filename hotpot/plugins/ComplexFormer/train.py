import os
import logging
from typing import Union, Optional, Iterable

import torch
import torch.nn as nn

from torch_geometric.data import Batch


import lightning as L

from .tasks import Task
from .configs import OptimizerConfigure
from ...utils import fmt_print

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
        self.test_metrics = {}
        self.pred_inspect = None

    # Forward process
    def f(self, batch):
        # Regularize dtype of Tensors in batch
        self.tasks.batch_dtype_preprocessor(batch)
        inputs = self.tasks.inputs_getter(self.tasks.batch_preprocessor(batch))
        xyz = self.tasks.get_xyz(inputs)
        sol_graph, sol_prop, sol_ratios = self.tasks.get_sol_info(batch)
        med_graph, med_prop, med_ratios = self.tasks.get_med_info(batch)
        inputs = self.tasks.inputs_preprocessor(inputs)

        # Mask inputs
        if self.trainer.state.stage in ('fit', 'validate'):
            inputs, masked_idx = self.tasks.x_masker(inputs)
        else:
            masked_idx = None

        # Forward pass through core
        core_output = self.core(
            *inputs,
            xyz=xyz,
            sol_graph=sol_graph,
            sol_props=sol_prop,
            sol_ratios=sol_ratios,
            med_graph=med_graph,
            med_props=med_prop,
        )

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

    def test_step(self, batch, batch_idx):
        pred, masked_idx = self.f(batch)
        target, loss_weight = self.get_target(batch, masked_idx)
        self.tasks.add_test_pred_target(pred, target)

    def on_test_epoch_end(self) -> None:
        self.tasks.store_save_metrics_table(self)
        self.tasks.log_plots(self)
