import os
import copy
import logging
from typing import Union, Optional, Iterable, Any

import torch
import torch.nn as nn

from torch_geometric.data import Batch

from sklearn.base import BaseEstimator, clone as sk_clone
import lightning as L

from ..tasks import Task
from ..configs import OptimizerConfigure
from hotpot.utils import fmt_print

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

        self.get_features: bool = False

        # Save an "initial snapshot" that can be used to restore the predictors
        # to their initial (post-__init__) state later.
        self._init_predictors_snapshot = self._create_predictors_snapshot(self.predictors)

    @property
    def lr(self) -> float:
        return self.optim_configure.lr

    @lr.setter
    def lr(self, lr: float):
        self.optim_configure.lr = lr

    def predict_step(self, batch) -> Any:
        feature, masked_idx = self.encode(batch)
        target, loss_weights = self.get_target(batch, masked_idx)

        if self.get_features:
            return feature, target
        else:
            pred = self.tasks.predict(self.predictors, feature)
            pred = self.tasks.inverse_pred(pred)
            return pred, target

    def encode(self, batch):
        # Regularize dtype of Tensors in batch
        self.tasks.batch_dtype_preprocessor(batch)
        inputs = self.tasks.inputs_getter(self.tasks.batch_preprocessor(batch))
        xyz = self.tasks.get_xyz(inputs, batch)
        sol_graph, sol_prop, sol_ratios = self.tasks.get_sol_info(batch)
        med_graph, med_prop, med_ratios = self.tasks.get_med_info(batch)
        inputs = self.tasks.inputs_preprocessor(inputs)

        # Mask inputs
        if self.trainer.state.stage in ('train', 'validate'):
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
        return feature, masked_idx

    # Forward process
    def f(self, batch):
        """ Forward propagation """
        feature, masked_idx = self.encode(batch)
        # Make predictor
        pred = self.tasks.predict(self.predictors, feature)
        return pred, masked_idx

    # Get target
    def get_target(
            self,
            batch: Batch,
            masked_idx: Optional[torch.Tensor] = None,
            norm: bool = False,
            **kwargs
    ):

        target = self.tasks.label2oh_conversion(
            self.tasks.peel_unmaksed_obj(
                self.tasks.target_getter(batch, norm=norm),
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
        target, loss_weight = self.get_target(batch, masked_idx, norm=True)

        # Calculation loss value
        loss = self.tasks.loss_fn(pred, target, loss_weight)

        # Log the loss and accuracy
        self.tasks.log_on_train_batch_end(self, loss, pred, target)

        return loss

    def on_train_end(self) -> None:
        self.tasks.export_hparams(self)

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
        pred = self.tasks.inverse_pred(pred)
        target, loss_weight = self.get_target(batch, masked_idx)
        self.tasks.add_val_pred_target(pred, target)

    def on_validation_epoch_end(self) -> None:
        self.tasks.eval_on_val_end(self)

    def configure_optimizers(self):
        return self.optim_configure(self)

    def test_step(self, batch, batch_idx):
        pred, masked_idx = self.f(batch)
        pred = self.tasks.inverse_pred(pred)
        target, loss_weight = self.get_target(batch, masked_idx)
        self.tasks.add_test_pred_target(pred, target)

    def on_test_epoch_end(self) -> None:
        self.tasks.store_save_metrics_table(self)
        self.tasks.log_plots(self)

    def on_test_end(self) -> None:
        self.tasks.export_hparams(self)

    def freeze_core(self):
        for p in self.core.parameters():
            p.requires_grad = False

    def freeze_(self):
        for p in self.parameters():
            p.requires_grad = False
        if isinstance(self.predictors, nn.Module):
            for p in self.parameters():
                p.requires_grad = False
        elif isinstance(self.predictors, dict):
            for name, predictor in self.predictors.items():
                for p in predictor.parameters():
                    p.requires_grad = False

    # ------------------------------------------------------------------
    # Refresh the predictors
    # ------------------------------------------------------------------
    def refresh_predictors(self) -> None:
        """ Reset the training state of self.predictors. """
        if not hasattr(self, "_init_predictors_snapshot"):
            raise RuntimeError(
                "No initial predictors snapshot found. Make sure you create "
                "_init_predictors_snapshot in __init__."
            )

        predictors = self.predictors
        init_snap = self._init_predictors_snapshot

        if isinstance(predictors, nn.Module):
            # Single torch module
            self._refresh_single_module(predictors, init_snap)

        elif isinstance(predictors, BaseEstimator):
            # Single sklearn estimator
            self.predictors = self._refresh_single_estimator(init_snap)

        elif isinstance(predictors, dict):
            # Dict of predictors
            for key, predictor in predictors.items():
                snap_k = init_snap[key]
                if isinstance(predictor, nn.Module):
                    self._refresh_single_module(predictor, snap_k)
                elif isinstance(predictor, BaseEstimator):
                    predictors[key] = self._refresh_single_estimator(snap_k)
                else:
                    raise TypeError(
                        f"Unsupported predictor type for key={key}: {type(predictor)}"
                    )
        else:
            raise TypeError(f"Unsupported predictors type: {type(predictors)}")

    @staticmethod
    def _create_predictors_snapshot(predictors: Union[nn.Module, BaseEstimator, dict[str, Any]]) -> Any:
        """
        Create a snapshot of the current predictors that can be used to
        restore them to the initial state.

        For nn.Module:
            - stores a deep-copied state_dict.
        For sklearn BaseEstimator:
            - stores an unfitted "template" estimator via sklearn.clone.
        For dict:
            - applies the above rules to each value.
        """
        # For torch modules, only save parameters (state_dict)
        if isinstance(predictors, nn.Module):
            return copy.deepcopy(predictors.state_dict())

        # For sklearn estimators, clone creates an unfitted copy
        if isinstance(predictors, BaseEstimator):
            return sk_clone(predictors)

        if isinstance(predictors, dict):
            snapshot: dict[str, Any] = {}
            for key, value in predictors.items():
                if isinstance(value, nn.Module):
                    snapshot[key] = copy.deepcopy(value.state_dict())
                elif isinstance(value, BaseEstimator):
                    snapshot[key] = sk_clone(value)
                else:
                    raise TypeError(
                        f"Unsupported predictor type in dict for key={key}: {type(value)}"
                    )
            return snapshot

        raise TypeError(f"Unsupported predictors type: {type(predictors)}")

    @staticmethod
    def _refresh_single_module(module: nn.Module, init_state_dict: dict) -> None:
        module.load_state_dict(init_state_dict)

    @staticmethod
    def _refresh_single_estimator(init_snapshot: BaseEstimator):
        return sk_clone(init_snapshot)

    def apply_lora_to_predictors(
            self, which: Union[str, Iterable[str]] = None,
            rank=4, alpha=4, dropout=0.3, force: bool = False,
            **kwargs
    ):
        from ..models.predictor import Predictor, apply_lora_to_predictor

        if isinstance(self.predictors, Predictor):
            self.predictors.apply_lora(rank, alpha, dropout, force, **kwargs)
        elif isinstance(self.predictors, dict) or isinstance(self.predictors, nn.ModuleDict):
            if not which:
                which = list(self.predictors.keys())
            elif isinstance(which, str):
                which = [which]
            elif isinstance(which, Iterable):
                which = list(which)
            else:
                raise TypeError(f"Unsupported predictors type: {type(which)}")

            for w in which:
                if isinstance(self.predictors[w], Predictor):
                    self.predictors[w].apply_lora(rank, alpha, dropout, force, **kwargs)
                elif isinstance(self.predictors[w], nn.Module):
                    self.predictors[w] = apply_lora_to_predictor(self.predictors[w], rank, alpha, dropout, **kwargs)
                else:
                    raise AttributeError(f"The predictor type {type(self.predictors[w])} is not applied to the LoRA")
        elif isinstance(self.predictors, nn.Module):
            self.predictors = apply_lora_to_predictor(self.predictors, rank, alpha, dropout, **kwargs)
        else:
            raise AttributeError(f"The predictor type {type(self.predictors)} is not applied to the LoRA")