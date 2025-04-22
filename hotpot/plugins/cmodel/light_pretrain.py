import logging
import os
import sys
import glob
import math
import os.path as osp
import typing
from pathlib import Path
from typing import Callable, Union, Sequence, Optional, Any, Type, Literal, Iterable
from abc import ABC, abstractmethod

from typing_extensions import override

from rich import console as rconsole
from operator import attrgetter

from lightning.pytorch.callbacks import EarlyStopping, TQDMProgressBar
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Optimizer, Adam
import torch.optim.lr_scheduler as lrs
from torch.utils.data import Dataset, IterableDataset

from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, Data

import lightning as L
from lightning.pytorch import loggers as pl_loggers

from hotpot.utils import fmt_print
from . import (
    models as M,
    types as tp,
    callbacks as cbs,
    dataset as D,
    loader as ldr
)


class FeatureExtractorTemplate(typing.Protocol):
    @staticmethod
    def __call__(
            seq: torch.Tensor,
            X_mask: torch.Tensor,
            R_mask: torch.Tensor,
            batch: Batch,
            batch_getter: Callable[[Batch], Union[tuple, torch.Tensor]]=None
    ) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        ...

class Hypers:
    """ A handle of hyperparameters. """
    def __init__(self):
        self.lr = 1e-3
        self.weight_decay = 4e-5
        self.batch_size = 256


class _Task(ABC):
    _expect_types = {}

    def __init__(
            self,
            feature_extractor: Union[FeatureExtractorTemplate, dict[str, FeatureExtractorTemplate]],
            target_getter: Union[tp.TargetGetter, dict[str, tp.TargetGetter]],
            loss_fn: Callable[[torch.Tensor, torch.Tensor, Optional[Any]], torch.Tensor],
            primary_metric: Union[str, dict[str, str]],
            metrics: dict[str, Callable[[tp.TensorArray, tp.TensorArray], Union[float, tp.TensorArray]]],
            extractor_attr_getter: Union[tp.ExtractorAttrGetter, dict[str, tp.ExtractorAttrGetter]] = None,
            loss_weight_calculator: Optional[Union[tp.LossWeightCalculator, dict[str, tp.LossWeightCalculator]]] = None,
            to_onehot: Union[bool, Iterable[str]] = False,
            onehot_types: Optional[Union[int, dict[str, int]]] = None,
            x_masker: Callable[[tuple[torch.Tensor, ...], torch.Tensor], tuple[torch.Tensor, torch.Tensor]] = None,
            mask_need_task: Optional[list[str]] = None,
            **kwargs
    ):
        # Mask
        self._x_masker = x_masker
        self._mask_need_task = mask_need_task if mask_need_task else []
        self._masked_idx = None

        # Feature extract
        self._feature_extractor = feature_extractor
        self._extractor_attr_getter = extractor_attr_getter

        # get target
        self._target_getter = target_getter

        # Onehot
        self._to_onehot = to_onehot
        self._onehot_types = onehot_types

        # Loss
        self._loss_fn = loss_fn
        self._loss_weight_calculator = loss_weight_calculator
        self.atl_weights: Optional[dict[str, float]] = None

        # Metrics
        self._primary_metric = primary_metric
        self._metrics = metrics

        self.console = rconsole.Console()

    def _type_check(self):
        for attr_name, attr_type in self._expect_types.items():
            if not isinstance(getattr(self, attr_name), attr_type):
                raise TypeError(
                    f'The type of  {self.__class__.__name__}.{attr_name} should be {attr_type}, '
                    f'got {type(getattr(self, attr_name))}')

    def x_masker(self, inputs: tuple[torch.Tensor, ...]) -> (tuple[torch.Tensor, ...], torch.Tensor):
        if self._x_masker:
            return self._x_masker(inputs)
        return inputs, None

    @abstractmethod
    def feature_extractor(self, *args, **kwargs) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        raise NotImplementedError

    @abstractmethod
    def loss_weight_calculator(self, target):
        raise NotImplementedError

    @abstractmethod
    def loss_fn(self, pred, target, loss_weight):
        raise NotImplementedError

    @abstractmethod
    def label2oh_conversion(self, target: Union[torch.Tensor, dict[str, torch.Tensor]]):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def peel_unmaksed_obj(
            feature_target: Union[torch.Tensor, dict[str, torch.Tensor]],
            mask_idx: Optional[Union[torch.Tensor, dict[str, torch.Tensor]]] = None,
    ):
        raise NotImplementedError

    @abstractmethod
    def log_on_train_batch_end(
            self,
            pl_module: L.LightningModule,
            loss: Union[torch.Tensor, dict[str, torch.Tensor]],
            pred: Union[torch.Tensor, dict[str, torch.Tensor]],
            target: Union[torch.Tensor, dict[str, torch.Tensor]],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_val_pred_target(
            self,
            pred: Union[torch.Tensor, dict[str, torch.Tensor]],
            target: Union[torch.Tensor, dict[str, torch.Tensor]]
    ):
        raise NotImplementedError

    @abstractmethod
    def eval_on_val_end(self,pl_module: L.LightningModule):
        raise NotImplementedError

    @staticmethod
    def _print_and_log_metrics(pl_module: L.LightningModule, metrics_dict: dict[str, float]):
        for metric_name, metric_value in metrics_dict.items():
            pl_module.log(metric_name, metric_value, sync_dist=True)  # Log metrics
        pl_module.val_metrics.update(metrics_dict)

    @property
    def slr_metric_track(self):
        return "metric_to_track"

    def configure_optimizers(self, pl_module: L.LightningModule):
        optimizer = pl_module.t.optimizer(
            pl_module.parameters(),
            lr=pl_module.t.hypers.lr,
            weight_decay=pl_module.t.hypers.weight_decay
        )

        if pl_module.t.constant_lr:
            return optimizer

        if pl_module.t.lr_scheduler:
            scheduler = pl_module.t.lr_scheduler(optimizer, **pl_module.t.lrs_kwargs)
        else:
            scheduler = lrs.ReduceLROnPlateau(optimizer, **pl_module.t.lrs_kwargs)

        return {
            'optimizer': optimizer,
            "lr_scheduler": {
            "scheduler": scheduler,
                "monitor": self.slr_metric_track,
                "frequency": pl_module.t.lr_scheduler_frequency,  # indicates how often the metric is updated
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }

    @staticmethod
    def dict_fmt_print(dict_: dict[str, float], print_func=fmt_print.bold_magenta, prefix=''):
        msg = f'{prefix}[' + ', '.join([f'{k}={v:.3g}' for k, v in dict_.items()]) + ']'
        print(type(print_func))
        print_func(msg)


class _SingleTask(_Task):
    """"""
    _expect_types = {
        '_feature_extractor': Callable,
        '_extractor_attr_getter': Callable,
        '_target_getter': Callable,
        '_loss_fn': Callable,
        '_primary_metric': str,
        '_metrics': dict[str, Callable],
        '_x_masker': Callable,
    }

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.val_pred = []
        self.val_target = []

    def feature_extractor(self, *args, **kwargs) -> torch.Tensor:
        return self._feature_extractor(*args, batch_getter=self._extractor_attr_getter, **kwargs)

    @staticmethod
    def predict(predictor: nn.Module, features: torch.Tensor) -> torch.Tensor:
        return predictor(features)

    def target_getter(self, batch: Batch) -> torch.Tensor:
        return self._target_getter(batch)

    def label2oh_conversion(self, target: Union[torch.Tensor, dict[str, torch.Tensor]]):
        return target.view(-1, 1)

    @staticmethod
    def peel_unmaksed_obj(
            feature_target: torch.Tensor,
            mask_idx: Optional[torch.Tensor] = None,
    ):
        if mask_idx is None:
            return feature_target
        elif isinstance(mask_idx, torch.Tensor):
            return feature_target[mask_idx]
        else:
            raise NotImplementedError('The mask_idx must be None or a torch.Tensor')

    def loss_weight_calculator(self, target) -> Optional[torch.Tensor]:
        if self._loss_weight_calculator:
            return self._loss_weight_calculator(target)
        return None

    def loss_fn(self, pred, target, loss_weight):
        return self._loss_fn(pred, target, loss_weight) \
            if isinstance(loss_weight, torch.Tensor) \
            else self._loss_fn(pred, target)

    @override
    def log_on_train_batch_end(
            self,
            pl_module: L.LightningModule,
            loss: torch.Tensor,
            pred: torch.Tensor,
            target: torch.Tensor,
    ) -> None:
        p_metric = self._metrics[self._primary_metric](pred, target)
        pl_module.log('loss', loss.item(), prog_bar=True)
        pl_module.log(self._primary_metric, p_metric, prog_bar=True)

    @override
    def add_val_pred_target(
            self,
            pred: torch.Tensor,
            target: torch.Tensor
    ):
        self.val_pred.append(pred.cpu().detach().float().numpy())
        self.val_target.append(target.cpu().detach().float().numpy())

    @override
    def eval_on_val_end(self, pl_module: L.LightningModule):
        pred = np.concatenate(self.val_pred)
        target = np.concatenate(self.val_target)

        # Calculating the metrics
        metrics_dict = {
            metric_name: metric_func(pred, target)
            for metric_name, metric_func in self._metrics.items()
        }

        metrics_dict['lr'] = pl_module.optimizers().param_groups[0]['lr']

        # Print and log metrics
        self._print_and_log_metrics(pl_module, metrics_dict)

        # free memory
        self.val_pred.clear()
        self.val_target.clear()


class _MultiTask(_Task):
    """"""
    _expect_types = {
        '_feature_extractor': dict[str, Callable],
        '_extractor_attr_getter': dict[str, Callable],
        'target_getter': dict[str, Callable],
        'loss_fn': dict[str, Callable],
        '_primary_metric': dict[str, str],
        '_metrics': dict[str, dict[str, Callable]],
        '_x_masker': dict[str, Callable],
    }

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loss_dict = None
        self.val_pred = {}
        self.val_target = {}

        try:
            self.atl_weights_calculators = kwargs['atl_weights_calculators']
        except KeyError:
            self.atl_weights_calculators = M.atl_calculator

        if not self.atl_weights_calculators:
            self.atl_weights_calculators = M.atl_calculator

    def feature_extractor(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        if self._extractor_attr_getter is None:
            extractor = {}
        elif isinstance(self._extractor_attr_getter, dict):
            extractor = self._extractor_attr_getter
        else:
            raise NotImplementedError

        return {
            k: ext(*args, batch_getter=extractor.get(k, None), **kwargs)
            for k, ext in self._feature_extractor.items()
        }

    @staticmethod
    def predict(predictor: dict[str, nn.Module], features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {n: predictor[n](f) for n, f in features.items()}

    def target_getter(self, batch: Batch) -> dict[str, torch.Tensor]:
        return {k: tg(batch) for k, tg in self._target_getter.items()}

    @override
    def label2oh_conversion(self, target: dict[str, torch.Tensor]):
        return {k: t.view(-1, 1) for k, t in target.items()}

    def peel_unmaksed_obj(
            self,
            feature_target: dict[str, torch.Tensor],
            mask_idx: Union[torch.Tensor] = None,
    ):
        if mask_idx is None and self._mask_need_task is None:
            return feature_target
        elif isinstance(mask_idx, torch.Tensor):
            for mask_task in self._mask_need_task:
                feature_target[mask_task] = feature_target[mask_task][mask_idx]
            return feature_target
        else:
            raise NotImplementedError('The mask_idx must be None or a torch.Tensor')

    def loss_weight_calculator(self, target) -> Optional[dict[str, torch.Tensor]]:
        if self._loss_weight_calculator is None:
            return None
        elif isinstance(self._loss_weight_calculator, dict):
            return {
                k: calculator(target[k])
                for k, calculator in self._loss_weight_calculator.items()
            }

    def loss_fn(
            self,
            pred: dict[str, torch.Tensor],
            target: dict[str, torch.Tensor],
            loss_weight: dict[str, torch.Tensor]
    ):
        """ Calculate the loss value for multi-task """
        assert len(pred) == len(target)
        assert all((kp in target and kt in pred) for kp, kt in zip(target.keys(), pred.keys()))

        # Calculate loss individually
        self.loss_dict = {}
        for k, p in pred.items():
            if (lw := loss_weight.get(k, None)) is not None:
                self.loss_dict[k] = self._loss_fn[k](p, target[k], lw)
            else:
                self.loss_dict[k] = self._loss_fn[k](p, target[k])

        # Calculate the total loss
        if isinstance(self.atl_weights, dict):
            return sum(lo * self.atl_weights.get(k, 1.) for k, lo in self.loss_dict.items())
        else:
            return sum(lo for lo in self.loss_dict.values())

    @override
    def log_on_train_batch_end(
            self,
            pl_module: L.LightningModule,
            loss: torch.Tensor,
            pred: dict[str, torch.Tensor],
            target: dict[str, torch.Tensor],
    ) -> None:
        # Log total loss
        pl_module.log('loss', loss.item(), prog_bar=True)

        # Calculate primary metrics for each task
        for k, t in target.items():
            pm = self._metrics[k][self._primary_metric[k]](pred[k], t)
            pl_module.log(
                # f"{k}-{self._primary_metric[k]}",
                k,
                pm,
                prog_bar=True
            )

    @override
    def add_val_pred_target(
            self,
            pred: dict[str, torch.Tensor],
            target: dict[str, torch.Tensor]
    ):
        for k, t in target.items():
            self.val_pred.setdefault(k, []).append(pred[k].cpu().detach().float().numpy())
            self.val_target.setdefault(k, []).append(t.cpu().detach().float().numpy())

    @override
    def eval_on_val_end(self,pl_module: L.LightningModule):
        val_target = {k: np.concatenate(t) for k, t in self.val_target.items()}
        val_pred = {k: np.concatenate(p) for k, p in self.val_pred.items()}

        # Calculate metrics
        metrics_dict = {
            k: self._metrics[k][self._primary_metric[k]](val_pred[k], t)
            # self._primary_metric[k]: self._metrics[k][self._primary_metric[k]](val_pred[k], t)
            for k, t in val_target.items()}

        # Update across loss weights
        if pl_module.current_epoch < 1 or not isinstance(self.atl_weights_calculators, Callable):
            self.atl_weights = None
        else:
            self.atl_weights = self.atl_weights_calculators(metrics_dict)
            # logging.debug(self.atl_weights)
            self.dict_fmt_print(self.atl_weights, prefix='\nalt_weights: ')

        # Add sum metrics
        metrics_dict['smtrc'] = np.mean([v for v in metrics_dict.values()])

        # Add learning rate information
        metrics_dict['lr'] = pl_module.optimizers().param_groups[0]['lr']

        # Print and log metrics
        self._print_and_log_metrics(pl_module, metrics_dict)

        self.val_pred.clear()
        self.val_target.clear()

    @property
    def slr_metric_track(self):
        return "smtrc"

class TrainTools:
    def __init__(
            self,
            work_dir: str,
            train_dataset: Union[D.MConcatDataset, Iterable[D.PretrainDataset], D.PretrainDataset],
            test_dataset: Union[D.MConcatDataset, Iterable[D.PretrainDataset], D.PretrainDataset],
            feature_extractor: Union[Callable, dict[str, Callable]],
            inputs_getter: Callable[[Batch], tuple[torch.Tensor, ...]],
            target_getter: Union[tp.TargetGetter, dict[str, tp.TargetGetter]],
            loss_fn: Callable[[torch.Tensor, torch.Tensor, Optional[Any]], torch.Tensor],
            hypers: Hypers,
            primary_metric: Union[str, dict[str, str]],
            metrics: dict[str, Callable[[tp.TensorArray, tp.TensorArray], Union[float, tp.TensorArray]]],
            batch_preprocessor: Callable[[Batch], Batch] = None,
            xyz_index: Union[list, torch.Tensor] = None,
            inputs_preprocessor: Callable[[tuple[torch.Tensor, ...], Union[list, torch.Tensor]], tuple[torch.Tensor, ...]] = None,
            x_masker: Callable[[tuple[torch.Tensor, ...], torch.Tensor], tuple[torch.Tensor, torch.Tensor]] = None,
            mask_need_task: list[str] = None,
            extractor_attr_getter: Union[tp.ExtractorAttrGetter, dict[str, tp.ExtractorAttrGetter]] = None,
            to_onehot: Union[bool, Iterable[str]] = False,
            onehot_types: Optional[Union[int, dict[str, int]]] = None,
            loss_weight_calculator: Callable[[torch.Tensor, int], torch.Tensor] = None,
            optimizer: Optional[Type[Optimizer]] = None,
            constant_lr: bool = False,
            lr_scheduler_frequency: int = 1,
            lr_scheduler: Optional[Type[torch.optim.lr_scheduler.LRScheduler]] = None,
            lr_scheduler_kwargs: Optional[dict] = None,
            task_name: Optional[str] = None,
            labeled_x: bool = False,
            input_x_index: Union[list, torch.Tensor] = None,
            xyz_perturb_sigma: Optional[float] = None,
            debug: bool = False,
            debug_batch_num: int = 8,
            atl_weights_calculators: Optional[Callable[[dict[str, float]], float]] = None,
            load_all_data: bool = False,
            # batch_size: int = 1,
            train_shuffle: bool = True,
            eval_shuffle: bool = False,
            **kwargs
    ):
        if isinstance(target_getter, dict):
            self.multi_target_mode = True
        else:
            self.multi_target_mode = False

        if isinstance(task_name, str):
            self.task_name = self.work_name = task_name
        elif isinstance(task_name, Sequence):
            self.task_name = task_name
            self.work_name = f'MultiTask({len(task_name)})'
        else:
            raise ValueError(f'task_name must be str or Sequence, not {type(task_name)}')

        # Specify the directories
        self.work_dir = work_dir
        self.model_dir = None

        # Datasets
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_loader, self.test_loader = self.prepare_dataset(
            train_dataset=self.train_dataset,
            test_dataset=self.test_dataset,
            load_all_data=load_all_data,
            batch_size=hypers.batch_size,
            train_shuffle=train_shuffle,
            eval_shuffle=eval_shuffle,
        )

        # A position for Lightning Logger
        self.logger = False

        self.hypers = hypers
        self.xyz_index = xyz_index
        self._feature_extractor = feature_extractor
        self._batch_preprocessor = batch_preprocessor
        self._inputs_getter = inputs_getter

        # Input preprocessor configures
        self.labeled_x = labeled_x
        self.input_x_index = input_x_index
        if isinstance(inputs_preprocessor, Callable):
            self._inputs_preprocessor = inputs_preprocessor
        elif self.labeled_x:
            self._inputs_preprocessor = M.get_labeled_x_input_attrs
        else:
            self._inputs_preprocessor = lambda inp: M.get_x_input_attrs(inp, input_x_index=self.input_x_index)

        self._x_masker = x_masker
        self._mask_need_task = mask_need_task

        self._extractor_attr_getter = extractor_attr_getter

        # Configure for target getting
        self._target_getter = target_getter
        self.to_onehot = to_onehot
        self.onehot_types = onehot_types
        self._loss_weight_calculator = loss_weight_calculator

        self._loss_fn = loss_fn
        self.optimizer = optimizer if optimizer is not None and issubclass(optimizer, Optimizer) else Adam
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_frequency = lr_scheduler_frequency
        self.constant_lr = constant_lr
        self.lrs_kwargs = lr_scheduler_kwargs if isinstance(lr_scheduler_kwargs, dict) else {}

        # Metrics
        self.primary_metric = primary_metric
        self.metrics = metrics
        self.val_pred = []
        self.val_target = []

        # Perturb xyz operation
        self._xyz_perturb_sigma = xyz_perturb_sigma

        # Debug configures
        self.debug = debug
        self.debug_batch_num = debug_batch_num

        #
        self._atl_weights_calculators = atl_weights_calculators

        # Check attributes types and length
        self._task = None
        self._align_attrs_types()

    def _align_attrs_types(self):
        """ Check attributes types and length """
        args = (
            self._feature_extractor,
            self._target_getter,
            self._loss_fn,
            self.primary_metric,
            self.metrics,
            self._extractor_attr_getter,
            self._loss_weight_calculator,
            self.to_onehot,
            self.onehot_types,
            self._x_masker,
            self._mask_need_task
        )

        # If this is a single target task
        if not isinstance(self.train_loader, ldr.CDataLoader):
            if isinstance(self._target_getter, Callable):
                self._task = _SingleTask(*args)
            elif isinstance(self._target_getter, dict):
                self._task = _MultiTask(*args, atl_weights_calculators=self._atl_weights_calculators)
            else:
                raise NotImplementedError('The target_getter should be a callable or a dict of callables.')
        else:
            self._task = _MultiTask(*args)

    @property
    def sample_num(self) -> Optional[int]:
        return self.debug_batch_num * self.hypers.batch_size if self.debug else None

    def predict(self, predictors, features) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        return self._task.predict(predictors, features)

    def prepare_dataset(
            self,
            train_dataset: Union[D.MConcatDataset, Iterable[D.PretrainDataset], D.PretrainDataset],
            test_dataset: Union[D.MConcatDataset, Iterable[D.PretrainDataset], D.PretrainDataset],
            load_all_data: bool = False,
            batch_size: Optional[int] = None,
            train_shuffle: bool = True,
            eval_shuffle: bool = False,
            **kwargs,
    ):
        datasets = [train_dataset, test_dataset]
        dataset_names = ['train', 'test']

        list_data = []
        for i, dataset in enumerate(datasets):
            if isinstance(dataset, D.PretrainDataset):
                if load_all_data:
                    list_data.append(dataset.load_data(self.sample_num))
                else:
                    list_data.append(dataset)

            elif isinstance(dataset, Dataset):
                list_data.append(dataset)

            elif isinstance(dataset, IterableDataset):
                if load_all_data:
                    list_data.append([d for d in dataset])
                else:
                    list_data.append(dataset)

            elif isinstance(dataset, Iterable):
                dataset = list(dataset)
                if isinstance(dataset[0], (Dataset, IterableDataset)):
                    assert all(isinstance(ds, (Dataset, IterableDataset, Iterable)) for ds in dataset), (
                        'Multi dataset should make sure all items is a Dataset, IterableDataset or Iterable')

                    # Check iterable-formatted dataset
                    for k, ds in enumerate(dataset):
                        if isinstance(ds, Iterable):
                            assert all(isinstance(d, Dataset) for d in ds), (
                                'When `dataset` given by Iterable, all items should be Data'
                            )

                    list_data.append(D.MConcatDataset(dataset))

                elif isinstance(dataset[0], Data):
                    list_data.append(dataset)

                else:
                    raise TypeError('When `dataset` given by Iterable, all items should be Data or Dataset')

            elif isinstance(dataset, D.MConcatDataset):
                list_data.append(dataset)

            else:
                raise TypeError(
                    f'{dataset_names[i]} dataset should be a Data, Iterable[Data], IterableDataset, Dataset,'
                    f'MConcatDataset or Iterable[Dataset]'
                )

        train_dataset, test_dataset = list_data

        if isinstance(train_dataset, D.MConcatDataset):
            train_loader = ldr.CDataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
            eval_loader = ldr.CDataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
        else:
            train_loader = ldr.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=train_shuffle,
            )
            eval_loader = ldr.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=eval_shuffle,
            )

        return train_loader, eval_loader

    def init_model_dir(self):
        self.model_dir = osp.join(self.work_dir, self.work_name)
        self.logger = pl_loggers.TensorBoardLogger(save_dir=self.logs_dir)

        fmt_print.bold_dark_green(f'ModelDir: {self.model_dir}')
        fmt_print.bold_dark_green(f'LogsDir: {self.logs_dir}')

    def _get_ckpt_files(self):
        # Use glob to find all .ckpt files in the specified directory
        ckpt_files = glob.glob(osp.join(self.work_dir, '**', '*.ckpt'), recursive=True)

        # Sort the files by creation time
        ckpt_files.sort(key=os.path.getctime)

        return ckpt_files

    def load_ckpt(self, which: Optional[Union[int, str]] = -1, prefix: str = 'best'):
        if isinstance(which, int):
            ckpt_files = self._get_ckpt_files()
            ckpt_file = ckpt_files[which]
        elif isinstance(which, str):
            ckpt_file = which
        else:
            raise NotImplementedError

        return torch.load(ckpt_file)

    @staticmethod
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

    def target_getter(self, batch: Batch) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        return self._task.target_getter(batch)

    @property
    def logs_dir(self) -> str:
        return osp.join(self.model_dir, "logs")

    def batch_preprocessor(self, batch: Batch) -> Batch:
        if isinstance(self._batch_preprocessor, Callable):
            return self._batch_preprocessor(batch)
        return batch

    def inputs_getter(self, batch: Batch) -> tuple[torch.Tensor, ...]:
        return self._inputs_getter(batch)

    def get_xyz(self, inputs: tuple[torch.Tensor, ...]) -> Optional[torch.Tensor]:
        if self.xyz_index is None:
            return None
        else:
            return self.perturb_xyz(inputs[0][:, self.xyz_index])

    def inputs_preprocessor(self, inputs: tuple[torch.Tensor, ...], **kwargs) -> tuple[torch.Tensor, ...]:
        if self._inputs_preprocessor:
            return self._inputs_preprocessor(*inputs, **kwargs)
        return inputs

    def x_masker(self, inputs: tuple[torch.Tensor, ...]) -> (tuple[torch.Tensor, ...], torch.Tensor):
        return self._task.x_masker(inputs)

    def feature_extractor(self, *args, **kwargs) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
      return self._task.feature_extractor(*args, **kwargs)

    def loss_weight_calculator(self, target):
        return self._task.loss_weight_calculator(target)

    def label2oh_conversion(self, target: Union[torch.Tensor, dict[str, torch.Tensor]]):
        return self._task.label2oh_conversion(target)

    def loss_fn(self, pred, target, loss_weight):
        return self._task.loss_fn(pred, target, loss_weight)

    def log_on_train_batch_end(
            self,
            pl_module: L.LightningModule,
            loss: Union[torch.Tensor, dict[str, torch.Tensor]],
            pred: Union[torch.Tensor, dict[str, torch.Tensor]],
            target: Union[torch.Tensor, dict[str, torch.Tensor]],
    ) -> None:
        self._task.log_on_train_batch_end(pl_module, loss, pred, target)

    def add_val_pred_target(
            self,
            pred: Union[torch.Tensor, dict[str, torch.Tensor]],
            target: Union[torch.Tensor, dict[str, torch.Tensor]]
    ):
        self._task.add_val_pred_target(pred, target)

    def eval_on_val_end(self, pl_module: L.LightningModule):
        self._task.eval_on_val_end(pl_module)

    def peel_unmaksed_obj(
            self,
            feature: Union[torch.Tensor, dict[str, torch.Tensor]],
            mask_idx: Optional[Union[torch.Tensor, dict[str, torch.Tensor]]] = None,
    ):
        return self._task.peel_unmaksed_obj(feature, mask_idx)

    @staticmethod
    def batch_dtype_preprocessor(batch):
        for name in batch.keys():
            if "index" in name:
                batch[name] = batch[name].long()
            elif name in ['batch', 'ptr']:
                batch[name] = batch[name].int()
            elif 'nums' in name:
                batch[name] = batch[name].int()
            elif isinstance(batch[name], torch.Tensor) and torch.is_floating_point(batch[name]):
                batch[name] = batch[name].bfloat16()

    def perturb_xyz(self, xyz):
        if isinstance(self._xyz_perturb_sigma, float):
            return M.perturb_xyz(xyz, self._xyz_perturb_sigma)
        return xyz

    def configure_optimizers(self, pl_module: L.LightningModule):
        return self._task.configure_optimizers(pl_module)


class LightPretrain(L.LightningModule):
    def __init__(
            self,
            core: Union[nn.Module, str],
            predictors: Union[nn.Module, dict[str, nn.Module]],
            train_tools: TrainTools,
    ):
        super().__init__()
        self.core = core
        if isinstance(predictors, nn.Module):
            self.predictors = predictors
        elif isinstance(predictors, dict):
            self.predictors = nn.ModuleDict(predictors)
        else:
            raise NotImplementedError('predictors must be a nn.Module or dict of nn.Module')
        self.t = train_tools

        self.val_metrics = {}

    # Forward process
    def f(self, batch):
        if (dataset_idx := getattr(batch, 'dataset_idx', None)) is not None:
            assert len(torch.unique(dataset_idx)) == 1
            dataset_idx = dataset_idx[0]

        # Regularize dtype of Tensors in batch
        self.t.batch_dtype_preprocessor(batch)
        inputs = self.t.inputs_getter(self.t.batch_preprocessor(batch))
        xyz = self.t.get_xyz(inputs)
        inputs = self.t.inputs_preprocessor(inputs)

        # Mask inputs
        inputs, masked_idx = self.t.x_masker(inputs)

        # Forward pass through core
        core_output = self.core(*inputs, xyz=xyz)

        # Extract features
        feature = self.t.peel_unmaksed_obj(
            self.t.feature_extractor(*core_output, batch),
            masked_idx
        )

        # Make predictor
        pred = self.t.predict(self.predictors, feature)
        return pred, masked_idx

    # Get target
    def get_target(
            self,
            batch: Batch,
            masked_idx: Optional[torch.Tensor] = None,
            **kwargs
    ):

        target = self.t.label2oh_conversion(
            self.t.peel_unmaksed_obj(
                self.t.target_getter(batch),
                masked_idx))

        # Calc loss weights
        loss_weight = self.t.loss_weight_calculator(target)
        return target, loss_weight

    def training_step(self, batch, batch_idx):
        # Forward
        pred, masked_idx = self.f(batch)

        # Retrieve target and loss_weight for categorical task
        target, loss_weight = self.get_target(batch, masked_idx)

        # Calculation loss value
        loss = self.t.loss_fn(pred, target, loss_weight)

        # Log the loss and accuracy
        self.t.log_on_train_batch_end(self, loss, pred, target)

        return loss

    def validation_step(self, batch, batch_idx):
        pred, masked_idx = self.f(batch)
        target, loss_weight = self.get_target(batch, masked_idx)
        self.t.add_val_pred_target(pred, target)

    def on_validation_epoch_end(self) -> None:
        self.t.eval_on_val_end(self)

    def configure_optimizers(self):
        return self.t.configure_optimizers(self)


class PBar(TQDMProgressBar):
    """ Waiting specification """
    def __init__(
            self,
            refresh_rate: int = 1,
            process_position: int = 0,
            leave: bool = False,
            show_val: bool = True,
            metric_len: int = None
    ):
        super().__init__(refresh_rate, process_position, leave)
        self.show_val = show_val

        if not metric_len:
            self.ncols = 100
        else:
            self.ncols = 100 + metric_len * 10

    @override
    def init_train_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for training."""
        return Tqdm(
            desc=self.train_description,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            # dynamic_ncols=True,
            ncols=self.ncols,
            file=sys.stdout,
            smoothing=0,
            # bar_format=self.BAR_FORMAT,
        )

    @override
    def on_train_batch_end(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        n = batch_idx + 1
        if self._should_update(n, self.train_progress_bar.total):
            _update_n(self.train_progress_bar, n)
            self.train_progress_bar.set_postfix_str(
                self.build_table_str(self.get_metrics(trainer, pl_module))
            )

    @override
    def on_validation_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        if not trainer.sanity_checking:
            self.val_progress_bar = None

    @override
    def on_validation_epoch_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        pass

    @override
    def on_validation_batch_start(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        pass  # ignore the TQDMProgressBar implementation

    @override
    def on_validation_batch_end(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        pass

    @override
    def on_validation_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        pass


    @staticmethod
    def _convert_inf(x: Optional[Union[int, float]]) -> Optional[Union[int, float]]:
        """The tqdm doesn't support inf/nan values.

        We have to convert it to None.

        """
        if x is None or math.isinf(x) or math.isnan(x):
            return None
        return x

    @staticmethod
    def build_table_str(data_dict, cols=5):
        """
        Build a table-style string in chunks of `cols` columns.
        """
        items = list(data_dict.items())
        lines = []

        # Process dict items in groups of `cols`
        for i in range(0, len(items), cols):
            chunk = items[i:i + cols]

            # Create header (keys)
            header = "  |  ".join(x[0] for x in chunk)
            # Create values row
            values = " | ".join(f"{x[1]:.3g}" for x in chunk)

            sep_line = "-" * max(len(header), len(values))
            lines.append(sep_line)
            lines.append(header)
            lines.append(sep_line)
            lines.append(values)
            lines.append("+" * max(len(header), len(values)))

        # Join all parts with newlines
        return '\r\n' + "\n".join(lines)

def _update_n(bar, value: int) -> None:
    if not bar.disable:
        bar.n = value
        bar.refresh()

############################## Pretrain Run ###################################
TargetTypeOption = ('xyz', 'onehot', 'binary', 'num')
MetricType = Literal['r2score', 'rmse', 'mse', 'mae', 'accuracy', 'binary_accuracy', 'metal_accuracy']
metrics_name_convert = {
    'r2score': 'r2',
    'accuracy': 'acc',
    'metal_accuracy': 'macc',
    'binary_accuracy': 'bacc',
}
metrics_options = {
    'r2': M.Metrics.r2_score,
    'rmse': M.Metrics.rmse,
    'mae': M.Metrics.mae,
    'mse': M.Metrics.mse,
    'acc': lambda p, t: M.Metrics.calc_oh_accuracy(p, t),
    'macc': lambda p,t: M.Metrics.metal_oh_accuracy(p, t),
    'bacc': M.Metrics.binary_accuracy,
    'amd': M.LossMethods.average_maximum_displacement
}
# extractor_options = {
#     "atom": M.FeatureExtractors.extract_atom_vec,
#     "pair": M.FeatureExtractors.extract_pair_vec,
#     "ring": M.FeatureExtractors.extract_ring_vec,
#     "mol": M.FeatureExtractors.extract_mol_vec,
#     "cbond": M.FeatureExtractors.extract_cbond_pair
# },
loss_options = {
    'mse': F.mse_loss,
    'cross_entropy': M.LossMethods.calc_atom_type_loss,
    'binary_cross_entropy': F.binary_cross_entropy_with_logits,
    'amd': M.LossMethods.average_maximum_displacement
}
x_masker_options = {
    'atom': M.mask_atom_type,
    'metal': M.mask_atom_type
}

# Contract
INPUT_X_ATTR = ('atomic_number', 'n', 's', 'p', 'd', 'f', 'g', 'x', 'y', 'z')
COORD_X_ATTR = ('x', 'y', 'z')


class TargetGetter:
    def __init__(self, first_data, data_item: str, attrs: Union[str, Iterable[str]] = None):
        self.data_item = data_item
        self.attrs = attrs
        self.data_idx =  _get_index(first_data, self.data_item, attrs)

    def __call__(self, batch: Batch) -> torch.Tensor:
        try:
            return getattr(batch, self.data_item)[:, self.data_idx]
        except Exception as e:
            msg = e.args[0] + f'data.item={self.data_item} attr={self.attrs}'
            raise type(e)(msg)


def _get_index(first_data, data_item: str, attrs: Union[str, Iterable[str]] = None) -> Union[int, list[int]]:
    item_names = first_data[f"{data_item}_names"]
    if attrs is None:
        return list(range(len(item_names)))
    elif isinstance(attrs, str):
        return item_names.index(attrs)
    elif isinstance(attrs, Iterable):
        return [item_names.index(a) for a in attrs]

##########################################################################################################
############################### Argument Regularization ##################################################
# Aligner
def _align_task_seq(
        arg_name: str, arg: Sequence, task_names: Sequence,
        judge: Callable[[Any], bool] = None,
        strict_align: bool = True
):
    assert isinstance(arg, Sequence), f"Expecting {arg_name} is a sequence, but got {type(arg)}"
    assert isinstance(task_names, Sequence), f"Expecting task_name is a sequence, but got {type(task_names)}"
    assert len(task_names) == len(set(task_names)), f"Expecting all task_names are unique, but not"
    if strict_align:
        assert len(arg) == len(task_names), (
            f"The length of {arg_name} should be equal to task_names, but {len(arg)} != {len(task_names)}")
    if isinstance(judge, Callable):
        for v in arg:
            if not judge(v):
                raise ValueError(f"The value {v} not satisfied the judge function {judge.__name__}")

def _align_task_names(
        arg_name: str, arg: dict, task_names: set,
        value_judge: Callable[[Any], bool],
        strict_align: bool = True
):
    """ Align the arg dict to task_names """
    assert isinstance(arg, dict), f"Arg {arg_name} should be a dict, but got {type(arg)}"
    if isinstance(strict_align, bool):
        assert len(arg) == len(task_names), f"Arg {arg_name} should be equal to task_names, but {len(arg)} != {len(task_names)}"
    assert all(k in task_names for k in arg.keys()), f"All keys of {arg_name} dict should be in {task_names}"
    assert all(value_judge(v) for v in arg.values()), f"All values of {arg_name} should satisfy the demand of {value_judge}"

# TargetGetter
def _specify_target_getter(
        task_names: Union[str, list[str]],
        target_getter: tp.TargetGetterInput,
        first_data: Data,
):
    if isinstance(task_names, (list, tuple)):
        task_names = list(task_names)
    elif isinstance(task_names, str):
        if isinstance(target_getter, dict):
            task_names = list(target_getter.keys())
        else:
            task_names = task_names
    else:
        raise TypeError(f"Expecting task_names str or dict, but got {type(task_names)}")

    if isinstance(task_names, str):
        if target_getter is None:
            if task_names == 'xyz':
                XYZ_INDEX = _get_index(first_data, 'x', ('x', 'y', 'z'))
                target_getter = lambda batch: batch.x[:, XYZ_INDEX]
            elif task_names == 'AtomType':
                ATOM_TYPE_INDEX = _get_index(first_data, 'x', 'atomic_number')
                target_getter = lambda batch: batch.x[:, ATOM_TYPE_INDEX]
            elif task_names == 'AtomCharge':
                ATOM_CHRG_INDEX = _get_index(first_data, 'x', 'partial_charge')
                target_getter = lambda batch: batch.x[:, ATOM_CHRG_INDEX]

        # Check the final target_getter
        assert isinstance(target_getter, Callable), (
            f"Expecting target_getter to be a callable when task_names is a str, but got {type(target_getter)}")

    elif isinstance(task_names, list):
        if isinstance(target_getter, dict):
            _align_task_names('target_getter', target_getter, task_names, lambda g: isinstance(g, Callable))
        elif isinstance(target_getter, (list, tuple)):
            _align_task_names('target_getter', target_getter, task_names, lambda g: isinstance(g, Callable))
            target_getter = dict(zip(task_names, target_getter))
        else:
            raise TypeError('Expecting target_getter to be a dict or Sequence of Callable, '
                            f'when task_names is a Sequence, got {type(target_getter)}')
    else:
        raise TypeError(f'Expecting task_names str or list, but got {type(task_names)}')

    return task_names, target_getter

def _specify_onehot_type(onehot_type: Optional[Union[int, dict[str, int]]], predictors: Optional[dict] = None):
    _default_oht = None
    if onehot_type is None:
        onehot_type = {}
    elif isinstance(onehot_type, int):
        onehot_type = {}
        _default_oht = int
    elif isinstance(onehot_type, dict):
        assert all(k in predictors for k in onehot_type), (
            f"name of `onehot_type` {list(onehot_type.keys())} "
            f"cannot match with the name in `predictor` {list(predictors.keys())}")
    else:
        raise TypeError(f"`onehot_type` should be an int or a dict")

    return onehot_type, _default_oht

# Predictor
def _str2callable_predictor(predictor: Union[str, Callable], core, onehot_type=None, **kwargs):
    if isinstance(predictor, Callable):
        return predictor
    elif isinstance(predictor, str):
        if predictor == 'onehot':
            if not onehot_type:
                print(RuntimeWarning(
                    'The default `onehot_type=119` for onehot predictor,'
                    'but explicitly specify the onehot_type is recommended.'))
                onehot_type = 119
            return M.Predictor(core.vec_size, 'onehot', onehot_type=onehot_type, **kwargs)
        elif predictor in ('xyz', 'binary', 'num'):
            return M.Predictor.link_with_core_module(core, predictor, **kwargs)
        else:
            raise ValueError(f'Unknown predictor type: {predictor}')
    else:
        raise TypeError(f'The predictor should be a callable or a str.')

def _specify_predictors(
        task_name: Union[str, Sequence[str]],
        core: M.CoreBase,
        predictors: tp.PredictorInput,
        onehot_type: Optional[Union[int, dict[str, int]]] = None,
        **kwargs
):
    """
    Normalize variable input to the format `Callable|dict[task_name, Callable]`,
    that can be used directly by `LightPretrain` and `pl.Trainer`
    """
    # If get a None, inferring according to the task_name
    if predictors is None and isinstance(task_name, str):
        if task_name == "AtomType":
            return M.Predictor(core.vec_size, 'onehot', onehot_type=119)
        elif task_name == "AtomCharge":
            return M.Predictor(core.vec_size, 'num')
        elif task_name.startswith("xyz"):
            return M.Predictor(core.vec_size, 'xyz')
        elif task_name in ['Cbond', 'RingAromatic']:
            return M.Predictor(core.vec_size, 'binary')
        else:
            ValueError(f'Unknown predictor types!')

    elif isinstance(predictors, (M.Predictor, Callable, str)):
        return _str2callable_predictor(predictors, core, onehot_type, **kwargs)

    elif isinstance(predictors, (list, tuple)):
        # If the given predictor is a Sequence, convert the Sequence one to dict one.
        _align_task_seq(
            'Predictor', predictors, task_name,
            lambda p: isinstance(p, (str, Callable, M.Predictor)))
        predictors = dict(zip(task_name, predictors))

    # Regularize the format of predictor
    if isinstance(predictors, dict):
        onehot_type, _default_oht = _specify_onehot_type(onehot_type, predictors)
        return {
            name: _str2callable_predictor(predictor, core, onehot_type.get(name, _default_oht), **kwargs)
            for name, predictor in predictors.items()}
    else:
        raise TypeError(f"`predictors` should be a str|Callable, or a Sequence|dict of str|Callable")


# FeatureExtractor
def _str2callable_convertor(extractors: dict[str, Union[Callable, str]], item_getter):
    return {n: item_getter[e] if isinstance(e, str) else e for n, e in extractors.items()}


def _specify_feature_extractor(
        task_names: Union[str, Sequence[str]],
        extractors: Union[str, Callable, Sequence[Union[str, Callable]], dict[str, Union[str, Callable]]],
        core: Optional[M.CoreBase]
):
    """"""
    if extractors is None and isinstance(task_names, str):
        if "Atom" in task_names or "xyz" in task_names:
            extractor_name = 'atom'
        elif "Ring" in task_names:
            extractor_name = 'ring'
        elif "Cbond" in task_names:
            extractor_name = 'cbond'
        elif "Pair" in task_names:
            extractor_name = 'pair'
        elif "Mol" in task_names:
            extractor_name = 'mol'
        else:
            raise ValueError("Unknown feature extractor type")
        return core.feature_extractor[extractor_name]

    elif isinstance(extractors, str):
        assert isinstance(task_names, str), (
            f"Expecting task_names is a str when feature_names is a str, but got {type(task_names)}")
        return core.feature_extractor[extractors]

    elif isinstance(extractors, Callable):
        assert isinstance(task_names, str), (
            f"Expecting task_names is a str when feature_names is a Callable, but got {type(task_names)}")
        return extractors

    elif isinstance(extractors, (list, tuple)):
        _align_task_seq('extractors', extractors, task_names, lambda g: isinstance(g, (Callable, str)))
        return _str2callable_convertor(dict(zip(task_names, extractors)), core.feature_extractor)

    elif isinstance(extractors, dict):
        _align_task_names('extractors', extractors, task_names, lambda g: isinstance(g, (Callable, str)))
        return _str2callable_convertor(extractors, core.feature_extractor)

    else:
        raise ValueError(
            f'The extractors should be a string, a callable, a sequence of '
            f'or a dict of str or Callable, not {type(extractors)}')

# Loss_fn
def _specify_loss_fn(
        task_names: Union[str, Sequence[str]],
        loss_fn: tp.LossFnInput,
        predictors: Union[M.Predictor, dict[str, M.Predictor]],
):
    if loss_fn is None and isinstance(predictors, M.Predictor):
        assert isinstance(task_names, str), (
            f"In Multi-task mode, the loss_fn should be explicitly specified, but got a None `loss_fn`")

        if predictors.target_type == 'onehot':
            return M.LossMethods.calc_atom_type_loss
        elif predictors.target_type == 'xyz':
            return M.LossMethods.average_maximum_displacement
        elif predictors.target_type == 'binary':
            return F.binary_cross_entropy_with_logits
        elif predictors.target_type == 'num':
            return F.mse_loss
        else:
            raise ValueError(f"Loss function has not been specified, pass by argument `loss_fn`")

    elif isinstance(loss_fn, str):
        assert isinstance(task_names, str), (
            f"In Multi-task mode, the loss_fn should be a sequence with same number with `task_names` or "
            f"a dict with its key aligning to the `task_names`!!")
        return loss_options[loss_fn]

    elif isinstance(loss_fn, Callable):
        assert isinstance(task_names, str), (
            f"In Multi-task mode, the loss_fn should be a sequence with same number with `task_names` or "
            f"a dict with its key aligning to the `task_names`!!")
        return loss_fn

    elif isinstance(loss_fn, (list, tuple)):
        _align_task_seq('loss_fn', loss_fn, task_names, lambda g: isinstance(g, (Callable, str)))
        return _str2callable_convertor(dict(zip(task_names, loss_fn)), loss_options)

    elif isinstance(loss_fn, dict):
        _align_task_names('loss_fn', loss_fn, task_names, lambda g: isinstance(g, (Callable, str)))
        return _str2callable_convertor(loss_fn, loss_options)

    else:
        raise ValueError(
            f'The loss_fn should be a string, a callable, a sequence of '
            f'or a dict of str and Callable, not {type(loss_fn)}')

# Metrics
def _specify_metrics(
        task_names: Union[str, Sequence[str]],
        primary_metrics: Union[str, Sequence[str], dict[str, str]],
        other_metrics: Union[str, Sequence[str], dict[str, Union[str, Callable]]],
        predictors: Union[M.Predictor, dict[str, M.Predictor]],
) -> (str, dict):
    _metrics = {}

    # Specify primary_metric name
    if primary_metrics is None:
        assert isinstance(task_names, str), (f'In Multi-task mode, the primary_metrics should be '
            'specified explicitly, but got None of `primary_metrics`')

        # Infer the primary_metric according to the task_names
        if task_names == 'AtomType':
            primary_metrics = 'acc'
            _metrics = {'acc': metrics_options['acc']}
        elif task_names == 'MetalType':
            primary_metrics = 'macc'
            _metrics = {'macc': metrics_options['macc']}
        elif task_names == 'AtomCharge':
            primary_metrics = 'r2'
            _metrics = {'r2': metrics_options['r2']}
        elif task_names in ['Cbond', 'RingAromatic']:
            primary_metrics = 'bacc'
            _metrics = {'bacc': metrics_options['bacc']}

        # Infer the primary_metric according to the type of predictor
        elif isinstance(predictors, M.Predictor):
            if predictors.target_type == 'onehot':
                primary_metrics = 'acc'
                _metrics = {'acc': metrics_options['acc']}
            elif predictors.target_type == 'xyz':
                primary_metrics = 'amd'
                _metrics = {'amd': metrics_options['amd']}
            elif predictors.target_type == 'binary':
                primary_metrics = 'bacc'
                _metrics = {'bacc': metrics_options['bacc']}
            elif predictors.target_type == 'num':
                primary_metrics = 'r2'
                _metrics = {'r2': metrics_options['r2']}
            else:
                raise ValueError(f'Fail to infer the `primary_metrics` according to predictor type {predictors.target_type}')

        else:
            raise ValueError(f'Unknown `primary_metrics` types, please specify the it explicitly')

    elif isinstance(primary_metrics, str):
        assert isinstance(task_names, str), (
            f"In Multi-task mode, the primary_metrics should be a sequence with same number with `task_names` or "
            f"a dict with its key aligning to the `task_names`!!")
        primary_metrics = primary_metrics
        _metrics = {primary_metrics: metrics_options[primary_metrics]}

    elif isinstance(primary_metrics, (list, tuple)):
        _align_task_seq('primary_metrics', primary_metrics, task_names, lambda g: isinstance(g, str))
        primary_metrics = dict(zip(task_names, primary_metrics))
        _metrics = {tn: {pmn: metrics_options[pmn]} for tn, pmn in zip(task_names, primary_metrics)}

    elif isinstance(primary_metrics, dict):
        _align_task_names('primary_metrics', primary_metrics, task_names, lambda v: isinstance(v, str))
        _metrics = {tn: {pmn: metrics_options[pmn]} for tn, pmn in primary_metrics.items()}

    else:
        raise TypeError(f'The `primary_metrics` should be a task_names[str], or a sequence|dict of task_names')

    # Specify other metrics
    if isinstance(other_metrics, str):
        assert isinstance(primary_metrics, str), (
            f'a str other metrics could be given only when the _primary metrics is also a str (in single task)')
        _metrics[other_metrics] = metrics_options[other_metrics]

    elif isinstance(other_metrics, (list, tuple)):
        assert isinstance(primary_metrics, str), (
            f'a sequence of `other_metrics` could be given only when the _primary metrics is a str (in single task)')
        _metrics.update({omn: metrics_options[omn] for omn in other_metrics})

    elif isinstance(other_metrics, dict):
        if isinstance(primary_metrics, str):  # In single task
            assert all(isinstance(v, Callable) for v in other_metrics.values()), (
                f'In single task mode, the values in other_metrics dict should be callable, check the input value')
            _metrics.update(other_metrics)
        elif isinstance(primary_metrics, dict):  # In multi task
            _align_task_names(
                'other_metrics', other_metrics, task_names, lambda v: isinstance(v, dict), strict_align=False)
            for tsk_name, tsk_metric in other_metrics.items():
                _metrics[tsk_name].update(tsk_metric)
        else:
            raise RuntimeError(f'the primary_metrics fails to specify')

    # Return
    return primary_metrics, _metrics

# Specify x masker
_default_mask_task = ['AtomType']
def _specify_masker(
        task_names: Union[str, Sequence[str]],
        x_masker: Union[bool, str, Callable[[tuple], tuple[tuple, Optional[torch.Tensor]]]],
        core: M.CoreBase
):
    if x_masker is False:
        return None

    elif x_masker is None:
        if isinstance(task_names, str):
            if task_names == 'AtomType':
                return lambda inp: M.mask_atom_type(inp, core.x_mask_vec)
            elif task_names == 'MetalType':
                return lambda inp: M.mask_metal_type(inp, core.x_mask_vec)
            else:
                return None

        if isinstance(task_names, (list, tuple)):  # default to mask atom types
            return lambda inp: M.mask_atom_type(inp, core.x_mask_vec)

    elif isinstance(x_masker, Callable):
        return x_masker

    elif isinstance(x_masker, str):
        mask_func = x_masker_options[x_masker]
        return lambda inp: mask_func(inp, core.x_mask_vec)

    else:
        raise TypeError(f'The `x_masker` should be a callable or a str')

def _single_calculator(task_names, predictor, _methods, _default_method, predictor_check: bool = False):
    assert isinstance(predictor, M.Predictor), f"the predictors should be a callable, got {type(predictor)}"
    if predictor.target_type == 'onehot':
        return lambda label: M.weight_labels(label, predictor.onehot_type, _methods.get(task_names, _default_method))
    elif predictor.target_type == 'binary':
        return lambda label: M.weight_labels(label, 2, _methods.get(task_names, _default_method))
    elif predictor_check:
        raise ValueError(f'The `loss_weights_calculator` just works for predictor `onehot` or `binary`')
    else:
        return None

def _specify_loss_weights_calculator(
        task_names: Union[str, Sequence[str]],
        loss_weights_calculator: Optional[Union[Callable, str, Sequence[str], dict[str, Callable]]],
        predictors: Union[M.Predictor, dict[str, M.Predictor]],
        loss_weight_method: Union[tp.LossWeightMethods, dict[str, tp.LossWeightMethods]] = 'inverse-count',
):
    # Specify the loss_weight_method
    _methods = {}
    _default_method = 'inverse-count'
    if isinstance(loss_weight_method, str):
        _default_method = loss_weight_method
    elif isinstance(loss_weight_method, dict):
        _align_task_names('loss_weight_methods', loss_weight_method, task_names, lambda v: isinstance(v, str), strict_align=False)
        _methods.update(loss_weight_method)

    if loss_weights_calculator is None or loss_weights_calculator is True:
        if isinstance(task_names, str):
            return _single_calculator(task_names, predictors, _methods, _default_method)
        elif isinstance(task_names, (list, tuple)):
            _align_task_names('predictors', predictors, task_names, lambda v: isinstance(v, M.Predictor))
            return {
                tsk_name: M.label_weights_calculator(
                    getattr(p, 'onehot_type', 2),
                    _methods.get(tsk_name, _default_method)
                )
                for tsk_name, p in predictors.items()
                if p.target_type in ('onehot', 'binary')
            }
        else:
            raise TypeError(f'the `task_names` should be a str or a (list, tuple) of str`')

    elif isinstance(loss_weights_calculator, str):
        if isinstance(task_names, str) and task_names == loss_weights_calculator:
            return _single_calculator(task_names, predictors, _methods, _default_method)
        elif isinstance(task_names, (list, tuple)):
            _align_task_names('predictors', predictors, task_names, lambda v: isinstance(v, M.Predictor))

            assert loss_weights_calculator in task_names, (
                f'the loss_weights_calculator should be one of {task_names}')

            assert predictors[loss_weights_calculator].target_type in ('onehot', 'binary'), (
                f'Only the onehot and binary predictors could be assign a loss weight calculator')

            tsk_name = loss_weights_calculator
            return {tsk_name: _single_calculator(tsk_name, predictors, _methods, _default_method)}

        return None

    elif isinstance(loss_weights_calculator, Callable):
        return loss_weights_calculator

    elif isinstance(loss_weights_calculator, (list, tuple)):
        assert isinstance(task_names, (list, tuple)), (
            f'Single task module should given the loss_weights_calculator by str, bool or None')
        _align_task_seq(
            'loss_weights_calculator',
            loss_weights_calculator,
            task_names, lambda tsk: task_names in task_names,
            strict_align=False
        )

        return {tsk: _single_calculator(tsk, predictors[tsk], _methods, _default_method) for tsk in loss_weights_calculator}

    elif isinstance(loss_weights_calculator, dict):
        _align_task_names(
            'loss_weights_calculator',
            loss_weights_calculator,
            task_names,
            lambda v: isinstance(v, Callable),
            strict_align=False
        )
        return loss_weights_calculator

    else:
        raise TypeError(f'The `loss_weights_calculator` should be a callable or a str, '
                        f'or a sequence of str, dict of Callable, got {type(loss_weights_calculator)}')


############################### Argument Regularization ##################################################
##########################################################################################################

##########################################################################################################
################################## Task Configuration ####################################################
def _config_task_args(
        work_name: str,
        task_names: Optional[Union[str, Sequence[str]]],
        first_data: Data,
        target_getter,
        feature_extractor,
        core,
        predictor: M.Predictor,
        loss_fn,
        primary_metric,
        other_metric,
        x_masker,
        mask_need_task,
        loss_weight_calculator,
        loss_weight_method: Optional[Literal['inverse-count', 'cross-entropy', 'sqrt-invert_count']],
):
    # Prepare
    if task_names is None:
        task_names = work_name
    elif isinstance(task_names, (list, tuple)):
        task_names = list(task_names)
    elif not isinstance(task_names, str):
        raise TypeError(f'The `task_names` should be a str or a Sequence of str')

    ###########################################################
    ##################### Important Args ######################
    # Specify target_getter
    task_names, target_getter = _specify_target_getter(task_names, target_getter, first_data)

    # Specify default predictor
    predictor = _specify_predictors(task_names, core, predictor)

    # Specify default feature extractor
    feature_extractor = _specify_feature_extractor(task_names, feature_extractor, core)

    # Specify loss func
    loss_fn = _specify_loss_fn(task_names, loss_fn, predictor)

    # Specify primary metric
    primary_metric, metrics = _specify_metrics(task_names, primary_metric, other_metric, predictor)

    ####################### Important Args #####################
    ############################################################

    ####################### Optional Args ######################
    # Specify x masker
    x_masker = _specify_masker(task_names, x_masker, core)
    if not isinstance(task_names, (list, tuple)):
        mask_need_task = None
    elif mask_need_task is None:
        mask_need_task = [t for t in task_names if t in _default_mask_task]
    elif isinstance(mask_need_task, (list, tuple)):
        assert all(mt in task_names for mt in mask_need_task), 'All `mask_need_task` should in the task_names list'
    else:
        raise TypeError(f'The `mask_need_task` should be a str or a Sequence of str')

    # Loss weight calculator
    loss_weight_calculator = _specify_loss_weights_calculator(
        task_names, loss_weight_calculator, predictor, loss_weight_method)

    #############################################################
    return{
        'task_name': task_names,
        'target_getter': target_getter,
        'feature_extractor': feature_extractor,
        'predictor': predictor,
        'loss_fn': loss_fn,
        'primary_metric': primary_metric,
        'metrics': metrics,
        'x_masker': x_masker,
        'loss_weight_calculator': loss_weight_calculator,
        'mask_need_task': mask_need_task,
    }


def _config_multi_task(
        task_names: Sequence[str],
        first_data: Data,
        target_getter,
        feature_extractor,
        core,
        predictor,
        loss_fn,
        primary_metric,
        other_metric,
        x_masker,
        loss_weight_calculator,
        loss_weight_method,
        onehot_type,
):
    # Check parameters
    # Arg: target_getter
    task_names = list(task_names)

    ###########################################################
    ##################### Important Args ######################
    # Specify target_getter
    task_names, target_getter = _specify_target_getter(task_names, target_getter, first_data)

    # Specify default predictor
    predictor = _specify_predictors(task_names, core, predictor)

    # Specify default feature extractor
    feature_extractor = _specify_feature_extractor(task_names, feature_extractor, core)

    # Specify loss func
    loss_fn = _specify_loss_fn(task_names, loss_fn, predictor)

    # Specify primary metric
    primary_metric, metrics = _specify_metrics(task_names, primary_metric, loss_fn, predictor)
    ####################### Important Args #####################
    ############################################################

    ####################### Optional Args ######################
    # Specify x masker
    x_masker = _specify_masker(task_names, x_masker, core)

    # Loss weight calculator
    loss_weight_calculator = _specify_loss_weights_calculator(
        task_names, loss_weight_calculator, predictor, loss_weight_method)

    #############################################################
    return {
        'target_getter': target_getter,
        'feature_extractor': feature_extractor,
        'predictor': predictor,
        'loss_fn': loss_fn,
        'primary_metric': primary_metric,
        'metrics': metrics,
        'x_masker': x_masker,
        'loss_weight_calculator': loss_weight_calculator,
    }

################################## Task Configuration ####################################################
##########################################################################################################


def run(
        work_name: str,
        work_dir: str,
        core: M.CoreBase,
        train_dataset: Union[Dataset, Iterable[Dataset], D.MConcatDataset],
        test_dataset: Union[Dataset, Iterable[Dataset], D.MConcatDataset],
        hypers: Union[dict, Hypers],
        target_getter: tp.TargetGetterInput = None,
        task_name: Union[str, Sequence[str]] = None,
        checkpoint_path: Union[str, int] = None,
        load_core_only: bool = True,
        epochs: int = 100,
        with_xyz: bool = True,
        save_model: bool = True,
        optimizer: Optional[Type[Optimizer]] = None,
        constant_lr: bool = False,
        lr_schedular: Optional[Callable] = None,
        lr_scheduler_frequency: int = 1,
        lr_schedular_kwargs: Optional[dict] = None,
        feature_extractor: Optional[tp.FeatureExtractorInput] = None,
        predictor: Optional[tp.PredictorInput] = None,
        loss_fn: Optional[tp.LossFnInput] = None,
        primary_metric: Optional[MetricType] = None,
        other_metric: Optional[Union[MetricType, Iterable[MetricType], dict[str, Callable]]] = None,
        device: Optional[Union[torch.device, str]] = None,
        eval_first: bool = True,
        eval_steps: int = 1,
        minimize_metric: bool = False,
        early_stopping: bool = True,
        early_stop_step: int = 5,
        loss_weight_calculator: Optional[Union[Callable, bool]] = None,
        loss_weight_method: Literal['inverse-count', 'cross-entropy', 'sqrt-invert_count'] = 'inverse-count',
        onehot_labels: Optional[int] = None,
        eval_each_step: Optional[int] = 1,
        freeze_core: Optional[bool] = None,
        keep_grad_state: bool = False,
        x_masker: Optional[Union[str, Callable]] = None,
        mask_need_task: Optional[list[str]] = None,
        load_all_data: bool = False,
        precision='bf16',
        float32_matmul_precision='medium',
        xyz_perturb_sigma: Optional[float] = None,
        profiler="simple",
        debug: bool = False,
        debug_batch_size: int = 8,
        **kwargs,
):
    """
    Key parameters:
        target_getter(Callable|dict[task_name, Callable]):
        feature_extractor(Callable|dict[task_name, Callable]):
        predictor(nn.Module|dict[task_name, nn.Module]):
        loss_fn(Callable|dict[task_name, Callable[[pred, target], float]]):
        primary_metric(metric_name|dict[task_name, metric_name]):
    """
    if debug:
        logging.basicConfig(level=logging.DEBUG)

    ##################### Base Args ##########################
    torch.set_float32_matmul_precision(float32_matmul_precision)
    first_data = train_dataset[0]
    inputs_getter = attrgetter(
        'x', 'edge_index', 'edge_attr', 'rings_node_index',
        'rings_node_nums', 'mol_rings_nums', 'batch', 'ptr')
    ###########################################################

    ##################### Documentation #######################
    # This function is constructed according to 7 key arguments,
    # which are sorted by priority to be specified:
    #   - target_getter
    #   - task_name
    #   - predictor
    #   - feature_extractor
    #   - loss_fn
    #   - primary_metrics
    #   - metrics

    ###########################################################
    ##################### Configure Args ######################
    configs = _config_task_args(
        work_name, task_name, first_data, target_getter, feature_extractor,
        core, predictor, loss_fn, primary_metric, other_metric,
        x_masker, mask_need_task, loss_weight_calculator, loss_weight_method
    )
    ####################### Configure Args #####################
    ############################################################

    ###################### Device configure #####################
    # Specify the device.
    if isinstance(device, list):
        accelerator = 'gpu'
        device = [d.index if isinstance(d, torch.device) else d for d in device]
    elif isinstance(device, torch.device):
        if device.type == 'cuda':
            accelerator = 'gpu'
            device = device.index
        else:
            accelerator = 'cpu'
            device = 'auto'
    elif isinstance(device, int):
        accelerator = 'gpu'
    elif isinstance(device, str):
        try:
            t, i = device.split(':')
            if t == 'cuda':
                accelerator = 'gpu'
                device = int(i)
            else:
                accelerator = 'cpu'
                device = 'auto'
        except ValueError:
            if device == 'cuda':
                accelerator = 'gpu'
            else:
                accelerator = 'cpu'
            device = 'auto'
    else:
        raise ValueError(f"Unknown device: {device}")
    ################################################################

    ###################### Run Preparation #########################
    train_tools = TrainTools(
        # task_name=work_name,
        work_dir=work_dir,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        hypers=hypers,
        optimizer=optimizer,
        constant_lr=constant_lr,
        lr_scheduler=lr_schedular,
        lr_scheduler_frequency=lr_scheduler_frequency,
        lr_schedular_kwargs=lr_schedular_kwargs,
        inputs_getter=inputs_getter,
        device=device,
        epochs=epochs,
        eval_first=eval_first,
        eval_steps=eval_steps,
        early_stopping=early_stopping,
        early_stop_steps=early_stop_step,
        minimize_metric=minimize_metric,
        keep_grad_state=keep_grad_state,
        xyz_index=_get_index(first_data, 'x', COORD_X_ATTR) if with_xyz else None,
        # x_masker=x_masker,
        labeled_x=isinstance(getattr(core, 'x_label_nums', None), int),
        # loss_weight_calculator=loss_weight_calculator,
        xyz_perturb_sigma=xyz_perturb_sigma,
        batch_size=hypers.batch_size,
        debug=debug,
        debug_batch_size=debug_batch_size,
        **configs,
        **kwargs)

    # Initialize model
    model = LightPretrain(core, configs['predictor'], train_tools)

    # Automatically loading Checkpoint
    if isinstance(checkpoint_path, (int, str, Path)):
        ckpt = train_tools.load_ckpt(checkpoint_path)
        train_tools.load_model_state_dict(model, ckpt)

    # Compile the model
    torch.compile(model)

    # Prepare dataset loader
    # train_loader, test_loader = train_tools.prepare_dataset(
    #     train_dataset,
    #     test_dataset,
    #     load_all_data=load_all_data,
    #     batch_size=hypers.batch_size,
    #     **kwargs
    # )

    # Initialize work directory
    if save_model:
        train_tools.init_model_dir()

    # configure EarlyStop
    early_stop_callback = EarlyStopping(
        monitor='smtrc' if isinstance(configs['primary_metric'], dict) else configs['primary_metric'],
        mode='min' if minimize_metric else 'max',
        patience=early_stop_step,
    )

    # progress_bar = PBar(metric_len=len(configs['metrics']))
    progress_bar = cbs.Pbar()

    ######################## Run ############################
    trainer = L.Trainer(
        default_root_dir=train_tools.model_dir,
        logger=train_tools.logger,
        max_epochs=epochs,
        callbacks=[early_stop_callback, progress_bar],
        precision=precision,
        accelerator='auto',
        devices='auto',
        strategy='ddp_find_unused_parameters_true',
        profiler = profiler)

    trainer.fit(model, train_tools.train_loader, train_tools.test_loader)