import os
import sys
import glob
import math
import os.path as osp
import datetime
import typing
from pathlib import Path
from typing import Callable, Union, Sequence, Optional, Any, Type, Literal, Iterable

from lightning.pytorch.accelerators import Accelerator
from typing_extensions import override

from tqdm import tqdm
from operator import attrgetter

from lightning.pytorch.callbacks import EarlyStopping, TQDMProgressBar
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm
from pytorch_lightning.strategies import DDPStrategy

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Optimizer, Adam
import torch.optim.lr_scheduler as lrs

from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

import lightning as L
from lightning.pytorch import loggers as pl_loggers

from hotpot.utils import print_fmt as p_fmt
from . import (
    models as M,
    types as tp
)


class FeatureExtractorTemplate(typing.Protocol):
    @staticmethod
    def __call__(
            seq: torch.Tensor,
            X_mask: torch.Tensor,
            R_mask: torch.Tensor,
            batch: Batch,
            batch_getter: Callable[[Batch], Union[tuple, torch.Tensor]]=None
    ):
        ...

class Hypers:
    """ A handle of hyperparameters. """
    def __init__(self):
        self.lr = 1e-3
        self.weight_decay = 4e-5
        self.batch_size = 256


class TrainTools:
    def __init__(
            self,
            work_dir: str,
            feature_extractor: Union[Callable, dict[str, Callable]],
            inputs_getter: Callable[[Batch], tuple[torch.Tensor, ...]],
            target_getter: Union[tp.TargetGetter, dict[str, tp.TargetGetter]],
            loss_fn: Callable[[torch.Tensor, torch.Tensor, Optional[Any]], torch.Tensor],
            hypers: Hypers,
            primary_metric: str,
            metrics: dict[str, Callable[[tp.TensorArray, tp.TensorArray], Union[float, tp.TensorArray]]],
            batch_preprocessor: Callable[[Batch], Batch] = None,
            xyz_index: Union[list, torch.Tensor] = None,
            inputs_preprocessor: Callable[[tuple[torch.Tensor, ...], Union[list, torch.Tensor]], tuple[torch.Tensor, ...]] = None,
            x_masker: Callable[[tuple[torch.Tensor, ...], torch.Tensor], tuple[torch.Tensor, torch.Tensor]] = None,
            extractor_attr_getter: Union[tp.ExtractorAttrGetter, dict[str, tp.ExtractorAttrGetter]] = None,
            to_onehot: Union[bool, Iterable[str]] = False,
            onehot_types: Optional[Union[int, dict[str, int]]] = None,
            loss_weight_calculator: Callable[[torch.Tensor, int], torch.Tensor] = None,
            optimizer: Optional[Type[Optimizer]] = None,
            constant_lr: bool = False,
            lr_scheduler_frequency: int = 1,
            lr_scheduler: Optional[Type[torch.optim.lr_scheduler.LRScheduler]] = None,
            lr_scheduler_kwargs: Optional[dict] = None,
            work_name: Optional[str] = None,
            labeled_x: bool = False,
            input_x_index: Union[list, torch.Tensor] = None,
            xyz_perturb_sigma: Optional[float] = None,
            debug: bool = False,
            debug_batch_num: int = 8,
            **kwargs
    ):
        if isinstance(target_getter, dict):
            self.multi_target_mode = True
        else:
            self.multi_target_mode = False

        self.work_name = work_name

        # Specify the directories
        self.work_dir = work_dir
        self.model_dir = None

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
        self.lrs_kwargs = lr_scheduler_kwargs

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

        # Check attributes types and length
        self._align_attrs_types()

    def _align_attrs_types(self):
        """ Check attributes types and length """

    @property
    def sample_num(self) -> Optional[int]:
        return self.debug_batch_num * self.hypers.batch_size if self.debug else None

    def prepare_dataset(
            self,
            train_dataset,
            test_dataset,
            load_all_data: bool = False,
            batch_size: Optional[int] = None,
            **kwargs,
    ):
        train_loader = DataLoader(
            train_dataset.load_all(self.sample_num) if load_all_data else train_dataset,
            batch_size=batch_size,
            shuffle=kwargs.get('train_shuffle', True),
        )
        eval_loader = DataLoader(
            test_dataset.load_all(self.sample_num) if load_all_data else test_dataset,
            batch_size=batch_size,
            shuffle=kwargs.get('eval_shuffle', False),
        )

        return train_loader, eval_loader

    def init_model_dir(self):
        self.model_dir = osp.join(self.work_dir, self.work_name)
        self.logger = pl_loggers.TensorBoardLogger(save_dir=self.logs_dir)

        print(f'\033[38;5;208mModelDir: {self.model_dir}\033[0m')
        print(f'\033[38;5;208mLogsDir: {self.logs_dir}\033[0m')

    def _get_ckpt_files(self):
        # Use glob to find all .ckpt files in the specified directory
        ckpt_files = glob.glob(osp.join(self.work_dir, '**.ckpt'))

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

    def target_getter(self, batch: Batch) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        if isinstance(self._target_getter, Callable):
            return self._target_getter(batch)
        elif isinstance(self._target_getter, dict):
            return {k: tg(batch) for k, tg in self._target_getter.items()}
        else:
            raise AttributeError('The TrainTools.target_getter must be callable or dict of callable!')

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

    def x_masker(self, inputs: tuple[torch.Tensor, ...], x_mask_vec) -> (tuple[torch.Tensor, ...], torch.Tensor):
        if self._x_masker:
            return self._x_masker(inputs, x_mask_vec)
        return inputs, None

    def feature_extractor(self, *args, **kwargs) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        if isinstance(self._feature_extractor, Callable):
            return self._feature_extractor(*args, self._extractor_attr_getter, **kwargs)
        elif isinstance(self._feature_extractor, dict):
            return {k: ext(*args, self._extractor_attr_getter[k], **kwargs) for k, ext in self._feature_extractor.items()}

    def loss_weight_calculator(self, target):
        if self._loss_weight_calculator:
            return self._loss_weight_calculator(target, self.onehot_types)
        return None

    def loss_fn(self, pred, target, loss_weight):
        return self._loss_fn(pred, target, loss_weight) \
            if isinstance(loss_weight, torch.Tensor) \
            else self._loss_fn(pred, target)

    def label2oh_conversion(self, target: Union[torch.Tensor, dict[str, torch.Tensor]]):
        if isinstance(target, torch.Tensor):
            if self.to_onehot is True:
                return F.one_hot(target.long(), num_classes=self.onehot_types)
            else:
                raise target.view(-1, 1)
        elif isinstance(target, dict):
            if not isinstance(self.to_onehot, Iterable):
                return {n: t.view(-1, 1) for n, t in target.items()}
            else:
                _to_onehot = list(self.to_onehot)
                if not isinstance(self.onehot_types, dict):
                    raise AttributeError('When the `to_onehot` is an Iterable of str, the `onehot_types` should be a dict.')
                if len(_to_onehot) != len(self.onehot_types):
                    raise ValueError('Then length of `to_onehot` and `onehot_types` should be the same.')

                _target = {}
                for n, t in target.items():
                    if n in _to_onehot:
                        _target[n] = F.one_hot(t.long(), num_classes=self.onehot_types[n])
                    else:
                        _target[n] = t.view(-1, 1)
                return _target

    @staticmethod
    def peel_unmaksed_obj(
            feature: Union[torch.Tensor, dict[str, torch.Tensor]],
            mask_idx: Optional[Union[torch.Tensor, dict[str, torch.Tensor]]] = None,
    ):
        if mask_idx is None:
            return feature
        if isinstance(feature, torch.Tensor) and isinstance(mask_idx, torch.Tensor):
            return feature[mask_idx]
        elif isinstance(feature, dict) and isinstance(mask_idx, dict):
            return {k: f[mask_idx[k]] for k, f in feature.items() if isinstance(mask_idx[k], torch.Tensor)}

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

    # Forward process
    def f(self, batch):
        # Regularize dtype of Tensors in batch
        self.t.batch_dtype_preprocessor(batch)
        inputs = self.t.inputs_getter(self.t.batch_preprocessor(batch))
        xyz = self.t.get_xyz(inputs)
        inputs = self.t.inputs_preprocessor(inputs)

        # Mask inputs
        inputs, masked_idx = self.t.x_masker(inputs, self.core.x_mask_vec)

        # Forward pass through core
        core_output = self.core(*inputs, xyz=xyz)

        # Extract features
        feature = self.t.feature_extractor(*core_output, batch)
        feature = self.t.peel_unmaksed_obj(feature, masked_idx)

        # Make predictor
        if isinstance(self.predictors, nn.Module):
            pred = self.predictors(feature)
        elif isinstance(self.predictors, nn.ModuleDict):
            pred = {k: p(feature[k]) for k, p in self.predictors.items()}
        else:
            raise NotImplementedError('predictors must be a nn.Module or nn.ModuleDict')

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
                masked_idx
            ))

        # if self.t.to_onehot:
        #     target = F.one_hot(target.long(), num_classes=self.t.onehot_types)  # Convert to OneHot label
        # else:
        #     target = target.view((-1, 1))

        loss_weight = self.t.loss_weight_calculator(target)
        return target, loss_weight

    def training_step(self, batch, batch_idx):
        pred, masked_idx = self.f(batch)
        target, loss_weight = self.get_target(batch, masked_idx)
        loss = self.t.loss_fn(pred, target, loss_weight)
        p_metric = self.t.metrics[self.t.primary_metric](pred, target)
        self.log('loss', loss.item(), prog_bar=True)
        self.log(self.t.primary_metric, p_metric, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred, masked_idx = self.f(batch)
        target, loss_weight = self.get_target(batch, masked_idx)
        self.t.val_pred.append(pred.cpu().detach().float().numpy())
        self.t.val_target.append(target.cpu().detach().float().numpy())

    def on_validation_epoch_end(self) -> None:
        pred = np.concatenate(self.t.val_pred)
        target = np.concatenate(self.t.val_target)

        # Calculating the metrics
        metrics_dict = {
            metric_name: metric_func(pred, target)
            for metric_name, metric_func in self.t.metrics.items()
        }

        # Print and log metrics
        metric_msg = []
        for metric_name, metric_value in metrics_dict.items():
            self.log(metric_name, metric_value, sync_dist=True)  # Log metrics
            metric_msg.append(f'{metric_name}={metric_value:.3f}')  # Add metrics
        p_fmt.dark_green('\tEval Metrics: [' + ', '.join(metric_msg) + ']')

        # free memory
        self.t.val_pred.clear()
        self.t.val_target.clear()

    def configure_optimizers(self):
        optimizer = self.t.optimizer(self.parameters(), lr=self.t.hypers.lr, weight_decay=self.t.hypers.weight_decay)
        if self.t.constant_lr:
            return optimizer

        if self.t.lr_scheduler:
            scheduler = self.t.lr_scheduler(optimizer, **self.t.lrs_kwargs)
        else:
            scheduler = lrs.ReduceLROnPlateau(optimizer, **self.t.lrs_kwargs)

        return {
            'optimizer': optimizer,
            "lr_scheduler": {
            "scheduler": scheduler,
                "monitor": "metric_to_track",
                "frequency": self.t.lr_scheduler_frequency,  # indicates how often the metric is updated
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }


class CustomPBar(TQDMProgressBar):
    """ Waiting specification """
    def __init__(
            self,
            refresh_rate: int = 1,
            process_position: int = 0,
            leave: bool = False,
            show_val: bool = True,
    ):
        super().__init__(refresh_rate, process_position, leave)
        self.show_val = show_val

    @override
    def init_train_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for training."""
        return Tqdm(
            desc=self.train_description,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            # dynamic_ncols=True,
            ncols=100,
            file=sys.stdout,
            smoothing=0,
            # bar_format=self.BAR_FORMAT,
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

def _update_n(bar, value: int) -> None:
    if not bar.disable:
        bar.n = value
        bar.refresh()

############################## Pretrain Run ###################################
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
    'acc': lambda p, t: M.Metrics.calc_oh_accuracy(p, t, is_onehot=True),
    'macc': lambda p,t: M.Metrics.metal_oh_accuracy(p, t, is_onehot=True),
    'bacc': M.Metrics.binary_accuracy,
    'AMD': M.LossMethods.mean_maximum_displacement
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
    'binary_cross_entropy': F.binary_cross_entropy,
    'mean_maximum_displace': M.LossMethods.mean_maximum_displacement
}
x_masker_options = {
    'atom': M.mask_atom_type,
    'metal': M.mask_atom_type
}

# Contract
INPUT_X_ATTR = ('atomic_number', 'n', 's', 'p', 'd', 'f', 'g', 'x', 'y', 'z')
COORD_X_ATTR = ('x', 'y', 'z')


def _get_index(first_data, data_item: str, attrs: Union[str, Iterable[str]] = None) -> Union[int, list[int]]:
    item_names = first_data[f"{data_item}_names"]
    if attrs is None:
        return list(range(len(item_names)))
    elif isinstance(attrs, str):
        return item_names.index(attrs)
    elif isinstance(attrs, Iterable):
        return [item_names.index(a) for a in attrs]

def run(
        work_name: str,
        work_dir: str,
        core: M.CoreBase,
        train_dataset,
        test_dataset,
        hypers: Union[dict, Hypers],
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
        target_type: Optional[M.TargetTypeName] = None,
        feature_extractor: Optional[Union[Callable, str]] = None,
        predictor: Optional[Union[nn.Module, str]] = None,
        target_getter: Optional[Union[str, Callable]] = None,
        loss_fn: Optional[Union[Callable, str]] = None,
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
        load_all_data: bool = False,
        precision='bf16',
        float32_matmul_precision='medium',
        xyz_perturb_sigma: Optional[float] = None,
        profiler="simple",
        debug: bool = False,
        debug_batch_size: int = 8,
        **kwargs,
):
    torch.set_float32_matmul_precision(float32_matmul_precision)
    first_data = train_dataset[0]
    inputs_getter = attrgetter(
        'x', 'edge_index', 'edge_attr', 'rings_node_index',
        'rings_node_nums', 'mol_rings_nums', 'batch', 'ptr')

    if target_type is None:
        if work_name == "AtomType":
            target_type = 'onehot'
        elif work_name.startswith("xyz"):
            target_type = 'xyz'
        elif work_name in ['Cbond', 'RingAromatic']:
            target_type = 'binary'
        else:
            target_type = 'num'

    # FeatureExtractor, Predictor, LossFunc, Metrics, and TargetGetter
    flmt = {}

    # Specify default feature extractor
    if isinstance(feature_extractor, Callable):
        flmt['feature_extractor'] = feature_extractor
    elif isinstance(feature_extractor, str):
        if feature_extractor.lower() in ['atom', 'pair', 'ring', 'cbond', 'mol']:
            flmt['feature_extractor'] = core.feature_extractor[feature_extractor.lower()]
        else:
            raise ValueError(f"Unknown feature extractor: Named {feature_extractor}")
    else:
        if "Atom" in work_name or "xyz" in work_name:
            extractor_name = 'atom'
        elif "Ring" in work_name:
            extractor_name = 'ring'
        elif "Cbond" in work_name:
            extractor_name = 'cbond'
        elif "Pair" in work_name:
            extractor_name = 'pair'
        elif "Mol" in work_name:
            extractor_name = 'mol'
        else:
            raise ValueError("Unknown feature extractor type")
        flmt['feature_extractor'] = core.feature_extractor[extractor_name]

    # Specify default predictor
    if isinstance(predictor, (Callable, nn.Module)):
        pass  # Do nothing
    elif isinstance(predictor, str):
        predictor = M.Predictor(core.vec_size, predictor.lower())
    elif target_type in ['onehot', 'xyz', 'binary', 'num']:
        predictor = M.Predictor(core.vec_size, target_type)
    else:
        raise ValueError(f"Unknown predictor type: {target_type}")

    # Specify loss func
    if isinstance(loss_fn, Callable):
        flmt['loss_fn'] = loss_fn
    elif isinstance(loss_fn, str):
        try:
            flmt['loss_fn'] = loss_options[loss_fn]
        except KeyError:
            raise ValueError(f"Unknown loss function: {loss_fn}")
    else:
        if target_type == 'onehot':
            flmt['loss_fn'] = M.LossMethods.calc_atom_type_loss
        elif target_type == 'xyz':
            flmt['loss_fn'] = M.LossMethods.mean_maximum_displacement
        elif target_type == 'binary':
            flmt['loss_fn'] = F.binary_cross_entropy
        elif target_type == 'num':
            flmt['loss_fn'] = F.mse_loss
        else:
            raise ValueError(f"Loss function has not been specified, pass by argument `loss_fn`")

    # Specify primary metric
    if isinstance(primary_metric, str):
        # Convert the old long metric name to new brief name
        primary_metric = metrics_name_convert.get(primary_metric, primary_metric)
        try:
            flmt['metrics'] = {primary_metric: metrics_options[primary_metric]}
        except KeyError:
            raise ValueError(f"Unknown primary metric: {primary_metric}\n, choose from: {list(metrics_options.keys())}")
    else:
        if target_type == 'onehot':
            primary_metric = 'acc'
            flmt['metrics'] = {primary_metric: lambda p, t: M.Metrics.calc_oh_accuracy(p, t, is_onehot=True)}
        elif target_type == 'xyz':
            primary_metric = 'AMD'  # Average maximum displacement
            flmt['metrics'] = {primary_metric: M.LossMethods.mean_maximum_displacement}
        elif target_type == 'binary':
            primary_metric = 'bacc'
            flmt['metrics'] = {primary_metric: M.Metrics.binary_accuracy}
        elif target_type == 'num':
            primary_metric = 'r2'
            flmt['metrics'] = {primary_metric: M.Metrics.r2_score}
        else:
            raise ValueError(f"The primary metric has not been specified, pass by argument `primary_metric`")

    # Specify other target getter
    if other_metric is None:
        pass
    elif isinstance(other_metric, str):
        # Convert the old long metric name to new brief name
        other_metric = metrics_name_convert.get(other_metric, other_metric)
        if other_metric in metrics_options:
            flmt['metrics'].update({other_metric: metrics_options[other_metric]})
        else:
            raise ValueError(f"Unknown other metric: {other_metric}\n, choose from: {list(metrics_options.keys())}")
    elif isinstance(other_metric, Iterable) and not isinstance(other_metric, dict):
        assert all(isinstance(m, str) for m in other_metric)  # all elements of other metrics must be str
        # Convert the old long metric name to new brief name
        other_metric = [metrics_name_convert.get(m, m) for m in other_metric]
        try:
            flmt['metrics'].update({n: metrics_options[n] for n in other_metric})
        except KeyError as e:
            print(e)
            raise ValueError(f"Unknown other metric, choose from: {list(metrics_options.keys())}")
    elif isinstance(other_metric, dict):
        for n, c in other_metric.items():
            if not isinstance(n, str):
                raise TypeError(f"The metric name should be a string, instead got {type(n)}")
            if not isinstance(c, Callable):
                raise TypeError(f"The metric value should be a callable, instead got {type(c)}")

            flmt['metrics'].update({n: c})

    # Specify target_getter
    if isinstance(target_getter, Callable):
        flmt['target_getter'] = target_getter
    elif isinstance(target_getter, str):
        if target_type == 'xyz':
            XYZ_INDEX = _get_index(first_data, 'x', ('x', 'y', 'z'))
            flmt['target_getter'] = lambda batch: batch.x[:, XYZ_INDEX]
        else:
            attr_type, attr_name = target_getter.rsplit('.')
            TARGETINDEX = _get_index(first_data, attr_type, attr_name)
            flmt['target_getter'] = lambda batch: _get_index(batch, attr_type)[:, TARGETINDEX]
    else:
        if target_type == 'xyz':
            XYZ_INDEX = _get_index(first_data, 'x', ('x', 'y', 'z'))
            flmt['target_getter'] = lambda batch: batch.x[:, XYZ_INDEX]
        elif work_name == 'AtomType':
            flmt['target_getter'] = lambda batch: batch.x[:, 0]
        elif work_name == "AtomCharge":
            ATOM_CHRG_INDEX = _get_index(first_data,'x', 'partial_charge')
            flmt['target_getter'] = lambda batch: batch.x[:, ATOM_CHRG_INDEX]

    # Specify x masker
    if isinstance(x_masker, Callable):
        x_masker = x_masker
    elif isinstance(x_masker, str):
        try:
            x_masker = x_masker_options[x_masker]
        except KeyError:
            raise ValueError(f"Unknown x_masker, choose from: {list(x_masker_options.keys())}")
    else:
        if work_name == 'AtomType':
            x_masker = M.mask_atom_type
        elif work_name == "MetalType":
            x_masker = M.mask_metal_type

    if loss_weight_calculator is None and target_type == 'onehot':
        loss_weight_calculator = lambda t, n: M.atom_label_weight_(t, n, loss_weight_method)
    else:
        loss_weight_calculator = None

    train_tools = TrainTools(
        work_name=work_name,
        work_dir=work_dir,
        hypers=hypers,
        optimizer=optimizer,
        primary_metric=primary_metric,
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
        onehot_types=119 if work_name == 'AtomType' else onehot_labels,
        to_onehot=True if target_type == 'onehot' else False,
        x_masker=x_masker,
        labeled_x=isinstance(getattr(core, 'x_label_nums', None), int),
        loss_weight_calculator=loss_weight_calculator,
        xyz_perturb_sigma=xyz_perturb_sigma,
        debug=debug,
        debug_batch_size=debug_batch_size,
        **flmt,
        **kwargs)

    # Prepare dataset loader
    train_loader, test_loader = train_tools.prepare_dataset(
        train_dataset,
        test_dataset,
        load_all_data=load_all_data,
        batch_size=hypers.batch_size,
        **kwargs
    )

    if save_model:
        train_tools.init_model_dir()

    if isinstance(checkpoint_path, (str, Path)):
        ckpt = train_tools.load_ckpt(checkpoint_path)

    model = LightPretrain(core, predictor, train_tools)
    torch.compile(model)

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

    # configure EarlyStop
    early_stop_callback = EarlyStopping(
        monitor=primary_metric,
        mode='min' if minimize_metric else 'max',
        patience=early_stop_step,
    )

    progress_bar = CustomPBar()
    trainer = L.Trainer(
        default_root_dir=train_tools.model_dir,
        logger=train_tools.logger,
        max_epochs=epochs,
        callbacks=[early_stop_callback, progress_bar],
        precision=precision,
        accelerator='auto',
        devices='auto',
        strategy='ddp_find_unused_parameters_true',
        profiler = profiler
    )
    trainer.fit(
        model,
        train_loader,
        test_loader
    )