import os
import re
import os.path as osp
import functools
from abc import ABC, abstractmethod
from typing import Union, Callable, Optional, Iterable, Any, Literal
from typing_extensions import override

import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn

from torch_geometric.data import Batch

import lightning as L

from hotpot.utils import fmt_print
from hotpot.plugins.ComplexFormer import (
    types as tp,
    models as M,
    tools
)


def specify_single_dataset_task(target_getter: Union[Callable, dict]):
    if isinstance(target_getter, Callable):
        return SingleTask
    elif isinstance(target_getter, dict):
        return MultiTask
    else:
        raise NotImplementedError(f'The target_getter should be a callable or a dict of callables, not{target_getter}')

def specify_task_types(
        is_multi_datasets: bool,
        target_getter: Callable,
):
    if is_multi_datasets:
        return MultiDataTask
    else:
        return specify_single_dataset_task(target_getter)


def _inspector(array: np.ndarray):
    array = np.array(array)
    return [np.mean(array), np.std(array), np.median(array), np.max(array), np.min(array)]


class BaseTask(ABC):
    ############################# General Task Utils #################################
    @staticmethod
    def _preprocess_batch(task: 'Task', batch: Batch) -> Batch:
        if (preprocessor := getattr(task, '_batch_preprocessor', None)) and isinstance(preprocessor, Callable):
            return preprocessor(batch)
        return batch

    #################### Preprocessor before forward #################################
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

    @abstractmethod
    def batch_preprocessor(self, batch: Batch) -> Batch:
        raise NotImplementedError

    @abstractmethod
    def inputs_getter(self, batch: Batch) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_xyz(self, inputs: tuple[torch.Tensor, ...]) -> Optional[torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def perturb_xyz(self, xyz):
        raise NotImplementedError

    @abstractmethod
    def inputs_preprocessor(self, inputs: tuple[torch.Tensor, ...], **kwargs) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError

    @abstractmethod
    def x_masker(self, inputs: tuple[torch.Tensor, ...]) -> (tuple[torch.Tensor, ...], torch.Tensor):
        raise NotImplementedError

    #########################################################################################################
    #########################################################################################################
    ################################### PostProcessor after forward #########################################
    @abstractmethod
    def feature_extractor(self, *args, **kwargs) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        raise NotImplementedError

    @abstractmethod
    def target_getter(self, batch: Batch) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def predict(
            predictor: Union[nn.Module, dict[str, nn.Module]],
            features: Union[torch.Tensor, dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
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
    def add_test_pred_target(
            self,
            pred: Union[torch.Tensor, dict[str, torch.Tensor]],
            target: Union[torch.Tensor, dict[str, torch.Tensor]]
    ):
        raise NotImplementedError

    @abstractmethod
    def summary_val_metrics(self, pl_module: L.LightningModule) -> dict[str, float]:
        raise NotImplementedError

    @staticmethod
    def _log_metrics_on_val_epoch_end(pl_module: L.LightningModule, metrics_dict: dict[str, float]):
        for metric_name, metric_value in metrics_dict.items():
            pl_module.log(metric_name, metric_value, sync_dist=True)  # Log metrics
        pl_module.val_metrics.update(metrics_dict)

    @staticmethod
    def _log_metrics_on_train_batch(pl_module: L.LightningModule, metrics_dict: dict[str, float]):
        for metric_name, metric_value in metrics_dict.items():
            pl_module.log(metric_name, metric_value, sync_dist=True, prog_bar=True)  # Log metrics
        pl_module.train_metrics.update(metrics_dict)

    @abstractmethod
    def eval_on_val_end(self,pl_module: L.LightningModule):
        raise NotImplementedError

    ################## Plot Make ############################
    @abstractmethod
    def make_plots(self, stage: str) -> dict[str, plt.Figure]:
        raise NotImplementedError

    def log_plots(self, pl_module: L.LightningModule):
        # Get current stage
        stage = pl_module.trainer.state.stage
        logdir = pl_module.logger.log_dir
        plotsdir = osp.join(logdir, 'plots')
        if not osp.exists(plotsdir):
            os.mkdir(plotsdir)

        # Make the plots, return a dict
        plots: dict[str, plt.Figure] = self.make_plots(stage)

        for fig_name, fig in plots.items():
            fig.savefig(osp.join(plotsdir, f"{fig_name.replace('/', '_')}.png"))

        # for fig_name, fig in plots.items():
        #     pl_module.logger.experiment.add_figure(f'{stage}/{fig_name}', fig)
    ##########################################################


class Task(BaseTask, ABC):
    _expect_types = {}

    def __init__(
            self,
            inputs_getter: Callable,
            feature_extractor: Union[Callable, dict[str, Callable]],
            target_getter: Union[tp.TargetGetter, dict[str, tp.TargetGetter]],
            loss_fn: Callable[[torch.Tensor, torch.Tensor, Optional[Any]], torch.Tensor],
            primary_metric: Union[str, dict[str, str]],
            metrics: dict[str, Callable[[tp.TensorArray, tp.TensorArray], Union[float, tp.TensorArray]]],
            hypers: tools.Hypers,
            batch_preprocessor: Optional[tp.BatchPreProcessor] = None,
            inputs_preprocessor: Optional[Callable] = None,
            xyz_index: Optional[Iterable[int]] = None,
            xyz_perturb_sigma: Optional[float] = None,
            extractor_attr_getter: Union[tp.ExtractorAttrGetter, dict[str, tp.ExtractorAttrGetter]] = None,
            loss_weight_calculator: Optional[Union[tp.LossWeightCalculator, dict[str, tp.LossWeightCalculator]]] = None,
            to_onehot: Union[bool, Iterable[str]] = False,
            onehot_types: Optional[Union[int, dict[str, int]]] = None,
            x_masker: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
            mask_need_task: Optional[list[str]] = None,
            pred_inspect: Union[bool, Iterable[str]] = False,
            plot_makers: Optional[dict[str, Union[tp.PlotMaker, tp.PlotMakerDict]]] = None,
            **kwargs
    ):
        # Inputs process control arguments
        self._batch_preprocessor = batch_preprocessor
        self._inputs_getter = inputs_getter
        self.xyz_index = xyz_index
        self._xyz_perturb_sigma = xyz_perturb_sigma
        self._inputs_preprocessor = inputs_preprocessor

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
        self.primary_metric = primary_metric
        self._metrics = metrics

        # Pred inspect
        self.pred_inspect = pred_inspect

        # Plot Maker for test step
        self.plot_makers = plot_makers
        self.plots = {}

        # Hyperparameters
        self.hypers = hypers

        # Args check and post process
        self._attr_post_process()


    #################### Args Check and Post Process #################################
    def _type_check(self):
        for attr_name, attr_type in self._expect_types.items():
            if not isinstance(getattr(self, attr_name), attr_type):
                raise TypeError(
                    f'The type of  {self.__class__.__name__}.{attr_name} should be {attr_type}, '
                    f'got {type(getattr(self, attr_name))}')

    def _attr_post_process(self):
        pass

    #################### Preprocessor before forward #################################
    def batch_preprocessor(self, batch: Batch) -> Batch:
        if isinstance(self._batch_preprocessor, Callable):
            return self._batch_preprocessor(batch)
        return batch

    def inputs_getter(self, batch: Batch) -> torch.Tensor:
        return self._inputs_getter(batch)

    def get_xyz(self, inputs: tuple[torch.Tensor, ...]) -> Optional[torch.Tensor]:
        if self.xyz_index is None:
            return None
        else:
            return self.perturb_xyz(inputs[0][:, self.xyz_index])

    def perturb_xyz(self, xyz):
        if isinstance(self._xyz_perturb_sigma, float):
            return M.perturb_xyz(xyz, self._xyz_perturb_sigma)
        return xyz

    def inputs_preprocessor(self, inputs: tuple[torch.Tensor, ...], **kwargs) -> tuple[torch.Tensor, ...]:
        if self._inputs_preprocessor:
            return self._inputs_preprocessor(*inputs, **kwargs)
        return inputs

    def x_masker(self, inputs: tuple[torch.Tensor, ...]) -> (tuple[torch.Tensor, ...], Optional[torch.Tensor]):
        if self._x_masker:
            return self._x_masker(inputs)
        return inputs, None

    @abstractmethod
    def _concat_pred_target(self, which: Literal['val', 'test']) -> (Union[dict, np.ndarray], Optional[np.ndarray]):
        raise NotImplementedError

    @abstractmethod
    def calc_train_batch_loss_metrics(
            self,
            pl_module: L.LightningModule,
            loss: torch.Tensor,
            pred: torch.Tensor,
            target: torch.Tensor,
    ) -> dict[str, float]:
        raise NotImplementedError

    def log_on_train_batch_end(
            self,
            pl_module: L.LightningModule,
            loss: Union[torch.Tensor, dict[str, torch.Tensor]],
            pred: Union[torch.Tensor, dict[str, torch.Tensor]],
            target: Union[torch.Tensor, dict[str, torch.Tensor]],
    ) -> None:
        train_metrics = self.calc_train_batch_loss_metrics(pl_module, loss, pred, target)
        # for name, metric in train_metrics.items():
        #     pl_module.log(name, metric, prog_bar=True)
        self._log_metrics_on_train_batch(pl_module, train_metrics)

    def eval_on_val_end(self,pl_module: L.LightningModule):
        metrics_dict = self.summary_val_metrics(pl_module)
        self._log_metrics_on_val_epoch_end(pl_module, metrics_dict)

    @staticmethod
    def dict_fmt_print(dict_: dict[str, float], print_func=fmt_print.bold_magenta, prefix=''):
        msg = f'{prefix}[' + ', '.join([f'{k}={v:.3g}' for k, v in dict_.items()]) + ']'
        print(type(print_func))
        print_func(msg)


class SingleTask(Task):
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

        self.test_pred = []
        self.test_target = []

    def feature_extractor(self, *args, **kwargs) -> torch.Tensor:
        return self._feature_extractor(*args, batch_getter=self._extractor_attr_getter, **kwargs)

    @staticmethod
    def predict(predictor: nn.Module, features: torch.Tensor) -> torch.Tensor:
        return predictor(features)

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

    def target_getter(self, batch: Batch) -> torch.Tensor:
        return self._target_getter(batch)

    def loss_weight_calculator(self, target) -> Optional[torch.Tensor]:
        if self._loss_weight_calculator:
            return self._loss_weight_calculator(target)
        return None

    def loss_fn(self, pred, target, loss_weight):
        return self._loss_fn(pred, target, loss_weight) \
            if isinstance(loss_weight, torch.Tensor) \
            else self._loss_fn(pred, target)

    def calc_train_batch_loss_metrics(
            self,
            pl_module: L.LightningModule,
            loss: torch.Tensor,
            pred: torch.Tensor,
            target: torch.Tensor,
    ) -> dict[str, float]:
        raise {
            'loss': loss.item(),
            self.primary_metric: self._metrics[self.primary_metric](pred, target),
        }

    @override
    def add_val_pred_target(
            self,
            pred: torch.Tensor,
            target: torch.Tensor
    ):
        self.val_pred.append(pred.cpu().detach().float().numpy())
        self.val_target.append(target.cpu().detach().float().numpy())

    @override
    def add_test_pred_target(
            self,
            pred: torch.Tensor,
            target: torch.Tensor
    ):
        self.test_pred.append(pred.cpu().detach().float().numpy())
        self.test_target.append(target.cpu().detach().float().numpy())

    @override
    def _concat_pred_target(self, which: Literal['val', 'test']) -> (np.ndarray, np.ndarray):
        if which == 'val':
            return np.concatenate(self.val_pred), np.concatenate(self.val_target)
        elif which == 'test':
            return np.concatenate(self.test_pred), np.concatenate(self.test_target)
        else:
            raise NotImplementedError

    def summary_val_metrics(self, pl_module: L.LightningModule) -> dict[str, float]:
        pred, target = self._concat_pred_target(which='val')

        # Calculating the metrics
        metrics_dict = {
            metric_name: float(metric_func(pred, target))
            for metric_name, metric_func in self._metrics.items()
        }

        metrics_dict['lr'] = pl_module.optimizers().param_groups[0]['lr']

        return metrics_dict

    def make_plots(self, stage: str) -> dict[str, plt.Figure]:
        pred, target = self._concat_pred_target(stage)
        return {plot_name: maker(pred, target) for plot_name, maker in self.plot_makers.items()}

    @override
    def eval_on_val_end(self, pl_module: L.LightningModule):
        # Print and log metrics
        super().eval_on_val_end(pl_module)

        # free memory
        self.val_pred.clear()
        self.val_target.clear()


class MultiTask(Task):
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
        self.test_pred = {}
        self.test_target = {}

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
        return {k: (t.view(-1, 1) if t.dim() == 1 else t) for k, t in target.items()}

    def peel_unmaksed_obj(
            self,
            feature_target: dict[str, torch.Tensor],
            mask_idx: Union[torch.Tensor] = None,
    ):
        if mask_idx is None or self._mask_need_task is None:
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
            try:
                if (lw := loss_weight.get(k, None)) is not None:
                    self.loss_dict[k] = self._loss_fn[k](p, target[k].to(p.dtype), lw)
                else:
                    self.loss_dict[k] = self._loss_fn[k](p, target[k].to(p.dtype))
            except RuntimeError as e:
                msg = e.args[0]
                raise RuntimeError(
                    msg + '\n'
                    f'{k}(pred dtype: {p.dtype}, target dtype: {target[k].dtype})'
                )

        # Calculate the total loss
        if isinstance(self.atl_weights, dict):
            return sum(lo * self.atl_weights.get(k, 1.) for k, lo in self.loss_dict.items())
        else:
            return sum(lo for lo in self.loss_dict.values())

    def calc_train_batch_loss_metrics(
            self,
            pl_module: L.LightningModule,
            loss: torch.Tensor,
            pred: dict[str, torch.Tensor],
            target: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        train_metrics = {'loss': loss.item()}
        for tsk_name, t in target.items():
            try:
                metric = self._metrics[tsk_name][self.primary_metric[tsk_name]](pred[tsk_name], t)
                if isinstance(metric, torch.Tensor):
                    train_metrics[tsk_name] = metric.item()
                else:
                    train_metrics[tsk_name] = float(metric)

            except KeyError as e:
                if tsk_name not in self._metrics:
                    raise KeyError(f'task_name: {tsk_name} not in self._metrics, {list(self._metrics.keys())}')
                elif tsk_name not in self.primary_metric:
                    raise KeyError(f'task_name: {tsk_name} not in self.primary_metric, {list(self._metrics.keys())}')
                elif tsk_name in pred:
                    raise KeyError(f'task_name: {tsk_name} not in pred, {list(self._metrics.keys())}')
                elif self.primary_metric[tsk_name] not in self._metrics[tsk_name]:
                    raise KeyError(f'primary_metric: {self.primary_metric[tsk_name]} not in '
                                   f'self._metrics["{tsk_name}"], {list(self._metrics.keys())}')
                else:
                    raise KeyError(f'Unknown Key error for:\n'
                                   f' task "{tsk_name}"\n'
                                   f' self._metrics: {self._metrics}\n'
                                   f' self.primary_metric: {self.primary_metric}\n'
                                   f' pred: {pred.keys()}\n')

        return train_metrics

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
    def add_test_pred_target(
            self,
            pred: Union[torch.Tensor, dict[str, torch.Tensor]],
            target: Union[torch.Tensor, dict[str, torch.Tensor]]
    ):
        for k, t in target.items():
            self.test_pred.setdefault(k, []).append(pred[k].cpu().detach().float().numpy())
            self.test_target.setdefault(k, []).append(t.cpu().detach().float().numpy())

    @override
    def _concat_pred_target(self, which: Literal['val', 'test']) -> (dict[str, np.ndarray], dict[str, np.ndarray]):
        if which == 'val':
            pred = {k: np.concatenate(p) for k, p in self.val_pred.items()}
            target = {k: np.concatenate(t) for k, t in self.val_target.items()}
        elif which == 'test':
            pred = {k: np.concatenate(p) for k, p in self.test_pred.items()}
            target = {k: np.concatenate(t) for k, t in self.test_target.items()}
        else:
            raise NotImplementedError

        return pred, target

    def summary_val_metrics(self, pl_module: L.LightningModule) -> dict[str, float]:
        val_pred, val_target = self._concat_pred_target('val')

        # Calculate metrics
        metrics_dict = {
            k: float(self._metrics[k][self.primary_metric[k]](val_pred[k], t))
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

        return metrics_dict

    @override
    def eval_on_val_end(self,pl_module: L.LightningModule):
        # Print and log metrics
        super().eval_on_val_end(pl_module)

        self.val_pred.clear()
        self.val_target.clear()

    def make_plots(self, stage: str) -> dict[str, plt.Figure]:
        pred, target = self._concat_pred_target(stage)

        plots = {}
        for tsk, maker_dict in self.plot_makers.items():
            for plot_name, maker in maker_dict.items():
                plots[f'{tsk}/{plot_name}'] = maker(pred[tsk], target[tsk])

        return plots


############################# MultiDataTask #####################################
def _perform_current_task(md_task_method: Callable):
    @functools.wraps(md_task_method)
    def wrapper(self, *args, **kwargs):
        return getattr(self.current_task, md_task_method.__name__)(*args, **kwargs)
    return wrapper

def _align_md_task_options(name, arg: Any, task_counts: int):
    if isinstance(arg, list):
        assert len(arg) == task_counts, f'Expecting {name} has same length as task_counts, but {len(arg)} != {task_counts}'
        return arg
    else:
        return [arg] * task_counts

class MultiDataTask(BaseTask):
    lr_matcher = re.compile(r'\d+-lr')

    def __init__(self, *tasks: MultiTask, list_kwargs: list[dict[str, Any]] = None):
        if tasks:
            self._tasks = tasks
        elif isinstance(list_kwargs, list) and all(isinstance(kw, dict) for kw in list_kwargs):
            self._tasks = [specify_single_dataset_task(kw['target_getter'])(**kw) for kw in list_kwargs]
        else:
            raise NotImplementedError('The args `tasks` and `list_kwargs` must be given at least one')

        self.current_task = None
        self.metrics_dict = {}

    @classmethod
    def init_from_args(
            cls,
            tasks_counts: int,
            inputs_getter: Callable,
            feature_extractor: list[dict[str, Callable]],
            target_getter: list[dict[str, tp.TargetGetter]],
            loss_fn: list[dict[str, tp.LossFn]],
            primary_metric: list[dict[str, str]],
            metrics: list[dict[str, tp.MetricFn]],
            batch_preprocessor: Optional[list[tp.BatchPreProcessor]] = None,
            inputs_preprocessor: Optional[list[Callable]] = None,
            xyz_index: Optional[list[Iterable[int]]] = None,
            xyz_perturb_sigma: Optional[list[float]] = None,
            extractor_attr_getter: Optional[list[dict[str, tp.ExtractorAttrGetter]]] = None,
            loss_weight_calculator: Optional[list[dict[str, tp.LossWeightCalculator]]] = None,
            # to_onehot: Optional[list[Iterable[str]]] = None,
            onehot_types: Optional[list[dict[str, int]]] = None,
            x_masker: Optional[list[tp.XMasker]] = None,
            mask_need_task: Optional[list[list[str]]] = None,
            **kwargs
    ):
        # Aligning requirements
        assert len(feature_extractor) == tasks_counts
        assert len(target_getter) == tasks_counts
        assert len(loss_fn) == tasks_counts
        assert len(primary_metric) == tasks_counts
        assert len(metrics) == tasks_counts
        if isinstance(inputs_getter, Callable):
            inputs_getter = [inputs_getter] * tasks_counts
        elif isinstance(inputs_getter, list):
            assert len(inputs_getter) == tasks_counts
        else:
            raise TypeError('inputs_getter must be a callable or a list of callable')

        # Aligning optionals
        batch_preprocessor = _align_md_task_options('batch_preprocessor', batch_preprocessor, tasks_counts)
        inputs_preprocessor = _align_md_task_options('inputs_preprocessor', inputs_preprocessor, tasks_counts)
        xyz_index = _align_md_task_options('xyz_index', xyz_index, tasks_counts)
        xyz_perturb_sigma = _align_md_task_options('xyz_perturb_sigma', xyz_perturb_sigma, tasks_counts)
        extractor_attr_getter = _align_md_task_options('extractor_attr_getter', extractor_attr_getter, tasks_counts)
        loss_weight_calculator = _align_md_task_options('loss_weight_calculator', loss_weight_calculator, tasks_counts)
        onehot_types = _align_md_task_options('onehot_types', onehot_types, tasks_counts)
        x_masker = _align_md_task_options('x_masker', x_masker, tasks_counts)
        mask_need_task = _align_md_task_options('mask_need_task', mask_need_task, tasks_counts)

        to_onehot = [(list(oh_types) if isinstance(oh_types, dict) else None) for oh_types in onehot_types]

        tasks = []
        for i in range(tasks_counts):
            task = MultiTask(
                inputs_getter=inputs_getter[i],
                feature_extractor=feature_extractor[i],
                target_getter=target_getter[i],
                loss_fn=loss_fn[i],
                primary_metric=primary_metric[i],
                metrics=metrics[i],
                batch_preprocessor=batch_preprocessor[i],
                inputs_preprocessor=inputs_preprocessor[i],
                xyz_index=xyz_index[i],
                xyz_perturb_sigma=xyz_perturb_sigma[i],
                extractor_attr_getter=extractor_attr_getter[i],
                loss_weight_calculator=loss_weight_calculator[i],
                to_onehot=to_onehot[i],
                onehot_types=onehot_types[i],
                x_masker=x_masker[i],
                mask_need_task=mask_need_task[i],
            )
            tasks.append(task)

        return cls(*tasks)

    def choose_task(self, batch):
        assert hasattr(batch, 'dataset_idx'), "The task choice depends on batch attr `dataset_idx`, but it's not found"
        dataset_idx = batch.dataset_idx
        assert len(torch.unique(dataset_idx)), (f'The implementation of MultiDataTask requires all Data in the Batch '
            f'from a same dataset\n, but they are from various Dataset: Index{torch.unique(dataset_idx)}')
        self.current_task = self._tasks[dataset_idx[0]]

    def batch_preprocessor(self, batch: Batch) -> Batch:
        self.choose_task(batch)
        return self.current_task.batch_preprocessor(batch)

    def log_on_train_batch_end(
            self,
            pl_module: L.LightningModule,
            loss: Union[torch.Tensor, dict[str, torch.Tensor]],
            pred: Union[torch.Tensor, dict[str, torch.Tensor]],
            target: Union[torch.Tensor, dict[str, torch.Tensor]],
    ) -> None:
        # total_metrics = {}
        # for i, task in enumerate(self._tasks):
        #     metrics_dict = task.calc_train_batch_loss_metrics(pl_module, loss, pred, target)
        #     for tsk_name, metrics in metrics_dict.items():
        #         total_metrics[f'{i}-{tsk_name}'] = metrics
        ds_idx = self._tasks.index(self.current_task)
        metrics_dict = self.current_task.calc_train_batch_loss_metrics(pl_module, loss, pred, target)
        metrics_dict = {f'{ds_idx}-{tsk_name}': metrics for tsk_name, metrics in metrics_dict.items()}

        self._log_metrics_on_train_batch(pl_module, metrics_dict)

    def eval_on_val_end(self, pl_module: L.LightningModule):
        total_metrics = {}
        for i, task in enumerate(self._tasks):
            metrics_dict = task.summary_val_metrics(pl_module)
            metrics_dict.pop('smtrc', None)
            for tsk_name, metrics in metrics_dict.items():
                total_metrics[f'{i}-{tsk_name}'] = float(metrics)

        metrics_values = [v for k, v in total_metrics.items() if not self.lr_matcher.fullmatch(k)]
        total_metrics['smtrc'] = sum(metrics_values) / len(metrics_values)
        self._log_metrics_on_val_epoch_end(pl_module, total_metrics)

    def make_plots(self, stage: str) -> dict[str, plt.Figure]:
        plots = {}
        for i, task in enumerate(self._tasks):
            tsk_plots = task.make_plots(stage)
            for plot_name, plot in tsk_plots.items():
                plots[f'Dataset{i}/{plot_name}'] = plot

        return plots

# Set all abstractmethod to do current tasks
for abc_method in MultiDataTask.__abstractmethods__:
    setattr(MultiDataTask, abc_method, _perform_current_task(getattr(MultiDataTask, abc_method)))
MultiDataTask.__abstractmethods__ = frozenset([])
######################################################################################

