import os
import re
import glob
import os.path as osp
import functools
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Union, Callable, Optional, Iterable, Any, Literal

from typing_extensions import override

import numpy as np
from matplotlib import font_manager as fm
from matplotlib import pyplot as plt

import torch
import torch.nn as nn

from torch_geometric.data import Batch

import lightning as L

import hotpot.plugins.opti.params_space
from hotpot.plugins.ComplexFormer import (
    types as tp,
    models as M,
    tools,
)

from hotpot.utils import fmt_print
from hotpot.utils.configs.logging_config import LoggerDict


__all__ = [
    'specify_task_types',
    'specify_single_dataset_task',
    'NaNMetricError',
    'BaseTask',
    'Task',
    'SingleTask',
    'MultiTask',
    'MultiDataTask'
]

# Add the fonts in hotpot
def add_mpl_font():
    file_dir = osp.dirname(__file__)
    fonts_dir = osp.abspath(osp.join(file_dir, '..', '..', 'cheminfo', 'fonts'))
    logging.debug(f'fonts_dir={fonts_dir}')

    fonts_paths = glob.glob(osp.join(fonts_dir, '*', '*.ttf'))
    for font_path in fonts_paths:
        fm.fontManager.addfont(font_path)
add_mpl_font()
#####################################


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


class NaNMetricError(Exception): pass


class BaseTask(ABC):
    ################################## Checkers ####################################
    
    ############################ Variable Declaration ###############################
    hypers: hotpot.plugins.opti.params_space.ParamSets
    show_pbar: bool = True
    test_primary_metrics: Union[float, dict[str, float]] = None

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
    def get_xyz(self, inputs: tuple, batch: Batch) -> Optional[torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def get_sol_info(self, batch: Batch) -> Optional[tuple[torch.Tensor, ...]]:
        raise NotImplementedError

    @abstractmethod
    def get_med_info(self, batch: Batch) -> Optional[tuple[torch.Tensor, ...]]:
        raise NotImplementedError

    @abstractmethod
    def perturb_xyz(self, xyz, batch: Batch) -> torch.Tensor:
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
    def inverse_pred(self, pred: Union[torch.Tensor, dict[str, torch.Tensor]]) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def target_getter(self, batch: Batch, norm: bool = False) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
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
    def summary_metrics(
            self, pl_module: L.LightningModule,
            stages: Literal['val', 'test'] = 'val'
    ) -> tuple[
        dict[str, Union[float, dict]],
        dict[str, float]
    ]:
        """ Summary all metrics results in stored in the Task instance, after a val or test epoch """
        raise NotImplementedError

    def _log_metrics(
            self,
            pl_module: L.LightningModule,
            metrics_dict: dict[str, Union[float, dict[str, float]]],
            stages: Literal['train', 'val', 'test'] = 'val'
    ) -> None:
        # Log metrics to Lightning
        for tsk_mtrc_name, metric_dict_value in metrics_dict.items():
            if isinstance(metric_dict_value, float):
                pl_module.log(tsk_mtrc_name, metric_dict_value, sync_dist=True)  # Log metrics
            elif isinstance(metric_dict_value, dict):
                for mtrc_name, mtrc_value in metric_dict_value.items():
                    if tsk_mtrc_name == 'smtrc':
                        pl_module.log('smtrc', mtrc_value, sync_dist=True)
                    else:
                        pl_module.log(f"{tsk_mtrc_name}_{mtrc_name}", mtrc_value, sync_dist=True)

        # Log metrics to PretrainModule
        pl_module_metrics = getattr(pl_module, f'{stages}_metrics')
        pl_module_metrics.update(metrics_dict)

        if not self.show_pbar and stages != 'train':
            table = fmt_print.dict_to_table(
                metrics_dict, {'style': 'magenta'},
                title=f'Eval in Validation Step (Epoch {pl_module.current_epoch})',
            )
            fmt_print.rich_print(table, width=200)

    @staticmethod
    def _log_metrics_on_val_epoch_end(
            pl_module: L.LightningModule,
            metrics_dict: dict[str, Union[float, dict[str, float]]]
    ):
        """ TODO: Deprecated """
        for tsk_mtrc_name, metric_dict_value in metrics_dict.items():
            if isinstance(metric_dict_value, float):
                pl_module.log(tsk_mtrc_name, metric_dict_value, sync_dist=True)  # Log metrics
            elif isinstance(metric_dict_value, dict):
                for mtrc_name, mtrc_value in metric_dict_value.items():
                    pl_module.log(f"{tsk_mtrc_name}_{mtrc_name}", mtrc_value, sync_dist=True)

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
    def make_plots(self, stages: Literal['val', 'test']) -> dict[str, plt.Figure]:
        raise NotImplementedError

    @staticmethod
    def retrieve_path_from_log_dir(pl_module, subdir: Optional[str] = None) -> str:
        logdir = pl_module.logger.log_dir
        if not isinstance(subdir, str):
            return logdir

        else:
            subpath = osp.join(logdir, 'plots')
            if not osp.exists(subpath):
                os.mkdir(subpath)
            return subpath

    def log_plots(self, pl_module: L.LightningModule):
        # Get current stage
        stages = pl_module.trainer.state.stage
        plotsdir = self.retrieve_path_from_log_dir(pl_module, 'plots')

        # Make the plots, return a dict
        plots: dict[str, plt.Figure] = self.make_plots(stages)

        for fig_name, fig in plots.items():
            fig.savefig(osp.join(plotsdir, f"{fig_name.replace('/', '_')}.png"))
    ##########################################################

    ################## Save Metric Sheet #####################
    def store_save_metrics_table(self, pl_module: L.LightningModule):
        metrics_dict, primary_metric = self.summary_metrics(pl_module, stages='test')
        table = fmt_print.dict_to_table(metrics_dict, title="Test Metrics")
        logdir = pl_module.logger.log_dir
        fmt_print.export_table(table, osp.join(logdir, 'test_metrics.txt'))
        fmt_print.rich_print(table, width=200)

        df = fmt_print.dict_to_df(metrics_dict)
        df.to_csv(osp.join(logdir, 'test_metrics.csv'))

        self.test_primary_metrics = primary_metric
        return metrics_dict, primary_metric

    def export_hparams(self, pl_module: L.LightningModule):
        if not osp.exists(save_dir := os.path.join(pl_module.logger.log_dir, 'hparams.json')):
            self.hypers.export(save_dir)

_default_sol_graph_inputs = ('x', 'edge_index', 'batch')
_default_med_graph_inputs = ('x', 'edge_index', 'batch')

class Task(BaseTask, ABC):
    _expect_types = {}
    _sol_key_matcher = re.compile(r'sol\d?_.+')
    _med_key_matcher = re.compile(r'med\d?_.+')

    def __init__(
            self,
            inputs_getter: Callable,
            feature_extractor: Union[Callable, dict[str, Callable]],
            target_getter: Union[tp.TargetGetter, dict[str, tp.TargetGetter]],
            loss_fn: Callable[[torch.Tensor, torch.Tensor, Optional[Any]], torch.Tensor],
            primary_metric: Union[str, dict[str, str]],
            metrics: dict[str, Union[tp.MetricFn, dict[str, tp.MetricFn]]],
            hypers: hotpot.plugins.opti.params_space.ParamSets,
            batch_preprocessor: Optional[tp.BatchPreProcessor] = None,
            inputs_preprocessor: Optional[Callable] = None,
            xyz_index: Optional[Iterable[int]] = None,
            xyz_perturb_sigma: Optional[float] = None,
            xyz_perturb_mode: M.PerturbMode = 'uniform',
            extractor_attr_getter: Union[tp.ExtractorAttrGetter, dict[str, tp.ExtractorAttrGetter]] = None,
            loss_weight_calculator: Optional[Union[tp.LossWeightCalculator, dict[str, tp.LossWeightCalculator]]] = None,
            to_onehot: Union[bool, Iterable[str]] = False,
            onehot_types: Optional[Union[int, dict[str, int]]] = None,
            x_masker: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
            mask_need_task: Optional[list[str]] = None,
            pred_inspect: Union[bool, Iterable[str]] = False,
            plot_makers: Optional[dict[str, Union[tp.PlotMaker, tp.PlotMakerDict]]] = None,
            target_normalizer = None,
            **kwargs
    ):
        # Inputs process control arguments
        self._batch_preprocessor = batch_preprocessor
        self._inputs_getter = inputs_getter
        self.xyz_index = xyz_index
        self._xyz_perturb_sigma = xyz_perturb_sigma
        self._xyz_perturb_mode = xyz_perturb_mode
        self._inputs_preprocessor = inputs_preprocessor

        # Mask
        self._x_masker = x_masker
        self._mask_need_task = mask_need_task if mask_need_task else []
        self._masked_idx = None

        # Feature extract
        self._feature_extractor = feature_extractor
        self._extractor_attr_getter = extractor_attr_getter

        # get target
        self._target_getter = self._check_target_getter(target_getter)

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

        self.with_sol = kwargs.get('with_sol', False)
        self.sol_graph_inputs = kwargs.get('sol_graph_inputs', _default_sol_graph_inputs)

        self.with_med = kwargs.get('with_med', False)
        self.med_graph_inputs = kwargs.get('med_graph_inputs', _default_med_graph_inputs)

        self.with_env = kwargs.get('with_env', False)

        # xyz_perturb recording
        self._target_normalizer = target_normalizer

        # init a logger
        self.info_logger = LoggerDict(interval_count=25)


    #################### Args Check and Post Process #################################
    @staticmethod
    @abstractmethod
    def _check_target_getter(target_getter) -> Union[tools.TargetGetter, dict[str, tools.TargetGetter]]:
        raise NotImplementedError
    
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

    def get_xyz(self, inputs: tuple, batch: Batch) -> Optional[torch.Tensor]:
        if self.xyz_index is None:
            return None
        else:
            return self.perturb_xyz(inputs[0][:, self.xyz_index], batch)

    @staticmethod
    def _extract_specific_data(
            flag: str,
            batch: Batch,
            pattern: re.Pattern
    ) -> tuple[
        Optional[Union[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]],  # graph info
        Optional[Union[torch.Tensor, list[torch.Tensor]]],  # attributes
        Optional[torch.Tensor]  # ratio
    ]:
        """ Extract solvent or media data from PyG batch """
        exclude_suffix = ['_names', '_metric']
        the_keys = [
            k for k in batch.keys()
            if (pattern.fullmatch(k) and not any(k.endswith(s) for s in exclude_suffix))
        ]
        assert len(the_keys) > 1, f"The task with flag `{flag}` but without corresponding keys in the data Batch"

        # Extract ratio information
        ratio_key = [k for k in the_keys if k.endswith('_ratio')]
        assert len(ratio_key) <= 1
        if len(ratio_key) == 1:
            ratio_key = ratio_key[0]
            the_keys.remove(ratio_key)

            the_ratio = batch[ratio_key]

        else:
            the_ratio = None

        # Check whether the names of all keys are as expected:
        # If `the_ratio` is None, there should be just only obj (denoted as XXX) matched the pattern, thus the
        # names of the keys should like `XXX_xxxx` without the series number
        # Conversely, If `the ratio` is a Tensor, there should be multiple obj matched the pattern,
        # thus the names of the keys should like `XXXN_xxxx`, where the N is the series number of the XXX objs.
        # The N is range from 1 to the length of `the_ratio` Tensor
        # forth_char = the_keys[0][3] if not the_keys[0].endwith('_ratio') else the_keys[1][3]

        if the_ratio is None:
            assert all(k[3] == '_' for k in the_keys)
        else:
            series_char = {k[3] for k in the_keys}
            assert all(c.isdigit() for c in series_char), \
                f'not all key[3] is digit, the_keys:\n{[k for k in the_keys if not k[3].isdigit()]}'
            assert len(series_char) == the_ratio.size(1)
            assert all(1 <= int(c) <= the_ratio.size(1) for c in series_char)

        group_indices = {int(c) for c in locals().get('series_char', set())}

        # For one object
        if not group_indices:
            extract_info = {k[4:]: batch[k] for k in the_keys}
            attributes = extract_info.pop('attr', None)

            graph_info = extract_info if extract_info else None  # Just a copy

            # Data structure:
            # graph_info: None or {'x': Tensor, 'edge_index': tensor, ...}
            # attributes: Tensor or None
            # graph_ratio: None
            return graph_info, attributes, None

        else:  # For multiply object
            extract_info = defaultdict(dict)
            for key in the_keys:
                extract_info[key[3]][key[5:]] = batch[key]

            graph_info, attributes = [], []
            for sol_idx in sorted(extract_info.keys()):
                attributes.append(extract_info[sol_idx].pop('attr', None))
                graph_info.append(extract_info[sol_idx])

            graph_info = graph_info if graph_info else None
            attributes = attributes if any(a is not None for a in attributes) else None

            # Data structure:
            # graph_info: None or [
            #     {'x': Tensor, 'edge_index': tensor, ...},  # graph 1
            #     {'x': Tensor, 'edge_index': tensor, ...},  # graph 2
            #     ...
            # ]
            # attribute: None or [
            #     Tensor, # attrs for graph 1
            #     Tensor, # attrs for graph 2
            # ]
            return graph_info, attributes, the_ratio

    @staticmethod
    def _avoid_empty_graphs(graph_dict, attrs, ratios):
        """ Avoid any empty graphs in whole a col components """
        if isinstance(ratios, torch.Tensor):
            assert graph_dict is None or (isinstance(graph_dict, list) and all(isinstance(g, dict) for g in graph_dict))
            assert attrs is None or (isinstance(attrs, list) and all(isinstance(a, torch.Tensor) for a in attrs))

            # Check if some components just contain empty graphs
            if isinstance(graph_dict, list):
                remove_idx = sorted([i for i, g in enumerate(graph_dict) if g['x'].numel() == 0])
                assert len(remove_idx) < len(graph_dict)  # Make sure that not all graphs are empty

                if remove_idx:
                    for i in remove_idx:
                        # all i col ratio should be 0
                        assert torch.count_nonzero(ratios[:, i]) == 0

                        if isinstance(attrs, list):
                            assert attrs is None or torch.count_nonzero(attrs[i]) == 0
                            del attrs[i]

                        del graph_dict[i]

                    assert attrs is None or len(attrs) == len(graph_dict)

                    # If only one component leave
                    if len(graph_dict) == 1:
                        graph_dict = graph_dict[0]
                        if attrs is not None:
                            attrs = attrs[0]

                        ratios = None

        else:
            if graph_dict is not None:
                assert isinstance(graph_dict, dict)
                if graph_dict['x'].numel() == 0:
                    assert attrs is None or (isinstance(attrs, torch.Tensor) and torch.count_nonzero(attrs) == 0)

                    graph_dict = attrs = None

        return graph_dict, attrs, ratios


    def envs_graph_postprocessing(self, graph_dict, attrs, ratios):
        # Eliminate empty graph
        graph_dict, attrs, ratios = self._avoid_empty_graphs(graph_dict, attrs, ratios)

        # Graph inputs preprocessing
        # In default, extracting node feature[n, s, p, ..., x, y, z, formal_charge, ...] from Batch.x, convert the dtype.
        if isinstance(graph_dict, dict):
            graph_dict = self.inputs_preprocessor({inp: graph_dict[inp] for inp in self.sol_graph_inputs})
        else:
            graph_dict = [self.inputs_preprocessor({inp: d[inp] for inp in self.sol_graph_inputs}) for d in graph_dict]

        return graph_dict, attrs, ratios

    def get_sol_info(self, batch: Batch):
        """"""
        if not self.with_sol:
            return None, None, None

        graph_dict, attrs, ratio = self.envs_graph_postprocessing(
            *self._extract_specific_data('sol', batch, self._sol_key_matcher)
        )

        return graph_dict, attrs, ratio

    def get_med_info(self, batch: Batch):
        if not self.with_med:
            return None, None, None

        return self.envs_graph_postprocessing(
            *self._extract_specific_data('med', batch, self._med_key_matcher)
        )

    def perturb_xyz(self, xyz, batch: Batch):
        if isinstance(self._xyz_perturb_sigma, float):
            self.info_logger['perturb_xyz'].info(f"[#c0fb2d]XYZ perturb[/], sigma={self._xyz_perturb_sigma}, mode={self._xyz_perturb_mode}")
            xyz, pert = M.perturb_xyz(xyz, self._xyz_perturb_sigma, self._xyz_perturb_mode)
            batch.pert_xyz = pert

        return xyz

    def inputs_preprocessor(self, inputs: Union[dict, list, tuple], **kwargs) -> tuple[torch.Tensor, ...]:
        if self._inputs_preprocessor:
            return self._inputs_preprocessor(inputs, **kwargs)
        return inputs

    def x_masker(self, inputs: tuple[torch.Tensor, ...]) -> (tuple[torch.Tensor, ...], Optional[torch.Tensor]):
        if self._x_masker:
            return self._x_masker(inputs)
        return inputs, None

    @abstractmethod
    def concat_pred_target(self, stage: Literal['val', 'test']) -> (Union[dict, np.ndarray], Optional[np.ndarray]):
        raise

    @abstractmethod
    def calc_metrics(self, pred, target, get_primary=False) -> (dict[str, float], dict[str, float]):
        raise NotImplementedError

    @staticmethod
    def _check_metrics(metrics: dict[str, Union[float, dict[str, float]]]) -> dict[str, Union[float, dict[str, float]]]:
        for name, value in metrics.items():
            if isinstance(value, float):
                if np.isnan(value):
                    raise NaNMetricError(f'metric {name} has NaN value')
            elif isinstance(value, dict):
                for m, v in value.items():
                    if np.isnan(v):
                        raise NaNMetricError(f'metric {name}-{m} has NaN value')
            else:
                raise TypeError(f'metric {name} has type {type(value)}')
        return metrics

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
        # The targets have been normalized
        train_metrics = self.calc_train_batch_loss_metrics(pl_module, loss, pred, target)
        self._log_metrics(pl_module, train_metrics, stages='train')

    def eval_on_val_end(self,pl_module: L.LightningModule):
        metrics_dict, primary_metric = self.summary_metrics(pl_module, stages='val')
        self._log_metrics(pl_module, metrics_dict, stages='val')

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

    @staticmethod
    def _check_target_getter(target_getter) -> tools.TargetGetter:
        assert isinstance(target_getter, tools.TargetGetter), 'target_getter must be a TargetGetter instance'
        return target_getter

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

    def target_getter(self, batch: Batch, norm=False) -> torch.Tensor:
        return self._target_getter(batch, norm)

    def loss_weight_calculator(self, target) -> Optional[torch.Tensor]:
        if self._loss_weight_calculator:
            return self._loss_weight_calculator(target)
        return None

    def loss_fn(self, pred, target, loss_weight):
        return self._loss_fn(pred, target, loss_weight) \
            if isinstance(loss_weight, torch.Tensor) \
            else self._loss_fn(pred, target)
    
    def inverse_pred(self, pred: torch.Tensor) -> torch.Tensor:
        return self._target_getter.inverse(pred)

    def calc_train_batch_loss_metrics(
            self,
            pl_module: L.LightningModule,
            loss: torch.Tensor,
            pred: torch.Tensor,
            target: torch.Tensor,
    ) -> dict[str, float]:
        return {
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
    def concat_pred_target(self, stage: Literal['val', 'test']) -> (np.ndarray, np.ndarray):
        if stage == 'val':
            return np.concatenate(self.val_pred), np.concatenate(self.val_target)
        elif stage == 'test':
            return np.concatenate(self.test_pred), np.concatenate(self.test_target)
        else:
            raise NotImplementedError

    def calc_primary_metrics(
            self,
            pred: Union[np.ndarray, torch.Tensor],
            target: Union[np.ndarray, torch.Tensor]
    ) -> dict[str, float]:
        return {self.primary_metric: self._metrics[self.primary_metric](pred, target)}

    def calc_metrics(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            get_primary: bool = False
    ) -> tuple[dict[str, float], dict[str, float]]:
        metrics_dict = {
            metric_name: float(metric_func(pred, target))
            for metric_name, metric_func in self._metrics.items()
        }
        primary_metric = {self.primary_metric: metrics_dict[self.primary_metric]}
        return self._check_metrics(metrics_dict), primary_metric

    def summary_metrics(self, pl_module: L.LightningModule, stages='val') -> tuple[dict[str, float], dict[str, float]]:
        pred, target = self.concat_pred_target(stages)

        # Calculating the metrics
        metrics_dict, primary_metrics = self.calc_metrics(pred, target)
        if stages == 'val':
            metrics_dict['lr'] = pl_module.optimizers().param_groups[0]['lr']

        return metrics_dict, primary_metrics

    def make_plots(self, stages: Literal['val', 'test']) -> dict[str, plt.Figure]:
        pred, target = self.concat_pred_target(stages)
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

    @staticmethod
    def _check_target_getter(target_getter) -> dict[str, tools.TargetGetter]:
        assert isinstance(target_getter, dict), 'target_getter must be a dict'
        assert all(isinstance(tg, tools.TargetGetter) for tg in target_getter.values()), (
            'all values in the target_getter must be a TargetGetter instance')
        return target_getter

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

    def target_getter(self, batch: Batch, norm=False) -> dict[str, torch.Tensor]:
        return {k: tg(batch, norm) for k, tg in self._target_getter.items()}

    def inverse_pred(self, pred: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: tg.inverse(pred[k]) for k, tg in self._target_getter.items()}

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
                    train_metrics[tsk_name] = {self.primary_metric[tsk_name]: metric.item()}
                else:
                    train_metrics[tsk_name] = {self.primary_metric[tsk_name]: float(metric)}

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
    def concat_pred_target(self, stages: Literal['val', 'test']) -> (dict[str, np.ndarray], dict[str, np.ndarray]):
        if stages == 'val':
            pred = {k: np.concatenate(p) for k, p in self.val_pred.items()}
            target = {k: np.concatenate(t) for k, t in self.val_target.items()}
        elif stages == 'test':
            pred = {k: np.concatenate(p) for k, p in self.test_pred.items()}
            target = {k: np.concatenate(t) for k, t in self.test_target.items()}
        else:
            raise NotImplementedError

        return pred, target

    def calc_metrics(
            self,
            pred: dict[str, Union[torch.Tensor, np.ndarray]],
            target: dict[str, torch.Tensor, np.ndarray],
            get_primary=False
    ) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
        metrics_dict ={
            tsk: {mtrc_name: float(mtrc(pred[tsk], tgt)) for mtrc_name, mtrc in self._metrics[tsk].items()}
            for tsk, tgt in target.items()
        }
        primary_metrics = {tsk: metrics_dict[tsk][self.primary_metric[tsk]] for tsk in target}
        return self._check_metrics(metrics_dict), primary_metrics

    def _update_across_tasks_loss_weights(self, pl_module: L.LightningModule, primary_metrics: dict[str, float]):
        """ Updata across tasks loss weights. """
        if pl_module.current_epoch < 1 or not isinstance(self.atl_weights_calculators, Callable):
            self.atl_weights = None
        else:
            self.atl_weights = self.atl_weights_calculators(primary_metrics)
            self.dict_fmt_print(self.atl_weights, prefix='\nalt_weights: ')

    def summary_metrics(self, pl_module: L.LightningModule, stages='val') -> tuple[dict[str, dict[str, float]], dict[str, float]]:
        pred, target = self.concat_pred_target(stages)

        # Calculate metrics
        metrics_dict, primary_metrics = self.calc_metrics(pred, target, get_primary=True)

        if stages == 'val':  # Update across tasks loss weights
            self._update_across_tasks_loss_weights(pl_module, primary_metrics)

        # Add sum metrics
        metrics_dict['smtrc'] = {'smtrc': np.mean([v for v in primary_metrics.values()]) if metrics_dict else 0.}

        if stages == 'val':
            logging.debug(f'[#3f51b5]The Optimizers: {pl_module.optimizers()}[/]')
            metrics_dict['lr'] = {'lr': pl_module.optimizers().param_groups[0]['lr']}

        return metrics_dict, primary_metrics

    @override
    def eval_on_val_end(self,pl_module: L.LightningModule):
        # Print and log metrics
        super().eval_on_val_end(pl_module)

        self.val_pred.clear()
        self.val_target.clear()

    def make_plots(self, stages: Literal['val', 'test']) -> dict[str, plt.Figure]:
        pred, target = self.concat_pred_target(stages)

        plots = {}
        for tsk, maker_dict in self.plot_makers.items():
            for plot_name, maker in maker_dict.items():
                logging.info(f'Plotting {tsk}/{plot_name} ...')
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

    def __getitem__(self, item: int):
        return self._tasks[item]

    @property
    def current_task_index(self) -> Optional[int]:
        if not self.current_task:
            return None
        return self._tasks.index(self.current_task)

    def __repr__(self):
        return f'MultiDataTask(total={len(self._tasks)}; current={self.current_task_index})'

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

        # self._log_metrics_on_train_batch(pl_module, metrics_dict)
        self._log_metrics(pl_module, metrics_dict, stages='train')

    def summary_metrics(self, pl_module: L.LightningModule, stages: Literal['val', 'test'] = 'val') -> tuple[dict[str, dict[str, float]], dict[str, float]]:
        total_metrics = {}
        primary_values = []
        for i, task in enumerate(self._tasks):
            # metrics_dict = task.summary_metrics(pl_module, stages)
            pred, target = task.concat_pred_target(stages)
            metrics_dict, primary = task.calc_metrics(pred, target, get_primary=True)
            primary_values.extend(primary.values())

            for tsk_name, tsk_mtrc_dict in metrics_dict.items():
                assert isinstance(tsk_mtrc_dict, dict)
                total_metrics[f'{i}-{tsk_name}'] = tsk_mtrc_dict

            for tsk_name, tsk_mtrc_dict in metrics_dict.items():
                if isinstance(tsk_mtrc_dict, dict):
                    total_metrics[f'{i}-{tsk_name}'] = tsk_mtrc_dict
                else:
                    total_metrics[f'{i}-{tsk_name}'] = {tsk_name: tsk_mtrc_dict}

        total_metrics['smtrc'] = {'smtrc': sum(primary_values) / len(primary_values)}
        if stages == 'val':
            logging.debug(f'[#3f51b5]The Optimizers: {pl_module.optimizers()}[/]')
            total_metrics['lr'] = {'lr': pl_module.optimizers().param_groups[0]['lr']}
        return total_metrics, primary_values

    def eval_on_val_end(self, pl_module: L.LightningModule):
        total_metrics, _ = self.summary_metrics(pl_module, stages='val')
        self._log_metrics(pl_module, total_metrics, stages='val')

    def make_plots(self, stages: Literal['val', 'test']) -> dict[str, plt.Figure]:
        plots = {}
        for i, task in enumerate(self._tasks):
            tsk_plots = task.make_plots(stages)
            for plot_name, plot in tsk_plots.items():
                plots[f'Dataset{i}/{plot_name}'] = plot

        return plots

# Set all abstractmethod to do current tasks
for abc_method in MultiDataTask.__abstractmethods__:
    setattr(MultiDataTask, abc_method, _perform_current_task(getattr(MultiDataTask, abc_method)))
MultiDataTask.__abstractmethods__ = frozenset([])
######################################################################################

