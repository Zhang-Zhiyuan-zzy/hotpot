"""
@File Name:        config_task
@Project:          
@Author:           Zhiyuan Zhang
@Created On:       2025/11/29 20:41
@Project:          Hotpot
"""
import copy
import logging
from typing import *
from operator import or_
from functools import wraps, reduce, cached_property, partial

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from dataclasses import dataclass, asdict
from typing import Any, Optional, Union, Sequence, Callable, get_args

from hotpot.plugins.opti.params_space import ParamSets
from . import (
    types as tp,
    models as M,
    tasks,
    tools
)
from .run.datacls import ConfigArgs


#################################### Options dict ##############################
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
    'amd': M.Metrics.average_inverse_distance,
    'precision': M.Metrics.precision,
    'recall': M.Metrics.recall,
    'f1': M.Metrics.f1_score,
    'mf1': M.Metrics.mf1_score,
    'auc': M.Metrics.auc,
}
predictor_plot_maker_map = {
    'num': ('r2',),
    'onehot': ('conf', 'mroc'),
    'binary': ('bconf', 'roc', 'det', 'prc'),
    'xyz': ('hist',)
}

loss_options = {
    'mse': F.mse_loss,
    'cross_entropy': M.LossMethods.calc_atom_type_loss,
    'binary_cross_entropy': F.binary_cross_entropy_with_logits,
    'amd': M.LossMethods.average_maximum_displacement
}
x_masker_options = {
    'atom': M.mask_atom_type,
    'metal': M.mask_atom_type  # TODO: ?
}

# Contract
INPUT_X_ATTR = ('atomic_number', 'n', 's', 'p', 'd', 'f', 'g', 'x', 'y', 'z')
COORD_X_ATTR = ('x', 'y', 'z')

######################################## Main Interface ##################################################
def config_task(cfg: ConfigArgs):
    builder = TaskBuilder(cfg)
    return builder.build_task()

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
        arg_name: str, arg: dict, task_names: Union[set, list, tuple],
        value_judge: Callable[[Any], bool],
        strict_align: bool = True
):
    """ Align the arg dict to task_names """
    assert isinstance(arg, dict), f"Arg {arg_name} should be a dict, but got {type(arg)}"
    if strict_align:
        assert len(arg) == len(task_names), f"Arg {arg_name} should be equal to task_names, but {len(arg)} != {len(task_names)}"
    assert all(k in task_names for k in arg.keys()), f"All keys of {arg_name} dict should be in {task_names}"
    assert all(value_judge(v) for v in arg.values()), f"All values of {arg_name} should satisfy the demand of {value_judge}"


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

def _config_onehot_predictor(
        core: M.CoreBase,
        onehot_type=None,
        hid_dim: int = 1024,
        num_layers: int = 4,
        **kwargs
):
    if not onehot_type:
        print(RuntimeWarning(
            'The default `onehot_type=119` for onehot predictor,'
            'but explicitly specify the onehot_type is recommended.'))
        onehot_type = 119
    return M.Predictor(
        core.vec_size,
        'onehot',
        onehot_type=onehot_type,
        hidden_dim=hid_dim,
        num_layers=num_layers,
        **kwargs
    )

# Predictor
def _str2callable_predictor(
        name: str,
        predictor: Union[str, Callable],
        core: M.CoreBase,
        onehot_type=None,
        hid_dim: int = 256,
        num_layers: int = 1,
        out_size: int = 1024,
        **kwargs
):
    if isinstance(predictor, Callable):
        logging.info(f'Predictor[{name}]=[blue]Callable[/]')
        return predictor
    elif isinstance(predictor, str):
        if predictor == 'onehot':
            logging.info(f"Predictor[{name}]=[blue]{predictor}[/]")
            return _config_onehot_predictor(core, onehot_type, hid_dim, num_layers, name=name, **kwargs)
        elif predictor in ('xyz', 'binary', 'num'):
            logging.info(f"Predictor[{name}]=[blue]{predictor}[/]")
            return M.Predictor(
                core.vec_size,
                predictor,
                hidden_dim=hid_dim,
                num_layers=num_layers,
                out_size=out_size,
                name=name,
                **kwargs
            )

        else:
            raise ValueError(f'Unknown predictor type: {predictor}')
    else:
        raise TypeError(f'The predictor should be a callable or a str.')


############################ Config Multi-data tasks arguments #############################
def _align_md_task_options(name, arg: Any, dataset_counts: int, lst_values: bool = False):
    """
    Align a parameter `arg` to match the number of datasets (Tasks).
    - list/tuple: length must match dataset_counts
    - None: expands to [None]*dataset_counts
    - others: replicated for all datasets
    Always returns a list.
    """
    if isinstance(arg, (list, tuple)):
        if not lst_values:
            assert len(arg) == dataset_counts, f'Expecting {name} has same length as task_counts, but {len(arg)} != {dataset_counts}'
            return list(arg)
        else:
            # There could be a bug when the structure of arg is an nested list or tuple, like value=[[[Any]]]
            # However, it can be foreseen that this situation is very rare, so I temporarily ignore it.
            # If the bug met, please modify the `lst_values: bool` to `lst_deep: int`
            if all(isinstance(a[0], (list, tuple)) for a in arg):
                assert len(arg) == dataset_counts, f'Expecting {name} has same length as task_counts, but {len(arg)} != {dataset_counts}'
                return list(arg)
            else:
                return [arg] * dataset_counts
    else:
        return [arg] * dataset_counts

def _extract_args_by_task_name(
        arg_name: str, arg: dict, task_names: Union[set, list, tuple],
        # default_value: Optional[Any] = None,
) -> dict[str, Any]:
    _values = {}
    for tsk_name in task_names:
        # if default_value is None and tsk_name not in arg:
        #     raise ValueError(f"Arg {arg_name} has not been set for task {tsk_name}")
        if (value := arg.get(tsk_name, None)) is not None:
            _values[tsk_name] = value
    return _values

def _extract_options_list(
        arg_name: str,
        list_task_names: list[list[str]],
        arg: Union[list, tuple, dict],
        dataset_counts: int,
        default_values: Optional[dict] = None,
) -> list[Any]:
    """
    This auxiliary function splitting global task-specific arguments to dataset-task-wise arguments.
    This function is useful when your arguments just are differentiated by the task signature, on
    matter which dataset to be applied. In the case, the user can just define a dataset-independent
    dict: {`task_name`: arg_value}, this function assign these values in to each dataset-specific
    tasks, according to the `list_task_names`.
    """
    assert len(list_task_names) == dataset_counts, (
        f"The number of `task_names` groups do not equal the dataset counts in extracting {arg_name}")
    if isinstance(arg, dict):
        return [_extract_args_by_task_name(arg_name, arg, task_names) for task_names in list_task_names]
    else:
        return list(arg)

# Add metric as part of loss_fn
# Wrapper decorator
def wrap_loss_fn(lofn: tp.LossFn, weight: float, mtr: Callable) -> tp.Callable:
    logging.info(f'[blue]Wrap metric in to loss_fn: loss_fn: {lofn}, weight: {weight}, mtr: {mtr}[/]')
    def wrapped_lofn(pred, tgt, acc=None):
        return lofn(pred, tgt) + weight * (1. - mtr(pred, tgt))

    return wrapped_lofn


# 1. 定义标准化的输出对象 (替代原本那个弱类型的 dict)
@dataclass
class TaskComponents:
    """
    这是一个单纯的数据容器，存放构建好的所有组件。
    下游代码可以用 task.predictor 而不是 task['predictor'] 来访问。
    """
    task_names: list[str]
    target_getter: dict[str, tools.TargetGetter]
    feature_extractor: dict[str, M.AttnExtractor]
    predictors: dict[str, M.Predictor]
    loss_fn: dict[str, tp.LossFn]
    primary_metrics: dict[str, str]
    metrics: dict[str, M.Metrics]
    plot_makers: dict
    x_masker: Optional[Callable]
    mask_need_task: Sequence[str]
    loss_weight_calculator: Optional[Callable]
    inputs_preprocessor: Callable
    batch_preprocessor: Any
    inputs_getter: Any
    xyz_index: Iterable[int]
    xyz_perturb_mode: str
    xyz_perturb_sigma: float
    hypers: ParamSets
    with_sol: bool
    with_med: bool
    with_env: bool
    sol_graph_inputs: Iterable[str]
    med_graph_inputs: Iterable[str]
    kwargs: dict

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# 2. 定义 TaskBuilder 类
class _Builder:
    _default_mask_task = ['AT']

    def __init__(self, cfg: ConfigArgs, core, first_data: Data, xyz_index: Iterable[int]):
        self.cfg = cfg
        self.core = core
        self.first_data = first_data
        self.xyz_index = xyz_index
        self.task_names = list(self.cfg.target_getter.keys())

    @cached_property
    def build_task(self):
        target_getter = self._specify_target_getter()

        # Specify default predictor
        predictor = self._specify_predictors()

        # Specify default feature extractor
        feature_extractor = self._specify_feature_extractor()

        # Specify loss func
        loss_fn = self._specify_loss_fn()

        # Specify primary metric
        primary_metrics, metrics = self._specify_metrics()

        # Specify test plot makers
        plot_makers = self._specify_test_plot_maker(predictor)

        ####################### Important Args #####################
        ############################################################

        ####################### Optional Args ######################
        # Wrap the loss_fn accounting the metrics into
        # Notes: the wrapped task name must be given by set of str, otherwise not work
        loss_fn = self._wrap_loss_fn_with_metric(loss_fn, metrics, primary_metrics)

        # Specify x masker
        x_masker = self._specify_masker()
        if self.cfg.mask_need_task is None:
            mask_need_task = [t for t in self.task_names if t in self._default_mask_task]
        elif isinstance(self.cfg.mask_need_task, (list, tuple)):
            _align_task_seq('mask_need_task', self.cfg.mask_need_task, self.task_names)
            mask_need_task = copy.copy(self.cfg.mask_need_task)
        else:
            mask_need_task = None

        # Loss weight calculator
        loss_weight_calculator = self._specify_loss_weights_calculator(predictor)

        # Input preprocessor
        inputs_preprocessor = self._specify_inputs_preprocessor()

        return TaskComponents(
            task_names=self.task_names,
            target_getter=target_getter,
            feature_extractor=feature_extractor,
            predictors=predictor,
            loss_fn=loss_fn,
            primary_metrics=primary_metrics,
            metrics=metrics,
            plot_makers=plot_makers,
            x_masker=x_masker,
            mask_need_task=mask_need_task,
            loss_weight_calculator=loss_weight_calculator,
            inputs_preprocessor=inputs_preprocessor,
            batch_preprocessor=self.cfg.batch_preprocessor,
            inputs_getter=self.cfg.inputs_getter,
            xyz_index=self.xyz_index,
            xyz_perturb_mode=self.cfg.xyz_perturb_mode,
            xyz_perturb_sigma=self.cfg.xyz_perturb_sigma,
            hypers=self.cfg.hypers,
            with_sol=self.cfg.with_sol,
            with_med=self.cfg.with_med,
            with_env=self.cfg.with_env,
            sol_graph_inputs=self.cfg.sol_graph_inputs,
            med_graph_inputs=self.cfg.med_graph_inputs,
            kwargs=self.cfg.kwargs
        )


    @staticmethod
    def _str2callable_convertor(item_getter: dict[str, Union[Callable, str]], options: dict[str, Any]) -> dict[str, Any]:
        return {n: options[e] if isinstance(e, str) else e for n, e in item_getter.items()}

    @staticmethod
    def _single_calculator(predictor, method, predictor_check: bool = True):
        assert isinstance(predictor, M.Predictor), f"the predictors should be a callable, got {type(predictor)}"
        if predictor.target_type == 'onehot':
            logging.info(f"Configure `onehot` weight calculator for Predictor[{predictor.name}] with types: {predictor.target_type}")
            # return lambda label: M.weight_labels(label, predictor.onehot_type, method)  # Shape of weight: [C]
            return partial(M.weight_labels, num_types=predictor.onehot_type, weight_method=method)
        elif predictor.target_type == 'binary':
            logging.info(f"Configure `binary` weight calculator for Predictor[{predictor.name}]")
            return M.weight_binary  # Shape of weight: [B, 1]
        elif predictor_check:
            raise ValueError(f'The `loss_weights_calculator` just works for predictor `onehot` or `binary`')
        else:
            return None

    def _specify_inputs_preprocessor(self):
        if isinstance(self.cfg.inputs_preprocessor, Callable):
            return self.cfg.inputs_preprocessor
        elif self.core.emb_type == 'atom':
            return wraps(M.get_x_input_attrs)(lambda inp: M.get_x_input_attrs(inp, input_x_index=0, dtype=torch.int))
        elif self.core.emb_type == 'orbital':
            return wraps(M.get_x_input_attrs)(
                lambda inp: M.get_x_input_attrs(inp, input_x_index=list(range(7)), dtype=torch.int))
        else:
            return wraps(M.get_x_input_attrs)(lambda inp: M.get_x_input_attrs(inp, input_x_index=list(range(7))))

    def _specify_all_classifier(
            self, predictors: dict[str, M.Predictor],
            _default_method: tp.LossWeightMethods= 'inverse-count'
    ) -> Optional[dict[str, tp.LossWeightCalculator]]:
        classifiers = {tsk_name: predictor for tsk_name, predictor in predictors.items() if predictor.target_type in ('onehot', 'binary')}
        if not classifiers:
            return None

        _default_method = _default_method or 'inverse-count'
        _align_task_names('predictors', predictors, self.task_names, lambda v: isinstance(v, M.Predictor))
        return {
            tsk_name: self._single_calculator(predictor, _default_method)
            for tsk_name, predictor in classifiers.items()
        }

    def _specify_loss_weights_calculator(self, predictors: dict[str, M.Predictor]) -> Optional[dict[str, tp.LossWeightCalculator]]:
        DEFAULT_METHOD: tp.LossWeightMethods = 'inverse-count'
        if self.cfg.loss_weight_method is False:
            return None

        if self.cfg.loss_weight_method is None or self.cfg.loss_weight_method is True:
            return self._specify_all_classifier(predictors, DEFAULT_METHOD)

        elif isinstance(self.cfg.loss_weight_method, str):
            assert self.cfg.loss_weight_method in get_args(tp.LossWeightMethods)
            return self._specify_all_classifier(predictors, self.cfg.loss_weight_method)

        elif isinstance(self.cfg.loss_weight_method, (list, tuple)):
            _align_task_seq('loss_weight_method', self.cfg.loss_weight_method, self.task_names, strict_align=False)
            _predictors = {tsk_name: predictors[tsk_name] for tsk_name in self.cfg.loss_weight_method}
            return self._specify_all_classifier(_predictors, DEFAULT_METHOD)

        elif isinstance(self.cfg.loss_weight_method, dict):
            _align_task_names(
                'loss_weight_method', self.cfg.loss_weight_method, self.task_names,
                value_judge=lambda v: isinstance(v, (str, Callable)), strict_align=False)

            return {
                tsk_name: (
                    method if isinstance(method, Callable)
                    else self._single_calculator(predictors[tsk_name], method)
                )
                for tsk_name, method in self.cfg.loss_weight_method.items()
            }
        else:
            raise TypeError(f'The `loss_weight_calculator` should be optional boolean, str, '
                            f'callable or dict, got {type(self.cfg.loss_weight_method)}')

    # Specify x masker
    def _specify_masker(self):
        if self.cfg.x_masker is False:
            return None

        elif self.cfg.x_masker is None:
            if isinstance(self.task_names, (list, tuple)):  # default to mask atom types
                return lambda inp: M.mask_atom_type(inp, self.core.node_mask)
            else:
                return None

        elif isinstance(self.cfg.x_masker, Callable):
            return self.cfg.x_masker

        elif isinstance(self.cfg.x_masker, str):
            mask_func = x_masker_options[self.cfg.x_masker]
            return lambda inp: mask_func(inp, self.core.node_mask)

        else:
            raise TypeError(f'The `x_masker` should be a callable or a str')

    def _wrap_loss_fn_with_metric(
            self,
            loss_fn: dict[str, tp.LossFn],
            metrics: dict[str, dict[str, tp.MetricFn]],
            primary_metrics: dict[str, str],
    ):
        # Not wrapping
        if self.cfg.lofn_wrap_tasks is None:
            return loss_fn

        if isinstance(self.cfg.lofn_wrap_tasks, bool):
            lofn_wrap_weights = {tsk_n: 1.0 for tsk_n in loss_fn}
        elif isinstance(self.cfg.lofn_wrap_tasks, float):
            lofn_wrap_weights = {tsk_n: self.cfg.lofn_wrap_tasks for tsk_n in loss_fn}
        else:
            _align_task_names('lofn_wrap_tasks', self.cfg.lofn_wrap_tasks, self.task_names, lambda v: isinstance(v, float), strict_align=False)
            lofn_wrap_weights = copy.copy(self.cfg.lofn_wrap_tasks)

        _align_task_names('loss_fn', loss_fn, self.task_names, lambda v: isinstance(v, Callable))
        _align_task_names('metrics', metrics, self.task_names, lambda v: isinstance(v, dict))
        _align_task_names('primary_metrics', primary_metrics, self.task_names, lambda v: isinstance(v, str))

        _primary_metrics_fn = {
            tsk_name: metrics[tsk_name][p_mtr_name]
            for tsk_name, p_mtr_name in primary_metrics.items()
        }

        return {
            tsk_name: wrap_loss_fn(lofn, lofn_wrap_weights[tsk_name], _primary_metrics_fn[tsk_name])
            for tsk_name, lofn in loss_fn.items()
        }

    # Test plot maker
    def _specify_test_plot_maker(
            self,
            predictors: Union[M.Predictor, dict[str, M.Predictor]],
    ) -> dict[str, tp.PlotMaker]:
        assert isinstance(predictors, dict), "The `predictors` should be a dict"
        plot_makers = {}
        for tsk, predictor in predictors.items():
            plot_name = predictor_plot_maker_map[predictor.target_type]
            plot_makers[tsk] = {n: M.plots_options[n] for n in plot_name}
        _align_task_names("plot_makers", plot_makers, self.task_names, lambda v: isinstance(v, dict))
        return plot_makers

    # Metrics
    def _specify_metrics(self) -> tuple[dict[str, str], dict[str, dict[str, tp.MetricFn]]]:
        # Specify primary metrics
        assert isinstance(self.cfg.primary_metrics, dict), 'The `primary_metrics` should be a dict`'
        _align_task_names('primary_metrics', self.cfg.primary_metrics, self.task_names, lambda v: isinstance(v, str))
        primary_metrics = copy.copy(self.cfg.primary_metrics)
        _metrics = {tn: {pmn: metrics_options[pmn]} for tn, pmn in primary_metrics.items()}

        # Specify other metrics
        if self.cfg.other_metrics is not None:
            assert isinstance(self.cfg.other_metrics, dict), "`other_metrics` should be a dict"
            _align_task_names(
                'other_metrics', self.cfg.other_metrics, self.cfg.task_names, lambda v: isinstance(v, (dict, tuple, list)),
                strict_align=False)
            for tsk_name, tsk_metric in self.cfg.other_metrics.items():
                if isinstance(tsk_metric, dict):
                    _metrics[tsk_name].update(self._str2callable_convertor(self.cfg.other_metrics, metrics_options))
                else:
                    _metrics[tsk_name].update({omn: metrics_options[omn] for omn in tsk_metric})

        return primary_metrics, _metrics

    # Loss_fn
    def _specify_loss_fn(self):
        if isinstance(self.cfg.loss_fn, (list, tuple)):
            _align_task_seq('loss_fn', self.cfg.loss_fn, self.task_names, lambda g: isinstance(g, (Callable, str)))
            return self._str2callable_convertor(dict(zip(self.task_names, self.cfg.loss_fn)), loss_options)

        elif isinstance(self.cfg.loss_fn, dict):
            _align_task_names('loss_fn', self.cfg.loss_fn, self.task_names, lambda g: isinstance(g, (Callable, str)))
            return self._str2callable_convertor(self.cfg.loss_fn, loss_options)

        else:
            raise ValueError(
                f'The loss_fn should be a string, a callable, a sequence of '
                f'or a dict of str and Callable, not {type(self.cfg.loss_fn)}')

    def _specify_feature_extractor(self):
        """"""
        if isinstance(self.cfg.feature_extractor, (list, tuple)):
            _align_task_seq('extractors', self.cfg.feature_extractor, self.task_names, lambda g: isinstance(g, (Callable, str)))
            return self._str2callable_convertor(dict(zip(self.task_names, self.cfg.feature_extractor)), self.core.feature_extractor)

        elif isinstance(self.cfg.feature_extractor, dict):
            _align_task_names('extractors', self.cfg.feature_extractor, self.task_names, lambda g: isinstance(g, (Callable, str)))
            return self._str2callable_convertor(self.cfg.feature_extractor, self.core.feature_extractor)

        else:
            raise ValueError(
                f'The extractors should be a string, a callable, a sequence of '
                f'or a dict of str or Callable, not {type(self.cfg.feature_extractor)}')


    def _specify_target_getter(self):
        if isinstance(self.cfg.target_getter, dict):
            _align_task_names('target_getter', self.cfg.target_getter, self.task_names, lambda g: isinstance(g, (Callable, dict)))
            target_getter = copy.copy(self.cfg.target_getter)
        elif isinstance(self.cfg.target_getter, (list, tuple)):
            _align_task_seq('target_getter', self.cfg.target_getter, self.task_names, lambda g: isinstance(g, (Callable, dict)))
            target_getter = dict(zip(self.task_names, self.cfg.target_getter))
        else:
            raise TypeError('Expecting target_getter to be a dict or Sequence of Callable, '
                            f'when task_names is a Sequence, got {type(self.cfg.target_getter)}')

        # Convert the target_getter dict definition to TargetGetter instance
        for name, getter in target_getter.items():
            if isinstance(getter, dict):
                target_getter[name] = tools.specify_target_getter(self.first_data, **getter)

        return target_getter

    def _specify_predictors(self):
        """
        Normalize variable input to the format `Callable|dict[task_name, Callable]`,
        that can be used directly by `LightPretrain` and `pl.Trainer`
        """
        if isinstance(self.cfg.predictor, (M.Predictor, Callable, str)):
            return _str2callable_predictor(
                self.cfg.work_name,
                self.cfg.predictor,
                self.core,
                self.cfg.onehot_type,
                hid_dim=getattr(self.cfg.hypers, 'HIDDEN_DIM', 256),
                num_layers=getattr(self.cfg.hypers, 'HIDDEN_LAYERS', 1),
                out_size=getattr(self.cfg.hypers, 'OUT_SIZE', 1024),
                **self.cfg.kwargs
            )

        if isinstance(self.cfg.predictor, (list, tuple)):
            # If the given predictor is a Sequence, convert the Sequence one to dict one.
            _align_task_seq(
                'Predictor', self.cfg.predictor, self.task_names,
                lambda p: isinstance(p, (str, Callable, M.Predictor)))
            predictors = dict(zip(self.task_names, self.cfg.predictor))
        else:
            assert isinstance(self.cfg.predictor, dict)
            predictors = copy.copy(self.cfg.predictor)

        # Regularize the format of predictor
        if isinstance(predictors, dict):
            onehot_type = copy.copy(self.cfg.onehot_type) if isinstance(self.cfg.onehot_type, dict) else {}
            _align_task_names('onehot_type', onehot_type, list(predictors.keys()), lambda v: isinstance(v, int), strict_align=False)

            return {
                name: _str2callable_predictor(
                    name,
                    predictor, self.core,
                    onehot_type.get(name, None),
                    hid_dim=getattr(self.cfg.hypers, 'HIDDEN_DIM', 256),
                    num_layers=getattr(self.cfg.hypers, 'HIDDEN_LAYERS', 1),
                    out_size=getattr(self.cfg.hypers, 'OUT_SIZE', 1024),
                    **self.cfg.kwargs
                )
                for name, predictor in self.cfg.predictor.items()}
        else:
            raise TypeError(f"`predictors` should be a str|Callable, or a Sequence|dict of str|Callable")


class TaskBuilder:
    INPUT_X_ATTR = ('atomic_number', 'n', 's', 'p', 'd', 'f', 'g', 'x', 'y', 'z')
    COORD_X_ATTR = ('x', 'y', 'z')

    def __init__(self, cfg: ConfigArgs):
        """"""
        self.core = self._init_core(cfg.hypers)
        self.task_type = self._specify_task_type(cfg)
        self.cfg = cfg

        self.first_data = cfg.dataModule.first_data
        self.xyz_index = self._specify_xyz_index(self.first_data, cfg.with_xyz)

        if self.task_type in (tasks.SingleTask, tasks.MultiTask):
            assert not cfg.dataModule.is_multi_datasets
            self.builder = _Builder(cfg, self.core, self.first_data, self.xyz_index)
        elif self.task_type is tasks.MultiDataTask:
            assert cfg.dataModule.is_multi_datasets
            self.builder = self._align_arguments(cfg, self.core, self.first_data, self.xyz_index)
        else:
            raise NotImplementedError(f'Unsupported task type: {self.task_type}')

    def _build_all(self):
        ...


    @staticmethod
    def _align_arguments(cfg, core, first_data, xyz_index):
        # Aligning requirements
        dataset_counts = cfg.dataModule.dataset_counts
        assert len(cfg.feature_extractor) == dataset_counts
        assert len(cfg.target_getter) == dataset_counts
        assert len(cfg.loss_fn) == dataset_counts
        assert len(cfg.primary_metrics) == dataset_counts
        assert len(first_data) == dataset_counts
        # if isinstance(cfg.inputs_getter, Callable):
        #     inputs_getter = [cfg.inputs_getter] * dataset_counts
        # elif isinstance(cfg.inputs_getter, list):
        #     assert len(cfg.inputs_getter) == dataset_counts
        #     inputs_getter = cfg.inputs_getter
        # else:
        #     raise TypeError('inputs_getter must be a callable or a list of callable')

        # Align task-specific options to the number of datasets.
        #
        # Each dataset corresponds to one Task instance, and each Task requires
        # a full set of parameter configs. Many parameters are identical across
        # Tasks, so instead of repeating the same values, users can provide a
        # single value. This code expands shared values to all Tasks automatically.
        #
        # If a parameter differs across Tasks, the number of provided values must
        # match the number of datasets; otherwise, an error is raised.
        #
        # Example:
        #   Suppose there are 3 datasets:
        #       xyz_index = True
        #   After alignment:
        #       xyz_index = [True, True, True]
        inputs_getter = _align_md_task_options('inputs_getter', cfg.inputs_getter, dataset_counts)
        task_names = _align_md_task_options('task_names', cfg.task_names, dataset_counts)
        batch_preprocessor = _align_md_task_options('batch_preprocessor', cfg.batch_preprocessor, dataset_counts)
        inputs_preprocessor = _align_md_task_options('inputs_preprocessor', cfg.inputs_preprocessor, dataset_counts)
        xyz_index = _align_md_task_options('xyz_index', xyz_index, dataset_counts)
        with_sol = _align_md_task_options('with_sol', cfg.with_sol, dataset_counts)
        with_med = _align_md_task_options('with_med', cfg.with_med, dataset_counts)
        with_env = _align_md_task_options('with_env', cfg.with_env, dataset_counts)
        xyz_perturb_sigma = _align_md_task_options('xyz_perturb_sigma', cfg.xyz_perturb_sigma, dataset_counts)
        xyz_perturb_mode = _align_md_task_options('xyz_perturb_mode', cfg.xyz_perturb_mode, dataset_counts)
        extractor_attr_getter = _align_md_task_options('extractor_attr_getter', cfg.extractor_attr_getter, dataset_counts)
        lofn_wrap_tasks = _align_md_task_options('lofn_wrap_tasks', cfg.lofn_wrap_tasks, dataset_counts, lst_values=True)
        loss_weight_calculator = _align_md_task_options('loss_weight_method', cfg.loss_weight_method, dataset_counts)
        onehot_types = _align_md_task_options('onehot_types', cfg.onehot_types, dataset_counts)
        x_masker = _align_md_task_options('x_masker', cfg.x_masker, dataset_counts)
        mask_need_task = _align_md_task_options('mask_need_task', cfg.mask_need_task, dataset_counts)
        hypers = _align_md_task_options('hypers', cfg.hypers, dataset_counts)
        # to_onehot = [(list(oh_types) if isinstance(oh_types, dict) else bool(oh_types)) for oh_types in onehot_types]

        # Extract list of dict from define task-wise dict
        other_metrics = _extract_options_list('other_metrics', task_names, cfg.other_metrics, dataset_counts, {})
        ############################## End of Aligning #######################################

        builders = []
        for i in range(dataset_counts):
            _cfg = ConfigArgs(
                work_name=cfg.work_name,
                task_names=task_names[i],
                inputs_preprocessor=inputs_preprocessor[i],
                target_getter=cfg.target_getter[i],
                feature_extractor=cfg.feature_extractor[i],
                hypers=hypers[i],
                predictor=cfg.predictor[i],
                loss_fn=cfg.loss_fn[i],
                primary_metrics=cfg.primary_metrics[i],
                other_metrics=other_metrics[i],
                x_masker=x_masker[i],
                mask_need_task=mask_need_task[i],
                loss_weight_method=loss_weight_calculator[i],
                lofn_wrap_tasks=lofn_wrap_tasks[i],
                batch_preprocessor=batch_preprocessor[i],
                inputs_getter=inputs_getter[i],
                xyz_perturb_sigma=xyz_perturb_sigma[i],
                xyz_perturb_mode=xyz_perturb_mode[i],
                # to_onehot=to_onehot[i],
                extractor_attr_getter=extractor_attr_getter[i],
                with_sol=with_sol[i],
                with_med=with_med[i],
                with_env=with_env[i],
            )
            builders.append(_Builder(_cfg, core, first_data[i], xyz_index[i]))

        return builders

    @staticmethod
    def _init_core(hypers: ParamSets) -> M.CoreBase:
        return M.Core(
            vec_dim=hypers.VEC_DIM,
            emb_type=hypers.EMB_TYPE,
            # x_label_nums=hypers.ATOM_TYPES,
            ring_layers=hypers.RING_LAYERS,
            ring_nheads=hypers.RING_HEADS,
            ring_encoder_kw={'dim_feedforward': hypers.DIM_FEEDFORWARD},
            mol_layers=hypers.MOL_LAYERS,
            mol_nheads=hypers.MOL_HEADS,
            mol_encoder_kw={'dim_feedforward': hypers.DIM_FEEDFORWARD},
            graph_layer=hypers.GRAPH_LAYERS,
            med_props_nums=22,
            sol_props_nums=34,
            with_sol_encoder=True,
            with_med_encoder=True,
        )

    @staticmethod
    def _specify_task_type(cfg: ConfigArgs):
        return tasks.specify_task_types(cfg.dataModule.is_multi_datasets, cfg.target_getter)

    def _specify_xyz_index(
            self,
            first_data: Union[Data, Iterable[Data]],
            with_xyz: Optional[Union[bool, Iterable[bool]]]
    ):
        if isinstance(first_data, Data):
            if with_xyz:
                return tools.get_index(first_data, 'x', self.COORD_X_ATTR)
            return None

        elif isinstance(first_data, Iterable):
            first_data = list(first_data)
            if not with_xyz:
                return [None] * len(first_data)
            elif with_xyz is True:
                return [tools.get_index(fd, 'x', self.COORD_X_ATTR) for fd in first_data]
            elif isinstance(with_xyz, Iterable):
                with_xyz = list(with_xyz)
                assert len(first_data) == len(with_xyz), (
                    f'Expected len(first_data) == len(with_xyz), got {len(first_data)} != {len(with_xyz)}')

                return [
                    tools.get_index(fd, 'x', self.COORD_X_ATTR) if opt is True else None
                    for fd, opt in zip(first_data, with_xyz)
                ]
            else:
                raise TypeError(f'The `with_xyz` should be a bool or a iterable of bool')

        else:
            raise TypeError(f'The `first_data` should be a Data, Iterable, or None')

    def build_task(self) -> tuple[M.CoreBase, tasks.BaseTask, dict[str, M.Predictor]]:
        # Initialize Task object
        if self.task_type is tasks.MultiDataTask:
            assert isinstance(self.builder, list)
            components = [builder.build_task.to_dict() for builder in self.builder]
            task = self.task_type(list_kwargs=components)
            predictors = reduce(or_, [builder.build_task.predictors for builder in self.builder])
        else:
            assert isinstance(self.builder, _Builder)
            task = self.task_type(**self.builder.build_task.to_dict())
            predictors = self.builder.build_task.predictors

        task.show_pbar = self.cfg.show_pbar
        task.hypers = self.cfg.hypers

        return self.core, task, predictors
