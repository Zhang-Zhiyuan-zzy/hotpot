import logging
from typing import *
from functools import wraps

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from hotpot.plugins.opti.params_space import ParamSets
from . import (
    types as tp,
    models as M,
    tasks,
    tools,
)
from .data import (
    DataModule
)
from ...utils import fmt_print

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
    'metal': M.mask_atom_type
}

# Contract
INPUT_X_ATTR = ('atomic_number', 'n', 's', 'p', 'd', 'f', 'g', 'x', 'y', 'z')
COORD_X_ATTR = ('x', 'y', 'z')


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
                XYZ_INDEX = tools.get_index(first_data, 'x', ('x', 'y', 'z'))
                target_getter = lambda batch: batch.x[:, XYZ_INDEX]
            elif task_names == 'AtomType':
                ATOM_TYPE_INDEX = tools.get_index(first_data, 'x', 'atomic_number')
                target_getter = lambda batch: batch.x[:, ATOM_TYPE_INDEX]
            elif task_names == 'AtomCharge':
                ATOM_CHRG_INDEX = tools.get_index(first_data, 'x', 'partial_charge')
                target_getter = lambda batch: batch.x[:, ATOM_CHRG_INDEX]

        # Check the final target_getter
        assert isinstance(target_getter, Callable), (
            f"Expecting target_getter to be a callable when task_names is a str, but got {type(target_getter)}")

    elif isinstance(task_names, list):
        if isinstance(target_getter, dict):
            _align_task_names('target_getter', target_getter, task_names, lambda g: isinstance(g, (Callable, dict)))
        elif isinstance(target_getter, (list, tuple)):
            _align_task_seq('target_getter', target_getter, task_names, lambda g: isinstance(g, (Callable, dict)))
            target_getter = dict(zip(task_names, target_getter))
        else:
            raise TypeError('Expecting target_getter to be a dict or Sequence of Callable, '
                            f'when task_names is a Sequence, got {type(target_getter)}')

        # Convert the target_getter dict definition to TargetGetter instance
        for name, getter in target_getter.items():
            if isinstance(getter, dict):
                target_getter[name] = tools.specify_target_getter(first_data, **getter)
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
def _str2callable_predictor(
        predictor: Union[str, Callable],
        core: M.CoreBase,
        onehot_type=None,
        hid_dim: int = 1024,
        num_layers: int = 4,
        **kwargs
):
    if isinstance(predictor, Callable):
        return predictor
    elif isinstance(predictor, str):
        if predictor == 'onehot':
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
        elif predictor in ('xyz', 'binary', 'num'):
            return M.Predictor(
                core.vec_size,
                predictor,
                hidden_dim=hid_dim,
                num_layers=num_layers,
                **kwargs
            )
            # return M.Predictor.link_with_core_module(core, predictor, **kwargs)
        else:
            raise ValueError(f'Unknown predictor type: {predictor}')
    else:
        raise TypeError(f'The predictor should be a callable or a str.')

def _specify_predictors(
        task_name: Union[str, Sequence[str]],
        core: M.CoreBase,
        predictors: tp.PredictorInput,
        hypers: ParamSets,
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
        return _str2callable_predictor(
            predictors,
            core,
            onehot_type,
            hid_dim=getattr(hypers, 'HIDDEN_DIM', 1024),
            num_layers=getattr(hypers, 'HIDDEN_LAYERS', 4),
            **kwargs
        )

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
            name: _str2callable_predictor(
                predictor, core,
                onehot_type.get(name, _default_oht),
                hid_dim=getattr(hypers, 'HIDDEN_DIM', 1024),
                num_layers=getattr(hypers, 'NUM_LAYERS', 4),
                **kwargs
            )
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

def _metric_name_func_convert(metric_or_name: Union[str, Callable]):
    return metric_or_name if isinstance(metric_or_name, Callable) else metrics_options[metric_or_name]

# Metrics
def _specify_metrics(
        task_names: Union[str, Sequence[str]],
        primary_metrics: Union[str, Sequence[str], dict[str, str]],
        other_metrics: Union[str, Sequence[str], dict[str, Union[str, Callable]]],
        predictors: Union[M.Predictor, dict[str, M.Predictor]],
) -> (Union[str, dict[str, str]], dict[str, Union[tp.MetricFn, dict[str, tp.MetricFn]]]):
    _metrics = {}

    # Specify primary_metric name
    if primary_metrics is None:
        assert isinstance(task_names, str), (f'In Multi-task mode, the primary_metrics should be '
            'specified explicitly, but got None of `primary_metrics`')

        # TODO: deprecated
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
        # TODO: deprecated

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
            _metrics.update({_metric_name_func_convert(om) for om in other_metrics})
        elif isinstance(primary_metrics, dict):  # In multi task
            _align_task_names(
                'other_metrics', other_metrics, task_names, lambda v: isinstance(v, (dict, tuple, list)), strict_align=False)
            for tsk_name, tsk_metric in other_metrics.items():
                if isinstance(tsk_metric, dict):
                    _metrics[tsk_name].update({omn: _metric_name_func_convert(om) for omn, om in tsk_metric.items()})
                else:
                    _metrics[tsk_name].update({omn: metrics_options[omn] for omn in tsk_metric})
        else:
            raise RuntimeError(f'the primary_metrics fails to specify')

    # Return
    return primary_metrics, _metrics

# Add metric as part of loss_fn
# Wrapper decorator
def wrap_loss_fn(lofn: tp.LossFn, weight: float, mtr: Callable) -> tp.Callable:
    logging.info(f'[blue]Wrap metric in to loss_fn: loss_fn: {lofn}, weight: {weight}, mtr: {mtr}[/]')
    def wrapped_lofn(pred, tgt, acc=None):
        return lofn(pred, tgt) + weight * (1. - mtr(pred, tgt))

    return wrapped_lofn

def _wrap_loss_fn_with_metric(
        task_names: Union[str, list[str], tuple[str]],
        loss_fn: Union[tp.LossFn, dict[str, tp.LossFn]],
        metrics: dict[str, Union[tp.MetricFn]],
        primary_metrics: Union[str, dict[str, str]],
        lofn_wrap_tasks: Optional[Union[bool, str, set[str]]],
        lofn_wrap_metric_names: Optional[Union[str, dict[str, str]]] = None,
        lofn_wrap_metric_weights: Optional[Union[float, dict[str, float]]] = None,
        **kwargs
):
    # SingleTask mode, the `primary_metrics` should be a str
    if isinstance(primary_metrics, str):
        assert isinstance(task_names, str), (f'task_names is a str which indicates a SingleTask mode, '
                                             f'do not match the type of primary_metrics `{type(primary_metrics)}`')
        assert lofn_wrap_tasks is True or (isinstance(lofn_wrap_tasks, str) and lofn_wrap_tasks == task_names)
        assert isinstance(loss_fn, Callable), "the `loss_fn` should be a callable in SingleTask mode"

        if lofn_wrap_metric_weights is None:
            lofn_wrap_metric_weights = 1.0
        assert isinstance(lofn_wrap_metric_weights, (float, int)), 'The `wrap_metric_weights` should be a float in SingleTask mode'
        assert primary_metrics in metrics and isinstance(metrics[primary_metrics], Callable)

        # Specify the wrapping metrics for loss_fn:
        #     1. if the `lofn_wrap_metric_names` is given as a str, the wrapping metric the give name.
        #     2. if the `lofn_wrap_metric_names` is not given, the primary metrics would be the one.
        assert lofn_wrap_metric_names is None or isinstance(lofn_wrap_metric_names, str)
        if isinstance(lofn_wrap_metric_names, str):
            assert lofn_wrap_metric_names in metrics
            wrap_mtrc = lofn_wrap_metric_names
        else:
            wrap_mtrc = primary_metrics

        # Wrap the loss_fn
        return wrap_loss_fn(loss_fn, lofn_wrap_metric_weights, metrics[wrap_mtrc])

    # MultiTask mode
    elif isinstance(primary_metrics, dict):
        _align_task_names('primary_metrics', primary_metrics, task_names, lambda v: isinstance(v, str))
        _align_task_names('metrics', metrics, task_names, lambda v: isinstance(v, dict))
        _align_task_names('loss_fn', loss_fn, task_names, lambda v: isinstance(v, Callable))
        assert all(pmn in metrics[tn] for tn, pmn in primary_metrics.items())

        # Specify the metric-wrapping Weight
        if lofn_wrap_metric_weights is None:
            lofn_wrap_metric_weights = {}
            _default_weight = 1.0
        elif isinstance(lofn_wrap_metric_weights, (float, int)):
            _default_weight = float(lofn_wrap_metric_weights)
            lofn_wrap_metric_weights = {}
        elif isinstance(lofn_wrap_metric_weights, dict):
            assert all(wtn in task_names for wtn in lofn_wrap_metric_weights), (
                f'not all keys in the `wrap_metric_weights` in the `task_names`,\n'
                f'Metric_weight_names: {lofn_wrap_metric_weights.keys()}\n'
                f'Task_names: {task_names}')
            assert all(isinstance(wtv, (float, int)) for wtv in lofn_wrap_metric_weights.values()), (
                f'all values in the `wrap_metric_weights` should be a float or int')
            _default_weight = 1.0
        else:
            raise TypeError(f'the `wrap_metric_weights` should be a float, int or dict')

        # Specify which tasks will be wrapped
        if isinstance(lofn_wrap_tasks, str):
            lofn_wrap_tasks = [lofn_wrap_tasks]
        elif isinstance(lofn_wrap_tasks, (set, list, tuple)):
            lofn_wrap_tasks = list(lofn_wrap_tasks)
        elif lofn_wrap_tasks is True:
            raise ValueError(
                'In MultiTask mode, the tasks should perform loss_fn wrapp must be specified in str or set of str')
        else:
            raise TypeError(f'the `wrapping_tasks` should be a str or a sequence of str')

        # Specify the metrics names to wrap the loss_fn
        if lofn_wrap_metric_names is None:
            wrap_mtrc = {}  # The primary_metrics will be set in the case
        elif isinstance(lofn_wrap_metric_names, dict):
            assert (tsk_name in metrics for tsk_name in lofn_wrap_metric_names)
            assert (mtr_name in metrics[tsk_name] for tsk_name, mtr_name in lofn_wrap_metric_names.items())
            wrap_mtrc = lofn_wrap_metric_names
        else:
            raise TypeError(f'the `wrap_metric_names` should a dict[task_name: metric_name]')

        for wrap_tsk in lofn_wrap_tasks:
            mtrc_fn = metrics[wrap_tsk][wrap_mtrc.get(wrap_tsk, primary_metrics[wrap_tsk])]
            mtrc_weight = lofn_wrap_metric_weights.get(wrap_tsk, 1.0)
            loss_fn[wrap_tsk] = wrap_loss_fn(loss_fn[wrap_tsk], mtrc_weight, mtrc_fn)
        logging.debug(f'[bold blue]Wrap metric in to loss_fn for task: {lofn_wrap_tasks}[/]')
        return loss_fn


# Test plot maker
def _specify_test_plot_maker(
        task_names: Union[str, Sequence[str]],
        predictors: Union[M.Predictor, dict[str, M.Predictor]],
):
    if isinstance(predictors, M.Predictor):
        plot_name = predictor_plot_maker_map[predictors.target_type]
        return {n: M.plots_options[n] for n in plot_name}
    elif isinstance(predictors, dict):
        plot_makers = {}
        for tsk, predictor in predictors.items():
            plot_name = predictor_plot_maker_map[predictor.target_type]
            plot_makers[tsk] = {n: M.plots_options[n] for n in plot_name}
        _align_task_names("plot_makers", plot_makers, task_names, lambda v: isinstance(v, dict))
        return plot_makers
    else:
        raise NotImplementedError

# Specify x masker
_default_mask_task = ['AT']
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
            return lambda inp: M.mask_atom_type(inp, core.node_mask)

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


def _specify_inputs_preprocessor(
        inputs_preprocessor: Callable,
        core: M.CoreBase,
        input_x_index: Optional[Union[list[int], torch.Tensor]] = None
):
    if isinstance(inputs_preprocessor, Callable):
        return inputs_preprocessor
    elif core.emb_type == 'atom':
        return wraps(M.get_x_input_attrs)(lambda inp: M.get_x_input_attrs(inp, input_x_index=0, dtype=torch.int))
    elif core.emb_type == 'orbital':
        return wraps(M.get_x_input_attrs)(lambda inp: M.get_x_input_attrs(inp, input_x_index=list(range(7)), dtype=torch.int))
    else:
        return wraps(M.get_x_input_attrs)(lambda inp: M.get_x_input_attrs(inp, input_x_index=list(range(7))))

def _specify_xyz_index(
        first_data: Union[Data, Iterable[Data]],
        with_xyz: Optional[Union[bool, Iterable[bool]]]
):
    if isinstance(first_data, Data):
        if with_xyz:
            return tools.get_index(first_data, 'x', COORD_X_ATTR)
        return None

    elif isinstance(first_data, Iterable):
        first_data = list(first_data)
        if not with_xyz:
            return [None] * len(first_data)
        elif with_xyz is True:
            return [tools.get_index(fd, 'x', COORD_X_ATTR) for fd in first_data]
        elif isinstance(with_xyz, Iterable):
            with_xyz = list(with_xyz)
            assert len(first_data) == len(with_xyz), (
                f'Expected len(first_data) == len(with_xyz), got {len(first_data)} != {len(with_xyz)}')

            return [
                tools.get_index(fd, 'x', COORD_X_ATTR) if opt is True else None
                for fd, opt in zip(first_data, with_xyz)
            ]
        else:
            raise TypeError(f'The `with_xyz` should be a bool or a iterable of bool')

    else:
        raise TypeError(f'The `first_data` should be a Data, Iterable, or None')


##########################################################################################################
################################## Task Configuration ####################################################
def _config_task_args(
        work_name: str,
        task_names: Optional[Union[str, Sequence[str]]],
        first_data: Data,
        target_getter,
        inputs_preprocessor: Optional[Callable],
        feature_extractor,
        core,
        hypers: ParamSets,
        predictor: M.Predictor,
        loss_fn,
        primary_metrics,
        other_metrics: tp.OtherMetricConfig,
        x_masker,
        mask_need_task,
        loss_weight_calculator,
        loss_weight_method: Union[tp.LossWeightMethods, dict[str, tp.LossWeightMethods]],
        loss_fn_wrap_tasks: bool = False,
        **kwargs
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
    predictor = _specify_predictors(task_names, core, predictor, hypers)

    # Specify default feature extractor
    feature_extractor = _specify_feature_extractor(task_names, feature_extractor, core)

    # Specify loss func
    loss_fn = _specify_loss_fn(task_names, loss_fn, predictor)

    # Specify primary metric
    primary_metrics, metrics = _specify_metrics(task_names, primary_metrics, other_metrics, predictor)

    # Specify test plot makers
    plot_makers = _specify_test_plot_maker(task_names, predictor)

    ####################### Important Args #####################
    ############################################################

    ####################### Optional Args ######################
    # Wrap the loss_fn accounting the metrics into
    # Notes: the wrapped task name must be given by set of str, otherwise not work
    if loss_fn_wrap_tasks is True or isinstance(loss_fn_wrap_tasks, (str, list, tuple, set)):
        loss_fn = _wrap_loss_fn_with_metric(task_names, loss_fn, metrics, primary_metrics, loss_fn_wrap_tasks)

    # Specify x masker
    x_masker = _specify_masker(task_names, x_masker, core)
    # if not isinstance(task_names, (list, tuple)):
    #     mask_need_task = None
    if mask_need_task is None:
        mask_need_task = [t for t in task_names if t in _default_mask_task]
    elif isinstance(mask_need_task, (list, tuple)):
        assert all(mt in task_names for mt in mask_need_task), 'All `mask_need_task` should in the task_names list'
    else:
        raise TypeError(f'The `mask_need_task` should be a str or a Sequence of str')

    # Loss weight calculator
    loss_weight_calculator = _specify_loss_weights_calculator(
        task_names, loss_weight_calculator, predictor, loss_weight_method)

    # Input preprocessor
    inputs_preprocessor = _specify_inputs_preprocessor(inputs_preprocessor, core, kwargs.pop('input_x_index', None))

    #############################################################
    return {
        'task_name': task_names,
        'target_getter': target_getter,
        'feature_extractor': feature_extractor,
        'predictor': predictor,
        'loss_fn': loss_fn,
        'primary_metric': primary_metrics,
        'metrics': metrics,
        'plot_makers': plot_makers,
        'x_masker': x_masker,
        'loss_weight_calculator': loss_weight_calculator,
        'mask_need_task': mask_need_task,
        'inputs_preprocessor': inputs_preprocessor,
    }

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


def config_tasks_from_multi_datasets(
        work_name: str,
        dataset_counts: int,
        inputs_getter: Callable,
        first_data: list[Data],
        core: M.CoreBase,
        predictor: M.Predictor,
        feature_extractor: list[dict[str, Callable]],
        target_getter: list[dict[str, tp.TargetGetter]],
        loss_fn: list[dict[str, tp.LossFn]],
        primary_metrics: list[dict[str, str]],
        other_metrics: Union[dict[str, Union[str, Iterable[str], tp.MetricFn]], list[tp.OtherMetricConfig]],
        hypers: Union[ParamSets, list[ParamSets]],
        task_names: Optional[list[Sequence[str]]] = None,
        batch_preprocessor: Optional[list[tp.BatchPreProcessor]] = None,
        inputs_preprocessor: Optional[list[Callable]] = None,
        xyz_index: Optional[list[Iterable[int]]] = None,
        with_sol: Optional[Union[bool, Iterable[bool]]] = None,
        with_med: Optional[Union[bool, Iterable[bool]]] = None,
        with_env: Optional[Union[bool, Iterable[bool]]] = None,
        xyz_perturb_sigma: Optional[list[float]] = None,
        xyz_perturb_mode: M.PerturbMode = 'uniform',
        extractor_attr_getter: Optional[list[dict[str, tp.ExtractorAttrGetter]]] = None,
        loss_weight_calculator: Optional[list[dict[str, tp.LossWeightCalculator]]] = None,
        loss_weight_method: Literal['inverse-count', 'cross-entropy', 'sqrt-invert_count'] = 'inverse-count',
        loss_fn_wrap_tasks: Optional[list[str, set[str]]] = None,
        onehot_types: Optional[list[dict[str, int]]] = None,
        x_masker: Optional[list[tp.XMasker]] = None,
        mask_need_task: Optional[list[list[str]]] = None,
        **kwargs
):
    # Aligning requirements
    assert len(feature_extractor) == dataset_counts
    assert len(target_getter) == dataset_counts
    assert len(loss_fn) == dataset_counts
    assert len(primary_metrics) == dataset_counts
    assert len(first_data) == dataset_counts
    if isinstance(inputs_getter, Callable):
        inputs_getter = [inputs_getter] * dataset_counts
    elif isinstance(inputs_getter, list):
        assert len(inputs_getter) == dataset_counts
    else:
        raise TypeError('inputs_getter must be a callable or a list of callable')

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
    task_names = _align_md_task_options('task_names', task_names, dataset_counts)
    batch_preprocessor = _align_md_task_options('batch_preprocessor', batch_preprocessor, dataset_counts)
    inputs_preprocessor = _align_md_task_options('inputs_preprocessor', inputs_preprocessor, dataset_counts)
    xyz_index = _align_md_task_options('xyz_index', xyz_index, dataset_counts)
    with_sol = _align_md_task_options('with_sol', with_sol, dataset_counts)
    with_med = _align_md_task_options('with_med', with_med, dataset_counts)
    with_env = _align_md_task_options('with_env', with_env, dataset_counts)
    xyz_perturb_sigma = _align_md_task_options('xyz_perturb_sigma', xyz_perturb_sigma, dataset_counts)
    xyz_perturb_mode = _align_md_task_options('xyz_perturb_mode', xyz_perturb_mode, dataset_counts)
    extractor_attr_getter = _align_md_task_options('extractor_attr_getter', extractor_attr_getter, dataset_counts)
    loss_fn_wrap_tasks = _align_md_task_options('loss_fn_wrap_tasks', loss_fn_wrap_tasks, dataset_counts, lst_values=True)
    loss_weight_calculator = _align_md_task_options('loss_weight_calculator', loss_weight_calculator, dataset_counts)
    loss_weight_method = _align_md_task_options('loss_weight_method', loss_weight_method, dataset_counts)
    onehot_types = _align_md_task_options('onehot_types', onehot_types, dataset_counts)
    x_masker = _align_md_task_options('x_masker', x_masker, dataset_counts)
    mask_need_task = _align_md_task_options('mask_need_task', mask_need_task, dataset_counts)
    hypers = _align_md_task_options('hypers', hypers, dataset_counts)
    to_onehot = [(list(oh_types) if isinstance(oh_types, dict) else bool(oh_types)) for oh_types in onehot_types]

    # Extract list of dict from define task-wise dict
    other_metrics = _extract_options_list('other_metrics', task_names, other_metrics, dataset_counts, {})
    ############################## End of Aligning #######################################

    tasks_arguments = []
    for i in range(dataset_counts):
        task_kwargs = _config_task_args(
            work_name=work_name,
            task_names=task_names[i],
            first_data=first_data[i],
            inputs_preprocessor=inputs_preprocessor[i],
            target_getter=target_getter[i],
            feature_extractor=feature_extractor[i],
            core=core,
            hypers=hypers[i],
            predictor=predictor[i],
            loss_fn=loss_fn[i],
            primary_metrics=primary_metrics[i],
            other_metrics=other_metrics[i],
            x_masker=x_masker[i],
            mask_need_task=mask_need_task[i],
            loss_weight_calculator=loss_weight_calculator[i],
            loss_weight_method=loss_weight_method[i],
            loss_fn_wrap_tasks=loss_fn_wrap_tasks[i],
        )
        task_kwargs.update(dict(
            batch_preprocessor=batch_preprocessor[i],
            inputs_getter=inputs_getter[i],
            xyz_index=xyz_index[i],
            xyz_perturb_sigma=xyz_perturb_sigma[i],
            xyz_perturb_mode=xyz_perturb_mode[i],
            to_onehot=to_onehot[i],
            extractor_attr_getter=extractor_attr_getter[i],
            hypers=hypers[i],
            with_sol=with_sol[i],
            with_med=with_med[i],
            with_env=with_env[i],
            **kwargs
        ))
        tasks_arguments.append(task_kwargs)

    return tasks_arguments


def config(
        work_name: str,
        task_names: Union[list[list], list[str]],
        task_type: type,
        dataModule: DataModule,
        inputs_getter: Callable,
        core: M.CoreBase,
        predictor: M.Predictor,
        feature_extractor,
        target_getter,
        loss_fn,
        primary_metrics,
        other_metrics: Union[tp.OtherMetricConfig, list[tp.OtherMetricConfig]],
        hypers: ParamSets,
        batch_preprocessor: Optional[list[tp.BatchPreProcessor]] = None,
        inputs_preprocessor: Optional[list[Callable]] = None,
        with_xyz: Optional[Union[bool, Iterable[bool]]] = None,
        with_sol: Optional[Union[bool, Iterable[bool]]] = None,
        with_med: Optional[Union[bool, Iterable[bool]]] = None,
        with_env: Optional[Union[bool, Iterable[bool]]] = None,
        xyz_perturb_sigma: Optional[list[float]] = None,
        xyz_perturb_mode: M.PerturbMode = 'uniform',
        extractor_attr_getter: Optional[list[dict[str, tp.ExtractorAttrGetter]]] = None,
        loss_weight_calculator: Optional[list[dict[str, tp.LossWeightCalculator]]] = None,
        loss_fn_wrap_tasks: Optional[Union[bool, set[str], Iterable[set[str]]]] = None,
        loss_weight_method: Literal['inverse-count', 'cross-entropy', 'sqrt-invert_count'] = 'inverse-count',
        onehot_types: Optional[Union[int, dict[str, int], list[dict[str, int]]]] = None,
        x_masker: Optional[list[tp.XMasker]] = None,
        mask_need_task: Optional[list[list[str]]] = None,
        **kwargs
):
    # Configure tasks in single dataset
    if task_type in (tasks.SingleTask, tasks.MultiTask):
        assert not dataModule.is_multi_datasets
        first_data = dataModule.first_data
        xyz_index = _specify_xyz_index(first_data, with_xyz)
        task_kwargs = _config_task_args(
            work_name=work_name,
            task_names=task_names,
            first_data=first_data,
            inputs_preprocessor=inputs_preprocessor,
            target_getter=target_getter,
            feature_extractor=feature_extractor,
            core=core,
            hypers=hypers,
            predictor=predictor,
            loss_fn=loss_fn,
            primary_metrics=primary_metrics,
            other_metrics=other_metrics,
            x_masker=x_masker,
            mask_need_task=mask_need_task,
            loss_weight_calculator=loss_weight_calculator,
            loss_weight_method=loss_weight_method,
            loss_fn_wrap_tasks=loss_fn_wrap_tasks,
        )
        task_kwargs.update(dict(
            hypers=hypers,
            batch_preprocessor=batch_preprocessor,
            inputs_getter=inputs_getter,
            xyz_index=xyz_index,
            xyz_perturb_sigma=xyz_perturb_sigma,
            xyz_perturb_mode=xyz_perturb_mode,
            extractor_attr_getter = extractor_attr_getter,
            onehot_types=onehot_types,
            to_onehot = list(onehot_types) if isinstance(onehot_types, dict) else bool(onehot_types),
            with_sol=with_sol,
            with_med=with_med,
            with_env=with_env,
            **kwargs
        ))

        return task_kwargs

    # Configure tasks for multi-datasets
    elif task_type is tasks.MultiDataTask:
        assert dataModule.is_multi_datasets

        first_data = dataModule.first_data  # list of PyG.Data: [Data]
        xyz_index = _specify_xyz_index(first_data, with_xyz)

        return config_tasks_from_multi_datasets(
            work_name=work_name,
            dataset_counts=dataModule.dataset_counts,
            inputs_getter=inputs_getter,
            first_data=first_data,
            core=core,
            predictor=predictor,
            feature_extractor=feature_extractor,
            target_getter=target_getter,
            loss_fn=loss_fn,
            primary_metrics=primary_metrics,
            other_metrics=other_metrics,
            hypers=hypers,
            task_names=task_names,
            batch_preprocessor=batch_preprocessor,
            inputs_preprocessor=inputs_preprocessor,
            xyz_index=xyz_index,
            xyz_perturb_sigma=xyz_perturb_sigma,
            xyz_perturb_mode=xyz_perturb_mode,
            extractor_attr_getter=extractor_attr_getter,
            loss_weight_calculator=loss_weight_calculator,
            loss_weight_method=loss_weight_method,
            loss_fn_wrap_tasks=loss_fn_wrap_tasks,
            onehot_types=onehot_types,
            x_masker=x_masker,
            mask_need_task=mask_need_task,
            with_sol=with_sol,
            with_med=with_med,
            **kwargs
        )
    else:
        raise NotImplementedError(f"Task type {task_type} is not implemented.")
