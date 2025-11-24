"""
@File Name:        datacls
@Project:          
@Author:           Zhiyuan Zhang
@Created On:       2025/11/19 20:14
@Project:          Hotpot
"""
from typing import *
from dataclasses import dataclass, field, fields
from operator import attrgetter

from torch.optim import Optimizer

from hotpot.plugins.ComplexFormer.data import DataModule
from hotpot.plugins.opti import ParamSpace, ParamSets

from .. import (
    types as tp,
    tasks
)

############################################################
# Constants
INPUT_GETTER = attrgetter(
        'x', 'edge_index', 'edge_attr', 'rings_node_index',
        'rings_node_nums', 'mol_rings_nums', 'batch', 'ptr'
)

################### Interface ########################

@dataclass
class GlobalInfo:
    work_name: str
    work_dir: str
    hypers: Union[ParamSets, ParamSpace]

@dataclass
class DataModuleArgs:
    dir_datasets: str
    dataset_names: Union[str, Sequence[str]] = None
    exclude_datasets: Union[str, Sequence[str]] = None
    shuffle_dataset: bool = True
    dataModule_seed: int = 315
    data_split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    external_datasets: Union[str, Sequence[str]] = None

@dataclass
class FlowCtrlArgs:
    stages: Optional[Union[tp.Stages, Iterable[tp.Stages]]] = None
    epochs: int = 100
    batch_size: int = 512
    early_stopping: bool = True
    early_stop_step: int = 10
    freeze_core: Optional[bool] = None
    keep_grad_state: bool = False
    target_metrics: Union[str, list[str], Callable[[dict], float]] = None
    num_trials: int = 10
    save_model: bool = True

@dataclass
class InputCtrlArgs:
    # Global inputs field
    xyz_perturb_sigma: Optional[float] = None
    batch_preprocessor: Optional[Union[tp.BatchPreProcessor, list[tp.BatchPreProcessor]]] = None
    inputs_preprocessor: Optional[Union[Callable, list[Callable]]] = None
    x_masker: Optional[Union[str, Callable]] = None
    mask_need_task: Optional[list[str]] = None

    # Individual datasets field
    with_xyz: Optional[Union[bool, Iterable[bool]]] = None
    with_sol: Optional[Union[bool, Iterable[bool]]] = None
    with_med: Optional[Union[bool, Iterable[bool]]] = None
    with_env: Optional[Union[bool, Iterable[bool]]] = None

@dataclass
class ModuleArgs:
    checkpoint_path: Union[str, int] = None
    load_core_only: bool = True

@dataclass
class TaskDefinitionArgs:
    task_names: Optional[Union[list[str], list[list[str]]]] = None
    target_getter: tp.TargetGetterInput = None
    feature_extractor: Optional[tp.FeatureExtractorInput] = None
    predictor: Optional[tp.PredictorInput] = None
    loss_fn: Optional[tp.LossFnInput] = None
    primary_metrics: Optional[tp.MetricType] = None
    other_metrics: Optional[Union[tp.OtherMetricConfig, list[tp.OtherMetricConfig]]] = None
    extractor_attr_getter: Optional[Union[Callable, dict[str, Callable], list[dict, Callable]]] = None
    minimize_metric: bool = False
    onehot_types: Optional[Union[int, dict[str, int], list[dict[str, int]]]] = None
    loss_fn_wrap_tasks: Optional[Union[bool, str, set[str], list[bool]]] = None

@dataclass
class EnvCfgArgs:
    devices: Optional[int] = None
    precision = 'bf16-mixed'
    float32_matmul_precision = 'medium'
    profiler = "simple"
    show_pbar: bool = True
    debug: bool = False
    use_debugger: bool = False
    warning_allowed: bool = True

@dataclass
class ConfigArgs:
    batch_preprocessor: Optional[Union[tp.BatchPreProcessor, list[tp.BatchPreProcessor]]] = None
    constant_lr: bool = False
    dataModule: Optional[Any] = None
    extractor_attr_getter: Optional[Union[Callable, dict[str, Callable], list[Union[dict, Callable]]]] = None
    feature_extractor: Optional[tp.FeatureExtractorInput] = None
    inputs_getter: Optional[Any] = INPUT_GETTER
    inputs_preprocessor: Optional[Union[Callable, list[Callable]]] = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    loss_fn: Optional[tp.LossFnInput] = None
    loss_fn_wrap_tasks: Optional[Union[bool, str, set[str], list[bool]]] = None
    loss_weight_calculator: Optional[Union[Callable, bool]] = None
    loss_weight_method: Literal["inverse-count", "cross-entropy", "sqrt-invert_count"] = "inverse-count"
    lr_scheduler: Optional[Callable] = None
    lr_scheduler_frequency: int = 2
    lr_scheduler_kwargs: Optional[dict[str, Any]] = None
    mask_need_task: Optional[list[str]] = None
    onehot_types: Optional[Union[int, dict[str, int], list[dict[str, int]]]] = None
    optimizer: Optional[Type[Optimizer]] = None
    other_metrics: Optional[Union[tp.OtherMetricConfig, list[tp.OtherMetricConfig]]] = None
    predictor: Optional[tp.PredictorInput] = None
    primary_metrics: Optional[tp.MetricType] = None
    target_getter: tp.TargetGetterInput = None
    task_names: Optional[Union[list[str], list[list[str]]]] = None
    with_med: Optional[Union[bool, Iterable[bool]]] = None
    with_sol: Optional[Union[bool, Iterable[bool]]] = None
    with_xyz: Optional[Union[bool, Iterable[bool]]] = None
    work_name: str = "default_work"
    x_masker: Optional[Union[str, Callable]] = None
    xyz_perturb_sigma: Optional[float] = None
    show_pbar: bool = True

@dataclass
class OptimConfig:
    lr: float = 1e-3
    weight_decay: float = 4e-5
    optimizer: Optional[Type[Optimizer]] = None
    constant_lr: bool = False
    lr_scheduler: Optional[Callable] = None
    lr_scheduler_kwargs: Optional[dict] = None
    lr_scheduler_frequency: int = 2

def build_from_kwargs(cls, kwargs: dict) -> Any:
    field_names = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in kwargs.items() if k in field_names}
    return cls(**filtered)
