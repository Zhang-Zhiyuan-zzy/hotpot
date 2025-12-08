"""
@File Name:        datacls
@Project:          
@Author:           Zhiyuan Zhang
@Created On:       2025/11/19 20:14
@Project:          Hotpot
"""
import logging
from typing import *
import datetime
from dataclasses import dataclass, field, fields
from operator import attrgetter

import torch
from torch.optim import Optimizer
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.strategies import DDPStrategy

from hotpot.plugins.opti import ParamSpace, ParamSets
from hotpot.utils.configs.logging_config import setup_logging

from .. import (
    types as tp,
    models as M,
    callbacks as cbs
)

__all__ = [
    'merge_dataclass',
    'RunArgs',
    'DataModuleArgs',
    'ConfigArgs'
]

############################################################
# Constants
INPUT_GETTER = attrgetter(
        'x', 'edge_index', 'edge_attr', 'rings_node_index',
        'rings_node_nums', 'mol_rings_nums', 'batch', 'ptr'
)

################### Interface ########################
T = TypeVar('T')
def merge_dataclass(target: T, *sources: Any, exclude_none: bool = False, **kwargs):
    """Merges fields from multiple sources into a copy of target.

    Sources are processed in order; later sources override earlier ones.

    Args:
        target: The base instance (immutable).
        *sources: One or more instances to pull values from.
        exclude_none: If True, skips fields in sources that are None.

    Returns:
        A new instance of T with aggregated values.
    """
    target_keys = {f.name for f in fields(target)}
    updates = {}

    for src in sources:
        updates.update({
            k: v for k, v in vars(src).items()
            if k in target_keys and (v is not None or not exclude_none)
        })
    updates.update(kwargs)

    return target(**updates)

@dataclass
class RunArgs:
    """Configuration arguments for pretraining the ComplexFormer.

    Attributes:
        work_name: The name of the work being trained. Standardized nomenclature is recommended.
        work_dir: The directory where the trained models and inspected info will be saved.
        hypers: Hyperparameters for optimizer, dataloader, and others except for model.

        dir_datasets: The directory where the datasets will be saved.
        dataset_names: Which datasets to use. Matches fold names under dir_datasets.
            If None, use all datasets.
        exclude_datasets: Mutual option with `dataset_names` to exclude specific datasets.
        shuffle_dataset: Whether to shuffle the dataset.
        dataModule_seed: Seed for dataloader.
        data_split_ratios: Ratio of train, validation and test splits.
        external_datasets: Path or names of external datasets.

        stages: Which stages will be performed during invoking the interface.
        eval_each_step: How many epochs to evaluate the model.

        epochs: The Maximum of epochs to train.
        batch_size: Batch size for training.
        early_stopping: Whether early stopping is enabled.
        early_stop_step: Steps without improvement to trigger early stopping.
        target_metrics: Metrics to monitor for early stopping or saving.
        num_trials: Number of trials (likely for hyperparameter tuning).

        checkpoint_path: Path (str) or index (int) of the checkpoint to load.

        optimizer: The type of optimizer to use. Defaults to Adam if None.
        constant_lr: Whether to use constant learning rate.
        lr_scheduler: Learning rate scheduler class/function.
        lr_scheduler_frequency: Frequency of scheduler stepping.
        lr_scheduler_kwargs: Keyword arguments passed to `lr_scheduler`.
        loss_weight_method: Method to calculate coefficients ki before sum of loss.

        xyz_perturb_sigma: Sigma value for Gaussian noise added to coordinates.
        xyz_perturb_mode: Mode to perturb xyz ('uniform' or 'normal').
        batch_preprocessor: Custom batch preprocessing logic.
        inputs_preprocessor: Custom inputs preprocessing logic.
        x_masker: Strategy or callable for masking inputs.
        mask_need_task: Tasks that require masking.

        with_xyz: Whether to load xyz to ComplexFormer.
        with_sol: Whether to extract solvents information.
        with_med: Whether to extract medium information.
        with_env: Whether to extract environment information.

        task_names: Define task tags for datasets. List[str] for single, List[List[str]] for multi.
        target_getter: Callable to extract target values from batch.
        feature_extractor: Which feature extractor to use.
        predictor: Which predictor to use (Module or key like 'onehot', 'num').
        loss_fn: Loss function configuration.
        primary_metrics: Primary metric to control training process.
        other_metrics: Metrics to measure performance without impacting training.
        extractor_attr_getter: Method to get attributes from extractor.
        minimize_metric: Whether the primary metric should be minimized.
        onehot_types: Specify types count for onehot predictors.
        loss_fn_wrap_tasks: Whether to add a metric wrapper to the loss func.

        save_model: Whether to save the model.

        devices: Number of GPU devices to use.
        precision: Precision used by PyTorch (e.g., 'bf16-mixed').
        float32_matmul_precision: Internal precision for float32 matmul.
        profiler: PyTorch Lightning profiler.
        show_pbar: Whether to show the progress bar.
        debug: Turn on debug mode.
        overfit_test: Run overfit test.
        use_debugger: Whether to use a debugger.
        warning_allowed: If False, warning messages will raise Errors.

        lofn_wrap_tasks: (Extra) Optional wrapper config for loss function.
        lofn_wrap_exclude_tasks: (Extra) Optional exclusion config for loss wrapper.
        lofn_wrap_metric_names: (Extra) Metric names for loss wrapper.
        lofn_wrap_metric_weights: (Extra) Metric weights for loss wrapper.
    """

    # -- Global information Arguments --
    work_name: str
    work_dir: str
    hypers: Union['ParamSets', 'ParamSpace']

    # -- DataModule Arguments --
    dir_datasets: str
    dataset_names: Optional[Union[str, Sequence[str]]] = None
    exclude_datasets: Optional[Union[str, Sequence[str]]] = None
    shuffle_dataset: bool = True
    dataModule_seed: int = 315
    data_split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    external_datasets: Optional[Union[str, Sequence[str]]] = None

    # -- Flow control Arguments --
    stages: Optional[Union[tp.Stages, list[tp.Stages]]] = None
    eval_each_step: Optional[int] = 1

    # -- Training loop control --
    epochs: int = 100
    batch_size: int = 512
    early_stopping: bool = True
    early_stop_step: int = 10
    target_metrics: Union[str, List[str], Callable[[dict], float], None] = None
    num_trials: int = 10

    # -- Arguments of checkpoints --
    checkpoint_path: Union[str, int, None] = None

    # -- Optimizer configuration --
    optimizer: Optional[Type['Optimizer']] = None
    constant_lr: bool = False
    lr_scheduler: Optional[Callable] = None
    lr_scheduler_frequency: int = 2
    lr_scheduler_kwargs: Optional[dict] = None
    loss_weight_method: Literal['inverse-count', 'cross-entropy', 'sqrt-invert_count'] = 'inverse-count'

    # -- Inputs specification --
    inputs_getter: Optional[Any] = INPUT_GETTER
    xyz_perturb_sigma: Optional[float] = None
    xyz_perturb_mode: 'M.PerturbMode' = 'uniform'
    batch_preprocessor: Optional[Union['tp.BatchPreProcessor', List['tp.BatchPreProcessor']]] = None
    inputs_preprocessor: Optional[Union[Callable, List[Callable]]] = None
    x_masker: Optional[Union[str, Callable]] = None
    mask_need_task: Optional[List[str]] = None

    # -- Dataset level Arguments --
    with_xyz: Optional[Union[bool, Iterable[bool]]] = None
    with_sol: Optional[Union[bool, Iterable[bool]]] = None
    with_med: Optional[Union[bool, Iterable[bool]]] = None
    with_env: Optional[Union[bool, Iterable[bool]]] = None

    # -- Task level Arguments --
    task_names: Optional[Union[List[str], List[List[str]]]] = None
    target_getter: 'tp.TargetGetterInput' = None
    feature_extractor: Optional['tp.FeatureExtractorInput'] = None
    predictor: Optional['tp.PredictorInput'] = None
    loss_fn: Optional['tp.LossFnInput'] = None
    primary_metrics: Optional['tp.MetricType'] = None
    other_metrics: Optional[Union['tp.OtherMetricConfig', List['tp.OtherMetricConfig']]] = None
    extractor_attr_getter: Optional[Union[Callable, Dict[str, Callable], List[Union[Dict, Callable]]]] = None
    minimize_metric: bool = False
    onehot_types: Optional[Union[int, Dict[str, int], List[Dict[str, int]]]] = None
    loss_fn_wrap_tasks: Optional[Union[bool, str, set[str], List[bool]]] = None

    # -- Postprocessing arguments --
    save_model: bool = True

    # -- Environmental configuration and device --
    devices: Optional[int] = None
    precision: str = 'bf16-mixed'
    accelerator: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    strategy = 'auto'
    float32_matmul_precision: str = 'medium'
    profiler: str = "simple"
    show_pbar: bool = True
    debug: bool = False
    overfit_test: bool = False
    use_debugger: bool = False
    warning_allowed: bool = True

    # -- Keyword Args (Explicitly defined for clarity) --
    lofn_wrap_tasks: Optional[Union[str, Sequence[str]]] = None
    lofn_wrap_exclude_tasks: Optional[Union[str, Sequence[str]]] = None
    lofn_wrap_metric_names: Optional[Union[str, Dict[str, str]]] = None
    lofn_wrap_metric_weights: Optional[Union[float, Dict[str, float]]] = None

    # Catch-all for any other kwargs if strictly necessary,
    # though explicit definition above is preferred.
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validations or post-processing can go here."""
        setup_logging(debug=self.debug)
        if self.stages is None:
            self.stages = ['train', 'test']
        elif isinstance(self.stages, str) and self.stages in get_args(tp.Stages):
            self.stages = [self.stages]
        elif isinstance(self.stages, Iterable):
            self.stages = list(self.stages)
            assert all(stage in get_args(tp.Stages) for stage in self.stages)
        else:
            raise ValueError(f"Unknown stages type: {type(self.stages)}, choose from {get_args(tp.Stages)}")

        ##################### Base Args ##########################
        torch.set_float32_matmul_precision(self.float32_matmul_precision)
        # Devices Configuration
        if self.devices is None:
            self.devices = 1
        if self.accelerator == 'cpu':
            self.devices = 1

        if self.accelerator == 'cuda' and self.strategy == 'auto':
            self.strategy = DDPStrategy(find_unused_parameters=True, timeout=datetime.timedelta(seconds=6000))

        logging.info(f"Using device: {self.accelerator} with {self.devices} devices")

        if self.debug or self.overfit_test:
            self.epochs = 5
            self.batch_num = 30
            logging.debug(f"Epoch={self.epochs} batch_num={self.batch_num} in Debug mode")
        else:
            self.batch_num = None


@dataclass
class DataModuleArgs:
    """Configuration arguments for the Lightning DataModule.

    Attributes:
        dir_datasets: Path to the parent folder containing dataset sub-directories.
        dataset_names: Specific dataset sub-folder(s) to load.
            If None, all sub-directories are used.
        exclude_datasets: Dataset(s) to exclude from selection.
        seed: Random seed for splitting data.
        batch_num: If set, limits the number of batches per dataset for debugging.
            Typically overrides `ratios` logic or dataset size.
        ratios: Fractions for (train, validation, test) splits.
        batch_size: Batch size for dataloaders.
        shuffle: Whether to shuffle the training dataloader.
        devices: Number of CUDA devices. Used for debug calculations and replica handling.
        num_replicas: Number of replicas for distributed training.
            Triggers `DistConcatLoader` if > 1.
        load_data_memory: Whether to load all tensors into RAM (True) or load lazily (False).
        test_only: Whether to setup only for testing (skips train/val splits).
    """

    # -- Core Path Arguments --
    dir_datasets: str
    dataset_names: Optional[Union[str, Sequence[str]]] = None
    exclude_datasets: Optional[Union[str, Sequence[str]]] = None

    # -- Split & Randomness --
    seed: int = 315
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)

    # -- Loader Configuration --
    batch_size: int = 1
    shuffle: bool = True
    load_data_memory: bool = True

    # -- Distributed & Hardware --
    devices: Optional[int] = None
    num_replicas: Optional[int] = None

    # -- Debug & Workflow Control --
    batch_num: Optional[int] = None
    stages: list[tp.Stages] = None
    test_only: bool = False

    def __post_init__(self):
        """Validate arguments after initialization."""
        self.test_only = ('test' in self.stages and 'train' not in self.stages)


@dataclass
class ConfigArgs:
    hypers: Union[ParamSets, ParamSpace]
    batch_preprocessor: Optional[Union[tp.BatchPreProcessor, list[tp.BatchPreProcessor]]] = None
    constant_lr: bool = False
    dataModule: Optional[Any] = None
    extractor_attr_getter: Optional[Union[Callable, dict[str, Callable], list[Union[dict, Callable]]]] = None
    feature_extractor: Optional[tp.FeatureExtractorInput] = None
    inputs_getter: Optional[Any] = INPUT_GETTER
    inputs_preprocessor: Optional[Union[Callable, list[Callable]]] = None
    loss_fn: Optional[tp.LossFnInput] = None
    lofn_wrap_tasks: Optional[Union[bool, float, dict[str, float]]] = None
    loss_weight_method: Optional[Union[Callable, bool]] = None
    lr_scheduler: Optional[Callable] = None
    lr_scheduler_frequency: int = 2
    lr_scheduler_kwargs: Optional[dict[str, Any]] = None
    mask_need_task: Optional[list[str]] = None
    onehot_types: Optional[Union[int, dict[str, int], list[dict[str, int]]]] = None
    optimizer: Optional[Type[Optimizer]] = None
    other_metrics: Optional[Union[tp.OtherMetricConfig, list[tp.OtherMetricConfig]]] = None
    predictor: Optional[tp.PredictorInput] = None
    primary_metrics: Optional[dict[str, str]] = None
    target_getter: tp.TargetGetterInput = None
    task_names: Optional[Union[list[str], list[list[str]]]] = None
    with_med: Optional[Union[bool, Iterable[bool]]] = None
    with_sol: Optional[Union[bool, Iterable[bool]]] = None
    with_xyz: Optional[Union[bool, Iterable[bool]]] = None
    with_env: Optional[Union[bool, Iterable[bool]]] = None
    sol_graph_inputs: Optional[Iterable[str]] = None
    med_graph_inputs: Optional[Iterable[str]] = None
    work_name: str = "default_work"
    x_masker: Optional[Union[str, Callable]] = None
    xyz_perturb_sigma: Optional[float] = None
    xyz_perturb_mode: M.PerturbMode = "uniform"
    show_pbar: bool = True
    onehot_type: Optional[Union[int, dict[str, int]]] = None
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class CallbackConfig:
    """
    Configuration class for generating training and testing callbacks.
    """
    optim_configure: Any
    stages: List[str]
    early_stop_step: int
    early_stopping: int
    minimize_metric: bool
    show_pbar: bool
    use_debugger: bool

    def _get_train_callbacks(self) -> List[Any]:
        callbacks = []

        if isinstance(self.early_stopping, int) and self.early_stopping > 0:
            callbacks.append(
                EarlyStopping(
                    monitor=self.optim_configure.primary_monitor,
                    mode='min' if self.minimize_metric else 'max',
                    patience=self.early_stop_step,
                )
            )

        if self.show_pbar:
            callbacks.append(cbs.Pbar())

        if self.use_debugger:
            callbacks.append(cbs.Debugger())

        return callbacks

    @staticmethod
    def _get_test_callbacks() -> List[Any]:
        return []

    def build(self) -> Optional[List[Any]]:
        callbacks = []

        if 'train' in self.stages:
            callbacks.extend(self._get_train_callbacks())

        if 'test' in self.stages:
            callbacks.extend(self._get_test_callbacks())

        return callbacks if callbacks else None

