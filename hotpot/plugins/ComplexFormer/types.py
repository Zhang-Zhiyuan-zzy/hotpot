from typing import Union, Callable, Sequence, Literal, Optional, Any, Iterable

import torch
from torch_geometric.data import Batch

import numpy as np
from matplotlib.pyplot import Figure

from . import models as M

Stages = Literal['train', 'test']
TargetType = Literal['xyz', 'onehot', 'binary', 'num']
TensorArray = Union[torch.Tensor, np.ndarray]
BatchPreProcessor = Callable[[Batch], Batch]
ExtractorAttrGetter = Callable[[Batch], Union[tuple, torch.Tensor]]
FeatureExtractorName = Literal['atom', 'bond', 'pair', 'ring', 'mol', 'cbond', 'metal']
TargetGetter = Callable[[Batch], torch.Tensor]
LossWeightCalculator = Callable[[torch.Tensor], torch.Tensor]
LossWeightMethods = Literal['inverse-count', 'cross-entropy', 'sqrt-invert_count']
LossFn = Callable[[torch.Tensor, torch.Tensor, Optional[Any]], torch.Tensor]
MetricFn = Callable[[TensorArray, TensorArray], Union[float, TensorArray]]
XMasker = Callable[[tuple[torch.Tensor, ...], torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
OtherMetricConfig = Union[str, Iterable[str], dict[str, Union[str, Iterable[str], MetricFn]]]


# types for run() arguments
WorkNameInput = Union[str, Sequence[str]]
TargetGetterInput = Union[str, Callable, Sequence[Callable], dict[str, Callable]]
FeatureExtractorInput = Union[str, Callable, Sequence[Callable], dict[str, Callable]]
PredictorInput = Union[str, M.Predictor, Sequence[M.Predictor], dict[str, M.Predictor]]
LossFnInput = Union[str, Callable, Sequence[Union[str, Callable]], dict[str, Callable]]
MetricType = Literal['r2score', 'rmse', 'mse', 'mae', 'accuracy', 'binary_accuracy', 'metal_accuracy']


PlotMaker = Callable[[np.ndarray, np.ndarray], Figure]
PlotMakerDict = dict[str, PlotMaker]

