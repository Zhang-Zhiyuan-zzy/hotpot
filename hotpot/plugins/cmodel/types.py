from typing import Union, Callable, Sequence, Literal

import torch
from torch_geometric.data import Batch

import numpy as np

from . import models as M

TargetType = Literal['xyz', 'onehot', 'binary', 'num']
TensorArray = Union[torch.Tensor, np.ndarray]
ExtractorAttrGetter = Callable[[Batch], Union[tuple, torch.Tensor]]
TargetGetter = Callable[[Batch], torch.Tensor]
LossWeightCalculator = Callable[[torch.Tensor, int], torch.Tensor]


# types for run() arguments
WorkNameInput = Union[str, Sequence[str]]
TargetGetterInput = Union[str, Callable, Sequence[Callable], dict[str, Callable]]
FeatureExtractorInput = Union[str, Callable, Sequence[Callable], dict[str, Callable]]
PredictorInput = Union[str, M.Predictor, Sequence[M.Predictor], dict[str, M.Predictor]]
LossFnInput = Union[str, Callable, Sequence[Union[str, Callable]], dict[str, Callable]]
