from typing import Union, Callable

import torch
from torch_geometric.data import Batch

import numpy as np

TensorArray = Union[torch.Tensor, np.ndarray]
ExtractorAttrGetter = Callable[[Batch], Union[tuple, torch.Tensor]]
TargetGetter = Callable[[Batch], torch.Tensor]