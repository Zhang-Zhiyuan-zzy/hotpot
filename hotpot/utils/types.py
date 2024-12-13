"""
python v3.9.0
@Project: hotpot
@File   : types
@Auther : Zhiyuan Zhang
@Data   : 2024/8/26
@Time   : 14:16

Notes:
    Define custom types used in this package
"""
from types import *
from typing import *
import numpy as np
import torch

ArrayLike = Union[Sequence, np.ndarray, torch.Tensor]
Vector = Sequence[Union[int, float]]


class ModelLike(Protocol):
    """ a typing Notation for object like sklearn models """
    def fit(self, *args, **kwargs) -> None:
        ...

    def predict(self, *args, **kwargs) -> Any:
        ...

    def score(self, *args, **kwargs) -> float:
        ...