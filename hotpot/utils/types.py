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
from typing import *
import numpy as np
import torch

ArrayLike = Union[Sequence, np.ndarray, torch.Tensor]
Vector = Sequence[Union[int, float]]
