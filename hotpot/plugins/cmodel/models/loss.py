from typing import Union, Literal
import numpy as np

import torch
import torch.nn.functional as F

from .utils import torch_numpy_exchanger, weight_labels, weight_binary

class LossMethods:
    """ A collection of loss functions """
    @staticmethod
    def calc_atom_type_loss(pred, target, weight=None, acc=torch.tensor(1.)):
        """ Cross Entropy Loss """
        if isinstance(weight, torch.Tensor):
            # return F.cross_entropy(pred, target.float(), weight=weight.to(pred.device)) - acc*torch.log(acc)
            return F.cross_entropy(pred, target.flatten().long(), weight=weight)
        else:
            return F.cross_entropy(pred, target.flatten().long())

    @staticmethod
    def average_maximum_displacement(
            pred: Union[torch.Tensor, np.ndarray],
            target: Union[torch.Tensor, np.ndarray],
            *args, **kwargs
    ) -> Union[torch.Tensor, np.ndarray, float]:
        if isinstance(target, torch.Tensor):
            norm = torch_numpy_exchanger(torch.norm, dim=-1)
        elif isinstance(target, np.ndarray):
            norm = torch_numpy_exchanger(np.linalg.norm, axis=-1)
        else:
            raise TypeError("The target and pred data should be torch.Tensor or np.ndarray")

        return norm(pred - target).mean()


######################### Labels loss weights calculation ################################
def label_weights_calculator(
        onehot_types: int,
        weight_method: Literal['inverse-count', 'cross-entropy', 'sqrt-invert_count'] = 'cross-entropy',
):
    def calculator(labels):
        if onehot_types == 2:
            return weight_binary(labels)
        else:
            return weight_labels(labels, onehot_types, weight_method)

    return calculator


######################## Across task loss weights calculation ############################
def _calc_weight(metrics: float, p: Union[int, float] = 3):
    assert isinstance(p, (int, float)) and p > 0
    if metrics < 0:
        base = 0
    elif metrics > 1:
        base = 1
    else:
        base = metrics

    return 1 - pow(base, p)


def atl_calculator(metrics: dict[str, float], p: Union[int, float] = 3, eps: float = 1e-4):
    """
    Across tasks losses (atl) weights calculator based on the primary metric results for each task.
    The values of each metric are supposed to in the [0, 1] range, otherwise, a ValueError is raised.
    """
    # Check all metric values are from 0 to 1
    for tsk_name, metric in metrics.items():
        if not metric <= 1:
            raise ValueError(f'metric {tsk_name} must be in [0, 1], but got {metric}')

    weights = {tsk: _calc_weight(m, p) for tsk, m in metrics.items()}
    multiplier = len(weights) / (sum(weights.values()) + eps)
    return {tsk: w*multiplier for tsk, w in weights.items()}