from typing import Union
import numpy as np

import torch
import torch.nn.functional as F

from .utils import torch_numpy_exchanger

class LossMethods:
    """ A collection of loss functions """
    @staticmethod
    def calc_atom_type_loss(pred, target, weight=None, acc=torch.tensor(1.)):
        """ Cross Entropy Loss """
        if isinstance(weight, torch.Tensor):
            # return F.cross_entropy(pred, target.float(), weight=weight.to(pred.device)) - acc*torch.log(acc)
            return F.cross_entropy(pred, target.float(), weight=weight)
        else:
            return F.cross_entropy(pred, target.float())

    @staticmethod
    def mean_maximum_displacement(
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