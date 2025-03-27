from typing import Union
import numpy as np
import torch
import torch.nn.functional as F

from . import utils


class Metrics:
    """ A collection of metrics functions """
    @staticmethod
    def calc_oh_accuracy(pred, target, is_onehot: bool = True):
        if is_onehot:
            pred_label, target_label = utils.oh2label(pred), utils.oh2label(target)
        else:
            pred_label, target_label = pred, target

        if isinstance(pred, torch.Tensor):
            return (pred_label == target_label).float().mean()
        elif isinstance(pred, np.ndarray):
            return (pred_label == target_label).mean()
        else:
            raise TypeError('pred_oh must be of type torch.Tensor or np.ndarray')

    @staticmethod
    def metal_oh_accuracy(pred, target, is_onehot: bool = True):
        if is_onehot:
            pred_label, target_label = utils.oh2label(pred), utils.oh2label(target)
        else:
            pred_label, target_label = pred, target

        metal_idx = utils.where_metal(target_label)
        pred_label = pred_label[metal_idx]
        target_label = target_label[metal_idx]

        if isinstance(pred, torch.Tensor):
            return (pred_label == target_label).float().mean()
        elif isinstance(pred, np.ndarray):
            return (pred_label == target_label).mean()
        else:
            raise TypeError('pred_oh must be of type torch.Tensor or np.ndarray')


    @staticmethod
    def binary_accuracy(pred: np.ndarray, target: np.ndarray):
        return (target == np.round(pred)).mean()

    @staticmethod
    def r2_score(
            pred: Union[np.ndarray, torch.Tensor],
            target: Union[np.ndarray, torch.Tensor]
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Computes the R^2 (coefficient of determination) score between y_true and y_pred.

        R^2 = 1 - (SS_res / SS_tot),
        where SS_res = Σ(y_true - y_pred)²
              SS_tot = Σ(y_true - mean(y_true))²
        """
        # Ensure y_true and y_pred are float tensors
        # target = target.float()
        # pred = pred.float()

        # Mean of true values
        mean_y_true = target.mean()

        # Sum of squares of residuals
        ss_res = ((target - pred) ** 2).sum()

        # Total sum of squares (relative to the mean)
        ss_tot = ((target - mean_y_true) ** 2).sum()

        # Handle the case where ss_tot can be zero (e.g., constant targets)
        if ss_tot <= 1e-8:
            if isinstance(pred, torch.Tensor):
                return torch.tensor(1.0 if torch.allclose(target, pred) else 0.0)
            else:
                return np.array(1.0 if np.allclose(target, pred) else 0.0)

        return 1 - ss_res / ss_tot

    @staticmethod
    def rmse(
            pred: Union[np.ndarray, torch.Tensor],
            target: Union[np.ndarray, torch.Tensor]
    ) -> Union[torch.Tensor, np.ndarray]:
        if isinstance(target, torch.Tensor):
            return torch.sqrt(F.mse_loss(pred, target))
        else:
            return np.sqrt(np.mean((pred - target) ** 2))

    @staticmethod
    def mse(
            pred: Union[np.ndarray, torch.Tensor],
            target: Union[np.ndarray, torch.Tensor]
    ) -> Union[torch.Tensor, np.ndarray]:
        if isinstance(target, torch.Tensor):
            return F.mse_loss(pred, target)
        else:
            return np.mean((pred - target) ** 2)

    @staticmethod
    def mae(
            pred: Union[np.ndarray, torch.Tensor],
            target: Union[np.ndarray, torch.Tensor]
    ) -> Union[torch.Tensor, np.ndarray]:
        if isinstance(target, torch.Tensor):
            return torch.mean(torch.abs(target - pred))
        else:
            return np.mean(np.abs(target - pred))