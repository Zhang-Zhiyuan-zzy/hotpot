from typing import Union
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc

from . import utils

class Metrics:
    """ A collection of metrics functions """
    @staticmethod
    def average_inverse_distance(pred, target):
        if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
            dist = torch.norm(pred-target, p=2, dim=-1)
            return torch.mean(1/(1+dist))
        elif isinstance(pred, np.ndarray) and isinstance(target, np.ndarray):
            dist = np.linalg.norm(pred-target, ord=2, axis=-1)
            return np.mean(1/(1+dist))
        else:
            raise TypeError('pred and target must be torch.Tensor or np.ndarray')

    @staticmethod
    def calc_oh_accuracy(pred, target):
        pred, target = map(utils.oh2label, (pred, target))
        if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
            return (pred == target).float().mean()
        elif isinstance(pred, np.ndarray) and isinstance(target, np.ndarray):
            return (pred == target).mean()
        else:
            raise TypeError('pred_oh must be of type torch.Tensor or np.ndarray')

    @staticmethod
    def metal_oh_accuracy(pred, target):
        pred, target = map(utils.oh2label, (pred, target))
        metal_idx = utils.where_metal(target)
        pred = pred[metal_idx]
        target = target[metal_idx]

        if isinstance(pred, torch.Tensor):
            return (pred == target).float().mean()
        elif isinstance(pred, np.ndarray):
            return (pred == target).mean()
        else:
            raise TypeError('pred_oh must be of type torch.Tensor or np.ndarray')


    @staticmethod
    def binary_accuracy(pred: Union[torch.Tensor, np.ndarray], target: Union[torch.Tensor, np.ndarray]):
        """ the pred is the output without Sigmoid activation """
        pred = utils.norm_binary_to_zero_one(pred)
        return (pred == target).mean() if isinstance(pred, np.ndarray) else (pred == target).float().mean()

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

    @staticmethod
    def precision(
            pred: Union[np.ndarray, torch.Tensor],
            target: Union[np.ndarray, torch.Tensor]
    ) -> Union[torch.Tensor, np.ndarray]:
        all_pred_true = pred >= 0.  # The value of pred is exponent of sigmoid, range from -oo to +oo
        tp = ((all_pred_true == target) & all_pred_true).sum()
        return tp / all_pred_true.sum()

    @staticmethod
    def recall(
            pred: Union[np.ndarray, torch.Tensor],
            target: Union[np.ndarray, torch.Tensor]
    ) -> Union[torch.Tensor, np.ndarray]:
        all_pred_true = pred > 0.  # The value of pred is exponent of sigmoid, range from -oo to +oo
        total_positive = target.sum()
        tp = ((all_pred_true == target) & np.bool_(target)).sum()
        return tp / total_positive

    @staticmethod
    def f1_score(
            pred: Union[np.ndarray, torch.Tensor],
            target: Union[np.ndarray, torch.Tensor]
    ) -> Union[torch.Tensor, np.ndarray]:
        """ Calculate the f1 score for the binary target """
        precision = Metrics.precision(pred, target)
        recall = Metrics.recall(pred, target)
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def auc(
            pred: Union[np.ndarray, torch.Tensor],
            target: Union[np.ndarray, torch.Tensor]
    ) -> float:
        """ Calculate the Area Under `ROC` Curve (AUC) for the binary target """
        fpr, tpr, threshold = Metrics.roc(pred, target)  # fpr: x, tpr: y
        return auc(fpr, tpr)

    @staticmethod
    def roc(
            pred: Union[np.ndarray, torch.Tensor],
            target: Union[np.ndarray, torch.Tensor]
    ) -> (Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray], float):
        """ Retrieve Receiver Operating Characteristic Curve (ROC) for the binary target """
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        elif isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        pred, target = pred.flatten(), target.flatten()
        return roc_curve(target, pred)

