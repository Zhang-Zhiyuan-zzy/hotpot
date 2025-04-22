from typing import Union
import numpy as np
import torch
import torch.nn.functional as F

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
        if isinstance(pred, np.ndarray):
            pred = torch.from_numpy(pred)
            pred = np.round(F.sigmoid(pred).numpy())
            return (pred == target).mean()

        else:
            pred = torch.round(F.sigmoid(pred))
            return (pred == target).float().mean()

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
        all_pred_true = pred > 0.5
        tp = (all_pred_true == target).sum()
        return tp / all_pred_true.sum()

    @staticmethod
    def recall(
            pred: Union[np.ndarray, torch.Tensor],
            target: Union[np.ndarray, torch.Tensor]
    ) -> Union[torch.Tensor, np.ndarray]:
        all_pred_true = pred > 0.5
        total_positive = target.sum()
        tp = (all_pred_true == target).sum()
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
    ) -> Union[torch.Tensor, np.ndarray]:
        """ Calculate the Area Under `ROC` Curve (AUC) for the binary target """
        fpr, tpr, threshold = Metrics.roc(target, pred)  # fpr: x, tpr: y
        return torch.trapezoid(tpr, fpr) if isinstance(pred, torch.Tensor) else np.trapz(tpr, fpr)

    @staticmethod
    def roc(
            pred: Union[np.ndarray, torch.Tensor],
            target: Union[np.ndarray, torch.Tensor]
    ) -> Union[torch.Tensor, np.ndarray]:
        """ Retrieve Receiver Operating Characteristic Curve (ROC) for the binary target """
        # Initialize functions
        if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
            _csum = torch.cumsum
            _arange = torch.arange
        elif isinstance(pred, np.ndarray) and isinstance(target, np.ndarray):
            _csum = np.cumsum
            _arange = np.arange
        else:
            raise ValueError('The `pred` and `target` must be simultaneously torch.Tensor or np.ndarray.')


        pred = pred.flatten()
        target = target.flatten()

        total_positive = target.sum()
        total_negative = len(target) - total_positive

        sort_idx = pred.argsort()
        thresholds = pred[sort_idx]  # Here, the sorted pred is equal to the thresholds

        sorted_target = target[sort_idx]
        inverse_sort_target = 1 - sorted_target  # 0 to 1, 1 to 0

        fn = _csum(sorted_target)  # False negative counts
        tp = total_negative - fn  # True positive counts
        tpr = tp / total_positive  # True positive ratio

        fp = _csum(inverse_sort_target)  # False positive counts
        fpr = fp / total_negative  # False positive ratio

        return fpr, tpr, thresholds

