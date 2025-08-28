# -*- coding: utf-8 -*-
"""
===========================================================
 Project   : hotpot
 File      : plot_makers
 Created   : 2025/5/21 21:38
 Author    : zhang
 Python    : 
-----------------------------------------------------------
 Description
 ----------------------------------------------------------
 
===========================================================
"""
from typing import Type
import numpy as np
import matplotlib.pyplot as plt

from hotpot.plugins.plots import (
    Plot,
    SciPlotter,
    R2Regression,
    ConfusionMatrix,
    ROCCurve,
    DETCurve,
    PrecisionRecallCurve,
    MultiClassROCCurve,
    Hist
)


__all__ = ['plots_options']


def _make_plot(plot_type: Type[Plot], pred, target):
    plotter = SciPlotter(plot_type(pred, target))
    fig, ax = plotter()
    return fig

def confusion_matrix(pred: np.ndarray, target: np.ndarray) -> plt.Figure:
    return _make_plot(ConfusionMatrix, pred, target)

def binary_confusion_metrix(pred: np.ndarray, target: np.ndarray) -> plt.Figure:
    return _make_plot(ConfusionMatrix, pred, target)

def r2_regression(pred: np.ndarray, target: np.ndarray) -> plt.Figure:
    return _make_plot(R2Regression, pred, target)

def roc_curve(pred: np.ndarray, target: np.ndarray) -> plt.Figure:
    return _make_plot(ROCCurve, pred, target)

def det_curve(pred: np.ndarray, target: np.ndarray) -> plt.Figure:
    return _make_plot(DETCurve, pred, target)

def precision_recall_curve(pred: np.ndarray, target: np.ndarray) -> plt.Figure:
    return _make_plot(PrecisionRecallCurve, pred, target)

def multiclass_roc_curve(pred: np.ndarray, target: np.ndarray) -> plt.Figure:
    return _make_plot(MultiClassROCCurve, pred, target)

def hist(pred: np.ndarray, target: np.ndarray) -> plt.Figure:
    distance = np.linalg.norm(pred - target, axis=1).flatten()
    assert len(distance) == pred.shape[0]

    plotter = SciPlotter(Hist(distance, "Distance"))
    fig, ax = plotter()
    return fig


plots_options = {
    # For num predictor
    'r2': r2_regression,

    # For binary predictor
    'bconf': binary_confusion_metrix,
    'roc': roc_curve,
    'det': det_curve,
    'prc': precision_recall_curve,

    # For onehot predictor
    'conf': confusion_matrix,
    'mroc': multiclass_roc_curve,

    # For xyz predictor
    "hist": hist,
}
