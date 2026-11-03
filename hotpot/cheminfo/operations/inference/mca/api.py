"""Explicit CPU and GPU MCA inference API."""

from __future__ import annotations

from typing import Literal

from .cpu_backend import MCACpuPredictor
from .gpu_backend import MCAGpuPredictor


def predict_mca_cpu(molecules, **kwargs):
    """Run one CPU prediction and close its ONNX session afterward."""

    with MCACpuPredictor(**kwargs) as predictor:
        return predictor.predict(molecules)


def predict_mca_gpu(molecules, **kwargs):
    """Run GPU-preferred inference and close any resources afterward."""

    with MCAGpuPredictor(**kwargs) as predictor:
        return predictor.predict(molecules)


def predict_mca(
    molecules,
    *,
    backend: Literal["cpu", "gpu"],
    **kwargs,
):
    """Predict MCA with one explicitly selected backend policy."""

    if backend == "cpu":
        return predict_mca_cpu(molecules, **kwargs)
    if backend == "gpu":
        return predict_mca_gpu(molecules, **kwargs)
    raise ValueError("backend must be 'cpu' or 'gpu'")


__all__ = [
    "MCACpuPredictor",
    "MCAGpuPredictor",
    "predict_mca",
    "predict_mca_cpu",
    "predict_mca_gpu",
]
