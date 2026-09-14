"""Standalone and hotpot-compatible MCA inference package."""

from .api import MCAPredictor, predict_mca
from .graph_adapter import MoleculeGraph
from .result_types import MoleculePrediction, SitePrediction

__all__ = [
    "MCAPredictor",
    "MoleculeGraph",
    "MoleculePrediction",
    "SitePrediction",
    "predict_mca",
]

