"""Standalone and hotpot-compatible MCA inference package."""

from .api import MCAPredictor, predict_mca
from .graph_adapter import MoleculeGraph
from .result_types import AtomPrediction, MoleculePrediction, SitePrediction

__all__ = [
    "MCAPredictor",
    "AtomPrediction",
    "MoleculeGraph",
    "MoleculePrediction",
    "SitePrediction",
    "predict_mca",
]
