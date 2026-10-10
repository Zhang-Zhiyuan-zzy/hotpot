"""MCA inference façade with lazy model-runtime loading."""

import os
from functools import lru_cache

from ..core import Molecule

__all__ = ["mca"]


@lru_cache(maxsize=8)
def _get_mca_predictor(
    device: str,
    allow_charged: bool,
    model_source: str = None,
    model_dir: str = None,
):
    from ..AImodels.mca import MCAPredictor

    return MCAPredictor(
        device=device,
        allow_charged=allow_charged,
        model_source=model_source,
        model_dir=model_dir,
    )


def mca(
    mol: Molecule,
    *,
    device: str = None,
    allow_charged: bool = False,
    model_source: str = None,
    model_dir: str = None,
):
    """Predict every atom's MCA and identify important nucleophilic sites."""
    selected_device = device or os.environ.get("HOTPOT_MCA_DEVICE", "auto")
    prediction = _get_mca_predictor(
        selected_device,
        allow_charged,
        model_source,
        model_dir,
    ).predict(mol)
    atom_values = {
        atom.atom_index: atom.mca_kj_mol
        for atom in prediction.atom_predictions
    }
    site_values = {site.atom_index: site.mca_kj_mol for site in prediction.sites}

    for atom in mol.atoms:
        object.__setattr__(atom, "_mca", atom_values[atom.idx])
    object.__setattr__(
        mol,
        "_mca_sites",
        {mol.atoms[index]: value for index, value in site_values.items()},
    )
    mol.properties["mca"] = prediction
    return prediction

