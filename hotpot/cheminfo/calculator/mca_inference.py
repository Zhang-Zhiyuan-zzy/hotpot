"""Attach explicit-backend MCA inference results to Hotpot molecules."""

from typing import Literal

from ..core import Molecule

__all__ = ["mca"]


def mca(
    mol: Molecule,
    *,
    backend: Literal["cpu", "gpu"],
    **predictor_options,
):
    """Predict every atom's MCA and identify important nucleophilic sites."""

    from ..AImodels.mca import predict_mca

    prediction = predict_mca(
        mol,
        backend=backend,
        **predictor_options,
    )
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
