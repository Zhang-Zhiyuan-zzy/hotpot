"""Attach explicit-backend MCA inference results to Hotpot molecules."""

from typing import Literal

from hotpot.cheminfo.core import Molecule
from hotpot.cheminfo.structure._results import _capture_mca_source, _select_mca_result

__all__ = ["mca"]


def mca(
    mol: Molecule,
    *,
    backend: Literal["cpu", "gpu"],
    **predictor_options,
):
    """Predict every atom's MCA and identify important nucleophilic sites."""

    from . import predict_mca
    from .serialization import normalize_molecule

    with mol._mutation_lock:
        molecule = normalize_molecule(mol)
        source = _capture_mca_source(mol)
        atom_uids = tuple(atom._uid for atom in mol._atoms)
    prediction = predict_mca(
        molecule,
        backend=backend,
        **predictor_options,
    )
    return _select_mca_result(mol, prediction, source, atom_uids)
