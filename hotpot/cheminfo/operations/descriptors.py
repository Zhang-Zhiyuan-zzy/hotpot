"""Retained molecular descriptor preparation over an operation-owned copy."""

from copy import copy

import numpy as np

from ..rules.mathematics.graph.matrix import linkmat2adj
from ..rules.chemistry.descriptors.graph_spectrum import GraphSpectrum


__all__ = ()


def _graph_spectral(mol, norm):
    clone = copy(mol)
    clone.add_hydrogens()
    adjacency = linkmat2adj(len(clone.atoms), clone.link_matrix)
    return GraphSpectrum.from_adj_atoms(
        adjacency, np.array([atom.atomic_number for atom in clone.atoms]), norm=norm)
