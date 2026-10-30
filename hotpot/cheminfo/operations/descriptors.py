"""Retained molecular descriptor preparation over an operation-owned copy."""

from copy import copy

import numpy as np

from ..rules.mathematics.graph.matrix import linkmat2adj
from ..rules.chemistry.descriptors.graph_spectrum import GraphSpectrum


__all__ = ()


def _element_counts(mol):
    from ..rules.chemistry.descriptors.formula import _element_counts as calculate
    with mol._mutation_lock:
        return calculate(tuple(int(value) for value in mol._atom_table.column('atomic_number')))


def _formula(mol):
    from ..rules.chemistry.descriptors.formula import _formula as calculate
    with mol._mutation_lock:
        return calculate(tuple(int(value) for value in mol._atom_table.column('atomic_number')))


def _weight(mol):
    from ..rules.chemistry.descriptors.formula import _weight as calculate
    with mol._mutation_lock:
        return calculate(tuple(int(value) for value in mol._atom_table.column('atomic_number')))


def _graph_spectral(mol, norm):
    clone = copy(mol)
    clone.add_hydrogens()
    adjacency = linkmat2adj(len(clone.atoms), clone.link_matrix)
    return GraphSpectrum.from_adj_atoms(
        adjacency, np.array([atom.atomic_number for atom in clone.atoms]), norm=norm)
