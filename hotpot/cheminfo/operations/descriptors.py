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


def _is_organic(mol):
    from .chemistry import _require_implicit_hydrogens
    from ..rules.chemistry.descriptors.composition import (
        _contains_carbon, _organic_element_allowed,
    )
    return (
        all(_organic_element_allowed(atom.is_metal) for atom in mol.atoms) and
        (bool(mol.hydrogens) or any(_require_implicit_hydrogens(mol))) and
        _contains_carbon(tuple(atom.atomic_number for atom in mol._atoms))
    )


def _is_full_halogenated(mol):
    if mol.is_organic:
        return False
    clone = mol.copy()
    for atom in clone.atoms:
        if atom.is_halogens:
            atom.atomic_number = 1
    return clone.is_organic


def _similarity(mol, other, method, norm):
    if method == "spectrum":
        return round(mol.graph_spectral(norm) | other.graph_spectral(norm), 15)
    raise NotImplementedError


def _cid(mol):
    from ..pubchem import pubchem_service
    return pubchem_service.smi_to_cid(mol.smiles)


def _name(mol):
    from ..pubchem import pubchem_service
    return pubchem_service.smi_to_name(mol.smiles)


def _cas(mol):
    from ..pubchem import pubchem_service
    return pubchem_service.smi_to_cas(mol.smiles)
