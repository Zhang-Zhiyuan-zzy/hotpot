"""Prepare carrier facts, then encode the existing model feature families."""
from __future__ import annotations
from typing import Optional, Iterable, Any
import numpy as np
from hotpot.cheminfo.structure.molecule import Molecule
from hotpot.cheminfo.views.pairs import AtomPair
from hotpot.cheminfo.structure._fields import ATOM_FIELDS
from hotpot.cheminfo.operations.chemistry import _atom_projection
from hotpot.cheminfo.rules.mathematics.graph.cycles import _cycle_basis
from hotpot.cheminfo.adapters.structure import inference as _encoding
from hotpot.cheminfo.adapters.structure.inference import direct_edge_to_indirect


def extract_potentials_cbonds(
        mol: Molecule,
        catoms_symbols: Optional[Iterable[str]] = None,
        data: dict[str, np.ndarray] = None,
) -> dict[str, np.ndarray]:
    return _encoding.extract_potentials_cbonds(mol, catoms_symbols, data)


def extract_atom_attrs(mol: Molecule, data: dict[str, Any] = None, atomic_number_only: bool = False) -> dict:
    with mol._mutation_lock:
        if atomic_number_only:
            extracted = _encoding.extract_atom_attrs(mol, atomic_number_only=True)
        else:
            fields = ATOM_FIELDS[:15]
            projected = _atom_projection(mol, fields=fields, strict_hydrogens=True)
            extracted = _encoding.extract_atom_attrs(mol, projected=projected, fields=fields)
    if isinstance(data, dict):
        data.update(extracted)
    else:
        data = extracted
    return data


def extract_bond_attrs(mol: Molecule, edge_attr_names: Iterable[str] = None, data: dict[str, Any] = None) -> dict:
    return _encoding.extract_bond_attrs(mol, edge_attr_names, data)


def extract_atom_pairs(mol: Molecule, data: dict[str, Any] = None) -> dict:
    atom_pairs = mol.atom_pairs
    atom_pairs.update_pairs()
    return _encoding.extract_atom_pairs(atom_pairs, AtomPair.attr_names, data)


def extract_ring_attrs(mol: Molecule, rings_attr_names: Iterable[str] = None, data: dict[str, Any] = None) -> dict:
    # Preserve the ring-tensor family used to train the existing models.
    rings = mol._rings_from_cycles(_cycle_basis(
        range(len(mol.atoms)),
        (
            (
                mol._topology_index[id(bond.atom1)][0],
                mol._topology_index[id(bond.atom2)][0],
            )
            for bond in mol.bonds
            if not bond.is_metal_ligand_bond
        ),
    ))

    return _encoding.extract_ring_attrs(rings, rings_attr_names, data)
