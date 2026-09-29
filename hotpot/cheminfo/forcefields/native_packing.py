"""Typed session boundary for native complex force-field stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ..obWrappers.packing import (
    _BOND_KIND_CODES,
    _MOLECULE_BUFFER_SCHEMA_VERSION,
    _empty_matrix,
    _pack_molecule,
)


if TYPE_CHECKING:
    from ..core import Bond, Molecule


__all__ = (
    "ComplexSessionInput",
    "pack_complex_session_input",
)


@dataclass(frozen=True)
class ComplexSessionInput:
    """Contiguous values copied at the Stage 1-to-native session boundary."""

    schema_version: int
    atomic_numbers: NDArray[np.int32]
    formal_charges: NDArray[np.int32]
    partial_charges: NDArray[np.float64]
    coordinates: NDArray[np.float64]
    atom_aromatic: NDArray[np.uint8]
    ligand_bond_indices: NDArray[np.int32]
    ligand_bond_orders: NDArray[np.float64]
    ligand_bond_kinds: NDArray[np.uint8]
    ligand_bond_aromatic: NDArray[np.uint8]
    metal_indices: NDArray[np.int32]
    intended_coordination_bonds: NDArray[np.int32]
    intended_coordination_orders: NDArray[np.float64]
    intended_coordination_kinds: NDArray[np.uint8]
    unit_cell: Optional[NDArray[np.float64]]


def _coordination_bonds(mol: "Molecule") -> tuple["Bond", ...]:
    active = tuple(bond for bond in mol.bonds if bond.is_metal_ligand_bond)
    hidden = tuple(mol._hided_metal_bonds)
    seen: set[int] = set()
    unique = []
    for bond in (*active, *hidden):
        identity = id(bond)
        if identity not in seen:
            seen.add(identity)
            unique.append(bond)
    return tuple(unique)


def pack_complex_session_input(mol: "Molecule") -> ComplexSessionInput:
    """Copy a complex into ligand topology and directed coordination buffers."""
    molecule = _pack_molecule(mol)
    atoms = tuple(mol.atoms)
    atom_rows = {id(atom): row for row, atom in enumerate(atoms)}
    active_bonds = tuple(mol.bonds)
    ligand_positions = tuple(
        index
        for index, bond in enumerate(active_bonds)
        if not bond.is_metal_ligand_bond
    )

    ligand_bond_indices = (
        np.ascontiguousarray(
            molecule.bond_indices[np.asarray(ligand_positions, dtype=np.intp)],
            dtype=np.int32,
        )
        if ligand_positions
        else _empty_matrix(2, np.dtype(np.int32))
    )
    ligand_bond_orders = np.ascontiguousarray(
        molecule.bond_orders[np.asarray(ligand_positions, dtype=np.intp)],
        dtype=np.float64,
    )
    ligand_bond_kinds = np.ascontiguousarray(
        molecule.bond_kinds[np.asarray(ligand_positions, dtype=np.intp)],
        dtype=np.uint8,
    )
    ligand_bond_aromatic = np.ascontiguousarray(
        molecule.bond_aromatic[np.asarray(ligand_positions, dtype=np.intp)],
        dtype=np.uint8,
    )

    intended_records = []
    for bond in _coordination_bonds(mol):
        first = atom_rows[id(bond.atom1)]
        second = atom_rows[id(bond.atom2)]
        if not bond.atom1.is_metal:
            first, second = second, first
        intended_records.append((first, second, bond))
    intended_records.sort(key=lambda record: (record[0], record[1]))

    intended_coordination_bonds = (
        np.ascontiguousarray(
            [(first, second) for first, second, _ in intended_records],
            dtype=np.int32,
        )
        if intended_records
        else _empty_matrix(2, np.dtype(np.int32))
    )
    intended_coordination_orders = np.fromiter(
        (bond.bond_order for _, _, bond in intended_records),
        dtype=np.float64,
        count=len(intended_records),
    )
    intended_coordination_kinds = np.fromiter(
        (
            _BOND_KIND_CODES[bond.bond_kind.value]
            for _, _, bond in intended_records
        ),
        dtype=np.uint8,
        count=len(intended_records),
    )
    metal_indices = np.fromiter(
        (index for index, atom in enumerate(atoms) if atom.is_metal),
        dtype=np.int32,
    )

    return ComplexSessionInput(
        schema_version=_MOLECULE_BUFFER_SCHEMA_VERSION,
        atomic_numbers=molecule.atomic_numbers,
        formal_charges=molecule.formal_charges,
        partial_charges=molecule.partial_charges,
        coordinates=molecule.coordinates,
        atom_aromatic=molecule.atom_aromatic,
        ligand_bond_indices=ligand_bond_indices,
        ligand_bond_orders=ligand_bond_orders,
        ligand_bond_kinds=ligand_bond_kinds,
        ligand_bond_aromatic=ligand_bond_aromatic,
        metal_indices=np.ascontiguousarray(metal_indices),
        intended_coordination_bonds=intended_coordination_bonds,
        intended_coordination_orders=np.ascontiguousarray(
            intended_coordination_orders
        ),
        intended_coordination_kinds=np.ascontiguousarray(
            intended_coordination_kinds
        ),
        unit_cell=molecule.unit_cell,
    )
