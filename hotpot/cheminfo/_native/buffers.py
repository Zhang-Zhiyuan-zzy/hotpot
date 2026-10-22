"""Field-selective contiguous molecule buffers for native consumers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..core import Molecule


class NativeBondKindCode(IntEnum):
    """Toolkit-neutral bond-kind codes shared with native views."""

    SINGLE = 1
    DOUBLE = 2
    TRIPLE = 3
    AROMATIC = 4
    ZERO = 5
    DATIVE = 6
    UNKNOWN = 7


class NativeMoleculeField(Enum):
    """Independently selectable fields in the common molecule payload."""

    ATOMIC_NUMBERS = "atomic_numbers"
    FORMAL_CHARGES = "formal_charges"
    PARTIAL_CHARGES = "partial_charges"
    COORDINATES = "coordinates"
    ATOM_AROMATIC = "atom_aromatic"
    BOND_INDICES = "bond_indices"
    BOND_ORDERS = "bond_orders"
    BOND_KINDS = "bond_kinds"
    BOND_AROMATIC = "bond_aromatic"
    UNIT_CELL = "unit_cell"


@dataclass(frozen=True)
class NativeMoleculeBuffers:
    """Contiguous values selected from one Hotpot molecule."""

    schema_version: int
    atom_count: int
    bond_count: int
    atomic_numbers: NDArray[np.int32] | None = None
    formal_charges: NDArray[np.int32] | None = None
    partial_charges: NDArray[np.float64] | None = None
    coordinates: NDArray[np.float64] | None = None
    atom_aromatic: NDArray[np.uint8] | None = None
    bond_indices: NDArray[np.int32] | None = None
    bond_orders: NDArray[np.float64] | None = None
    bond_kinds: NDArray[np.uint8] | None = None
    bond_aromatic: NDArray[np.uint8] | None = None
    unit_cell: NDArray[np.float64] | None = None


_BOND_KIND_CODES = {
    "single": NativeBondKindCode.SINGLE,
    "double": NativeBondKindCode.DOUBLE,
    "triple": NativeBondKindCode.TRIPLE,
    "aromatic": NativeBondKindCode.AROMATIC,
    "zero": NativeBondKindCode.ZERO,
    "dative": NativeBondKindCode.DATIVE,
    "unknown": NativeBondKindCode.UNKNOWN,
}

MOLECULE_BUFFER_SCHEMA_VERSION = 1


def _empty_matrix(columns: int, dtype: np.dtype) -> np.ndarray:
    return np.empty((0, columns), dtype=dtype)


def pack_molecule(
    mol: Molecule,
    fields: frozenset[NativeMoleculeField],
) -> NativeMoleculeBuffers:
    """Copy only the requested molecule fields into contiguous arrays."""

    atoms = tuple(mol.atoms)
    bonds = tuple(mol.bonds)
    atom_rows = (
        {id(atom): row for row, atom in enumerate(atoms)}
        if NativeMoleculeField.BOND_INDICES in fields
        else None
    )

    def atom_values(field, values, dtype):
        if field not in fields:
            return None
        return np.ascontiguousarray(
            np.fromiter(values, dtype=dtype, count=len(atoms))
        )

    def bond_values(field, values, dtype):
        if field not in fields:
            return None
        return np.ascontiguousarray(
            np.fromiter(values, dtype=dtype, count=len(bonds))
        )

    atomic_numbers = atom_values(
        NativeMoleculeField.ATOMIC_NUMBERS,
        (atom.atomic_number for atom in atoms),
        np.int32,
    )
    formal_charges = atom_values(
        NativeMoleculeField.FORMAL_CHARGES,
        (atom.formal_charge for atom in atoms),
        np.int32,
    )
    partial_charges = atom_values(
        NativeMoleculeField.PARTIAL_CHARGES,
        (atom.partial_charge for atom in atoms),
        np.float64,
    )
    coordinates = None
    if NativeMoleculeField.COORDINATES in fields:
        coordinates = (
            np.ascontiguousarray(
                [atom.coordinates for atom in atoms],
                dtype=np.float64,
            )
            if atoms
            else _empty_matrix(3, np.dtype(np.float64))
        )
    atom_aromatic = atom_values(
        NativeMoleculeField.ATOM_AROMATIC,
        (atom.is_aromatic for atom in atoms),
        np.uint8,
    )

    bond_indices = None
    if NativeMoleculeField.BOND_INDICES in fields:
        bond_indices = (
            np.ascontiguousarray(
                [
                    (atom_rows[id(bond.atom1)], atom_rows[id(bond.atom2)])
                    for bond in bonds
                ],
                dtype=np.int32,
            )
            if bonds
            else _empty_matrix(2, np.dtype(np.int32))
        )
    bond_orders = bond_values(
        NativeMoleculeField.BOND_ORDERS,
        (bond.bond_order for bond in bonds),
        np.float64,
    )
    bond_kinds = bond_values(
        NativeMoleculeField.BOND_KINDS,
        (_BOND_KIND_CODES[bond.bond_kind.value] for bond in bonds),
        np.uint8,
    )
    bond_aromatic = bond_values(
        NativeMoleculeField.BOND_AROMATIC,
        (
            bond.bond_kind.value == "aromatic" or bond.is_aromatic
            for bond in bonds
        ),
        np.uint8,
    )

    unit_cell = None
    if NativeMoleculeField.UNIT_CELL in fields and mol.crystal is not None:
        crystal = mol.crystal
        unit_cell = np.ascontiguousarray(
            (
                crystal.a,
                crystal.b,
                crystal.c,
                crystal.alpha,
                crystal.beta,
                crystal.gamma,
            ),
            dtype=np.float64,
        )

    return NativeMoleculeBuffers(
        schema_version=MOLECULE_BUFFER_SCHEMA_VERSION,
        atom_count=len(atoms),
        bond_count=len(bonds),
        atomic_numbers=atomic_numbers,
        formal_charges=formal_charges,
        partial_charges=partial_charges,
        coordinates=coordinates,
        atom_aromatic=atom_aromatic,
        bond_indices=bond_indices,
        bond_orders=bond_orders,
        bond_kinds=bond_kinds,
        bond_aromatic=bond_aromatic,
        unit_cell=unit_cell,
    )
