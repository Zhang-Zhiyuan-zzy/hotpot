"""Typed NumPy boundary between Hotpot molecules and native Open Babel code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .contracts import BondKindCode


if TYPE_CHECKING:
    from ..core import Molecule


__all__ = ()


_BOND_KIND_CODES = {
    "single": BondKindCode.SINGLE,
    "double": BondKindCode.DOUBLE,
    "triple": BondKindCode.TRIPLE,
    "aromatic": BondKindCode.AROMATIC,
    "zero": BondKindCode.ZERO,
    "dative": BondKindCode.DATIVE,
    "unknown": BondKindCode.UNKNOWN,
}

_MOLECULE_BUFFER_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class _MoleculeBuffers:
    """Contiguous values copied from one Hotpot molecule for native use."""

    schema_version: int
    atomic_numbers: NDArray[np.int32]
    formal_charges: NDArray[np.int32]
    partial_charges: NDArray[np.float64]
    coordinates: NDArray[np.float64]
    atom_aromatic: NDArray[np.uint8]
    bond_indices: NDArray[np.int32]
    bond_orders: NDArray[np.float64]
    bond_kinds: NDArray[np.uint8]
    bond_aromatic: NDArray[np.uint8]
    unit_cell: Optional[NDArray[np.float64]]


def _empty_matrix(columns: int, dtype: np.dtype) -> np.ndarray:
    return np.empty((0, columns), dtype=dtype)


def _pack_molecule(mol: "Molecule") -> _MoleculeBuffers:
    """Copy one molecule into exact, contiguous native-boundary arrays."""
    atoms = mol.atoms
    bonds = mol.bonds
    atom_rows = {id(atom): row for row, atom in enumerate(atoms)}

    atomic_numbers = np.fromiter(
        (atom.atomic_number for atom in atoms),
        dtype=np.int32,
        count=len(atoms),
    )
    formal_charges = np.fromiter(
        (atom.formal_charge for atom in atoms),
        dtype=np.int32,
        count=len(atoms),
    )
    partial_charges = np.fromiter(
        (atom.partial_charge for atom in atoms),
        dtype=np.float64,
        count=len(atoms),
    )
    coordinates = (
        np.ascontiguousarray(
            [atom.coordinates for atom in atoms],
            dtype=np.float64,
        )
        if atoms
        else _empty_matrix(3, np.dtype(np.float64))
    )
    atom_aromatic = np.fromiter(
        (atom.is_aromatic for atom in atoms),
        dtype=np.uint8,
        count=len(atoms),
    )

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
    bond_orders = np.fromiter(
        (bond.bond_order for bond in bonds),
        dtype=np.float64,
        count=len(bonds),
    )
    bond_kinds = np.fromiter(
        (_BOND_KIND_CODES[bond.bond_kind.value] for bond in bonds),
        dtype=np.uint8,
        count=len(bonds),
    )
    bond_aromatic = np.fromiter(
        (
            bond.bond_kind.value == "aromatic" or bond.is_aromatic
            for bond in bonds
        ),
        dtype=np.uint8,
        count=len(bonds),
    )

    crystal = mol.crystal
    unit_cell = None
    if crystal is not None:
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

    return _MoleculeBuffers(
        schema_version=_MOLECULE_BUFFER_SCHEMA_VERSION,
        atomic_numbers=np.ascontiguousarray(atomic_numbers),
        formal_charges=np.ascontiguousarray(formal_charges),
        partial_charges=np.ascontiguousarray(partial_charges),
        coordinates=coordinates,
        atom_aromatic=np.ascontiguousarray(atom_aromatic),
        bond_indices=bond_indices,
        bond_orders=np.ascontiguousarray(bond_orders),
        bond_kinds=np.ascontiguousarray(bond_kinds),
        bond_aromatic=np.ascontiguousarray(bond_aromatic),
        unit_cell=unit_cell,
    )
