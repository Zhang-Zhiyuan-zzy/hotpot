"""Typed input values for native complex force-field sessions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray


__all__ = ("ComplexSessionInput",)


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
