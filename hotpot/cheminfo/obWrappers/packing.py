"""Typed NumPy boundary between Hotpot molecules and native Open Babel code."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from .._native.buffers import (
    MOLECULE_BUFFER_SCHEMA_VERSION,
    NativeMoleculeBuffers,
    NativeMoleculeField,
    _BOND_KIND_CODES as _COMMON_BOND_KIND_CODES,
    _empty_matrix,
    pack_molecule as _pack_common_molecule,
)
from .contracts import BondKindCode


if TYPE_CHECKING:
    from ..core import Molecule


__all__ = ()


_BOND_KIND_CODES = {
    kind_name: BondKindCode(kind_code)
    for kind_name, kind_code in _COMMON_BOND_KIND_CODES.items()
}
_MOLECULE_BUFFER_SCHEMA_VERSION = MOLECULE_BUFFER_SCHEMA_VERSION
_MoleculeBuffers = NativeMoleculeBuffers
_OPENBABEL_FIELDS = frozenset(
    {
        NativeMoleculeField.ATOMIC_NUMBERS,
        NativeMoleculeField.FORMAL_CHARGES,
        NativeMoleculeField.PARTIAL_CHARGES,
        NativeMoleculeField.COORDINATES,
        NativeMoleculeField.ATOM_AROMATIC,
        NativeMoleculeField.BOND_INDICES,
        NativeMoleculeField.BOND_ORDERS,
        NativeMoleculeField.BOND_KINDS,
        NativeMoleculeField.BOND_AROMATIC,
        NativeMoleculeField.UNIT_CELL,
    }
)


def _pack_molecule(mol: "Molecule") -> _MoleculeBuffers:
    """Pack native tasks, encoding unavailable charge rows as finite zero.

    These task-only zeros are never selected as molecular charge observations.
    """
    from ..structure._results import _partial_charge_projection

    with mol._mutation_lock:
        buffers = _pack_common_molecule(
            mol, _OPENBABEL_FIELDS - {NativeMoleculeField.PARTIAL_CHARGES},
        )
        charges = _partial_charge_projection(mol)
        return replace(
            buffers,
            partial_charges=np.fromiter(
                (0.0 if value is None else value for value in charges),
                dtype=np.float64, count=buffers.atom_count,
            ),
        )
