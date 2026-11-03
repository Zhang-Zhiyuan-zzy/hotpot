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
    from hotpot.cheminfo.structure.molecule import Molecule


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


def _pack_molecule(mol: "Molecule", *, selection=None) -> _MoleculeBuffers:
    """Pack native tasks, encoding unavailable charge rows as finite zero.

    These task-only zeros are never selected as molecular charge observations.
    A selection filters Bond buffers after strict preparation of the intact source.
    """
    from ..structure._results import _partial_charge_projection
    from ..structure._selection import _check_topology_selection

    with mol._mutation_lock:
        if selection is not None:
            _check_topology_selection(mol, selection)
        buffers = _pack_common_molecule(
            mol, _OPENBABEL_FIELDS - {NativeMoleculeField.PARTIAL_CHARGES},
        )
        charges = _partial_charge_projection(mol)
        buffers = replace(
            buffers,
            partial_charges=np.fromiter(
                (0.0 if value is None else value for value in charges),
                dtype=np.float64, count=buffers.atom_count,
            ),
        )
        if selection is not None:
            _check_topology_selection(mol, selection)
            mask = np.frombuffer(selection.bond_mask, dtype=np.bool_)
            buffers = replace(
                buffers,
                bond_count=int(np.count_nonzero(mask)),
                bond_indices=buffers.bond_indices[mask],
                bond_orders=buffers.bond_orders[mask],
                bond_kinds=buffers.bond_kinds[mask],
                bond_aromatic=buffers.bond_aromatic[mask],
            )
        return buffers
