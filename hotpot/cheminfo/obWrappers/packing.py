"""Typed NumPy boundary between Hotpot molecules and native Open Babel code."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
_OPENBABEL_FIELDS = frozenset(NativeMoleculeField)


def _pack_molecule(mol: "Molecule") -> _MoleculeBuffers:
    """Copy every Open Babel molecule field into common native buffers."""

    return _pack_common_molecule(mol, _OPENBABEL_FIELDS)
