"""Lazy access to the direct native Open Babel backend."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

from .packing import _pack_molecule


if TYPE_CHECKING:
    from ..core import Molecule
    from . import _ob_native


__all__ = ()


def _native_module() -> ModuleType:
    try:
        return import_module("hotpot.cheminfo.obWrappers._ob_native")
    except ImportError as exc:
        raise ImportError(
            "the hotpot.cheminfo.obWrappers native extension is unavailable; "
            "install a compatible Hotpot wheel or rebuild Hotpot from source"
        ) from exc


def _native_molecule_data(mol: "Molecule") -> "_ob_native.MoleculeData":
    native = _native_module()
    buffers = _pack_molecule(mol)
    return native.MoleculeData(
        buffers.schema_version,
        buffers.atomic_numbers,
        buffers.formal_charges,
        buffers.partial_charges,
        buffers.coordinates,
        buffers.atom_aromatic,
        buffers.bond_indices,
        buffers.bond_orders,
        buffers.bond_kinds,
        buffers.bond_aromatic,
        buffers.unit_cell,
    )
