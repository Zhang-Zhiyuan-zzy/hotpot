"""Lazy access to the direct native Open Babel backend."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

from .packing import _pack_molecule


if TYPE_CHECKING:
    from hotpot.cheminfo.structure.molecule import Molecule
    from . import _ob_native


__all__ = ()


def _native_module() -> ModuleType:
    try:
        # Initialize the external toolkit stack at the actual native-use boundary.
        from ..adapters.openbabel import _runtime  # noqa: F401
        return import_module("hotpot.cheminfo.obWrappers._ob_native")
    except ImportError as exc:
        raise ImportError(
            "the hotpot.cheminfo.obWrappers native extension is unavailable; "
            "install a compatible Hotpot wheel or rebuild Hotpot from source"
        ) from exc


def _native_molecule_data(mol: "Molecule", *, selection=None) -> "_ob_native.MoleculeData":
    native = _native_module()
    buffers = _pack_molecule(mol, selection=selection)
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
