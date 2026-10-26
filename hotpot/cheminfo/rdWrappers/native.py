"""Internal access to the compiled RDKit wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .contracts import (
    ConformerBatch,
    CoordinationEmbeddingConstraints,
    ETKDGOptions,
    RuntimeInfo,
)

from .._native.buffers import (
    NativeMoleculeField,
    pack_molecule,
)

if TYPE_CHECKING:
    from hotpot.cheminfo.core import Molecule


_RDKIT_ETKDG_FIELDS = frozenset(
    {
        NativeMoleculeField.ATOMIC_NUMBERS,
        NativeMoleculeField.FORMAL_CHARGES,
        NativeMoleculeField.ATOM_AROMATIC,
        NativeMoleculeField.BOND_INDICES,
        NativeMoleculeField.BOND_KINDS,
        NativeMoleculeField.BOND_AROMATIC,
        NativeMoleculeField.STEREO_RECORDS,
        NativeMoleculeField.ATOM_TRANSPORT,
        NativeMoleculeField.BOND_TRANSPORT,
    }
)

def _backend():
    try:
        from . import _rd_native
    except ImportError as exc:
        raise RuntimeError("the native RDKit wrapper is not built") from exc
    return _rd_native


def generate(
    mol: Molecule,
    options: ETKDGOptions,
    constraints: CoordinationEmbeddingConstraints | None,
) -> ConformerBatch:
    """Dispatch one generation request to the compiled component."""

    return _backend().generate(
        pack_molecule(mol, _RDKIT_ETKDG_FIELDS),
        options,
        constraints,
    )


def runtime_info() -> RuntimeInfo:
    """Read compiled dependency identities from the native component."""

    return _backend().runtime_info()
