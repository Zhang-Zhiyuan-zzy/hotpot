"""Internal access to the compiled RDKit wrapper."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from .contracts import (
    ConformerBatch,
    CoordinationEmbeddingConstraints,
    ETKDGOptions,
    RuntimeInfo,
)

from .._native.buffers import (
    NativeAtomTransportBuffers,
    NativeHydrogenTransportFieldCode,
    NativeMoleculeBuffers,
    NativeMoleculeField,
    _pack_utf8,
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

_HOTPOT_LEGACY_NAMESPACE = "hotpot-legacy/1"


def _backend():
    try:
        from . import _rd_native
    except ImportError as exc:
        raise RuntimeError("the native RDKit wrapper is not built") from exc
    return _rd_native


def _utf8_values(table) -> tuple[str | None, ...]:
    values = []
    for offset, length, present in zip(
        table.offsets, table.lengths, table.present, strict=True
    ):
        if not present:
            values.append(None)
            continue
        start = int(offset)
        stop = start + int(length)
        values.append(table.data[start:stop].tobytes().decode("utf-8"))
    return tuple(values)


def _add_legacy_hydrogen_transport(
    mol: Molecule, buffers: NativeMoleculeBuffers
) -> NativeMoleculeBuffers:
    """Match the established Hotpot-to-RDKit fallback for payload-free atoms."""

    transport = buffers.atom_transport
    if transport is None:
        return buffers

    transported_atoms = {int(atom) for atom in transport.atoms}
    missing_atoms = tuple(
        row for row in range(buffers.atom_count) if row not in transported_atoms
    )
    if not missing_atoms:
        return buffers

    count = len(missing_atoms)

    def append(array, values, dtype):
        return np.ascontiguousarray(
            np.concatenate((array, np.asarray(tuple(values), dtype=dtype)))
        )

    atom_transport = NativeAtomTransportBuffers(
        schema_version=transport.schema_version,
        atoms=append(transport.atoms, missing_atoms, np.uint64),
        presence_masks=append(transport.presence_masks, np.zeros(count), np.uint32),
        isotopes=append(transport.isotopes, np.zeros(count), np.int32),
        radical_electrons=append(
            transport.radical_electrons, np.zeros(count), np.int32
        ),
        has_current_hydrogens=append(
            transport.has_current_hydrogens, np.ones(count), np.uint32
        ),
        current_hydrogen_presence_masks=append(
            transport.current_hydrogen_presence_masks,
            np.full(
                count,
                NativeHydrogenTransportFieldCode.IMPLICIT,
                dtype=np.uint32,
            ),
            np.uint32,
        ),
        current_implicit_hydrogens=append(
            transport.current_implicit_hydrogens,
            (mol.atoms[row].implicit_hydrogens for row in missing_atoms),
            np.int32,
        ),
        current_bracket_explicit_hydrogens=append(
            transport.current_bracket_explicit_hydrogens,
            np.zeros(count),
            np.int32,
        ),
        current_no_implicit=append(
            transport.current_no_implicit, np.zeros(count), np.uint32
        ),
        has_source_hydrogens=append(
            transport.has_source_hydrogens, np.zeros(count), np.uint32
        ),
        source_hydrogen_presence_masks=append(
            transport.source_hydrogen_presence_masks,
            np.zeros(count),
            np.uint32,
        ),
        source_implicit_hydrogens=append(
            transport.source_implicit_hydrogens, np.zeros(count), np.int32
        ),
        source_bracket_explicit_hydrogens=append(
            transport.source_bracket_explicit_hydrogens,
            np.zeros(count),
            np.int32,
        ),
        source_no_implicit=append(
            transport.source_no_implicit, np.zeros(count), np.uint32
        ),
        source_namespaces=_pack_utf8(
            _utf8_values(transport.source_namespaces)
            + (_HOTPOT_LEGACY_NAMESPACE,) * count
        ),
    )
    return replace(buffers, atom_transport=atom_transport)


def generate(
    mol: Molecule,
    options: ETKDGOptions,
    constraints: CoordinationEmbeddingConstraints | None,
) -> ConformerBatch:
    """Dispatch one generation request to the compiled component."""

    buffers = pack_molecule(mol, _RDKIT_ETKDG_FIELDS)
    buffers = _add_legacy_hydrogen_transport(mol, buffers)
    return _backend().generate(buffers, options, constraints)


def runtime_info() -> RuntimeInfo:
    """Read compiled dependency identities from the native component."""

    return _backend().runtime_info()
