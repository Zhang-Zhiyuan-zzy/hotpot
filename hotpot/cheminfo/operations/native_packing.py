"""Prepare selected molecule facts and pack native transport buffers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..adapters.native.molecule_buffers import (
    NativeMoleculeBuffers,
    NativeMoleculeField,
    _pack_atom_transport,
    _pack_bond_transport,
    _pack_molecule_fields,
    _pack_stereo_records,
)

if TYPE_CHECKING:
    from ..structure.molecule import Molecule


def pack_molecule(
    mol: Molecule,
    fields: frozenset[NativeMoleculeField],
) -> NativeMoleculeBuffers:
    """Copy only the requested molecule fields into contiguous arrays."""
    with mol._mutation_lock:
        if fields & {NativeMoleculeField.ATOM_AROMATIC, NativeMoleculeField.BOND_AROMATIC}:
            from .chemistry import _require_aromatic_representation
            _require_aromatic_representation(mol)

        atoms = tuple(mol.atoms)
        bonds = tuple(mol.bonds)
        molecule_fields = _pack_molecule_fields(mol, fields, atoms, bonds)

        stereo_records = None
        atom_transport = None
        bond_transport = None
        stereo_fields = fields & frozenset(
            {
                NativeMoleculeField.STEREO_RECORDS,
                NativeMoleculeField.ATOM_TRANSPORT,
                NativeMoleculeField.BOND_TRANSPORT,
            }
        )
        if stereo_fields:
            from ..structure._stereo import _get_stereo_payload

            payload = _get_stereo_payload(mol)
            if NativeMoleculeField.STEREO_RECORDS in fields:
                stereo_records = _pack_stereo_records(payload)
            if NativeMoleculeField.ATOM_TRANSPORT in fields:
                from .chemistry import _require_non_graph_hydrogens

                hydrogens = _require_non_graph_hydrogens(mol)
                atom_transport = _pack_atom_transport(payload, atoms, hydrogens)
            if NativeMoleculeField.BOND_TRANSPORT in fields:
                bond_transport = _pack_bond_transport(payload)

        return NativeMoleculeBuffers(
            **molecule_fields,
            stereo_records=stereo_records,
            atom_transport=atom_transport,
            bond_transport=bond_transport,
        )
