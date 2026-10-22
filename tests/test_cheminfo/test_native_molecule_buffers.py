"""Contracts for the consumer-neutral field-selective molecule payload."""

from __future__ import annotations

import numpy as np

from hotpot import read_mol
from hotpot.cheminfo._native.buffers import (
    NativeBondKindCode,
    NativeMoleculeBuffers,
    NativeMoleculeField,
    pack_molecule,
)


def test_empty_selection_preserves_dimensions_without_materializing_fields():
    mol = read_mol("CCO")

    buffers = pack_molecule(mol, frozenset())

    assert isinstance(buffers, NativeMoleculeBuffers)
    assert buffers.atom_count == len(mol.atoms)
    assert buffers.bond_count == len(mol.bonds)
    assert all(
        getattr(buffers, field.value) is None for field in NativeMoleculeField
    )


def test_selected_fields_are_exact_contiguous_arrays():
    mol = read_mol("c1ccccc1P(=O)(O)O")
    fields = frozenset(
        {
            NativeMoleculeField.ATOMIC_NUMBERS,
            NativeMoleculeField.COORDINATES,
            NativeMoleculeField.BOND_INDICES,
            NativeMoleculeField.BOND_KINDS,
        }
    )

    buffers = pack_molecule(mol, fields)

    assert buffers.atomic_numbers.dtype == np.int32
    assert buffers.coordinates.dtype == np.float64
    assert buffers.coordinates.shape == (buffers.atom_count, 3)
    assert buffers.bond_indices.dtype == np.int32
    assert buffers.bond_indices.shape == (buffers.bond_count, 2)
    assert buffers.bond_kinds.dtype == np.uint8
    assert all(
        array.flags.c_contiguous
        for array in (
            buffers.atomic_numbers,
            buffers.coordinates,
            buffers.bond_indices,
            buffers.bond_kinds,
        )
    )
    assert buffers.formal_charges is None
    assert buffers.partial_charges is None


def test_empty_present_bond_field_differs_from_absence():
    mol = read_mol("[Eu+3]")

    present = pack_molecule(
        mol,
        frozenset({NativeMoleculeField.BOND_INDICES}),
    )
    absent = pack_molecule(mol, frozenset())

    assert present.bond_indices.shape == (0, 2)
    assert absent.bond_indices is None


def test_common_bond_codes_retain_dative_semantics():
    assert NativeBondKindCode.ZERO == 5
    assert NativeBondKindCode.DATIVE == 6
    assert NativeBondKindCode.UNKNOWN == 7
