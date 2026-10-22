"""Tests for the typed native Open Babel input boundary."""

from __future__ import annotations

import numpy as np

from hotpot import read_mol
from hotpot.cheminfo._native.buffers import (
    MOLECULE_BUFFER_SCHEMA_VERSION,
    NativeMoleculeBuffers,
    NativeMoleculeField,
)
from hotpot.cheminfo.obWrappers.contracts import BondKindCode
from hotpot.cheminfo.obWrappers.packing import (
    _BOND_KIND_CODES,
    _MOLECULE_BUFFER_SCHEMA_VERSION,
    _MoleculeBuffers,
    _OPENBABEL_FIELDS,
    _empty_matrix,
    _pack_molecule,
)


def test_openbabel_compatibility_names_forward_to_common_buffers():
    mol = read_mol("CCO")

    buffers = _pack_molecule(mol)

    assert _MoleculeBuffers is NativeMoleculeBuffers
    assert isinstance(buffers, _MoleculeBuffers)
    assert buffers.atom_count == len(mol.atoms)
    assert buffers.bond_count == len(mol.bonds)
    assert _OPENBABEL_FIELDS == frozenset(
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
    assert all(
        getattr(buffers, field.value) is not None
        for field in _OPENBABEL_FIELDS
        if field is not NativeMoleculeField.UNIT_CELL
    )
    assert buffers.stereo_records is None
    assert buffers.atom_transport is None
    assert buffers.bond_transport is None
    assert _MOLECULE_BUFFER_SCHEMA_VERSION == MOLECULE_BUFFER_SCHEMA_VERSION
    assert all(
        isinstance(code, BondKindCode) for code in _BOND_KIND_CODES.values()
    )
    assert _empty_matrix(2, np.dtype(np.int32)).shape == (0, 2)


def test_pack_molecule_preserves_order_semantics_and_exact_dtypes():
    mol = read_mol("c1ccccc1P(=O)(O)O")
    mol.atoms[0].partial_charge = -0.125

    buffers = _pack_molecule(mol)

    assert buffers.schema_version == 1
    assert buffers.atomic_numbers.dtype == np.int32
    assert buffers.formal_charges.dtype == np.int32
    assert buffers.partial_charges.dtype == np.float64
    assert buffers.coordinates.dtype == np.float64
    assert buffers.atom_aromatic.dtype == np.uint8
    assert buffers.bond_indices.dtype == np.int32
    assert buffers.bond_orders.dtype == np.float64
    assert buffers.bond_kinds.dtype == np.uint8
    assert buffers.bond_aromatic.dtype == np.uint8
    assert all(
        array.flags.c_contiguous
        for array in (
            buffers.atomic_numbers,
            buffers.formal_charges,
            buffers.partial_charges,
            buffers.coordinates,
            buffers.atom_aromatic,
            buffers.bond_indices,
            buffers.bond_orders,
            buffers.bond_kinds,
            buffers.bond_aromatic,
        )
    )

    assert buffers.coordinates.shape == (len(mol.atoms), 3)
    assert buffers.bond_indices.shape == (len(mol.bonds), 2)
    assert buffers.partial_charges[0] == -0.125
    assert np.array_equal(
        buffers.bond_indices,
        np.asarray(
            [(bond.atom1.idx, bond.atom2.idx) for bond in mol.bonds],
            dtype=np.int32,
        ),
    )
    assert set(buffers.bond_kinds).issubset(
        {
            BondKindCode.SINGLE,
            BondKindCode.DOUBLE,
            BondKindCode.TRIPLE,
            BondKindCode.AROMATIC,
        }
    )
    assert np.array_equal(
        buffers.atom_aromatic,
        np.asarray([atom.is_aromatic for atom in mol.atoms], dtype=np.uint8),
    )


def test_pack_molecule_represents_empty_bond_table_and_optional_unit_cell():
    mol = read_mol("[Eu+3]")

    no_cell = _pack_molecule(mol)

    assert no_cell.bond_indices.shape == (0, 2)
    assert no_cell.bond_orders.shape == (0,)
    assert no_cell.bond_kinds.shape == (0,)
    assert no_cell.bond_aromatic.shape == (0,)
    assert no_cell.unit_cell is None

    mol.create_crystal(10.0, 11.0, 12.0, 90.0, 100.0, 120.0)
    with_cell = _pack_molecule(mol)

    assert with_cell.unit_cell is not None
    assert with_cell.unit_cell.dtype == np.float64
    assert with_cell.unit_cell.flags.c_contiguous
    assert np.array_equal(
        with_cell.unit_cell,
        np.asarray((10.0, 11.0, 12.0, 90.0, 100.0, 120.0)),
    )


def test_packing_does_not_replace_openbabel_hybridization_semantics():
    mol = read_mol("CP(=O)(OC)OC")

    buffers = _pack_molecule(mol)

    assert not hasattr(buffers, "hybridizations")
    assert not hasattr(buffers, "is_metal")
