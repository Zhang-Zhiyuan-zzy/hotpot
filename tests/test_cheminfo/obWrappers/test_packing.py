"""Tests for the typed native Open Babel input boundary."""

from __future__ import annotations

import numpy as np

from hotpot import read_mol
from hotpot.cheminfo.obWrappers.contracts import BondKindCode
from hotpot.cheminfo.obWrappers.packing import _pack_molecule


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
