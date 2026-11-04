"""Tests for the typed native Open Babel input boundary."""

from __future__ import annotations

import numpy as np
import pytest

from hotpot import Molecule, read_mol
from hotpot.cheminfo.adapters.native.molecule_buffers import (
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
    from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES
    from hotpot.cheminfo.structure._results import _import_partial_charges

    _import_partial_charges(
        mol, (-0.125,) + (None,) * (len(mol.atoms) - 1),
        producer=ATOM_SOURCE_CODES["explicit"],
    )

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
    np.testing.assert_allclose(
        with_cell.unit_cell,
        np.asarray((10.0, 11.0, 12.0, 90.0, 100.0, 120.0)),
        rtol=0,
        atol=1e-12,
    )


def test_packing_does_not_replace_openbabel_hybridization_semantics():
    mol = read_mol("CP(=O)(OC)OC")

    buffers = _pack_molecule(mol)

    assert not hasattr(buffers, "hybridizations")
    assert not hasattr(buffers, "is_metal")


def test_task_packing_uses_zero_for_absent_charge_without_selecting_a_result():
    mol = read_mol("CCO")

    buffers = _pack_molecule(mol)

    assert buffers.partial_charges.tolist() == [0.0, 0.0, 0.0]
    assert np.isfinite(buffers.partial_charges).all()
    assert tuple(atom.partial_charge for atom in mol.atoms) == (None,) * 3
    assert mol._partial_charge_result is None


def test_task_packing_preserves_current_rows_and_encodes_stale_rows_without_writeback():
    from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES
    from hotpot.cheminfo.structure._results import _import_partial_charges

    mol = read_mol("CCO")
    _import_partial_charges(mol, (None, 0.0, -0.25), producer=ATOM_SOURCE_CODES["explicit"])
    selected = mol._partial_charge_result
    assert _pack_molecule(mol).partial_charges.tolist() == [0.0, 0.0, -0.25]
    assert mol._partial_charge_result is selected
    assert tuple(atom.partial_charge for atom in mol.atoms) == (None, 0.0, -0.25)

    mol.atoms[0].x = 2.0
    stale = mol._partial_charge_result
    assert stale.invalidated_by
    assert _pack_molecule(mol).partial_charges.tolist() == [0.0, 0.0, 0.0]
    assert mol._partial_charge_result is stale
    assert stale.payload.values == (None, 0.0, -0.25)


@pytest.mark.parametrize('excluded_rows', [(), (0,), (1, 4, 7)])
def test_selected_packing_keeps_intact_atom_inputs_and_ordered_bond_rows(excluded_rows):
    from hotpot.cheminfo.structure._selection import _capture_topology_selection
    from tests.test_cheminfo._stage5_fixtures import factual_state

    from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES
    from hotpot.cheminfo.structure._results import _import_partial_charges

    mol = read_mol("c1ccccc1C(=O)N")
    mol.create_crystal(10., 11., 12., 90., 90., 90.)
    _import_partial_charges(
        mol, (None, 0., -0.25) + (None,) * (len(mol.atoms) - 3),
        producer=ATOM_SOURCE_CODES["explicit"],
    )
    full = _pack_molecule(mol)
    before = factual_state(mol)
    atoms, bonds = tuple(mol.atoms), tuple(mol.bonds)
    selection = _capture_topology_selection(
        mol, excluded_bonds=tuple(bonds[row] for row in excluded_rows),
    )

    selected = _pack_molecule(mol, selection=selection)

    assert selected.atom_count == full.atom_count == len(atoms)
    assert selected.bond_count == len(bonds) - len(excluded_rows)
    assert selected.schema_version == full.schema_version
    retained_rows = [row for row in range(len(bonds)) if row not in excluded_rows]
    bond_fields = ('bond_indices', 'bond_orders', 'bond_kinds', 'bond_aromatic')
    atom_fields = (
        'atomic_numbers', 'formal_charges', 'partial_charges', 'coordinates',
        'atom_aromatic', 'unit_cell',
    )
    for name in (*atom_fields, *bond_fields):
        actual, original = getattr(selected, name), getattr(full, name)
        expected = original[retained_rows] if name in bond_fields else original
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == original.dtype and actual.flags.c_contiguous
        assert not np.shares_memory(actual, original)
        actual[...] = 0
    assert factual_state(mol) == before
    assert tuple(mol.atoms) == atoms and tuple(mol.bonds) == bonds
    assert full.atom_aromatic[:6].tolist() == [1] * 6


def test_selected_packing_can_exclude_every_bond_without_changing_atom_typing_inputs():
    from hotpot.cheminfo.structure._selection import _capture_topology_selection

    mol = read_mol("c1ccccc1")
    selection = _capture_topology_selection(mol, excluded_bonds=tuple(mol.bonds))
    buffers = _pack_molecule(mol, selection=selection)

    assert buffers.atom_count == 6 and buffers.bond_count == 0
    assert buffers.atom_aromatic.tolist() == [1] * 6
    assert buffers.bond_indices.shape == (0, 2)
    assert buffers.bond_indices.dtype == np.int32
    for name, dtype in [('bond_orders', np.float64), ('bond_kinds', np.uint8),
                        ('bond_aromatic', np.uint8)]:
        values = getattr(buffers, name)
        assert values.shape == (0,) and values.dtype == dtype
        assert values.flags.c_contiguous
    assert len(mol.bonds) == 6


@pytest.mark.parametrize('selected', [False, True])
def test_native_packing_rejects_noncyclic_source_even_when_every_bond_is_excluded(selected):
    from hotpot.cheminfo.structure._selection import _capture_topology_selection
    from tests.test_cheminfo._stage5_fixtures import factual_state, supplied_aromatic_flags

    mol = Molecule()
    mol.create_atom(atomic_number=6)
    mol.create_atom(atomic_number=6)
    mol.add_bond(0, 1)
    mol = supplied_aromatic_flags(mol, (True, False))
    selection = _capture_topology_selection(mol, excluded_bonds=tuple(mol.bonds)) if selected else None
    before = factual_state(mol)

    with pytest.raises(RuntimeError, match=r'AROMATIC\.NONCYCLIC_SOURCE'):
        _pack_molecule(mol, selection=selection)

    assert factual_state(mol) == before
    assert mol._atom_table.column('is_aromatic').tolist() == [1., 0.]


@pytest.mark.parametrize('invalid', ['foreign', 'reordered'])
def test_selected_packing_rejects_a_mask_bound_to_another_source(invalid):
    from hotpot.cheminfo.structure._selection import _capture_topology_selection
    from tests.test_cheminfo._stage5_fixtures import factual_state

    mol = read_mol("CCC")
    selection = _capture_topology_selection(mol, excluded_bonds=(mol.bonds[0],))
    if invalid == 'foreign':
        mol = read_mol("CCC")
    else:
        mol._bonds.reverse()
    before = factual_state(mol)

    with pytest.raises(RuntimeError):
        _pack_molecule(mol, selection=selection)

    assert factual_state(mol) == before
