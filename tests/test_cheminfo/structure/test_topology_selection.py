"""Task graphs preserve complete factual membership and independent source bindings."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from hotpot import Bond, BondKind, Molecule
from hotpot.cheminfo.operations.topology_queries import _components, _project_molecule
from hotpot.cheminfo.structure._dependencies import _dependency_state
from hotpot.cheminfo.structure._identity import _capture_source, _check_source
from hotpot.cheminfo.structure._selection import (
    _TopologySelection, _capture_topology_selection, _check_topology_selection, _selected_bonds,
)
from tests.test_cheminfo._stage5_fixtures import factual_state


def _complex_with_a_ligand_ring():
    mol = Molecule()
    for row, number in enumerate((30, 7, 6, 6, 6)):
        mol.create_atom(atomic_number=number, coordinates=(row, row % 2, row % 3))
    metal = mol.add_bond(0, 1, bond_order=1., constraint=True, id=17,
                         bond_kind=BondKind.DATIVE, bond_direction='atom1_to_atom2',
                         bond_source='test', bond_source_metadata={'label': 'coordination'})
    ring = mol.add_bond(1, 2, bond_order=1., id=23)
    for first, second in ((2, 3), (3, 4), (4, 1)):
        mol.add_bond(first, second, bond_order=1.)
    return mol, metal, ring


def test_selections_keep_complete_handles_order_views_dependencies_and_frames():
    mol, metal, ring = _complex_with_a_ligand_ring()
    mol.conformer_add(mol.coordinates)
    # Settle the chemistry consumed by views before taking deterministic snapshots.
    rings, angle = tuple(mol.rings), mol.angles[0]
    expected_degrees = angle.degrees
    bonds, index, frames = tuple(mol.bonds), mol._topology_index, mol._source_frames
    frame_coordinates = mol.conformer_get(slice(None))['coordinates'].copy()
    attrs = tuple(bond.attr_dict for bond in bonds)
    state = factual_state(mol)
    status, seeds = _dependency_state(mol._atom_table)
    source = _capture_source(mol, ('topology', 'representation', 'source_frames'))
    first = _capture_topology_selection(mol, excluded_bonds=(metal, ring))
    second = _capture_topology_selection(mol, excluded_bonds=(ring,))
    assert first.bond_mask == b'\x00\x00\x01\x01\x01'
    assert _selected_bonds(mol, first) == bonds[2:]
    assert _selected_bonds(mol, second) == (metal, *bonds[2:])
    assert _selected_bonds(mol, None) == bonds
    assert tuple(mol.bonds) == bonds
    assert tuple(bond._uid for bond in mol.bonds) == tuple(bond._uid for bond in bonds)
    assert tuple(bond.attr_dict for bond in bonds) == attrs
    assert mol.bond(0, 1) is metal and mol.bond(1, 2) is ring
    assert mol.atoms[1] in mol.atoms[0].neighbours
    assert len(mol.rings) == len(rings) == 1
    assert angle.degrees == pytest.approx(expected_degrees)
    assert mol._topology_index is index and mol._source_frames is frames
    np.testing.assert_array_equal(mol.conformer_get(slice(None))['coordinates'], frame_coordinates)
    assert factual_state(mol) == state
    after_status, after_seeds = _dependency_state(mol._atom_table)
    np.testing.assert_array_equal(after_status, status)
    assert after_seeds == seeds
    _check_source(mol, source)


def test_full_zero_empty_and_immutable_selection_buffers():
    mol, _, _ = _complex_with_a_ligand_ring()
    full = _capture_topology_selection(mol)
    zero = _capture_topology_selection(mol, excluded_bonds=mol.bonds)
    assert full.bond_mask == bytes([1]) * len(mol.bonds)
    assert _selected_bonds(mol, zero) == ()
    assert full.atom_uids == tuple(atom._uid for atom in mol.atoms)
    mutable = bytearray(full.bond_mask)
    isolated = _TopologySelection(full.source, list(full.atom_uids), mutable)
    mutable[0] = 0
    assert isolated.bond_mask == full.bond_mask and type(isolated.atom_uids) is tuple
    with pytest.raises(FrozenInstanceError):
        full.bond_mask = b''
    empty = Molecule()
    assert _capture_topology_selection(empty).bond_mask == b''
    assert _selected_bonds(empty, _capture_topology_selection(empty)) == ()


def test_ligand_predicate_excludes_only_metal_nonmetal_edges():
    mol = Molecule()
    for number in (30, 29, 7, 8, 6):
        mol.create_atom(atomic_number=number)
    metal_metal = mol.add_bond(0, 1)
    coordination = mol.add_bond(0, 2)
    dative = mol.add_bond(2, 3, bond_kind=BondKind.DATIVE, bond_direction='atom1_to_atom2')
    unknown = mol.add_bond(3, 4, bond_order=0., bond_kind=BondKind.UNKNOWN)
    selection = _capture_topology_selection(mol, ligand_skeleton=True)
    assert selection.bond_mask == b'\x01\x00\x01\x01'
    assert _selected_bonds(mol, selection) == (metal_metal, dative, unknown)
    assert mol.bonds == [metal_metal, coordination, dative, unknown]


@pytest.mark.parametrize('invalid', ['duplicate', 'foreign', 'equal_endpoints', 'removed'])
def test_exclusions_validate_every_exact_handle_before_changing_any_source(invalid):
    mol, metal, ring = _complex_with_a_ligand_ring()
    if invalid == 'duplicate':
        bad = metal
    elif invalid == 'foreign':
        other, bad, _ = _complex_with_a_ligand_ring()
        assert bad._uid == metal._uid
    elif invalid == 'equal_endpoints':
        bad = Bond(metal.atom1, metal.atom2, bond_order=1.)
    else:
        bad = ring
        mol.remove_bond(bad)
    before = factual_state(mol)
    index = mol._topology_index
    with pytest.raises(ValueError):
        _capture_topology_selection(mol, excluded_bonds=(metal, bad))
    assert factual_state(mol) == before and mol._topology_index is index


@pytest.mark.parametrize('change', ['bond_order', 'atom_order', 'endpoints', 'remove_readd'])
def test_same_size_topology_changes_reject_stale_selection(change):
    mol, metal, ring = _complex_with_a_ligand_ring()
    selection = _capture_topology_selection(mol, excluded_bonds=(metal,))
    if change == 'bond_order':
        mol._bonds.reverse()
    elif change == 'atom_order':
        mol._atoms.reverse()
    elif change == 'endpoints':
        ring._atoms = (ring.atom2, ring.atom1)
    else:
        endpoints = ring.atoms
        mol.remove_bond(ring)
        mol.add_bond(*endpoints)
    before = tuple(mol._bonds)
    with pytest.raises(RuntimeError):
        _selected_bonds(mol, selection)
    assert tuple(mol._bonds) == before


def test_foreign_selection_rejects_equal_uids_and_identical_graph():
    mol, _, _ = _complex_with_a_ligand_ring()
    other, _, _ = _complex_with_a_ligand_ring()
    selection = _capture_topology_selection(mol)
    assert selection.atom_uids == tuple(atom._uid for atom in other.atoms)
    before = factual_state(other)
    with pytest.raises(RuntimeError, match='owner'):
        _project_molecule(other, Molecule, selection=selection)
    assert factual_state(other) == before


def test_coordinate_only_changes_preserve_selection_but_ligand_classification_is_bound():
    mol, metal, _ = _complex_with_a_ligand_ring()
    explicit = _capture_topology_selection(mol, excluded_bonds=(metal,))
    ligand = _capture_topology_selection(mol, ligand_skeleton=True)
    mol.coordinates = mol.coordinates + 1.
    _check_topology_selection(mol, explicit)
    _check_topology_selection(mol, ligand)
    mol.atoms[0].atomic_number = 6
    _check_topology_selection(mol, explicit)
    with pytest.raises(RuntimeError, match='representation'):
        _check_topology_selection(mol, ligand)


def test_ligand_selection_preserves_transported_nested_bytes_metadata_and_checks_changes():
    from hotpot.cheminfo.structure._builder import _freeze_value

    original, _, _ = _complex_with_a_ligand_ring()
    metadata = tuple({'row': row, 'raw': [b'C0', (row, None)]}
                     for row in range(len(original.bonds)))
    record, _ = original._capture_factual_record()
    mol = Molecule._build_molecule(replace(record, bond_metadata=tuple(map(_freeze_value, metadata))))
    assert tuple(bond.bond_source_metadata for bond in mol.bonds) == metadata
    selection = _capture_topology_selection(mol, ligand_skeleton=True)
    _check_topology_selection(mol, selection)
    assert selection.bond_mask == b'\x00\x01\x01\x01\x01'
    selected, selected_metadata = mol._capture_factual_record(selection=selection)
    assert len(selected.bond_endpoints) == 4
    assert selected_metadata == metadata[1:]
    projected = _project_molecule(mol, Molecule, selection=selection)
    assert tuple(bond.bond_source_metadata for bond in projected.bonds) == metadata[1:]
    parts = _components(mol, Molecule, with_bindings=True, selection=selection)
    assert [tuple(atom.idx for atom in bindings) for _, bindings in parts] == [(0,), (1, 2, 3, 4)]
    assert [tuple(bond.bond_source_metadata for bond in part.bonds) for part, _ in parts] == [(), metadata[1:]]
    mol.bonds[1]._bond_source_metadata['raw'][0] = b'C1'
    with pytest.raises(RuntimeError, match='representation'):
        _check_topology_selection(mol, selection)


@pytest.mark.parametrize('mask', [b'', b'\x01' * 4, b'\x01' * 6, b'\x02' * 5])
def test_invalid_masks_are_rejected_before_projection(mask):
    mol, _, _ = _complex_with_a_ligand_ring()
    selection = replace(_capture_topology_selection(mol), bond_mask=mask)
    before = factual_state(mol)
    with pytest.raises(ValueError):
        _project_molecule(mol, Molecule, selection=selection)
    assert factual_state(mol) == before


def test_projection_and_components_use_the_same_edges_without_reintroducing_chords():
    mol, metal, ring = _complex_with_a_ligand_ring()
    selection = _capture_topology_selection(mol, excluded_bonds=(metal, ring))
    before = factual_state(mol)
    projected = _project_molecule(mol, Molecule, selection=selection)
    assert len(projected.atoms) == len(mol.atoms)
    assert len(projected.bonds) == 3 and projected.rings == []
    assert [bond.id for bond in projected.bonds] == [bond.id for bond in mol.bonds[2:]]
    assert all(left is not right for left, right in zip(projected.atoms, mol.atoms))
    assert all(bond.mol is projected for bond in projected.bonds)
    parts = _components(mol, Molecule, with_bindings=True, selection=selection)
    assert [tuple(atom.idx for atom in bindings) for _, bindings in parts] == [(0,), (1, 2, 3, 4)]
    assert [len(part.bonds) for part, _ in parts] == [0, 3]
    assert len(mol.components) == 1 and len(mol.copy().bonds) == 5
    projected.atoms[1].formal_charge = -1
    projected.bonds[0].constraint = True
    assert factual_state(mol) == before


def test_selected_capture_preserves_aligned_raw_bond_rows_and_atom_mapping():
    mol, metal, ring = _complex_with_a_ligand_ring()
    record, metadata = mol._capture_factual_record()
    raw = np.frombuffer(record.bond_values, dtype='<u8').reshape(-1, 3).copy()
    raw[:, 1] = (0x8000000000000000, 0x7ff8000000000023, 0, 0x3ff0000000000000, 0xfff8000000000017)
    mol._install_molecule(replace(record, bond_values=raw.tobytes()), bond_metadata=metadata)
    selection = _capture_topology_selection(mol, excluded_bonds=(mol.bonds[0], mol.bonds[1]))
    complete, _ = mol._capture_factual_record()
    selected, selected_metadata = mol._capture_factual_record(selection=selection, rows=(4, 3, 2, 1))
    assert selected.bond_values == raw[2:].tobytes()
    assert selected.bond_endpoints == ((2, 1), (1, 0), (0, 3))
    assert selected.bond_kinds == complete.bond_kinds[2:]
    assert selected.bond_directions == complete.bond_directions[2:]
    assert selected.bond_sources == complete.bond_sources[2:]
    assert selected.bond_aromatic == complete.bond_aromatic[2:]
    assert selected_metadata == metadata[2:]
    atom_rows = np.frombuffer(complete.atom_values, dtype='<u8').reshape(-1, 19)
    assert selected.atom_values == atom_rows[[4, 3, 2, 1]].tobytes()
    assert selected.atom_presence == np.frombuffer(complete.atom_presence, dtype='<u4')[[4, 3, 2, 1]].tobytes()
    assert selected.atom_sources == np.frombuffer(complete.atom_sources, dtype='u1').reshape(-1, 19)[[4, 3, 2, 1]].tobytes()


def test_retired_public_methods_and_hidden_storage_are_absent():
    mol, _, _ = _complex_with_a_ligand_ring()
    for name in ('hide_bonds', 'hide_metal_ligand_bonds', 'restore_bonds',
                 'recover_hided_metal_ligand_bonds', 'recover_hided_covalent_bonds',
                 'clear_metal_ligand_bonds', 'clear_hided_covalent_bonds', 'clear_hided_bonds'):
        assert not hasattr(Molecule, name)
    assert not {'_hided_metal_bonds', '_hided_covalent_bonds', '_hided_bonds'} & mol.__dict__.keys()


def test_projection_recalculates_inferred_hydrogens_for_changed_inputs_only():
    from hotpot.cheminfo.operations.chemistry import _require_implicit_hydrogens
    mol = Molecule()
    mol.create_atom(atomic_number=6)
    mol.create_atom(atomic_number=6)
    bond = mol.add_bond(0, 1, bond_order=1.)
    assert _require_implicit_hydrogens(mol) == (3, 3)
    before = factual_state(mol)
    selection = _capture_topology_selection(mol, excluded_bonds=(bond,))
    projected = _project_molecule(mol, Molecule, selection=selection)
    assert _require_implicit_hydrogens(projected) == (4, 4)
    assert _require_implicit_hydrogens(mol) == (3, 3)
    assert factual_state(mol) == before
