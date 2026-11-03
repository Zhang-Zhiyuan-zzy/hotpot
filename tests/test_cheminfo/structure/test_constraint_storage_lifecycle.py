"""Public constraint values and factual transport survive owner lifecycle edits."""

from copy import copy, deepcopy
from dataclasses import replace
import pickle

import numpy as np
import pytest

from hotpot import Atom, Bond, Molecule
from hotpot.cheminfo.structure._commit import _FactTransaction
from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES, FIELD_INDEX
from hotpot.cheminfo.structure._identity import _capture_source, _check_source
from tests.test_cheminfo._stage5_fixtures import factual_state
from tests.test_cheminfo.structure.test_constraints import molecule


_AXES = ('x_constraint', 'y_constraint', 'z_constraint')
_AXIS_COLUMNS = tuple(FIELD_INDEX[name] for name in _AXES)
_AXIS_MASK = sum(1 << column for column in _AXIS_COLUMNS)


def _axes(atom):
    return tuple(getattr(atom, name) for name in _AXES)


def _constrained_chain():
    mol = molecule()
    for row, atom in enumerate(mol.atoms):
        atom.id = 101 + row
    mol.atoms[0].x_constraint = True
    mol.atoms[1].y_constraint = False
    mol.atoms[2].x_constraint = True
    mol.atoms[2].z_constraint = True
    mol.atoms[3].constraint = True
    for row, bond in enumerate(mol.bonds):
        bond.id = 71 + row
        bond.constraint = row > 0
    return mol


def _mapped_constraints(mol):
    record, _ = mol._capture_factual_record()
    masks = np.frombuffer(record.atom_presence, dtype='<u4')
    sources = np.frombuffer(record.atom_sources, dtype='u1').reshape(-1, 19)
    atoms = {
        atom.id: (_axes(atom), int(masks[row]) & _AXIS_MASK,
                  tuple(sources[row, column] for column in _AXIS_COLUMNS))
        for row, atom in enumerate(mol.atoms)
    }
    bonds = {
        bond.id: ((bond.atom1.id, bond.atom2.id), bond.bond_order, bond.constraint)
        for bond in mol.bonds
    }
    return atoms, bonds


def test_standalone_atom_partial_constructor_setters_and_current_pickle():
    atom = Atom(atomic_number=6, id=43, coordinates=(1, 2, 3),
                x_constraint=True, y_constraint=False, z_constraint=True)
    assert _axes(atom) == (True, False, True)
    assert not atom.constraint
    atom.y_constraint = True
    assert atom.constraint
    atom.x_constraint = False
    restored = pickle.loads(pickle.dumps(atom))
    assert _axes(restored) == (False, True, True)
    assert not restored.constraint
    assert restored.id == 43 and restored.coordinates == (1, 2, 3)
    restored.z_constraint = False
    assert _axes(atom) == (False, True, True)


def test_removed_atom_is_inspectable_readonly_and_current_pickle_preserves_values():
    mol = _constrained_chain()
    atom = mol.atoms[2]
    mol.remove_atom(atom)
    before = atom.attrs.copy()
    remaining_facts = factual_state(mol)
    assert _axes(atom) == (True, False, True)
    with pytest.raises(KeyError):
        atom.y_constraint = True
    np.testing.assert_array_equal(atom.attrs, before)
    assert factual_state(mol) == remaining_facts
    restored = pickle.loads(pickle.dumps(atom))
    np.testing.assert_array_equal(restored.attrs, before)
    assert _axes(restored) == (True, False, True) and not restored.constraint
    assert restored.id == 103 and restored.coordinates == atom.coordinates
    with pytest.raises(KeyError):
        restored.y_constraint = True
    np.testing.assert_array_equal(restored.attrs, before)
    np.testing.assert_array_equal(atom.attrs, before)


def test_direct_nonmember_bond_setter_and_pickle_preserve_other_facts():
    mol = molecule()
    first, second = mol.atoms[0], mol.atoms[-1]
    bond = Bond(first, second, constraint=True, id=87)
    assert bond.mol is mol and not any(member is bond for member in mol.bonds)
    assert bond.constraint and bond.id == 87 and bond.bond_order == 0
    bond.constraint = False
    assert (bond.atom1, bond.atom2) == (first, second)
    assert bond.id == 87 and bond.bond_order == 0 and not bond.constraint
    restored = pickle.loads(pickle.dumps(bond))
    assert restored.id == 87 and restored.bond_order == 0 and not restored.constraint
    assert not any(member is restored for member in restored.mol.bonds)
    restored.constraint = True
    assert not bond.constraint


def test_combined_atom_constraint_reads_all_axes_and_writes_all_axes():
    atom = molecule().atoms[0]
    atom.x_constraint = True
    atom.z_constraint = True
    assert not atom.constraint
    atom.y_constraint = True
    assert atom.constraint
    atom.constraint = False
    assert _axes(atom) == (False, False, False)
    atom.constraint = True
    assert _axes(atom) == (True, True, True)
    atom.y_constraint = False
    assert _axes(atom) == (True, False, True) and not atom.constraint


def test_public_constraint_projections_keep_width_order_and_buffer_isolation():
    mol = _constrained_chain()
    atom, bond = mol.atoms[2], mol.bonds[1]
    atom_values, bond_values, matrix = atom.attrs, bond.attrs, mol.atom_matrix
    assert atom_values.shape == (19,) and matrix.shape == (4, 19)
    assert bond_values.shape == (3,)
    np.testing.assert_array_equal(atom_values[list(_AXIS_COLUMNS)], (1, 0, 1))
    np.testing.assert_array_equal(matrix[2, list(_AXIS_COLUMNS)], (1, 0, 1))
    np.testing.assert_array_equal(bond_values, (bond.bond_order, 1, 72))
    atom_values[list(_AXIS_COLUMNS)] = 0
    bond_values[1:] = 0
    matrix[:, list(_AXIS_COLUMNS)] = 0
    assert _axes(atom) == (True, False, True)
    assert bond.constraint and bond.id == 72


@pytest.mark.parametrize('clone', [copy, deepcopy, lambda mol: pickle.loads(pickle.dumps(mol))],
                         ids=['copy', 'deepcopy', 'current-pickle'])
def test_atom_and_bond_constraints_remap_with_uid_gaps_without_sharing(clone):
    mol = _constrained_chain()
    assert mol.atoms[0]._uid != mol.atoms[0].idx
    expected = _mapped_constraints(mol)
    other = clone(mol)
    assert _mapped_constraints(other) == expected
    other.atoms[2].x_constraint = False
    other.bonds[1].constraint = False
    assert _mapped_constraints(mol) == expected
    assert _axes(other.atoms[2]) == (False, False, True)
    assert other.bonds[1].id == 72


def test_removal_compacts_rows_without_reassigning_constraint_subjects():
    mol = _constrained_chain()
    atoms, bonds = _mapped_constraints(mol)
    retained_atom, retained_bond = mol.atoms[2], mol.bonds[1]
    atom_uid, bond_uid = retained_atom._uid, retained_bond._uid
    mol.remove_atom(mol.atoms[0])
    assert retained_atom.idx == 1 and retained_atom._uid == atom_uid
    assert retained_bond._uid == bond_uid and retained_bond.id == 72
    assert _mapped_constraints(mol) == (
        {key: value for key, value in atoms.items() if key != 101},
        {key: value for key, value in bonds.items() if key != 71})


def test_component_extraction_and_merge_preserve_mapped_axes_bond_values_and_sources():
    source = _constrained_chain()
    expected = _mapped_constraints(source)
    destination = Molecule()
    destination.create_atom(atomic_number=8, id=900)
    destination.add_component(source)
    component = next(part for part in destination.components if len(part.atoms) == 4)
    assert _mapped_constraints(component) == expected
    merged_atoms, merged_bonds = _mapped_constraints(destination)
    assert {key: merged_atoms[key] for key in expected[0]} == expected[0]
    assert merged_bonds == expected[1]
    component.clear_constraints()
    assert _mapped_constraints(source) == expected
    assert _mapped_constraints(destination) == (merged_atoms, merged_bonds)
    assert all(_axes(atom) == (False, False, False) for atom in component.atoms)
    assert not any(bond.constraint for bond in component.bonds)


def test_explicit_false_axis_retains_presence_and_producer_through_transport():
    mol = molecule()
    atom = mol.atoms[0]
    atom.id = 43
    before, _ = _mapped_constraints(mol)
    atom.y_constraint = False
    after, _ = _mapped_constraints(mol)
    assert not before[atom.id][1] & (1 << FIELD_INDEX['y_constraint'])
    assert after[atom.id][1] & (1 << FIELD_INDEX['y_constraint'])
    # The value remains false while its supplied presence/source becomes explicit.
    assert before[atom.id][0] == after[atom.id][0] == (False, False, False)
    record, _ = mol._capture_factual_record()
    presence = np.frombuffer(record.atom_presence, dtype='<u4')
    sources = np.frombuffer(record.atom_sources, dtype='u1').reshape(-1, 19)
    column = FIELD_INDEX['y_constraint']
    assert int(presence[0]) & (1 << column)
    assert sources[0, column] == ATOM_SOURCE_CODES['explicit']
    restored = pickle.loads(pickle.dumps(mol))
    assert _mapped_constraints(restored) == _mapped_constraints(mol)


def test_mixed_outer_rollback_restores_constraints_facts_ids_coordinates_and_sources():
    mol = _constrained_chain()
    angle, torsion = mol.angles[0], mol.torsions[0]
    angle.constraint = True
    coordinates = mol.coordinates.copy()
    expected = factual_state(mol)
    sources = [_capture_source(mol, (domain,))
               for domain in ('constraints', 'geometry', 'topology', 'representation')]
    original = MemoryError('interrupted mixed constraint transaction')
    with pytest.raises(MemoryError) as caught:
        with _FactTransaction(mol, producer=ATOM_SOURCE_CODES['geometry']):
            mol.atoms[0].constraint = True
            mol.atoms[0].id = 999
            mol.atoms[0].coordinates = (7, 8, 9)
            mol.bonds[1].constraint = False
            mol.bonds[1].id = 888
            angle.constraint = False
            torsion.constraint = True
            with pytest.raises(RuntimeError, match='constraints'):
                _check_source(mol, sources[0])
            with pytest.raises(RuntimeError, match='geometry'):
                _check_source(mol, sources[1])
            raise original
    assert caught.value is original
    assert factual_state(mol) == expected
    np.testing.assert_array_equal(mol.coordinates, coordinates)
    assert angle.constraint and not torsion.constraint
    assert mol.atoms[0].id == 101 and mol.bonds[1].id == 72
    for source in sources:
        _check_source(mol, source)


def _assert_canonical_constraint_bindings(mol):
    complete = mol._bonds
    axes, bonds = mol._constraints.atom_axes, mol._constraints.bond_values
    assert set(axes) == {atom._uid for atom in mol.atoms}
    assert set(bonds) == {bond._uid for bond in complete}
    assert all(atom._constraint_values is axes for atom in mol.atoms)
    assert all(bond._constraint_values is bonds for bond in complete)
    assert all(type(cell) is bytes and len(cell) == 28 for cell in axes.values())
    assert all(type(cell) is bytes and len(cell) == 8 for cell in bonds.values())
    assert len(mol._atom_table._columns) == 16
    assert mol._atom_table.sources().shape == (len(mol.atoms), 16)
    assert not (mol._atom_table.presence() >> 16).any()
    assert all(bond._attrs.shape == (2,) and bond.attrs.shape == (3,) for bond in complete)


def _raw_constraint_chain():
    mol = _constrained_chain()
    record, metadata = mol._capture_factual_record()
    values = np.frombuffer(record.atom_values, dtype='<f8').reshape(4, 19).copy()
    bits = np.array([
        0x8000000000000000, 0x7ff8000000000042, 0xc004000000000000,
        0x7ff8000000000037, 0, 0x3fe0000000000000,
        0x400a000000000000, 0x8000000000000000, 0xfff8000000000021,
        0, 0x3ff0000000000000, 0xbff8000000000000,
    ], dtype=np.uint64).reshape(4, 3)
    values[:, 16:] = bits.view(np.float64)
    presence = np.frombuffer(record.atom_presence, dtype='<u4').copy() & np.uint32(0xffff)
    presence |= np.array([0, 5, 2, 7], dtype=np.uint32) << 16
    sources = np.frombuffer(record.atom_sources, dtype='u1').reshape(4, 19).copy()
    sources[:, 16:] = np.array([[4, 14, 13], [0, 1, 2], [3, 5, 6], [17, 18, 19]], dtype=np.uint8)
    bonds = np.frombuffer(record.bond_values, dtype='<f8').reshape(3, 3).copy()
    bonds[:, 1] = np.array([0x8000000000000000, 0x7ff8000000000023,
                          0xc004000000000000], dtype=np.uint64).view(np.float64)
    raw = replace(record, atom_values=values.tobytes(), atom_presence=presence.tobytes(),
                  atom_sources=sources.tobytes(), bond_values=bonds.tobytes())
    mol._install_molecule(raw, bond_metadata=metadata)
    return mol


def _raw_constraint_transport(mol, *, rows=None):
    record, _ = mol._capture_factual_record(rows=rows)
    values = np.frombuffer(record.atom_values, dtype='<f8').reshape(record.atom_count, 19)
    sources = np.frombuffer(record.atom_sources, dtype='u1').reshape(record.atom_count, 19)
    masks = np.frombuffer(record.atom_presence, dtype='<u4') & np.uint32(_AXIS_MASK)
    return (values[:, 16:].tobytes(), masks.tobytes(), sources[:, 16:].tobytes(),
            record.bond_values, record.bond_endpoints)


@pytest.mark.parametrize('transport', ['copy', 'deepcopy', 'current-pickle', 'install', 'component', 'merge'])
def test_raw_constraint_bits_presence_and_sources_survive_all_current_transports(transport):
    source = _raw_constraint_chain()
    expected = _raw_constraint_transport(source)
    _assert_canonical_constraint_bindings(source)
    if transport == 'copy':
        other = copy(source)
    elif transport == 'deepcopy':
        other = deepcopy(source)
    elif transport == 'current-pickle':
        other = pickle.loads(pickle.dumps(source))
        assert [atom._uid for atom in other.atoms] == [atom._uid for atom in source.atoms]
        assert other._next_atom_uid == source._next_atom_uid
    elif transport == 'install':
        record, metadata = source._capture_factual_record()
        other = Molecule()
        other._install_molecule(record, bond_metadata=metadata)
    elif transport == 'component':
        other = source.components[0]
    elif transport == 'merge':
        destination = Molecule()
        destination.create_atom(atomic_number=8)
        destination.add_component(source)
        _assert_canonical_constraint_bindings(destination)
        assert _raw_constraint_transport(destination, rows=(1, 2, 3, 4)) == expected
        other = next(part for part in destination.components if len(part.atoms) == 4)
    assert _raw_constraint_transport(other) == expected
    _assert_canonical_constraint_bindings(other)
    assert other._constraints.atom_axes is not source._constraints.atom_axes
    assert other._constraints.bond_values is not source._constraints.bond_values
    other.atoms[0].constraint = False
    other.bonds[0].constraint = True
    assert _raw_constraint_transport(source) == expected


def test_removed_and_replaced_members_isolate_cells_without_mutating_previous_maps():
    mol = _raw_constraint_chain()
    atom, bond = mol.atoms[0], mol.bonds[0]
    axes, bonds = mol._constraints.atom_axes, mol._constraints.bond_values
    old_axes, old_bonds = dict(axes), dict(bonds)
    atom_cell, bond_cell = axes[atom._uid], bonds[bond._uid]
    mol.remove_atom(atom)
    _assert_canonical_constraint_bindings(mol)
    assert axes == old_axes and bonds == old_bonds
    assert atom._constraint_values == {atom._uid: atom_cell}
    assert bond._constraint_values == {bond._uid: bond_cell}
    assert atom._constraint_values is not axes and bond._constraint_values is not bonds
    with pytest.raises(KeyError):
        atom.x_constraint = True
    with pytest.raises(ValueError, match='endpoints'):
        bond.constraint = True
    assert atom._constraint_values[atom._uid] is atom_cell
    assert bond._constraint_values[bond._uid] is bond_cell
    restored = pickle.loads(pickle.dumps(atom))
    assert restored._constraint_values == {restored._uid: atom_cell}
    destination = Molecule()
    inserted = destination.add_atom(restored)
    assert inserted._constraint_values[inserted._uid] == atom_cell
    _assert_canonical_constraint_bindings(destination)

    incoming = _raw_constraint_chain().atoms[2]
    incoming_cell = incoming._constraint_values[incoming._uid]
    replaced = mol.atoms[1]
    replaced_cell = replaced._constraint_values[replaced._uid]
    mol._replace_atom(1, incoming)
    _assert_canonical_constraint_bindings(mol)
    assert replaced._constraint_values == {replaced._uid: replaced_cell}
    assert mol.atoms[1]._constraint_values[mol.atoms[1]._uid] == incoming_cell
    assert all(replaced is not endpoint for member in mol.bonds for endpoint in member.atoms)


def test_endpoint_equal_nonmember_removal_isolates_the_actual_removed_bond():
    mol = _constrained_chain()
    actual = mol.bonds[0]
    actual.constraint = True
    alias = Bond(actual.atom1, actual.atom2, constraint=False, id=87)
    actual_cell = actual._constraint_values[actual._uid]
    alias_map, alias_cell = alias._constraint_values, alias._constraint_values[alias._uid]
    remaining = tuple(mol.bonds[1:])
    mol.remove_bond(alias)
    assert tuple(mol.bonds) == remaining
    _assert_canonical_constraint_bindings(mol)
    assert actual._uid not in mol._constraints.bond_values
    assert alias._uid not in mol._constraints.bond_values
    assert actual._constraint_values == {actual._uid: actual_cell}
    assert actual._constraint_values is not mol._constraints.bond_values
    assert alias._constraint_values is alias_map and alias_map[alias._uid] is alias_cell
    actual.constraint = False
    alias.constraint = True
    assert not actual.constraint and alias.constraint and alias.id == 87
    assert tuple(mol.bonds) == remaining


@pytest.mark.parametrize('removed_rows', [(0,), (1,), (0, 1)])
def test_complete_constraints_clear_and_permanent_removal_detaches_raw_cells(removed_rows):
    from tests.test_cheminfo.structure.test_topology_selection import _complex_with_a_ligand_ring

    mol, metal, covalent = _complex_with_a_ligand_ring()
    metal.constraint = covalent.constraint = True
    mol.atoms[0].constraint = True
    mol.angles[0].constraint = True
    mol.clear_constraints()
    assert not metal.constraint and not covalent.constraint
    assert all(_axes(atom) == (False, False, False) for atom in mol.atoms)
    assert not any(bond.constraint for bond in mol.bonds)
    assert not mol._constraints.angles and not mol._constraints.torsions
    _assert_canonical_constraint_bindings(mol)
    metal.constraint = covalent.constraint = True
    cells = {bond._uid: bond._constraint_values[bond._uid] for bond in (metal, covalent)}
    previous_map = mol._constraints.bond_values
    previous_cells = dict(previous_map)
    removed = tuple((metal, covalent)[row] for row in removed_rows)
    mol.remove_bonds(removed)
    _assert_canonical_constraint_bindings(mol)
    assert previous_map == previous_cells
    for bond in removed:
        assert bond._uid not in mol._constraints.bond_values
        assert bond._constraint_values == {bond._uid: cells[bond._uid]}
        bond.constraint = False
        assert not bond.constraint
    for bond in (metal, covalent):
        if all(bond is not member for member in removed):
            assert bond.constraint and bond._constraint_values is mol._constraints.bond_values


def test_nonmember_constructor_failure_and_later_registration_do_not_leak_cells():
    mol = molecule()
    before = dict(mol._constraints.bond_values)
    first, last = mol.atoms[0], mol.atoms[-1]
    with pytest.raises(TypeError, match='Unsupported'):
        Bond(first, last, constraint=True, id=87, unknown=True)
    assert mol._constraints.bond_values == before
    nonmember = Bond(first, last, constraint=True, id=87)
    assert mol._constraints.bond_values == before
    member = mol.add_bond(first, last, constraint=True, id=88)
    assert member is not nonmember
    _assert_canonical_constraint_bindings(mol)
    assert nonmember._uid not in mol._constraints.bond_values
    assert nonmember._constraint_values == {nonmember._uid: nonmember._constraint_values[nonmember._uid]}
    assert member.id == 88 and nonmember.id == 87
