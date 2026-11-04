"""Independent connected cell facts, aliases, member maps and real rollback."""

import builtins
import pickle

import numpy as np
import pytest

from hotpot import Crystal, Molecule, SpaceGroup
from hotpot.cheminfo.structure._builder import _restore_molecule_record
from hotpot.cheminfo.structure._commit import _FactTransaction
from hotpot.cheminfo.structure._identity import _capture_source, _check_source


def molecule():
    value = Molecule()
    for number, xyz in ((6, (1., 0., 0.)), (6, (4., 0., 0.)), (8, (2., 1., 0.))):
        value.create_atom(atomic_number=number, coordinates=xyz)
    value.add_bond(0, 1, bond_order=1)
    return value


def relations(group):
    return {'equivalent_images': (((0, group.identity(), (0, 0, 0)),
                                  (1, group.lift(1), (1, 0, 0))),),
            'occupancies': (.5, .5, None), 'disorder': (('A', 1), ('A', 2), None)}


def associated():
    mol = molecule()
    group = SpaceGroup(2)
    cell = Crystal.from_matrix(((0., 5., 0.), (-6., 0., 0.), (0., 0., 7.)), molecule=mol,
                               space_group=group, site_relations=relations(group),
                               bond_images=((1, -2, 0),))
    return mol, cell


def test_basis_orientation_parameters_coordinates_and_read_isolation():
    mol, cell = associated()
    np.testing.assert_array_equal(cell.matrix, ((0., 5., 0.), (-6., 0., 0.), (0., 0., 7.)))
    assert (cell.a, cell.b, cell.c) == pytest.approx((5, 6, 7))
    assert cell.volume == pytest.approx(210)
    assert cell.molecule is mol and mol.crystal is cell
    view = cell.matrix
    view[:] = 0
    assert cell.volume == pytest.approx(210)
    xyz = np.array([[[1., 2., 3.], [-1., 0., 7.]]])
    np.testing.assert_allclose(cell.to_cartesian(cell.to_fractional(xyz)), xyz, atol=1e-10)
    assert cell.symm_units == cell.space_group.symm_units
    with pytest.raises(TypeError):
        cell.site_relations['occupancies'] = ()
    with pytest.raises(AttributeError):
        cell.space_group = SpaceGroup(1)


def test_cell_replacement_preserves_declared_coordinate_mode_and_scalar_orientation():
    mol, cell = associated()
    before = mol.coordinates.copy()
    original = cell.matrix
    cell.set_cell(original * 2)
    np.testing.assert_array_equal(mol.coordinates, before)
    fractional = cell.to_fractional(before)
    cell.set_cell(original, preserve='fractional')
    np.testing.assert_allclose(mol.coordinates, before / 2, atol=1e-10)
    np.testing.assert_allclose(cell.to_fractional(mol.coordinates), fractional, atol=1e-10)
    before = mol.coordinates.copy()
    cell.a = 10
    np.testing.assert_allclose(cell.matrix[0], (0., 10., 0.), atol=1e-10)
    np.testing.assert_array_equal(mol.coordinates, before)


def test_direct_constructor_cannot_steal_association_and_explicit_replacement_detaches():
    mol, old = associated()
    with pytest.raises(ValueError, match='already has an active Crystal'):
        Crystal(1, 1, 1, 90, 90, 90, molecule=mol)
    before = old.matrix
    new = mol.create_crystal(8, 8, 8, 90, 90, 90)
    assert new is mol.crystal and new.molecule is mol
    assert old.molecule is None
    np.testing.assert_array_equal(old.matrix, before)
    assert old.space_group == SpaceGroup(2)
    assert old.bond_images == () and old.image_map is None
    assert old.site_relations['occupancies'] == ()
    assert mol.create_crystal(8, 8, 8, 90, 90, 90) is new


@pytest.mark.parametrize('protocol', range(pickle.HIGHEST_PROTOCOL + 1))
@pytest.mark.parametrize('crystal_first', [False, True])
def test_combined_pickle_aliases_keep_actual_member_uids_and_independent_mutable_facts(protocol, crystal_first):
    mol, cell = associated()
    mol.remove_atom(2)
    mol.create_atom(atomic_number=7, coordinates=(3, 0, 0))
    atom, bond = mol.atoms[-1], mol.bonds[0]
    items = (cell, mol, atom, bond) if crystal_first else (mol, cell, atom, bond)
    restored = pickle.loads(pickle.dumps(items, protocol=protocol))
    newcell, newmol, newatom, newbond = restored if crystal_first else (restored[1], restored[0], *restored[2:])
    assert newcell is newmol.crystal and newcell.molecule is newmol
    assert newatom is newmol.atoms[-1] and newbond is newmol.bonds[0]
    assert newatom._uid == atom._uid and newbond._uid == bond._uid
    assert newcell.bond_images == cell.bond_images
    assert newcell.site_relations == cell.site_relations
    assert newcell is not cell
    newcell.a = 10
    assert cell.a == pytest.approx(5)
    newmol.atoms[0].coordinates = (99., 0., 0.)
    assert mol.atoms[0].coordinates[0] == 1


def test_copy_components_merge_and_member_reorder_bind_actual_subjects():
    mol, cell = associated()
    clone = mol.copy()
    assert clone.crystal.molecule is clone and clone.crystal is not cell
    assert clone.crystal.bond_images == ((1, -2, 0),)
    assert clone.crystal.site_relations == cell.site_relations
    pieces = mol.components
    assert [len(part.atoms) for part in pieces] == [2, 1]
    assert pieces[0].crystal.bond_images == ((1, -2, 0),)
    assert pieces[1].crystal.site_relations['occupancies'] == (None,)
    merged = Molecule()
    for part in pieces:
        merged.add_component(part)
    assert merged.crystal.molecule is merged
    assert merged.crystal.site_relations == cell.site_relations
    assert merged.crystal.bond_images == cell.bond_images
    # The existing topology boundary can reorder members without replacing their UIDs.
    with _FactTransaction(mol, producer=1) as transaction:
        transaction.touch_topology()
        mol._atoms.reverse()
        mol._atom_table = mol._atom_table.take((2, 1, 0))
        mol._update_topology()
    assert cell.site_relations['occupancies'] == (None, .5, .5)
    assert cell.site_relations['disorder'] == (None, ('A', 2), ('A', 1))
    mol.remove_atom(1)
    assert cell.site_relations['equivalent_images'] == ()
    assert cell.bond_images == ()


def test_incompatible_merge_rejects_context_free_nonempty_or_different_oriented_basis():
    mol, cell = associated()
    other = molecule()
    with pytest.raises(ValueError, match='shared explicit cell context'):
        mol.add_component(other)
    assert mol.crystal is cell and len(mol.atoms) == 3
    other.create_crystal(5, 6, 7, 90, 90, 90)
    with pytest.raises(ValueError, match='compatible unit cells'):
        mol.add_component(other)
    assert mol.crystal is cell and len(mol.atoms) == 3


def test_group_relation_replacement_is_atomic_and_does_not_dirty_physical_cell_identity():
    mol, cell = associated()
    source = _capture_source(mol, ('cell',))
    before = cell._state
    with pytest.raises(ValueError, match='IMAGE-MAP-INTEGRITY'):
        cell.set_space_group(None)
    assert cell._state is before
    cell.set_space_group(None, site_relations={'equivalent_images': (), 'occupancies': (None,) * 3,
                                             'disorder': (None,) * 3})
    assert cell.space_group is None and cell.symm_units is None
    _check_source(mol, source)
    cell.set_space_group(SpaceGroup(1))
    _check_source(mol, source)
    cell.set_cell(cell.matrix * 2)
    with pytest.raises(RuntimeError, match='cell'):
        _check_source(mol, source)


def test_group_metric_failure_preserves_all_cell_facts():
    mol, cell = associated()
    before = cell._state
    with pytest.raises(ValueError, match='CRYSTAL-CONTEXT-INTEGRITY'):
        cell.set_space_group(SpaceGroup(225), site_relations={'equivalent_images': (),
             'occupancies': (None,) * 3, 'disorder': (None,) * 3})
    assert cell._state is before


def test_outer_failure_restores_old_and_new_associations_mutated_cell_and_members():
    mol, cell = associated()
    state, xyz, atoms, bonds = cell._state, mol.coordinates.copy(), mol.atoms, mol.bonds
    with pytest.raises(KeyboardInterrupt, match='restore'):
        with _FactTransaction(mol, producer=1):
            cell.set_cell(cell.matrix * 2, preserve='fractional')
            mol.remove_atom(0)
            new = mol.create_crystal(11, 12, 13, 90, 90, 90)
            raise KeyboardInterrupt('restore')
    assert mol.crystal is cell and cell.molecule is mol and cell._state is state
    assert new.molecule is None and new.bond_images == ()
    assert mol.atoms == atoms and mol.bonds == bonds
    np.testing.assert_array_equal(mol.coordinates, xyz)


def test_late_coordinate_failure_rolls_back_cell_and_real_coordinate_write(monkeypatch):
    mol, cell = associated()
    before, xyz = cell._state, mol.coordinates.copy()
    original = Molecule.coordinates
    def fail_after_write(mol, value):
        original.fset(mol, value)
        raise RuntimeError('late coordinate failure')
    monkeypatch.setattr(Molecule, 'coordinates', property(original.fget, fail_after_write))
    with pytest.raises(RuntimeError, match='late coordinate failure'):
        cell.set_cell(cell.matrix * 2, preserve='fractional')
    assert cell._state is before and mol.crystal is cell and cell.molecule is mol
    np.testing.assert_array_equal(mol.coordinates, xyz)


def test_record_generation7_hard_cut_and_native_six_value_projection():
    from hotpot.cheminfo.adapters.native.molecule_buffers import _pack_molecule_fields, NativeMoleculeField
    mol, cell = associated()
    record = mol._capture_factual_record()[0]
    assert record.cell[0] == 'hotpot.crystal/1' and record.cell[1] == tuple(map(tuple, cell.matrix))
    restore, (kind, generation, values) = record.__reduce__()
    assert generation == 7 and len(values) == 24
    with pytest.raises(ValueError, match='version 7'):
        _restore_molecule_record(kind, 6, values)
    assert restore(kind, generation, values).cell == record.cell
    packed = _pack_molecule_fields(mol, frozenset((NativeMoleculeField.UNIT_CELL,)), mol.atoms, mol.bonds)
    np.testing.assert_allclose(packed['unit_cell'], (5, 6, 7, 90, 90, 90), atol=1e-10)


def test_crystal_lifecycle_does_not_import_prohibited_toolkits(monkeypatch):
    original = builtins.__import__
    def no_toolkit(name, *args, **kwargs):
        if name.split('.')[0] in ('openbabel', 'rdkit', 'ccdc', 'gemmi'):
            raise AssertionError('prohibited Crystal dependency: ' + name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', no_toolkit)
    mol, cell = associated()
    cell.set_cell(cell.matrix * 2, preserve='fractional')
    cell.copy()
    pickle.loads(pickle.dumps((cell, mol)))
    cell.to_fractional(mol.coordinates)


@pytest.mark.parametrize('supplied', [{}, {'x': 1.}, {'x': 0., 'y': 0.}])
def test_fractional_preservation_rejects_missing_coordinate_facts_without_publication(supplied):
    mol = Molecule()
    mol.create_atom(atomic_number=6, **supplied)
    cell = Crystal(5, 6, 7, 90, 90, 90, molecule=mol)
    before = mol._capture_factual_record()[0]
    state = cell._state
    with pytest.raises(ValueError, match='complete supplied Cartesian coordinates'):
        cell.set_cell(cell.matrix * 2, preserve='fractional')
    after = mol._capture_factual_record()[0]
    assert cell._state is state and after.atom_presence == before.atom_presence
    assert after.atom_values == before.atom_values and after.atom_sources == before.atom_sources
    cell.set_cell(cell.matrix, preserve='fractional')  # identity publishes no invented facts
    assert mol._capture_factual_record()[0].atom_presence == before.atom_presence


@pytest.mark.parametrize('active', [None, SpaceGroup(2)])
def test_generated_record_and_setter_require_explicit_active_p1(active):
    from dataclasses import replace
    mol, cell = associated()
    record = mol._capture_factual_record()[0]
    source = cell.space_group
    origin = (record.cell[1], source, ((6,), (6,), (8,)), ((0, 1, 'single'),))
    maps = tuple(((row, source.identity(), (0, 0, 0)),) for row in range(3))
    bond_origins = (((0, source.identity(), (0, 0, 0)),),)
    generated = ('hotpot.crystal/1', record.cell[1], SpaceGroup(1), None,
                 record.cell[4], 'cell', origin, maps, bond_origins)
    valid = replace(record, cell=generated)
    with pytest.raises(ValueError, match='explicit active P1'):
        replace(record, cell=(*generated[:2], active, *generated[3:]))
    restored = Molecule._build_molecule(valid)
    before = restored.crystal._state
    with pytest.raises(ValueError, match='explicit active P1'):
        restored.crystal.set_space_group(active)
    assert restored.crystal._state is before


def test_concurrent_replacement_rejects_stale_association_after_real_lock(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from hotpot.cheminfo.operations import crystal as operations
    mol, old = associated()
    state = old._state
    selected = Event()
    transaction_type = operations._FactTransaction
    def select(molecule, **kwargs):
        if not selected.is_set():
            selected.set()
        return transaction_type(molecule, **kwargs)
    monkeypatch.setattr(operations, '_FactTransaction', select)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with mol._mutation_lock:
            future = pool.submit(old.set_cell, old.matrix * 2)
            assert selected.wait(5)
            new = mol.create_crystal(8, 8, 8, 90, 90, 90)
        with pytest.raises(RuntimeError, match='active association changed'):
            future.result(timeout=5)
    assert mol.crystal is new and new.molecule is mol and old.molecule is None
    assert old._state[0] == state[0]
