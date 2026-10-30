"""Source/member contracts for Stage4's on-demand mathematical views."""

from copy import deepcopy
from dataclasses import replace
from itertools import combinations
import pickle

import numpy as np
import pytest

from hotpot import Molecule
from hotpot.cheminfo.structure._commit import _FactTransaction, _commit_atom_facts
from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES, FIELD_INDEX, FIELD_PRESENCE_BITS
from hotpot.cheminfo.structure import _identity
from hotpot.cheminfo.structure._mutation import _normalize_atom_scalar


def molecule():
    mol = Molecule()
    for coordinates in ((0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (1., 1., 1.)):
        mol.create_atom(atomic_number=6, coordinates=coordinates)
    for row in range(3):
        mol.add_bond(row, row + 1)
    return mol


def test_views_are_fresh_and_membership_is_fixed_without_carrier_caches():
    mol = molecule()
    angle, torsion = mol.angles[0], mol.torsions[0]
    pair = mol.atom_pairs[(mol.atoms[0], mol.atoms[-1])]
    assert mol.angles[0] is not angle and mol.torsions[0] is not torsion
    assert not {'_angles', '_torsions', '_atom_pairs'}.intersection(mol.__dict__)
    for view in (angle, torsion, pair):
        with pytest.raises(AttributeError):
            view.atom1 = mol.atoms[-1]
    with pytest.raises(AttributeError):
        angle._replace_atom(angle.atom1, mol.atoms[-1])
    table = mol.atom_pairs
    with pytest.raises(TypeError):
        table[(mol.atoms[0], mol.atoms[1])] = pair
    with pytest.raises(AttributeError):
        table.clear()
    assert angle.degrees == pytest.approx(90.)
    assert pair.distance == pytest.approx(np.sqrt(3.))
    assert pair.length_shortest_path == 4


def test_read_container_copies_keep_the_existing_list_and_dict_contracts():
    mol = molecule()
    angle = mol.angles[0]
    bonds = angle.bonds
    assert isinstance(bonds, list)
    assert bonds == [mol.bonds[0], mol.bonds[1]]
    bonds.clear()
    assert angle.bonds == [mol.bonds[0], mol.bonds[1]]
    pairs = mol.atom_pairs
    copied = pairs.copy()
    assert type(copied) is dict
    assert tuple(copied) == tuple(pairs)
    copied.clear()
    assert len(pairs) == 6


@pytest.mark.parametrize('change', ['atom_append', 'bond_append', 'member_remove'])
def test_topology_change_rejects_calculation_but_keeps_member_inspection(change):
    mol = molecule()
    angle = mol.angles[0]
    members = angle.atoms
    if change == 'atom_append':
        mol.create_atom(atomic_number=6)
    elif change == 'bond_append':
        mol.add_bond(0, 3)
    else:
        mol.remove_atom(members[0])
    assert angle.atoms == members
    with pytest.raises(RuntimeError, match='topology'):
        _ = angle.degrees


def test_geometry_binding_uses_only_the_relevant_member_rows():
    mol = molecule()
    angle = mol.angles[0]
    mol.atoms[-1].x = 5.
    assert angle.degrees == pytest.approx(90.)
    mol.atoms[0].y = 2.
    assert angle.atoms[0] is mol.atoms[0]
    with pytest.raises(RuntimeError, match='geometry'):
        _ = angle.degrees


def test_collection_and_standalone_geometry_bindings_preserve_raw_bits_and_metadata():
    mol = molecule()
    for row in range(4, 7):
        mol.create_atom(atomic_number=6, coordinates=(row, 0., 0.))
        mol.add_bond(row - 1, row)
    record, _ = mol._capture_factual_record()
    values = np.frombuffer(record.atom_values, dtype='<f8').reshape(7, -1).copy()
    bits = np.resize(np.array((0x8000000000000000, 0, 0x7FF8000000000001,
                               0x3FF0000000000000, 0xFFF800000000002A), dtype=np.uint64), 21)
    values[:, 10:13] = bits.view(np.float64).reshape(7, 3)
    coordinate_bits = tuple(FIELD_PRESENCE_BITS[FIELD_INDEX[name]] for name in ('x', 'y', 'z'))
    presence = np.frombuffer(record.atom_presence, dtype='<u4').copy()
    presence &= ~np.uint32(sum(coordinate_bits))
    presence |= np.array([sum(bit for axis, bit in enumerate(coordinate_bits) if (row + axis) % 2)
                          for row in range(7)], dtype=np.uint32)
    sources = np.frombuffer(record.atom_sources, dtype=np.uint8).reshape(7, -1).copy()
    sources[:, 10:13] = np.resize(tuple(ATOM_SOURCE_CODES.values()), 21).reshape(7, 3)
    mol = Molecule._build_molecule(replace(record, atom_values=values.tobytes(),
                                          atom_presence=presence.tobytes(), atom_sources=sources.tobytes()))
    np.testing.assert_array_equal(mol.coordinates.view(np.uint64).ravel(), bits)
    for view in mol.angles + mol.torsions:
        direct = type(view)(*view.atoms)
        assert view._view_geometry == direct._view_geometry
        assert tuple(map(len, view._view_geometry)) == (len(view) * 24, len(view) * 4, len(view) * 3)
    table = mol.atom_pairs
    for first, second in combinations(mol.atoms, 2):
        for members in ((first, second), (second, first)):
            pair = table[members]
            assert pair._view_geometry == type(pair)(*members)._view_geometry


@pytest.mark.parametrize('kind', ['angle', 'torsion', 'pair'])
@pytest.mark.parametrize('dependency', ['coordinates', 'presence', 'producer'])
def test_collection_and_standalone_views_check_only_relevant_geometry_sources(kind, dependency):
    mol = molecule()
    mol.create_atom(atomic_number=6, coordinates=(2., 2., 2.))
    view = (mol.angles[0] if kind == 'angle' else mol.torsions[0] if kind == 'torsion'
            else mol.atom_pairs[(mol.atoms[0], mol.atoms[3])])
    direct = type(view)(*view.atoms)

    def query(value):
        return value.distance if kind == 'pair' else value.degrees

    def change(row):
        atom = mol.atoms[row]
        update = _normalize_atom_scalar(
            _identity._capture_source(mol), row, atom._uid, 'x',
            atom.x + 1. if dependency == 'coordinates' else atom.x,
            producer=ATOM_SOURCE_CODES['geometry'] if dependency == 'producer' else ATOM_SOURCE_CODES['explicit'],
            present=dependency != 'presence')
        _commit_atom_facts(mol, update)

    expected = query(view)
    change(4)
    assert query(view) == pytest.approx(expected)
    assert query(direct) == pytest.approx(expected)
    change(0)
    for value in (view, direct):
        with pytest.raises(RuntimeError, match='geometry'):
            query(value)


def test_collection_geometry_and_constraint_bindings_survive_provenance_rollback():
    mol = molecule()
    angle, torsion = mol.angles[0], mol.torsions[0]
    pair = mol.atom_pairs[(mol.atoms[0], mol.atoms[3])]
    views = (angle, type(angle)(*angle.atoms), torsion, type(torsion)(*torsion.atoms))
    pairs = (pair, type(pair)(*pair.atoms))
    expected = [value.degrees for value in views] + [value.distance for value in pairs]
    owner = mol._constraints
    original = MemoryError('aborted geometry provenance and constraint edit')
    with pytest.raises(MemoryError) as caught:
        with _FactTransaction(mol, producer=ATOM_SOURCE_CODES['geometry']):
            angle.constraint = True
            torsion.constraint = True
            _commit_atom_facts(mol, _normalize_atom_scalar(
                _identity._capture_source(mol), 0, mol.atoms[0]._uid, 'x', 5.,
                producer=ATOM_SOURCE_CODES['geometry'], present=False))
            with pytest.raises(RuntimeError, match='geometry'):
                _ = angle.degrees
            raise original
    assert caught.value is original
    assert mol._constraints is owner and not angle.constraint and not torsion.constraint
    actual = [value.degrees for value in views] + [value.distance for value in pairs]
    np.testing.assert_allclose(actual, expected)


def test_imported_overlapping_bond_partitions_keep_active_first_and_hidden_fallback_order():
    record, _ = molecule()._capture_factual_record()
    mol = Molecule._build_molecule(replace(
        record, bond_endpoints=record.bond_endpoints + ((1, 0), (0, 1), (0, 3), (3, 0)),
        bond_values=record.bond_values + record.bond_values[:24] * 4,
        bond_partitions=record.bond_partitions + (1, 2, 1, 2),
        bond_kinds=record.bond_kinds + (record.bond_kinds[0],) * 4,
        bond_directions=record.bond_directions + (None,) * 4,
        bond_sources=record.bond_sources + (None,) * 4,
        bond_metadata=record.bond_metadata + (record.bond_metadata[0],) * 4,
        bond_aromatic=record.bond_aromatic + (record.bond_aromatic[0],) * 4,
        angle_constraints=((0, 3, 1),), torsion_constraints=((1, 0, 3, 2),)))
    angle = mol.angles[0]
    direct = type(angle)(*angle.atoms)
    assert angle.bond1 is direct.bond1 is mol.bond(0, 1)
    assert angle.bond2 is direct.bond2 is mol.bond(1, 2)
    extra = next(view for view in mol.angles if view.constraint)
    assert extra.bond1 is mol._hided_metal_bonds[1] and extra.bond2 is None
    torsion = next(view for view in mol.torsions if view.constraint)
    assert torsion.bond1 is mol.bond(1, 0)
    assert torsion.bond2 is mol._hided_metal_bonds[1]
    assert torsion.bond3 is mol.bond(3, 2)
    with pytest.raises(KeyError):
        mol.bond(0, 3)


def test_triangle_torsions_preserve_repeated_members_and_constraint_projection():
    mol = molecule()
    mol.add_bond(0, 2)
    repeated = [view for view in mol.torsions if view.atoms[0] is view.atoms[-1]]
    assert repeated
    for view in repeated:
        assert view._view_geometry == type(view)(*view.atoms)._view_geometry
    repeated[0].constraint = True
    assert next(view for view in mol.torsions if view._view_uids == repeated[0]._view_uids).constraint


def test_rolled_back_fast_topology_append_keeps_the_original_view_valid():
    mol = molecule()
    angle = mol.angles[0]
    original = KeyboardInterrupt('aborted append')
    with pytest.raises(KeyboardInterrupt) as caught:
        with _FactTransaction(mol, producer=1):
            mol.add_bond(0, 3)
            raise original
    assert caught.value is original
    assert angle.degrees == pytest.approx(90.)


def test_mixed_fast_and_complete_topology_undo_restores_the_original_view_capability():
    mol = molecule()
    angle = mol.angles[0]
    index = mol._topology_index
    entries = dict(index)
    atoms, bonds = tuple(mol.atoms), tuple(mol.bonds)
    original = MemoryError('aborted mixed topology')
    with pytest.raises(MemoryError) as caught:
        with _FactTransaction(mol, producer=1) as transaction:
            mol._add_bond(0, 3)
            transaction.touch_topology()
            added = mol.create_atom(atomic_number=6)
            mol._add_bond(0, added)
            mol.remove_atom(mol.atoms[2])
            replacement = mol.create_atom(atomic_number=8)
            mol._add_bond(0, replacement)
            raise original
    assert caught.value is original
    assert mol._topology_index is index
    assert tuple(mol.atoms) == atoms and tuple(mol.bonds) == bonds
    assert index.keys() == entries.keys()
    assert all(index[key] is value for key, value in entries.items())
    assert angle.degrees == pytest.approx(90.)


def test_stale_sequence_members_remain_inspectable_and_row_queries_reject_the_source():
    mol = molecule()
    angle, torsion = mol.angles[0], mol.torsions[0]
    bond = angle.bond1
    mol.remove_bond(bond)
    assert angle.bond1 is bond
    assert angle.bonds[0] is bond
    for query in (lambda: angle.a1idx, lambda: angle.atoms_indices,
                  lambda: angle.has_metal, angle.to_mol, lambda: torsion.rotatable):
        with pytest.raises(RuntimeError, match='topology'):
            query()


def test_per_view_reads_do_not_compute_whole_molecule_content_digests(monkeypatch):
    mol = molecule()

    def forbidden(*args, **kwargs):
        raise AssertionError('view access computed a molecular content digest')

    monkeypatch.setattr(_identity, '_digest', forbidden)
    assert all(np.isfinite(angle.degrees) for angle in mol.angles)
    pairs = mol.atom_pairs
    assert len(pairs) == 6
    assert pairs.pair_distance.shape == (6,)
    assert pairs[(mol.atoms[0], mol.atoms[-1])].length_shortest_path == 4


def test_branched_topology_query_preserves_the_existing_canonical_tree():
    mol = Molecule()
    for row in range(4):
        mol.create_atom(atomic_number=6)
    for first, second in ((0, 1), (1, 2), (1, 3)):
        mol.add_bond(first, second)
    assert mol.canonical_tree() == ([0, 1, 2], {1: [([3], {})]})


@pytest.mark.parametrize('clone', [deepcopy, lambda value: pickle.loads(pickle.dumps(value))])
@pytest.mark.parametrize('kind', ['angle', 'torsion', 'pair', 'table'])
@pytest.mark.parametrize('owner_first', [False, True])
def test_current_view_serialization_rebinds_one_restored_owner_in_both_root_orders(clone, kind, owner_first):
    mol = molecule()
    value = (mol.angles[0] if kind == 'angle' else mol.torsions[0] if kind == 'torsion'
             else mol.atom_pairs if kind == 'table'
             else mol.atom_pairs[(mol.atoms[0], mol.atoms[-1])])
    restored = clone((mol, value) if owner_first else (value, mol))
    restored_mol, view = restored if owner_first else restored[::-1]
    assert view.mol is restored_mol
    if kind == 'angle':
        assert view.atoms[0] is restored_mol.atoms[0]
        assert view.degrees == pytest.approx(90.)
    elif kind == 'torsion':
        assert view.atoms[0] is restored_mol.atoms[0]
        assert np.isfinite(view.degrees)
    elif kind == 'pair':
        assert view.atom1 is restored_mol.atoms[0]
        assert view.distance == pytest.approx(np.sqrt(3.))
    else:
        assert view.pair_distance.shape == (6,)


@pytest.mark.parametrize('clone', [deepcopy, lambda value: pickle.loads(pickle.dumps(value))])
@pytest.mark.parametrize('kind', ['angle', 'torsion', 'pair', 'table'])
def test_standalone_view_serialization_preserves_owner_and_numeric_projection(clone, kind):
    mol = molecule()
    value = (mol.angles[0] if kind == 'angle' else mol.torsions[0] if kind == 'torsion'
             else mol.atom_pairs if kind == 'table'
             else mol.atom_pairs[(mol.atoms[0], mol.atoms[-1])])
    restored = clone(value)
    if kind == 'table':
        assert restored.pair_distance.shape == (6,)
    else:
        assert restored.atoms[0] is restored.mol.atoms[0]
        assert np.isfinite(restored.distance if kind == 'pair' else restored.degrees)


@pytest.mark.parametrize('clone', [deepcopy, lambda value: pickle.loads(pickle.dumps(value))])
@pytest.mark.parametrize('kind', ['angle', 'torsion', 'pair', 'table'])
@pytest.mark.parametrize('dependency', ['topology', 'geometry', 'membership'])
def test_serialization_does_not_turn_stale_view_sources_into_current_ones(clone, kind, dependency):
    mol = molecule()
    value = (mol.angles[0] if kind == 'angle' else mol.torsions[0] if kind == 'torsion'
             else mol.atom_pairs if kind == 'table'
             else mol.atom_pairs[(mol.atoms[0], mol.atoms[-1])])
    if dependency == 'topology':
        mol.add_bond(0, 3)
    elif dependency == 'membership':
        mol.remove_atom(mol.atoms[0])
    else:
        mol.atoms[0].x = 2.
    restored = clone(value)
    expected = 'geometry' if dependency == 'geometry' else 'topology'
    with pytest.raises(RuntimeError, match=expected):
        _ = (restored.pair_distance if kind == 'table' else restored.distance if kind == 'pair'
             else restored.degrees)
