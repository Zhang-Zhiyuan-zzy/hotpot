"""Connected carrier contracts for the Stage3 sole-table checkpoint."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import pickle
import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from hotpot import Atom, Bond, Molecule
core = importlib.import_module(Molecule.__module__)
from hotpot.cheminfo.structure._commit import _FactTransaction, _commit_topology
from hotpot.cheminfo.structure._dependencies import _dependency_state
from hotpot.cheminfo.structure._identity import _capture_source, _check_source
from hotpot.cheminfo.structure._mutation import _normalize_atom_facts
from hotpot.cheminfo.operations import chemistry
from hotpot.utils.chem.atom import calc_electron_config


def molecule():
    mol = Molecule()
    atoms = [mol.create_atom(atomic_number=number) for number in (6, 6, 8)]
    mol.add_bond(atoms[0], atoms[1])
    mol.add_bond(atoms[1], atoms[2])
    return mol


def facts(mol):
    status, seeds = _dependency_state(mol._atom_table)
    return (mol._atom_table.matrix().tobytes(), mol._atom_table.presence().tobytes(),
            mol._atom_table.sources().tobytes(), status.tobytes(), seeds,
            tuple(mol.atoms), tuple(mol._bonds + mol._hided_bonds))


def test_sole_numeric_owner_and_public_snapshot_isolation():
    values = np.arange(19, dtype=np.float64) + 0.25
    values[0] = 6
    values[9] = 0
    atom = Molecule()._create_atom_from_array(values)
    expected = values.copy()
    expected[1:7] = (2, 2, 2, 0, 0, 0)
    expected[13] = 4
    values[:] = -1
    np.testing.assert_array_equal(atom.attrs, expected)
    row, matrix = atom.attrs, atom.mol.atom_matrix
    row[:] = matrix[:] = -2
    np.testing.assert_array_equal(atom.attrs, expected)
    assert 'attrs' not in atom.__dict__ and '_attrs' not in atom.__dict__
    with pytest.raises(AttributeError):
        atom.attrs = expected
    assert Molecule().atom_matrix.shape == (0, 19)


def test_scalar_getter_and_coordinate_export_do_not_build_atom_rows(monkeypatch):
    mol = molecule()
    atom = mol.atoms[0]

    def forbidden(*args):
        raise AssertionError('scalar projection copied the complete row')

    monkeypatch.setattr(Atom, 'attrs', property(forbidden))
    monkeypatch.setattr(type(mol._atom_table), 'row', forbidden)
    monkeypatch.setattr(type(mol._atom_table), 'matrix', forbidden)
    assert atom.atomic_number == 6 and atom.formal_charge == 0
    assert atom.n == calc_electron_config(6)[0]
    assert atom.coordinates == (0., 0., 0.)
    assert mol.coordinates.shape == (3, 3)


def test_eager_element_failure_rolls_back_every_value_and_metadata(monkeypatch):
    """Keep the frozen regression name; current contracts are asserted below."""
    mol = molecule()
    atom = mol.atoms[0]
    before = facts(mol)
    observed = []
    calculate = chemistry._calculate_implicit_hydrogens

    def failure(*inputs):
        observed.append(inputs)
        raise ValueError('injected hydrogen calculation')

    monkeypatch.setattr(chemistry, '_calculate_implicit_hydrogens', failure)
    atom.atomic_number = 7
    assert observed == []
    after_write = facts(mol)
    assert after_write != before and atom.atomic_number == 7
    with pytest.raises(ValueError, match='injected hydrogen'):
        _ = atom.implicit_hydrogens
    assert len(observed) == 1 and observed[0][0] == 7
    assert facts(mol) == after_write
    assert 'atomic_number' not in atom.__dict__
    monkeypatch.setattr(chemistry, '_calculate_implicit_hydrogens', calculate)
    assert atom.implicit_hydrogens == 2
    assert atom.atomic_number == 7


def test_bond_endpoint_failure_restores_order_kind_and_first_endpoint(monkeypatch):
    """Keep the frozen regression name; current contracts are asserted below."""
    mol = molecule()
    bond = mol.bonds[0]
    old_raw, old_kind = bond.attrs, bond.bond_kind
    first, second = bond.atoms
    calculate = chemistry._calculate_implicit_hydrogens
    observed = []

    def hydrogen(*inputs):
        observed.append(inputs)
        if len(observed) == 2:
            raise RuntimeError('second endpoint')
        return 17

    monkeypatch.setattr(chemistry, '_calculate_implicit_hydrogens', hydrogen)
    bond.bond_order = 2
    assert observed == []
    after_write = facts(mol)
    written_raw, written_kind = bond.attrs, bond.bond_kind
    assert bond.attrs[0] == 2 and old_raw[0] == 1
    assert written_kind is core.BondKind.DOUBLE and written_kind is not old_kind
    with pytest.raises(RuntimeError, match='second endpoint'):
        mol.calc_implicit_hydrogens()
    assert len(observed) == 2
    # The factual write already committed; the later lazy preparation restores
    # its own proposal and retains the complete written DOUBLE representation.
    assert facts(mol) == after_write and bond.bond_kind is written_kind
    np.testing.assert_array_equal(bond.attrs, written_raw)
    np.testing.assert_array_equal(bond.attrs[1:], old_raw[1:])
    assert 'attrs' not in bond.__dict__
    with pytest.raises(AttributeError):
        bond.attrs = old_raw
    monkeypatch.setattr(chemistry, '_calculate_implicit_hydrogens', calculate)
    assert first.implicit_hydrogens != 17
    assert second.implicit_hydrogens != 17


def test_failed_constructor_leaves_no_member_and_burns_allocated_uid():
    mol = Molecule()
    original = Atom(mol, atomic_number=6)
    next_uid = mol._next_atom_uid
    with pytest.raises(ValueError, match="[Cc]onflict"):
        Atom(mol, symbol="C", atomic_number=8)
    assert mol.atoms == [original]
    assert len(mol._atom_table) == 1
    added = Atom(mol, atomic_number=8)
    assert added._uid > next_uid
    assert set(mol._topology_index) == {id(original), id(added)}


def test_batch_stages_all_inputs_and_preserves_explicit_downstream_values(monkeypatch):
    mol = molecule()
    first, second = mol.atoms[:2]
    seen = []

    def valence(self):
        seen.append((self._uid, tuple(atom.atomic_number for atom in mol.atoms)))
        return 4

    monkeypatch.setattr(Atom, 'get_valence', valence)
    monkeypatch.setattr(Atom, '_calc_implicit_hydrogens', lambda self: 3)

    def rule(*args):
        seen.append(args)
        return 3

    monkeypatch.setattr(chemistry, '_calculate_valence', rule)
    monkeypatch.setattr(chemistry, '_calculate_implicit_hydrogens', rule)
    assert mol.updata_atoms(atoms=[second, first], symbol=['N', 'O'],
        coordinates=[[1, 2, 3], [4, 5, 6]], implicit_hydrogens=[8, 9]) is None
    assert seen == []
    assert tuple(atom.atomic_number for atom in mol.atoms) == (8, 7, 8)
    assert (second.implicit_hydrogens, first.implicit_hydrogens) == (8, 9)
    np.testing.assert_array_equal(mol.coordinates[:2], [[4, 5, 6], [1, 2, 3]])
    before = facts(mol)
    with pytest.raises(ValueError):
        mol.updata_atoms(atoms=[first], symbol='N', atomic_number=8)
    assert facts(mol) == before


def test_remove_and_replace_preserve_old_inspection_without_row_retargeting():
    mol = molecule()
    old = mol.atoms[1]
    row, old_uid = old.attrs, old._uid
    bond = mol.bonds[0]
    assert mol.replace_atom(1, 7) is old
    assert old._uid == old_uid and old.mol is mol
    np.testing.assert_array_equal(old.attrs, row)
    assert old.bonds == [] and old.neighbours == []
    assert bond.atom2 is mol.atoms[1]
    with pytest.raises(KeyError):
        old.x = 2
    with pytest.raises(ValueError):
        old._detached_table._blocks[0].setflags(write=True)
    survivor = mol.atoms[-1]
    survivor_uid = survivor._uid
    mol.remove_atom(0)
    assert survivor._uid == survivor_uid and survivor.idx == 1


def test_copy_and_merge_preserve_raw_bond_values_opaque_metadata_and_local_handles():
    source = molecule()
    bond = source.bonds[0]
    bond.id = 2.75
    bond._set_bond_metadata(bond_kind=bond.bond_kind,
                            bond_source_metadata={'opaque': SimpleNamespace(values=[1, 2])})
    source.charge = 0
    source.spin_mult = 3
    source.create_crystal(10, 11, 12, 90, 91, 92)
    annotation = []
    source.properties['user'] = annotation
    clone = source.copy()
    np.testing.assert_array_equal(clone.atom_matrix, source.atom_matrix)
    np.testing.assert_array_equal(clone.bonds[0].attrs, bond.attrs)
    assert clone.bonds[0]._bond_source_metadata['opaque'] is not bond._bond_source_metadata['opaque']
    assert clone.properties is not source.properties and clone.properties['user'] is annotation
    assert clone._charge_present and clone.charge == 0 and clone.spin_mult == 3
    target = Molecule()
    retained = target.create_atom(atomic_number=1)
    target.create_crystal(10, 11, 12, 90, 91, 92)
    retained_uid = retained._uid
    target.add_component(source)
    assert target.atoms[0] is retained and retained._uid == retained_uid
    np.testing.assert_array_equal(target.bonds[0].attrs, bond.attrs)
    assert not target._charge_present and not target._spin_present
    incompatible = molecule()
    incompatible.create_crystal(20, 20, 20, 90, 90, 90)
    before = facts(target)
    with pytest.raises(ValueError, match='unit cells'):
        target.add_component(incompatible)
    assert facts(target) == before


def test_factual_installer_builds_index_once_and_never_runs_chemistry(monkeypatch):
    source = molecule()
    source.atoms[0].isotope = 13
    source.atoms[1].frac_valence = 3.75
    source.bonds[0].constraint = True
    source.charge = 0
    record, metadata = source._capture_factual_record()
    build = core._build_topology_index
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return build(*args, **kwargs)

    def forbidden(*args):
        raise AssertionError('factual install invoked chemistry')

    monkeypatch.setattr(core, '_build_topology_index', counted)
    monkeypatch.setattr(Atom, 'get_valence', forbidden)
    monkeypatch.setattr(Atom, '_calc_implicit_hydrogens', forbidden)
    monkeypatch.setattr(chemistry, '_calculate_valence', forbidden)
    monkeypatch.setattr(chemistry, '_calculate_implicit_hydrogens', forbidden)
    target = Molecule()
    _commit_topology(target, _capture_source(target), record,
        atom_bindings=(None,) * record.atom_count,
        bond_bindings=(None,) * len(record.bond_endpoints), bond_metadata=metadata)
    assert calls == [1]
    np.testing.assert_array_equal(target._atom_table.matrix(), source._atom_table.matrix())
    np.testing.assert_array_equal(target._atom_table.presence(), source._atom_table.presence())
    np.testing.assert_array_equal(target._atom_table.sources(), source._atom_table.sources())
    assert target.atoms[0].isotope == 13 and target.atoms[1].frac_valence == 3.75
    assert target.bonds[0].constraint and target._charge_present


@pytest.mark.parametrize('roundtrip', [lambda value: pickle.loads(pickle.dumps(value)), deepcopy])
def test_root_members_and_self_references_bind_the_restored_graph(roundtrip):
    mol = molecule()
    atom, bond = mol.atoms[1], mol.bonds[0]
    mol.properties.update(self=mol, atom=atom, bond=bond)
    restored_atom = roundtrip(atom)
    assert restored_atom is restored_atom.mol.atoms[1]
    assert restored_atom.mol.properties['self'] is restored_atom.mol
    assert restored_atom.mol.properties['atom'] is restored_atom
    restored_bond = roundtrip(bond)
    assert restored_bond is restored_bond.mol.bonds[0]
    assert restored_bond.mol.properties['bond'] is restored_bond
    assert restored_bond.atom1 is restored_bond.mol.atoms[0]


def test_joint_pickle_preserves_source_owner_and_uid_high_water():
    mol = molecule()
    mol.remove_atom(0)
    mol.create_atom(atomic_number=7)
    update = _normalize_atom_facts(_capture_source(mol), (0,), (mol.atoms[0]._uid,), {'x': 2})
    restored, restored_update = pickle.loads(pickle.dumps((mol, update)))
    _check_source(restored, restored_update.source)
    assert restored._next_atom_uid == mol._next_atom_uid
    assert [atom._uid for atom in restored.atoms] == [atom._uid for atom in mol.atoms]
    detached = Atom(atomic_number=6)
    detached.mol.remove_atom(detached)
    copied = pickle.loads(pickle.dumps(detached))
    assert copied.bonds == [] and copied.mol.atoms == []
    np.testing.assert_array_equal(copied.attrs, detached.attrs)


def test_nine_frame_domains_are_independent_and_survive_serialization():
    mol = molecule()
    n = len(mol.atoms)
    values = (np.arange(2*n*3.).reshape(2, n, 3), np.array([1., 2.]),
              np.arange(2*n).reshape(2, n), 4.5, np.ones((2, n, 3)),
              0., np.array([1, 3]), np.array([[8., 9.], [10., 11.]]), np.array([12., 13.]))
    mol._install_frame_state(values, 1)
    selected = mol.conformer_get(1)
    selected['coordinates'][:] = -1
    assert np.all(mol.conformer_get(1)['coordinates'] >= 0)
    frame = next(iter(mol.conformers))
    frame[:] = -2
    assert np.all(mol.conformers.coordinates(0) >= 0)
    restored = pickle.loads(pickle.dumps(mol))
    captured, index = restored._capture_frame_state()
    assert index == 1
    for expected, observed in zip(values, captured):
        np.testing.assert_array_equal(observed, expected)


def test_concurrent_pickle_sees_one_complete_coordinate_and_charge_commit():
    mol = molecule()

    def writer():
        for value in range(1, 25):
            with _FactTransaction(mol, producer=1):
                mol.coordinates = np.full((len(mol.atoms), 3), value, dtype=float)
                mol.charge = value

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(writer)
        for _ in range(25):
            restored = pickle.loads(pickle.dumps(mol))
            assert np.all(restored.coordinates == restored.charge)
        future.result()


def test_isolated_registration_does_not_scan_or_copy_existing_graph(monkeypatch):
    mol = molecule()

    def forbidden(*args, **kwargs):
        raise AssertionError('isolated registration scanned or copied the graph')

    monkeypatch.setattr(core, '_build_topology_index', forbidden)
    for name in ('matrix', 'copy', 'take'):
        monkeypatch.setattr(type(mol._atom_table), name, forbidden)
    added = Atom(mol, atomic_number=7)
    assert added.idx == 3 and added.neighbours == []


def test_complete_topology_failure_restores_sparse_facts_endpoints_and_identifier():
    from dataclasses import replace

    mol = molecule()
    mol.identifier = 'original'
    mol.atoms[0].isotope = 13
    mol.atoms[1].frac_valence = 3.75
    record, metadata = mol._capture_factual_record()
    candidate = replace(record, atom_isotopes=((0, 15),), atom_frac_valences=((1, 5.25),),
                        identifier=(True, 'candidate'), bond_endpoints=((1, 0), (1, 2)))
    before = facts(mol)
    endpoints = tuple(bond.atoms for bond in mol.bonds)
    with pytest.raises(KeyboardInterrupt):
        with _FactTransaction(mol, producer=1):
            _commit_topology(mol, _capture_source(mol), candidate,
                atom_bindings=tuple(mol.atoms), bond_bindings=tuple(mol.bonds), bond_metadata=metadata)
            raise KeyboardInterrupt('after complete publication')
    assert facts(mol) == before
    assert tuple(bond.atoms for bond in mol.bonds) == endpoints
    assert mol.identifier == 'original'
    assert mol.atoms[0].isotope == 13 and mol.atoms[1].frac_valence == 3.75


@pytest.mark.parametrize('projection', ['members', 'endpoint', 'index'])
def test_topology_projections_wait_for_failure_restoration(projection):
    from concurrent.futures import TimeoutError
    from threading import Event

    mol = molecule()
    old = mol.atoms[1]
    bond = mol.bonds[0]
    original_uids = tuple(atom._uid for atom in mol.atoms)
    entered, release = Event(), Event()

    def writer():
        with _FactTransaction(mol, producer=1):
            mol.replace_atom(1, 7)
            entered.set()
            release.wait(5)
            raise RuntimeError('restore speculative topology')

    def reader():
        if projection == 'members':
            return tuple(atom._uid for atom in mol.atoms)
        if projection == 'endpoint':
            return bond.atom2
        return old.idx

    with ThreadPoolExecutor(max_workers=2) as executor:
        writing = executor.submit(writer)
        assert entered.wait(5)
        reading = executor.submit(reader)
        try:
            with pytest.raises(TimeoutError):
                reading.result(timeout=0.05)
        finally:
            release.set()
        with pytest.raises(RuntimeError, match='speculative topology'):
            writing.result()
        value = reading.result()
    expected = original_uids if projection == 'members' else old if projection == 'endpoint' else 1
    assert value == expected


def test_pickle_constraint_capture_cannot_be_overwritten_by_live_view_extras(monkeypatch):
    mol = molecule()
    mol.atoms[0].constraint = True
    mol.bonds[0].constraint = True
    mol.angles[0].constraint = True
    original_getstate = Molecule.__getstate__

    def capture_then_change(self):
        captured = original_getstate(self)
        self.clear_constraints()
        return captured

    monkeypatch.setattr(Molecule, '__getstate__', capture_then_change)
    restored = pickle.loads(pickle.dumps(mol))
    assert restored.atoms[0].constraint
    assert restored.bonds[0].constraint
    assert restored.angles[0].constraint
    assert not mol.atoms[0].constraint and not mol.angles[0].constraint


def test_local_bond_append_undo_does_not_snapshot_the_complete_graph(monkeypatch):
    mol = Molecule()
    atoms = [Atom(mol, atomic_number=6) for _ in range(3)]
    before = facts(mol)

    def forbidden(*args):
        raise AssertionError('local Bond append copied the complete graph')

    monkeypatch.setattr(_FactTransaction, 'touch_topology', forbidden)
    with pytest.raises(KeyboardInterrupt):
        with _FactTransaction(mol, producer=1):
            mol._add_bond(0, 1)
            mol._add_bond(1, 2)
            raise KeyboardInterrupt('local append failure')
    assert facts(mol) == before
    assert all(atom.bonds == [] for atom in atoms)
    assert mol._next_bond_uid == 2


def test_local_atom_and_bond_append_undo_removes_the_new_index_entry():
    mol = Molecule()
    original = Atom(mol, atomic_number=6)
    with pytest.raises(RuntimeError):
        with _FactTransaction(mol, producer=1):
            added = Atom(mol, atomic_number=8)
            mol._add_bond(original, added)
            raise RuntimeError('mixed append failure')
    assert mol.atoms == [original]
    assert set(mol._topology_index) == {id(original)}
    assert len(mol._atom_table) == 1 and mol.bonds == []


def test_standalone_constraint_projects_the_owner_and_joins_undo():
    mol = molecule()
    angle = core.Angle(*mol.atoms)
    owner = mol._constraints
    assert owner.angles == ()
    with pytest.raises(RuntimeError):
        with _FactTransaction(mol, producer=1):
            angle.constraint = True
            assert mol._constraints.angles == (tuple(atom._uid for atom in angle.atoms),)
            raise RuntimeError('constraint failure')
    assert angle.constraint is False
    assert 'constraint' not in angle.__dict__
    assert mol._constraints is owner
    assert '_angles' not in mol.__dict__


def test_public_batch_reuses_membership_errors_without_repr_of_detached_atom():
    mol = molecule()
    foreign = Atom(atomic_number=7)
    with pytest.raises(core.NotInSameMolecule) as caught:
        mol.updata_atoms(atoms=[foreign], x=1)
    assert caught.value.obj1 is mol and caught.value.obj2 is foreign
    detached = mol.atoms[-1]
    mol.remove_atom(detached)
    with pytest.raises(core.ObjNotInMolecule) as caught:
        mol.updata_atoms(atoms=[detached], x=1)
    assert caught.value.obj is detached and caught.value.mol is mol
    with pytest.raises(KeyError):
        detached.x = 1
    with pytest.raises(ValueError):
        mol.updata_atoms(atoms=[mol.atoms[0], mol.atoms[0]], x=1)


def test_fresh_bond_root_binding_uses_uid_hint_without_scanning_members():
    mol = molecule()
    bond = mol.bonds[1]

    class IndexedOnly(list):
        def __iter__(self):
            raise AssertionError('ordinary Bond binding scanned membership')

    mol._bonds = IndexedOnly(mol._bonds)
    restore, args = bond.__reduce_ex__(4)
    assert restore(*args) is bond


@pytest.mark.parametrize('name', ['atom1', 'atom2'])
@pytest.mark.parametrize('foreign', [False, True])
def test_bond_endpoint_projection_rejects_shadow_assignment(name, foreign):
    mol = molecule()
    bond = mol.bonds[0]
    replacement = Atom(atomic_number=7) if foreign else mol.atoms[2]
    endpoints = bond.atoms
    before = facts(mol)
    index = dict(mol._topology_index)
    with pytest.raises(AttributeError):
        setattr(bond, name, replacement)
    assert bond.atoms == endpoints
    assert (bond.atom1, bond.atom2) == endpoints
    assert name not in bond.__dict__
    assert facts(mol) == before and mol._topology_index == index


@pytest.mark.parametrize('form', ['bound_keywords', 'bound_mixed',
                                 'unbound_keywords', 'unbound_self_keyword'])
@pytest.mark.parametrize('exception', [KeyboardInterrupt, MemoryError])
def test_atom_assignment_retains_keyword_binding_and_original_exceptions(form, exception):
    mol = molecule()
    atom = mol.atoms[0]
    atom.x = 0.

    def assign(key, value):
        if form == 'bound_keywords':
            return atom.__setattr__(key=key, value=value)
        if form == 'bound_mixed':
            return atom.__setattr__(key, value=value)
        if form == 'unbound_keywords':
            return Atom.__setattr__(atom, key=key, value=value)
        return Atom.__setattr__(self=atom, key=key, value=value)

    assert assign('formal_charge', 2.5) is None
    assert atom.formal_charge == 2
    assert atom.attrs[core._ATOM_FIELD_INDEX['formal_charge']] == 2.5
    assert 'formal_charge' not in atom.__dict__
    before = facts(mol)
    original = exception('conversion interruption')

    class InterruptedValue:
        def __float__(self):
            raise original

    with pytest.raises(exception) as caught:
        assign('x', InterruptedValue())
    assert caught.value is original
    assert facts(mol) == before
