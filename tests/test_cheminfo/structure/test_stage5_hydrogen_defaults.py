"""Finite H defaults, display/current separation and consumed-input ownership."""

from copy import copy
from dataclasses import replace
import pickle

import numpy as np
import pytest

from hotpot import Atom, Molecule
from hotpot.cheminfo.operations import chemistry
from hotpot.cheminfo.rules.chemistry.bonding.hydrogen import _calculate_implicit_hydrogens
from hotpot.cheminfo.stereo import (
    AtomTransportRecord, HydrogenTransportState, StereoAnchor, StereoAnchorKind,
    StereoRecord, get_stereo_payload, set_stereo_payload,
)
from hotpot.cheminfo.structure._commit import _FactTransaction, _commit_topology
from hotpot.cheminfo.structure._dependencies import GROUP_INDEX, _FactStatus, _dependency_state
from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES, FIELD_INDEX, FIELD_PRESENCE_BITS
from hotpot.cheminfo.structure._identity import _capture_source, _check_source
from tests.test_cheminfo._stage5_fixtures import factual_state




H_FIELD = FIELD_INDEX['implicit_hydrogens']
H_GROUP = GROUP_INDEX['IMPLICIT_H']


def _rule(number=6, neighbours=(), orders=None, aromatic=False, charge=0,
          kinds=None, flags=None, roles=None, radicals=0, bracket=0, no_implicit=False):
    orders = (1.,) * len(neighbours) if orders is None else orders
    kinds = tuple({1.: 'single', 1.5: 'aromatic', 2.: 'double', 3.: 'triple'}[order]
                  for order in orders) if kinds is None else kinds
    flags = (False,) * len(neighbours) if flags is None else flags
    roles = (None,) * len(neighbours) if roles is None else roles
    return _calculate_implicit_hydrogens(number, neighbours, orders, aromatic, charge,
                                         kinds, flags, roles, radicals, bracket, no_implicit)


@pytest.mark.parametrize('inputs,expected', [
    pytest.param({'number': 5}, 3, id='boron-neutral'),
    pytest.param({}, 4, id='carbon-neutral'),
    pytest.param({'number': 7}, 3, id='nitrogen-neutral'),
    pytest.param({'number': 8}, 2, id='oxygen-neutral'),
    pytest.param({'number': 14}, 4, id='silicon-neutral'),
    pytest.param({'number': 15}, 3, id='phosphorus-neutral'),
    pytest.param({'number': 33}, 3, id='arsenic-neutral'),
    pytest.param({'number': 16}, 2, id='sulfur-neutral'),
    pytest.param({'number': 34}, 2, id='selenium-neutral'),
    pytest.param({'number': 52}, 2, id='tellurium-neutral'),
    pytest.param({'number': 9}, 1, id='fluorine-neutral'),
    pytest.param({'number': 17}, 1, id='chlorine-neutral'),
    pytest.param({'number': 35}, 1, id='bromine-neutral'),
    pytest.param({'number': 53}, 1, id='iodine-neutral'),
    pytest.param({'number': 85}, 1, id='astatine-neutral'),
    pytest.param({'number': 5, 'charge': -1}, 4, id='borate'),
    pytest.param({'number': 5, 'charge': 1}, 2, id='boron-cation'),
    pytest.param({'charge': -1}, 3, id='carbanion'),
    pytest.param({'charge': 1}, 3, id='carbocation'),
    pytest.param({'number': 7, 'charge': -1}, 2, id='nitrogen-anion'),
    pytest.param({'number': 7, 'charge': 1}, 4, id='ammonium'),
    pytest.param({'number': 8, 'charge': -2}, 0, id='oxide'),
    pytest.param({'number': 8, 'charge': -1}, 1, id='hydroxide'),
    pytest.param({'number': 8, 'charge': 1}, 3, id='hydronium'),
    pytest.param({'number': 14, 'charge': -1}, 3, id='silicon-anion-lowest'),
    pytest.param({'number': 14, 'charge': -1, 'neighbours': (6,) * 4}, 1, id='silicon-anion-next-valence'),
    pytest.param({'number': 15, 'charge': -1, 'neighbours': (6,) * 3}, 1, id='phosphorus-anion-valence4'),
    pytest.param({'number': 15, 'charge': -1, 'neighbours': (6,) * 5}, 1, id='phosphorus-anion-valence6'),
    pytest.param({'number': 15, 'neighbours': (6,) * 4}, 1, id='phosphorus-neutral-valence5'),
    pytest.param({'number': 15, 'charge': 1, 'neighbours': (6,) * 3}, 1, id='phosphonium'),
    pytest.param({'number': 15, 'neighbours': (8,) * 3, 'orders': (2., 1., 1.)}, 1, id='phosphorus-oxo-occupied4'),
    pytest.param({'number': 15, 'neighbours': (8,) * 4, 'orders': (2., 1., 1., 1.)}, 0, id='phosphorus-oxo-occupied5'),
    pytest.param({'number': 16, 'neighbours': (6,) * 3}, 1, id='sulfur-neutral-valence4'),
    pytest.param({'number': 16, 'neighbours': (6,) * 5}, 1, id='sulfur-neutral-valence6'),
    pytest.param({'number': 16, 'charge': -1, 'neighbours': (6,) * 2}, 1, id='sulfur-anion-valence3'),
    pytest.param({'number': 34, 'charge': 1, 'neighbours': (6,) * 4}, 1, id='selenium-cation-valence5'),
    pytest.param({'number': 52, 'charge': 2, 'neighbours': (6,) * 3}, 1, id='tellurium-dication-valence4'),
    pytest.param({'number': 17, 'charge': -1}, 0, id='chloride'),
    pytest.param({'number': 17, 'charge': 1}, 2, id='chlorine-cation'),
    pytest.param({'number': 7, 'neighbours': (8,)}, 2, id='neutral-nitrogen-does-not-promote-to5'),
    pytest.param({'number': 7, 'charge': 1, 'neighbours': (6, 8, 8), 'orders': (1., 1., 2.)}, 0, id='nitro-nitrogen'),
    pytest.param({'charge': 2, 'neighbours': (1,), 'bracket': 1}, 2, id='unsupported-carbon-charge-retains-declarations'),
    pytest.param({'number': 2, 'neighbours': (1,)}, 1, id='unsupported-known-nonmetal-retains-graph-H'),
    pytest.param({'number': 30}, 0, id='zinc-no-default-hydride'),
    pytest.param({'number': 30, 'neighbours': (1,), 'bracket': 1, 'no_implicit': True}, 0, id='metal-raw0-precedes-bracket-policy'),
    pytest.param({'number': 1, 'bracket': 1, 'no_implicit': True}, 0, id='real-H-precedes-bracket-policy'),
    pytest.param({'number': 0, 'bracket': 1, 'no_implicit': True}, None, id='missing-element-precedes-bracket-policy'),
    pytest.param({'neighbours': (0,), 'no_implicit': True}, None, id='no-implicit-needs-known-graph-H-inventory'),
    pytest.param({'number': 32}, 0, id='existing-germanium-metal-classification'),
    pytest.param({'number': 51}, 0, id='existing-antimony-metal-classification'),
    pytest.param({'number': 84}, 0, id='existing-polonium-metal-classification'),
    pytest.param({'number': 1}, 0, id='isolated-H'),
    pytest.param({'number': 1, 'charge': 1}, 0, id='isolated-proton'),
    pytest.param({'number': 1, 'charge': -1}, 0, id='isolated-hydride'),
    pytest.param({'number': 1, 'neighbours': (1,)}, 1, id='H2-site-total'),
    pytest.param({'neighbours': (1,) * 4}, 4, id='fully-explicit-methane-site-total'),
    pytest.param({'neighbours': (1,)}, 4, id='partly-explicit-methane-site-total'),
    pytest.param({'neighbours': (6, 1)}, 3, id='alkyl-site-total-includes-graph-H'),
    pytest.param({'neighbours': (6,), 'radicals': 1}, 2, id='alkyl-radical-vacancy'),
    pytest.param({'radicals': 2}, 2, id='carbon-two-radicals'),
    pytest.param({'number': 7, 'neighbours': (6,), 'radicals': 1}, 1, id='nitrogen-radical'),
    pytest.param({'number': 8, 'radicals': 1}, 1, id='oxygen-radical'),
    pytest.param({'number': 14, 'neighbours': (6,), 'radicals': 1}, 2, id='silicon-radical'),
    pytest.param({'number': 35, 'radicals': 1}, 0, id='bromine-radical'),
    pytest.param({'number': 15, 'radicals': 1, 'bracket': 1}, 1, id='unsupported-phosphorus-radical'),
    pytest.param({'number': 16, 'radicals': 1}, 0, id='unsupported-sulfur-radical'),
    pytest.param({'aromatic': True, 'neighbours': (6, 6), 'orders': (1.5, 1.5)}, 1, id='aromatic-carbon-sigma'),
    pytest.param({'aromatic': True, 'neighbours': (6, 6, 6), 'orders': (1.5, 1.5, 1.)}, 0, id='fused-aromatic-carbon'),
    pytest.param({'aromatic': True, 'neighbours': (6, 6), 'orders': (2., 1.), 'flags': (True, True)}, 1, id='kekule-carbon-sigma'),
    pytest.param({'aromatic': True, 'neighbours': (6, 6), 'orders': (1.5, 1.5), 'radicals': 1}, 0, id='aromatic-carbon-radical'),
    pytest.param({'number': 5, 'aromatic': True, 'charge': -1, 'neighbours': (6, 6), 'orders': (1.5, 1.5)}, 1, id='aromatic-borate'),
    pytest.param({'number': 5, 'aromatic': True, 'neighbours': (6, 6), 'orders': (1.5, 1.5)}, 0, id='aromatic-neutral-boron-no-fill'),
    pytest.param({'number': 14, 'aromatic': True, 'neighbours': (6, 6), 'orders': (1.5, 1.5)}, 1, id='aromatic-silicon'),
    pytest.param({'number': 7, 'aromatic': True, 'charge': 1, 'neighbours': (6, 6), 'orders': (1.5, 1.5)}, 1, id='aromatic-positive-nitrogen'),
    pytest.param({'number': 7, 'aromatic': True, 'neighbours': (6, 6)}, 0, id='neutral-aromatic-n-does-not-select-nH'),
    pytest.param({'number': 7, 'aromatic': True, 'charge': -1, 'neighbours': (6, 6)}, 0, id='negative-aromatic-n-no-fill'),
    pytest.param({'number': 7, 'aromatic': True, 'charge': 1, 'neighbours': (6, 6), 'radicals': 1}, 0, id='aromatic-nitrogen-radical-no-fill'),
    pytest.param({'aromatic': True, 'charge': -1, 'neighbours': (6, 6), 'orders': (1.5, 1.5)}, 0, id='aromatic-negative-carbon-no-fill'),
    pytest.param({'aromatic': True, 'charge': 1, 'neighbours': (6, 6), 'orders': (1.5, 1.5)}, 0, id='aromatic-positive-carbon-no-fill'),
    pytest.param({'aromatic': True, 'charge': -1, 'neighbours': (6, 6), 'orders': (1.5, 1.5), 'bracket': 1}, 1, id='declared-cH-minus-retained'),
    pytest.param({'number': 7, 'aromatic': True, 'neighbours': (6, 6), 'bracket': 1}, 1, id='declared-nH-retained'),
    pytest.param({'neighbours': (30,)}, 3, id='covalent-zinc-carbon-consumes-valence'),
    pytest.param({'number': 7, 'neighbours': (30,), 'kinds': ('dative',), 'roles': ('donor',)}, 3, id='nitrogen-dative-donor'),
    pytest.param({'number': 7, 'neighbours': (5,), 'kinds': ('dative',), 'roles': ('acceptor',)}, 2, id='nonmetal-dative-acceptor'),
    pytest.param({'neighbours': (8,), 'orders': (0.,), 'kinds': ('zero',)}, 4, id='known-zero-order'),
    pytest.param({'neighbours': (8,), 'orders': (0.,), 'kinds': ('unknown',)}, None, id='unknown-order-is-missing'),
    pytest.param({'neighbours': (30,), 'kinds': ('dative',)}, None, id='missing-dative-direction'),
    pytest.param({'neighbours': (0,)}, None, id='missing-neighbour-element'),
    pytest.param({'number': 0}, None, id='missing-site-element'),
    pytest.param({'neighbours': (6, 1), 'bracket': 1, 'no_implicit': True}, 2, id='no-implicit-keeps-graph-and-bracket-only'),
], ids=None)
def test_finite_hydrogen_completion_scientific_cases(inputs, expected):
    assert _rule(**inputs) == expected


def _transport(mol, row, *, radicals=None, current=None, source=None, isotope=None, namespace='test'):
    payload = get_stereo_payload(mol)
    fact = AtomTransportRecord(row, isotope, radicals, current, source, namespace)
    set_stereo_payload(mol, replace(payload, atom_facts=tuple(
        item for item in payload.atom_facts if item.atom != row) + (fact,)))


def _state(mol):
    status, seeds = _dependency_state(mol._atom_table)
    return factual_state(mol), status.tobytes(), seeds, tuple(atom._uid for atom in mol.atoms)


def _present(mol, row=0):
    return bool(int(mol._atom_table._presence[row]) & FIELD_PRESENCE_BITS[H_FIELD])


def test_missing_display_is_not_a_zero_fact_and_strict_consumers_reject_it_once(monkeypatch):
    from hotpot.cheminfo.AImodels.data_extract import extract_atom_attrs
    from hotpot.cheminfo._native.buffers import NativeMoleculeField, pack_molecule
    from hotpot.cheminfo.rdconvert import to_rdmol
    atom = Atom()
    mol = atom.mol
    calls = []
    calculate = chemistry._calculate_implicit_hydrogens
    def observe(*args):
        calls.append(args[0])
        return calculate(*args)
    monkeypatch.setattr(chemistry, '_calculate_implicit_hydrogens', observe)
    for _ in range(3):
        assert atom.implicit_hydrogens == 0 and type(atom.implicit_hydrogens) is int
        assert atom.attrs[14] == 0. and mol.atom_matrix[0, 14] == 0.
        for consume in (
                lambda: chemistry._require_implicit_hydrogens(mol),
                lambda: extract_atom_attrs(mol),
                lambda: pack_molecule(mol, frozenset({NativeMoleculeField.ATOM_TRANSPORT})),
                lambda: to_rdmol(mol)):
            with pytest.raises(ValueError, match='resolved implicit-H'):
                consume()
        assert not _present(mol)
        assert mol._atom_table._sources[0, H_FIELD] == ATOM_SOURCE_CODES['default']
        assert mol._atom_table._group_status[0, H_GROUP] == _FactStatus.UNRESOLVED
    assert calls == [0]
    atom.atomic_number = 8
    assert atom.implicit_hydrogens == 2
    assert chemistry._require_implicit_hydrogens(mol) == (2,)
    assert calls == [0, 8]


def test_unresolved_recalculation_clears_old_inferred_presence_without_zero_fill():
    atom = Atom(atomic_number=6)
    assert atom.implicit_hydrogens == 4
    atom.atomic_number = 0
    assert atom.implicit_hydrogens == 0
    assert atom.attrs[14] == 0 and atom.mol.atom_matrix[0, 14] == 0
    assert not _present(atom.mol)
    assert atom.mol._atom_table._sources[0, H_FIELD] == ATOM_SOURCE_CODES['default']
    assert atom.mol._atom_table._group_status[0, H_GROUP] == _FactStatus.UNRESOLVED
    assert atom.mol._atom_table._columns[H_FIELD][0] == 4  # unused bytes, not a current fact
    with pytest.raises(ValueError, match='resolved implicit-H'):
        chemistry._require_implicit_hydrogens(atom.mol)


@pytest.mark.parametrize('number,charge', [(0, 0), (1, 0), (1, -1), (1, 1)])
def test_supplied_zero_and_real_hydrogen_zero_are_current_facts(number, charge):
    kwargs = {'implicit_hydrogens': 0} if number == 0 else {}
    atom = Atom(atomic_number=number, formal_charge=charge, **kwargs)
    assert atom.implicit_hydrogens == 0
    assert chemistry._require_implicit_hydrogens(atom.mol) == (0,)
    assert _present(atom.mol)
    assert atom.mol._atom_table._group_status[0, H_GROUP] == _FactStatus.CURRENT
    expected = 'explicit' if number == 0 else 'implicit_hydrogens'
    assert atom.mol._atom_table._sources[0, H_FIELD] == ATOM_SOURCE_CODES[expected]


@pytest.mark.parametrize('missing', ['neighbour', 'kind', 'direction'])
def test_needed_missing_bond_input_retries_after_its_writer(missing, monkeypatch):
    mol = Molecule()
    atom = mol.create_atom(atomic_number=6)
    other = mol.create_atom(atomic_number=0 if missing == 'neighbour' else 8)
    kind = 'unknown' if missing == 'kind' else 'dative' if missing == 'direction' else 'single'
    bond = mol.add_bond(atom, other, 0. if missing == 'kind' else 1., bond_kind=kind)
    calls = []
    calculate = chemistry._calculate_implicit_hydrogens
    def observe(*args):
        if args[0] == 6:
            calls.append(args)
        return calculate(*args)
    monkeypatch.setattr(chemistry, '_calculate_implicit_hydrogens', observe)
    assert atom.implicit_hydrogens == 0
    assert atom.implicit_hydrogens == 0 and len(calls) == 1
    assert not _present(mol)
    if missing == 'neighbour':
        other.atomic_number = 8
    else:
        bond._set_bond_metadata(bond_kind='zero' if missing == 'kind' else 'dative',
                                bond_direction='atom1_to_atom2' if missing == 'direction' else None)
    expected = 3 if missing == 'neighbour' else 4
    assert atom.implicit_hydrogens == expected
    assert atom.implicit_hydrogens == expected and len(calls) == 2
    assert _present(mol)


@pytest.mark.parametrize('direction,expected', [('atom1_to_atom2', 3), ('atom2_to_atom1', 2)])
def test_dative_direction_is_site_relative_and_direction_only_writes_invalidate(direction, expected):
    mol = Molecule()
    atom = mol.create_atom(atomic_number=7)
    mol.create_atom(atomic_number=5)
    bond = mol.add_bond(0, 1, 1., bond_kind='dative', bond_direction=direction)
    assert atom.implicit_hydrogens == expected
    source = _capture_source(mol, ('bonding_inputs',))
    reverse = 'atom2_to_atom1' if direction == 'atom1_to_atom2' else 'atom1_to_atom2'
    bond._set_bond_metadata(bond_kind='dative', bond_direction=reverse)
    assert atom._uid in mol._atom_table._dirty_seeds[H_GROUP]
    with pytest.raises(RuntimeError, match='bonding_inputs'):
        _check_source(mol, source)
    assert atom.implicit_hydrogens == 5 - expected


@pytest.mark.parametrize('raw', [np.float64(-0.), np.float64(-3.25), np.float64(65537.25),
                                np.array([0x7ff8000000000035], dtype=np.uint64).view(np.float64)[0]])
def test_supplied_numeric_H_snapshots_preserve_float64_bytes(raw):
    mol = Molecule()
    values = np.zeros(19)
    values[0], values[14] = 6, raw
    atom = mol._create_atom_from_array(values)
    expected = values[14:15].tobytes()
    for snapshot in (atom.attrs[14:15], mol.atom_matrix[0, 14:15],
                     chemistry._atom_projection(mol, fields=('implicit_hydrogens',))[0]):
        assert snapshot.tobytes() == expected
    assert mol._atom_table._sources[0, H_FIELD] == ATOM_SOURCE_CODES['legacy_array']
    if np.isfinite(raw):
        assert atom.implicit_hydrogens == int(raw)


@pytest.mark.parametrize('change,expected', [('radical', 2), ('bracket', 3), ('no_implicit', 0)])
def test_consumed_current_transport_changes_binding_and_recomputes(change, expected):
    mol = Molecule()
    atom = mol.create_atom(atomic_number=6)
    mol.create_atom(atomic_number=6)
    mol.add_bond(0, 1, 1.)
    assert atom.implicit_hydrogens == 3
    source = _capture_source(mol, ('bonding_inputs',))
    current = HydrogenTransportState(900, 1 if change == 'bracket' else 0, change == 'no_implicit')
    _transport(mol, 0, radicals=1 if change == 'radical' else 0, current=current)
    assert atom._uid in mol._atom_table._dirty_seeds[H_GROUP]
    with pytest.raises(RuntimeError, match='bonding_inputs'):
        _check_source(mol, source)
    assert atom.implicit_hydrogens == expected


def test_source_isotope_and_unconsumed_current_count_do_not_dirty_h_or_change_binding(monkeypatch):
    atom = Atom(atomic_number=6)
    mol = atom.mol
    current = HydrogenTransportState(71, 0, False)
    _transport(mol, 0, radicals=0, current=current)
    assert atom.implicit_hydrogens == 4
    source = _capture_source(mol, ('bonding_inputs',))
    before = _dependency_state(mol._atom_table)
    def forbidden(*args):
        raise AssertionError('unconsumed transport edit reran H')
    monkeypatch.setattr(chemistry, '_calculate_implicit_hydrogens', forbidden)
    _transport(mol, 0, radicals=None, current=HydrogenTransportState(999, None, None),
               source=HydrogenTransportState(17, 9, True), isotope=13, namespace='different_source')
    _check_source(mol, source)
    after = _dependency_state(mol._atom_table)
    assert before[0].tobytes() == after[0].tobytes() and before[1] == after[1]
    assert atom.implicit_hydrogens == 4
    assert not mol._atom_table._dirty_seeds[H_GROUP]


def test_supplied_zero_wins_over_current_radical_bracket_and_no_implicit():
    atom = Atom(atomic_number=6, implicit_hydrogens=0)
    _transport(atom.mol, 0, radicals=1, current=HydrogenTransportState(9, 2, True))
    assert atom.implicit_hydrogens == 0
    assert _present(atom.mol)
    assert atom.mol._atom_table._sources[0, H_FIELD] == ATOM_SOURCE_CODES['explicit']


def test_payload_writer_is_factual_and_rolls_back_binding_status_and_seeds(monkeypatch):
    atom = Atom(atomic_number=6)
    assert atom.implicit_hydrogens == 4
    mol = atom.mol
    before = _state(mol)
    source = _capture_source(mol, ('bonding_inputs',))
    def forbidden(*args):
        raise AssertionError('payload setter executed chemistry')
    monkeypatch.setattr(chemistry, '_calculate_implicit_hydrogens', forbidden)
    with pytest.raises(RuntimeError, match='rollback payload'):
        with _FactTransaction(mol, producer=ATOM_SOURCE_CODES['explicit']):
            _transport(mol, 0, radicals=1)
            assert atom._uid in mol._atom_table._dirty_seeds[H_GROUP]
            raise RuntimeError('rollback payload')
    assert _state(mol) == before
    _check_source(mol, source)


@pytest.mark.parametrize('missing', [False, True])
@pytest.mark.parametrize('change', ['radical', 'bracket', 'no_implicit', 'direction'])
def test_later_consumed_input_change_rejects_complete_H_proposal(change, missing, monkeypatch):
    mol = Molecule()
    for _ in range(2):
        mol.create_atom(atomic_number=0 if missing else 6)
    bond = mol.add_bond(0, 1, 1., bond_kind='dative', bond_direction='atom1_to_atom2') if change == 'direction' else None
    before = _state(mol)
    calls = []
    calculate = chemistry._calculate_implicit_hydrogens
    def alter(*args):
        calls.append(args[0])
        if len(calls) == 2:
            if change == 'direction':
                bond._set_bond_metadata(bond_kind='dative', bond_direction='atom2_to_atom1')
            else:
                _transport(mol, 0, radicals=1 if change == 'radical' else None,
                           current=HydrogenTransportState(None, 1 if change == 'bracket' else 0,
                                                          change == 'no_implicit'))
        return calculate(*args)
    monkeypatch.setattr(chemistry, '_calculate_implicit_hydrogens', alter)
    with pytest.raises(RuntimeError, match='changed'):
        chemistry._prepare_hydrogens(mol)
    assert len(calls) == 2
    assert _state(mol) == before


def test_retained_dependency_install_seeds_changed_consumed_payload_and_rolls_back():
    mol = Molecule()
    mol.create_atom(atomic_number=6)
    mol.create_atom(atomic_number=6)
    mol.add_bond(0, 1, 1.)
    assert chemistry._require_implicit_hydrogens(mol) == (3, 3)
    record, metadata = mol._capture_factual_record()
    payload = replace(record.stereo_payload, atom_facts=(AtomTransportRecord(0, None, 1, None, None, 'test'),))
    changed = replace(record, stereo_payload=payload)
    deps = _dependency_state(mol._atom_table)
    before = _state(mol)
    for rollback in (True, False):
        try:
            with _FactTransaction(mol, producer=ATOM_SOURCE_CODES['explicit']):
                _commit_topology(mol, _capture_source(mol), changed,
                                 atom_bindings=tuple(mol.atoms), bond_bindings=tuple(mol.bonds),
                                 dependency_state=deps, bond_metadata=metadata)
                assert mol.atoms[0]._uid in mol._atom_table._dirty_seeds[H_GROUP]
                if rollback:
                    raise RuntimeError('rollback install')
        except RuntimeError as error:
            assert rollback and str(error) == 'rollback install'
        if rollback:
            assert _state(mol) == before
    assert chemistry._require_implicit_hydrogens(mol) == (2, 3)


def test_copy_component_and_pickle_retain_current_input_status_and_uid_mapping(monkeypatch):
    mol = Molecule()
    removed = mol.create_atom(atomic_number=8)
    atom = mol.create_atom(atomic_number=6)
    mol.remove_atom(removed)
    _transport(mol, 0, radicals=1, current=HydrogenTransportState(99, 1, False),
               source=HydrogenTransportState(77, 3, True))
    assert atom.implicit_hydrogens == 3
    copied = copy(mol)
    restored = pickle.loads(pickle.dumps(mol))
    combined = Molecule()
    combined.create_atom(atomic_number=8, implicit_hydrogens=0)
    combined.add_component(mol)
    candidates = ((copied, 0), (restored, 0), (combined, 1))
    def forbidden(*args):
        raise AssertionError('matching copied input/status was needlessly recomputed')
    monkeypatch.setattr(chemistry, '_calculate_implicit_hydrogens', forbidden)
    for candidate, row in candidates:
        assert candidate.atoms[row].implicit_hydrogens == 3
        assert candidate._atom_table._group_status[row, H_GROUP] == _FactStatus.CURRENT
        fact = next(fact for fact in get_stereo_payload(candidate).atom_facts if fact.atom == row)
        assert fact.radical_electrons == 1 and fact.current_hydrogens.bracket_explicit_hydrogens == 1
        assert fact.source_hydrogens == HydrogenTransportState(77, 3, True)
    assert restored.atoms[0]._uid == atom._uid
    assert copied.atoms[0]._uid == 0


@pytest.mark.parametrize('no_implicit,expected', [(False, 3), (True, 1)])
def test_inferred_bracket_H_materializes_once_and_keeps_source_history(no_implicit, expected):
    mol = Molecule()
    atom = mol.create_atom(atomic_number=6)
    mol.create_atom(atomic_number=6, implicit_hydrogens=0)
    mol.add_bond(0, 1, 1.)
    history = HydrogenTransportState(17, 9, True)
    _transport(mol, 0, radicals=0, current=HydrogenTransportState(91, 1, no_implicit), source=history)
    assert atom.implicit_hydrogens == expected
    assert len(atom.add_hydrogen()) == expected
    assert atom.explicit_hydrogens == expected
    fact = next(fact for fact in get_stereo_payload(mol).atom_facts if fact.atom == 0)
    assert fact.current_hydrogens == HydrogenTransportState(0, 0, no_implicit)
    assert fact.source_hydrogens == history
    assert atom.implicit_hydrogens == expected
    assert atom.add_hydrogen() == []
    assert atom.explicit_hydrogens == expected


@pytest.mark.parametrize('number,kind,direction,expected', [
    (6, 'single', None, 3), (7, 'dative', 'atom1_to_atom2', 3),
    (8, 'dative', 'atom1_to_atom2', 2), (8, 'single', None, 1),
])
def test_inferred_metal_site_expands_its_actual_residual_without_polar_subtraction(number, kind, direction, expected):
    mol = Molecule()
    atom = mol.create_atom(atomic_number=number)
    mol.create_atom(atomic_number=30)
    mol.add_bond(0, 1, 1., bond_kind=kind, bond_direction=direction)
    mol.add_hydrogens(rng=np.random.default_rng(42))
    assert atom.explicit_hydrogens == expected
    assert atom.implicit_hydrogens == expected
    assert mol._atom_table._sources[0, H_FIELD] == ATOM_SOURCE_CODES['implicit_hydrogens']
    mol.add_hydrogens(rng=np.random.default_rng(42))
    assert atom.explicit_hydrogens == expected


def test_supplied_legacy_polar_count_keeps_its_characterized_expansion_interpretation():
    mol = Molecule()
    atom = mol.create_atom(atomic_number=8, implicit_hydrogens=2)
    mol.create_atom(atomic_number=30)
    mol.add_bond(0, 1, 1.)
    mol.add_hydrogens(rng=np.random.default_rng(42))
    assert atom.explicit_hydrogens == 1
    assert atom.implicit_hydrogens == 2
    assert mol._atom_table._sources[0, H_FIELD] == ATOM_SOURCE_CODES['explicit']


@pytest.mark.parametrize('failure', ['graph', 'payload'])
def test_inferred_expansion_failure_restores_graph_payload_target_and_dependencies(failure, monkeypatch):
    atom = Atom(atomic_number=6)
    mol = atom.mol
    _transport(mol, 0, current=HydrogenTransportState(None, 1, False),
               source=HydrogenTransportState(8, 7, True))
    before = _state(mol)
    if failure == 'graph':
        add = Atom._add_atom
        calls = []
        def injected(self, *args, **kwargs):
            calls.append(self._uid)
            if len(calls) == 2:
                raise RuntimeError('injected graph expansion')
            return add(self, *args, **kwargs)
        monkeypatch.setattr(Atom, '_add_atom', injected)
    else:
        materialize = chemistry._materialize_current_hydrogens
        def injected(self):
            materialize(self)
            raise RuntimeError('injected payload transfer')
        monkeypatch.setattr(chemistry, '_materialize_current_hydrogens', injected)
    with pytest.raises(RuntimeError, match='injected'):
        mol.add_hydrogens(rng=np.random.default_rng(3))
    assert _state(mol) == before
    assert not _present(mol)
    assert not mol.hydrogens


def test_graph_D_and_T_count_as_H_and_keep_their_isotopes():
    mol = Molecule()
    atom = mol.create_atom(atomic_number=6)
    for isotope in (2, 3):
        hydrogen = mol.create_atom(atomic_number=1)
        hydrogen.isotope = isotope
        mol.add_bond(atom, hydrogen, 1.)
    assert atom.implicit_hydrogens == 4
    assert len(atom.add_hydrogen()) == 2
    assert atom.explicit_hydrogens == 4
    assert sorted(h.isotope for h in mol.hydrogens if 'isotope' in h.__dict__) == [2, 3]
    assert atom.add_hydrogen() == []


@pytest.mark.parametrize('case', ['supplied', 'inferred', 'unresolved', 'stale'])
def test_detached_H_reads_keep_current_raw_values_and_hide_absent_or_stale_counts(case):
    if case == 'supplied':
        mol = Molecule()
        values = np.zeros(19)
        values[0], values[14] = 6, 65537.25
        atom = mol._create_atom_from_array(values)
    else:
        atom = Atom(atomic_number=0 if case == 'unresolved' else 6)
        mol = atom.mol
        assert atom.implicit_hydrogens == (0 if case == 'unresolved' else 4)
        if case == 'stale':
            atom.atomic_number = 0
    mol.remove_atom(atom)
    expected = 65537.25 if case == 'supplied' else 4 if case == 'inferred' else 0
    before = atom._detached_table.__getstate__()
    assert atom.implicit_hydrogens == int(expected)
    assert atom.attrs[14] == expected
    after = atom._detached_table.__getstate__()
    assert pickle.dumps(before) == pickle.dumps(after)


def test_stereo_descriptor_only_edit_keeps_H_binding_and_does_not_run_rule(monkeypatch):
    atom = Atom(atomic_number=6)
    assert atom.implicit_hydrogens == 4
    mol = atom.mol
    source = _capture_source(mol, ('bonding_inputs',))
    before = _dependency_state(mol._atom_table)
    payload = get_stereo_payload(mol)
    descriptor = StereoRecord(StereoAnchor(StereoAnchorKind.ATOM, (0,)), (),
                              'R', None, 'test', 'descriptor-only')
    def forbidden(*args):
        raise AssertionError('stereo descriptor reran H')
    monkeypatch.setattr(chemistry, '_calculate_implicit_hydrogens', forbidden)
    set_stereo_payload(mol, replace(payload, records=(descriptor,)))
    _check_source(mol, source)
    after = _dependency_state(mol._atom_table)
    assert before[0].tobytes() == after[0].tobytes() and before[1] == after[1]
    assert atom.implicit_hydrogens == 4


@pytest.mark.parametrize('change', ['bracket', 'no_implicit'])
def test_current_transport_change_retries_an_unchanged_unresolved_site_once(change, monkeypatch):
    atom = Atom()
    mol = atom.mol
    calls = []
    calculate = chemistry._calculate_implicit_hydrogens
    def observe(*args):
        calls.append(args[0])
        return calculate(*args)
    monkeypatch.setattr(chemistry, '_calculate_implicit_hydrogens', observe)
    assert atom.implicit_hydrogens == 0
    assert atom.implicit_hydrogens == 0 and calls == [0]
    _transport(mol, 0, current=HydrogenTransportState(None, 1 if change == 'bracket' else 0,
                                                    change == 'no_implicit'))
    assert atom.implicit_hydrogens == 0
    assert atom.implicit_hydrogens == 0 and calls == [0, 0]
    assert not _present(mol)
    assert mol._atom_table._group_status[0, H_GROUP] == _FactStatus.UNRESOLVED


@pytest.mark.parametrize('entry', ['atom', 'molecule'])
@pytest.mark.parametrize('graph_H', [0, 1])
def test_inferred_current_bracket_inventory_is_nonoverlapping_after_materialization(entry, graph_H):
    atom = Atom(atomic_number=6)
    mol = atom.mol
    history = HydrogenTransportState(17, 1, False)
    _transport(mol, 0, current=HydrogenTransportState(None, 1, False), source=history)
    for _ in range(graph_H):
        atom.add_atom(1)
    assert atom.implicit_hydrogens == 4
    source_bytes = pickle.dumps(get_stereo_payload(mol).atom_facts[0].source_hydrogens)
    if entry == 'atom':
        assert len(atom.add_hydrogen()) == 4 - graph_H
    else:
        mol.add_hydrogens(rng=np.random.default_rng(42))
    fact = next(fact for fact in get_stereo_payload(mol).atom_facts if fact.atom == 0)
    assert atom.explicit_hydrogens == 4 and atom.implicit_hydrogens == 4
    assert fact.current_hydrogens == HydrogenTransportState(0, 0, False)
    assert pickle.dumps(fact.source_hydrogens) == source_bytes
    if entry == 'atom':
        assert atom.add_hydrogen() == []
    else:
        mol.add_hydrogens(rng=np.random.default_rng(42))
    assert atom.explicit_hydrogens == 4


@pytest.mark.parametrize('entry', ['atom', 'molecule'])
@pytest.mark.parametrize('no_implicit', [None, False, True])
def test_inferred_zero_residual_cleans_current_H_and_preserves_target_and_history(entry, no_implicit):
    atom = Atom(atomic_number=6)
    mol = atom.mol
    for _ in range(4):
        atom.add_atom(1)
    history = HydrogenTransportState(17, 7, True)
    _transport(mol, 0, isotope=13, radicals=0,
               current=HydrogenTransportState(91, 0, no_implicit), source=history)
    assert atom.implicit_hydrogens == 4
    graph = tuple(mol.atoms), tuple(mol.bonds)
    source_bytes = pickle.dumps(get_stereo_payload(mol).atom_facts[0].source_hydrogens)
    for _ in range(2):
        if entry == 'atom':
            assert atom.add_hydrogen() == []
        else:
            mol.add_hydrogens(rng=np.random.default_rng(42))
        assert (tuple(mol.atoms), tuple(mol.bonds)) == graph
        fact = get_stereo_payload(mol).atom_facts[0]
        assert fact.current_hydrogens == HydrogenTransportState(0, 0, no_implicit)
        assert pickle.dumps(fact.source_hydrogens) == source_bytes
        assert fact.isotope == 13 and fact.radical_electrons == 0
        assert atom.implicit_hydrogens == 4 and atom.explicit_hydrogens == 4
        assert mol._atom_table._columns[H_FIELD][0] == 4
        assert mol._atom_table._sources[0, H_FIELD] == ATOM_SOURCE_CODES['implicit_hydrogens']
        assert _present(mol) and mol._atom_table._group_status[0, H_GROUP] == _FactStatus.CURRENT


@pytest.mark.parametrize('number', [1, 30], ids=['real-H', 'metal'])
@pytest.mark.parametrize('no_implicit', [None, False, True])
def test_direct_automatic_H_and_metal_calls_preserve_current_transport(number, no_implicit):
    atom = Atom(atomic_number=number)
    mol = atom.mol
    _transport(mol, 0, current=HydrogenTransportState(91, 3, no_implicit),
               source=HydrogenTransportState(17, 7, True))
    assert atom.implicit_hydrogens == 0
    graph = tuple(mol.atoms), tuple(mol.bonds)
    payload_bytes = pickle.dumps(get_stereo_payload(mol))
    for _ in range(2):
        assert atom.add_hydrogen() == []
        assert (tuple(mol.atoms), tuple(mol.bonds)) == graph
        assert pickle.dumps(get_stereo_payload(mol)) == payload_bytes
        assert atom.implicit_hydrogens == 0
        assert mol._atom_table._sources[0, H_FIELD] == ATOM_SOURCE_CODES['implicit_hydrogens']
        assert _present(mol)


@pytest.mark.parametrize('entry,count_source', [
    ('atom', 'inferred-num0'), ('atom', 'explicit'), ('molecule', 'explicit'),
    ('atom', 'legacy_array'), ('molecule', 'legacy_array'),
])
@pytest.mark.parametrize('no_implicit', [None, False, True])
def test_zero_addition_preserves_transport_for_explicit_num_and_supplied_counts(entry, count_source, no_implicit):
    mol = Molecule()
    if count_source == 'legacy_array':
        values = np.zeros(19)
        values[0], values[H_FIELD] = 6, 4
        atom = mol._create_atom_from_array(values)
    elif count_source == 'explicit':
        atom = mol.create_atom(atomic_number=6, implicit_hydrogens=4)
    else:
        atom = mol.create_atom(atomic_number=6)
    for _ in range(4):
        atom.add_atom(1)
    _transport(mol, 0, current=HydrogenTransportState(91, 0 if count_source == 'inferred-num0' else 1,
                                                    no_implicit),
               source=HydrogenTransportState(17, 7, True))
    assert atom.implicit_hydrogens == 4
    producer = ATOM_SOURCE_CODES['implicit_hydrogens' if count_source == 'inferred-num0' else count_source]
    assert mol._atom_table._sources[0, H_FIELD] == producer
    graph = tuple(mol.atoms), tuple(mol.bonds)
    payload_bytes = pickle.dumps(get_stereo_payload(mol))
    for _ in range(2):
        if count_source == 'inferred-num0':
            assert atom.add_hydrogen(num=0) == []
        elif entry == 'atom':
            assert atom.add_hydrogen() == []
        else:
            mol.add_hydrogens(rng=np.random.default_rng(42))
        assert (tuple(mol.atoms), tuple(mol.bonds)) == graph
        assert pickle.dumps(get_stereo_payload(mol)) == payload_bytes
        assert atom.implicit_hydrogens == 4 and _present(mol)
        assert mol._atom_table._sources[0, H_FIELD] == producer


@pytest.mark.parametrize('entry', ['atom', 'molecule'])
def test_zero_residual_current_H_cleanup_failure_rolls_back_whole_entry(entry, monkeypatch):
    atom = Atom(atomic_number=6)
    mol = atom.mol
    for _ in range(4):
        atom.add_atom(1)
    _transport(mol, 0, current=HydrogenTransportState(91, 0, None),
               source=HydrogenTransportState(17, 7, True))
    assert atom.implicit_hydrogens == 4
    before = _state(mol)
    materialize = chemistry._materialize_current_hydrogens
    def injected(member):
        materialize(member)
        raise RuntimeError('zero residual cleanup')
    monkeypatch.setattr(chemistry, '_materialize_current_hydrogens', injected)
    with pytest.raises(RuntimeError, match='zero residual cleanup'):
        if entry == 'atom':
            atom.add_hydrogen()
        else:
            mol.add_hydrogens(rng=np.random.default_rng(42))
    assert _state(mol) == before


@pytest.mark.parametrize('entry', ['atom', 'molecule'])
def test_aromatic_n_bracket_H_with_no_implicit_materializes_once(entry):
    from hotpot.cheminfo.rdconvert import from_rdmol
    from rdkit import Chem
    rdmol = Chem.MolFromSmiles('c1cc[nH]c1')
    next(atom for atom in rdmol.GetAtoms() if atom.GetAtomicNum() == 7).SetNoImplicit(True)
    mol = from_rdmol(rdmol, Molecule())
    atom = next(atom for atom in mol.atoms if atom.atomic_number == 7)
    # Request a fresh default for n while retaining its real bracket/current facts.
    from hotpot.cheminfo.structure._mutation import _normalize_atom_fields
    from hotpot.cheminfo.structure._commit import _commit_atom_facts
    _commit_atom_facts(mol, _normalize_atom_fields(_capture_source(mol), (atom.idx,), (atom._uid,),
        {'implicit_hydrogens': (0,)}, producer=ATOM_SOURCE_CODES['default'], present=False))
    fact = next(fact for fact in get_stereo_payload(mol).atom_facts if fact.atom == atom.idx)
    assert fact.current_hydrogens.bracket_explicit_hydrogens == 1
    assert fact.current_hydrogens.no_implicit is True
    history_bytes = pickle.dumps(fact.source_hydrogens)
    assert atom.implicit_hydrogens == 1
    if entry == 'atom':
        assert len(atom.add_hydrogen()) == 1
    else:
        mol.add_hydrogens(rng=np.random.default_rng(42))
    fact = next(fact for fact in get_stereo_payload(mol).atom_facts if fact.atom == atom.idx)
    assert atom.explicit_hydrogens == 1 and atom.implicit_hydrogens == 1
    assert fact.current_hydrogens == HydrogenTransportState(0, 0, True)
    assert pickle.dumps(fact.source_hydrogens) == history_bytes
    if entry == 'atom':
        assert atom.add_hydrogen() == []
    else:
        mol.add_hydrogens(rng=np.random.default_rng(42))
    assert atom.explicit_hydrogens == 1


@pytest.mark.parametrize('entry', ['atom', 'molecule'])
@pytest.mark.parametrize('failure', ['after_payload', 'final_topology'])
def test_expansion_entry_rolls_back_first_row_when_a_later_step_fails(entry, failure, monkeypatch):
    mol = Molecule()
    for _ in range(2 if entry == 'molecule' else 1):
        atom = mol.create_atom(atomic_number=6)
        _transport(mol, atom.idx, current=HydrogenTransportState(None, 1, False),
                   source=HydrogenTransportState(11, 1, False))
    before = _state(mol)
    visited = []
    if failure == 'after_payload':
        materialize = chemistry._materialize_current_hydrogens
        def injected(atom):
            materialize(atom)
            visited.append(atom._uid)
            if len(visited) == (2 if entry == 'molecule' else 1):
                raise RuntimeError('later expansion step')
        monkeypatch.setattr(chemistry, '_materialize_current_hydrogens', injected)
    else:
        update = Molecule._update_topology
        def injected(owner, *args, **kwargs):
            update(owner, *args, **kwargs)
            visited.append(len(owner.hydrogens))
            raise RuntimeError('later expansion step')
        monkeypatch.setattr(Molecule, '_update_topology', injected)
    with pytest.raises(RuntimeError, match='later expansion step'):
        if entry == 'atom':
            mol.atoms[0].add_hydrogen()
        else:
            mol.add_hydrogens(rng=np.random.default_rng(42))
    assert visited
    assert _state(mol) == before
    assert not mol.hydrogens and not _present(mol)


def test_metal_graph_hydride_keeps_inferred_raw_zero_and_is_not_removed():
    mol = Molecule()
    zinc = mol.create_atom(atomic_number=30)
    hydrogen = mol.create_atom(atomic_number=1)
    hydrogen.isotope = 2
    mol.add_bond(zinc, hydrogen, 1.)
    _transport(mol, 0, current=HydrogenTransportState(None, 1, True))
    assert zinc.implicit_hydrogens == 0 and _present(mol)
    before = tuple(mol.atoms), tuple(mol.bonds), get_stereo_payload(mol)
    mol.add_hydrogens(rng=np.random.default_rng(42))
    assert zinc.add_hydrogen() == []
    assert zinc.implicit_hydrogens == 0
    assert (tuple(mol.atoms), tuple(mol.bonds), get_stereo_payload(mol)) == before
    assert hydrogen.isotope == 2


@pytest.mark.parametrize('consumer', ['sum', 'organic', 'oxidation', 'default_charge', 'stereo'])
def test_direct_chemical_consumers_do_not_treat_missing_display_as_ready(consumer):
    atom = Atom()
    mol = atom.mol
    consume = {
        'sum': lambda: mol.sum_implicit_hydrogens,
        'organic': lambda: mol.is_organic,
        'oxidation': lambda: atom.oxidation_state,
        'default_charge': lambda: mol.calc_mol_default_charge(),
    }
    if consumer == 'stereo':
        from hotpot.cheminfo.operations.stereochemistry import _gate6_projection
        action = lambda: _gate6_projection(mol)
    else:
        action = consume[consumer]
    assert atom.implicit_hydrogens == 0 and not _present(mol)
    with pytest.raises(ValueError, match='resolved implicit-H'):
        action()
    assert not _present(mol)


def test_detached_H_snapshot_accounts_for_a_pending_neighbour_input():
    mol = Molecule()
    atom = mol.create_atom(atomic_number=6)
    neighbour = mol.create_atom(atomic_number=6)
    mol.add_bond(atom, neighbour, 1.)
    assert atom.implicit_hydrogens == 3
    neighbour.atomic_number = 0
    mol.remove_atom(atom)
    assert atom.implicit_hydrogens == 0 and atom.attrs[14] == 0
    assert atom._detached_table._columns[H_FIELD][0] == 3
    assert atom._detached_table._group_status[0, H_GROUP] == _FactStatus.NEEDS_RECALC


def test_batch_expansion_retains_prewrite_eligibility_when_earlier_writes_dirty_later_rows(monkeypatch):
    from hotpot.cheminfo.structure._dependencies import _ensure_group
    mol = Molecule()
    carbon = mol.create_atom(atomic_number=6)
    oxygen = mol.create_atom(atomic_number=8)
    zinc = mol.create_atom(atomic_number=30)
    mol.add_bond(carbon, oxygen, 1.)
    mol.add_bond(oxygen, zinc, 1., bond_kind='dative', bond_direction='atom1_to_atom2')
    for atom in (carbon, oxygen):
        _transport(mol, atom.idx, current=HydrogenTransportState(None, 1, False),
                   source=HydrogenTransportState(17, 1, False))
    materialize = chemistry._materialize_current_hydrogens
    visited = []
    def expand_pending(atom):
        materialize(atom)
        visited.append(atom._uid)
        if atom is carbon:
            _ensure_group(mol, 'IMPLICIT_H')
            assert mol._atom_table._group_status[oxygen.idx, H_GROUP] == _FactStatus.NEEDS_RECALC
    monkeypatch.setattr(chemistry, '_materialize_current_hydrogens', expand_pending)
    mol.add_hydrogens(rng=np.random.default_rng(42))
    assert visited == [carbon._uid, oxygen._uid]
    assert carbon.explicit_hydrogens == 3 and oxygen.explicit_hydrogens == 1
    assert (carbon.implicit_hydrogens, oxygen.implicit_hydrogens) == (3, 1)
    assert all(fact.current_hydrogens.bracket_explicit_hydrogens == 0
               for fact in get_stereo_payload(mol).atom_facts)
    monkeypatch.setattr(chemistry, '_materialize_current_hydrogens', materialize)
    mol.add_hydrogens(rng=np.random.default_rng(42))
    assert carbon.explicit_hydrogens == 3 and oxygen.explicit_hydrogens == 1


@pytest.mark.parametrize('number,charge,radicals', [(15, 0, 1), (6, 2, 0)])
def test_unsupported_known_state_materializes_declared_bracket_H_and_preserves_None_policy(number, charge, radicals):
    atom = Atom(atomic_number=number, formal_charge=charge)
    mol = atom.mol
    history = HydrogenTransportState(8, 1, None)
    _transport(mol, 0, radicals=radicals, current=HydrogenTransportState(None, 1, None), source=history)
    assert atom.implicit_hydrogens == 1
    assert len(atom.add_hydrogen()) == 1
    fact = next(fact for fact in get_stereo_payload(mol).atom_facts if fact.atom == 0)
    assert fact.radical_electrons == radicals
    assert fact.current_hydrogens == HydrogenTransportState(0, 0, None)
    assert fact.source_hydrogens == history
    assert atom.implicit_hydrogens == 1 and atom.add_hydrogen() == []


def test_payload_seed_failure_restores_old_payload_and_native_dependency_state():
    from hotpot.cheminfo.structure._commit import _native_fail_after
    atom = Atom(atomic_number=6)
    assert atom.implicit_hydrogens == 4
    mol = atom.mol
    before = _state(mol)
    source = _capture_source(mol, ('bonding_inputs',))
    with pytest.raises(KeyboardInterrupt):
        with _native_fail_after('seed_add'):
            _transport(mol, 0, radicals=1)
    assert _state(mol) == before
    _check_source(mol, source)


@pytest.mark.parametrize('change,expected', [
    ('element', (1, 3)), ('charge', (2, 3)), ('bond_order', (2, 2)),
    ('bond_kind', (4, 4)), ('direction', (3, 4)), ('clear_H', (3, 3)),
])
def test_retained_dependency_install_retries_every_changed_H_input_family(change, expected):
    mol = Molecule()
    for _ in range(2):
        mol.create_atom(atomic_number=6)
    mol.add_bond(0, 1, 1., bond_kind='dative' if change == 'direction' else 'single',
                 bond_direction='atom1_to_atom2' if change == 'direction' else None)
    chemistry._require_implicit_hydrogens(mol)
    record, metadata = mol._capture_factual_record()
    values = np.frombuffer(record.atom_values, dtype='<f8').reshape(2, 19).copy()
    presence = np.frombuffer(record.atom_presence, dtype='<u4').copy()
    sources = np.frombuffer(record.atom_sources, dtype='u1').reshape(2, 19).copy()
    changes = {}
    if change in ('element', 'charge'):
        field = FIELD_INDEX['atomic_number' if change == 'element' else 'formal_charge']
        values[0, field] = 8 if change == 'element' else 1
        presence[0] |= FIELD_PRESENCE_BITS[field]
        sources[0, field] = ATOM_SOURCE_CODES['explicit']
        changes.update(atom_values=values.tobytes(), atom_presence=presence.tobytes(), atom_sources=sources.tobytes())
    elif change in ('bond_order', 'bond_kind'):
        bond_values = np.frombuffer(record.bond_values, dtype='<f8').reshape(1, 3).copy()
        bond_values[0, 0] = 2 if change == 'bond_order' else 0
        changes.update(bond_values=bond_values.tobytes(),
                       bond_kinds=('double' if change == 'bond_order' else 'zero',))
    elif change == 'direction':
        changes['bond_directions'] = ('atom2_to_atom1',)
    else:
        presence[0] &= np.uint32(~FIELD_PRESENCE_BITS[H_FIELD] & ((1 << 19) - 1))
        sources[0, H_FIELD] = ATOM_SOURCE_CODES['default']
        changes.update(atom_presence=presence.tobytes(), atom_sources=sources.tobytes())
    _commit_topology(mol, _capture_source(mol), replace(record, **changes),
                     atom_bindings=tuple(mol.atoms), bond_bindings=tuple(mol.bonds),
                     dependency_state=_dependency_state(mol._atom_table), bond_metadata=metadata)
    assert mol.atoms[0]._uid in mol._atom_table._dirty_seeds[H_GROUP]
    assert chemistry._require_implicit_hydrogens(mol) == expected


def test_retained_dependency_install_promotes_new_source_zero_over_old_unresolved_status():
    atom = Atom()
    mol = atom.mol
    assert atom.implicit_hydrogens == 0 and not _present(mol)
    assert mol._atom_table._group_status[0, H_GROUP] == _FactStatus.UNRESOLVED
    record, metadata = mol._capture_factual_record()
    presence = np.frombuffer(record.atom_presence, dtype='<u4').copy()
    sources = np.frombuffer(record.atom_sources, dtype='u1').reshape(1, 19).copy()
    presence[0] |= FIELD_PRESENCE_BITS[H_FIELD]
    sources[0, H_FIELD] = ATOM_SOURCE_CODES['explicit']
    record = replace(record, atom_presence=presence.tobytes(), atom_sources=sources.tobytes())
    _commit_topology(mol, _capture_source(mol), record, atom_bindings=(atom,), bond_bindings=(),
                     dependency_state=_dependency_state(mol._atom_table), bond_metadata=metadata)
    assert chemistry._require_implicit_hydrogens(mol) == (0,)
    assert _present(mol) and mol._atom_table._group_status[0, H_GROUP] == _FactStatus.CURRENT
    assert mol._atom_table._sources[0, H_FIELD] == ATOM_SOURCE_CODES['explicit']


def test_retained_dependency_install_of_source_only_history_keeps_matching_H_status(monkeypatch):
    mol = Molecule()
    for _ in range(2):
        mol.create_atom(atomic_number=6)
    mol.add_bond(0, 1, 1.)
    assert chemistry._require_implicit_hydrogens(mol) == (3, 3)
    record, metadata = mol._capture_factual_record()
    payload = replace(record.stereo_payload, atom_facts=(AtomTransportRecord(0, 13, None, None,
                       HydrogenTransportState(17, 7, True), 'history-only'),))
    deps = _dependency_state(mol._atom_table)
    def forbidden(*args):
        raise AssertionError('history-only install reran H')
    monkeypatch.setattr(chemistry, '_calculate_implicit_hydrogens', forbidden)
    _commit_topology(mol, _capture_source(mol), replace(record, stereo_payload=payload),
                     atom_bindings=tuple(mol.atoms), bond_bindings=tuple(mol.bonds),
                     dependency_state=deps, bond_metadata=metadata)
    after = _dependency_state(mol._atom_table)
    assert deps[0].tobytes() == after[0].tobytes() and deps[1] == after[1]
    assert chemistry._require_implicit_hydrogens(mol) == (3, 3)
