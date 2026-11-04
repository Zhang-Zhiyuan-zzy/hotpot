"""Actual operations retain chemistry, selected topology and atomic recovery."""

from dataclasses import replace
import math

import pytest

from hotpot import BondKind, Molecule
from hotpot.cheminfo.operations import chemistry
from hotpot.cheminfo.operations._aromatic_protocol import _evaluate_aromatic_assignment
from hotpot.cheminfo.rules.chemistry.bonding import kekule as implementation
from hotpot.cheminfo.structure._dependencies import GROUP_INDEX, _FactStatus, _dependency_state
from tests.test_cheminfo._stage5_fixtures import factual_state


def _molecule(coordinates, edges):
    mol = Molecule()
    for row, point in enumerate(coordinates):
        mol.create_atom(atomic_number=6, coordinates=point, id=101 + 3 * row)
    for first, second in edges:
        mol.add_bond(first, second, 1.)
    return mol


def _benzene():
    return _molecule(
        tuple((math.cos(row * math.pi / 3), math.sin(row * math.pi / 3), 0.)
              for row in range(6)),
        tuple((row, (row + 1) % 6) for row in range(6)),
    )


def _state(mol):
    statuses, seeds = _dependency_state(mol._atom_table)
    return factual_state(mol), statuses.tobytes(), seeds


def test_operation_binding_is_the_defined_sole_implementation():
    assert chemistry._calculate_aromatic_assignment is _evaluate_aromatic_assignment
    assert _evaluate_aromatic_assignment is implementation._calculate_aromatic_assignment


@pytest.mark.parametrize('case', ('ordinary', 'fused', 'coordination'))
def test_public_assignment_uses_real_rule_with_complete_selected_mapping(case, monkeypatch):
    if case == 'fused':
        mol = _molecule(
            ((-1., 0., 0.), (0., .5, 0.), (0., 1.5, 0.), (-1., 2., 0.),
             (-2., 1.5, 0.), (-2., .5, 0.), (1., 0., 0.), (2., .5, 0.),
             (2., 1.5, 0.), (1., 2., 0.)),
            ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
             (1, 6), (6, 7), (7, 8), (8, 9), (9, 2)),
        )
        expected_orders = (2., 1., 2., 1., 2., 1., 1., 2., 1., 2., 1.)
    else:
        mol = _benzene()
        expected_orders = (2., 1., 2., 1., 2., 1.)
    ring_atom_count = len(mol.atoms)
    ring_bonds = tuple(mol.bonds)
    if case == 'coordination':
        zinc = mol.create_atom(atomic_number=30, coordinates=(4., 0., 0.), id=901)
        coordination = mol.add_bond(0, zinc, 1., bond_kind=BondKind.DATIVE,
            bond_direction='atom1_to_atom2', bond_source='fixture',
            bond_source_metadata={'label': 'coordination'})
        coordination_before = coordination.attr_dict
    atoms, bonds, index = tuple(mol.atoms), tuple(mol.bonds), mol._topology_index
    observed = []

    def observe(snapshot, *, kekule):
        before = _state(mol)
        proposal = _evaluate_aromatic_assignment(snapshot, kekule=kekule)
        assert _state(mol) == before
        observed.append((snapshot, proposal, kekule))
        return proposal

    monkeypatch.setattr(chemistry, '_calculate_aromatic_assignment', observe)
    assert mol.determine_rings_aromatic() is None

    assert len(observed) == 1
    snapshot, proposal, mode = observed[0]
    assert mode is True and proposal.source is snapshot.source
    assert proposal.atom_uids == tuple(atom._uid for atom in atoms)
    assert proposal.bond_uids == tuple(bond._uid for bond in ring_bonds)
    assert proposal.covered_rows == tuple(range(len(atoms)))
    assert proposal.method == 'kekule'
    assert proposal.bond_orders == expected_orders
    assert tuple(bond.bond_order for bond in ring_bonds) == expected_orders
    assert tuple(bond.bond_kind.value for bond in ring_bonds) == tuple(
        'double' if order == 2. else 'single' for order in expected_orders)
    assert all(atom.is_aromatic for atom in atoms[:ring_atom_count])
    assert all(bond.is_aromatic for bond in ring_bonds)
    assert tuple(mol.atoms) == atoms and tuple(mol.bonds) == bonds
    assert mol._topology_index is index
    if case == 'coordination':
        assert coordination._uid not in proposal.bond_uids
        assert coordination.attr_dict == coordination_before
        assert not zinc.is_aromatic and not coordination.is_aromatic


def test_lazy_geometry_refresh_preserves_aromatic_mode_and_stable_clean_reads(monkeypatch):
    mol = _benzene()
    planar = mol.coordinates.copy()
    mol.atoms[0].z = 2.
    mol.calc_atom_valence(assign_aromatic=True)
    assert not any(atom.is_aromatic for atom in mol.atoms)
    atoms, bonds = tuple(mol.atoms), tuple(mol.bonds)
    observed = []

    def observe(snapshot, *, kekule):
        observed.append((kekule, snapshot.atom_uids, snapshot.ring_has_3d))
        return _evaluate_aromatic_assignment(snapshot, kekule=kekule)

    monkeypatch.setattr(chemistry, '_calculate_aromatic_assignment', observe)
    mol.coordinates = planar
    assert mol.atoms[-1].is_aromatic is True
    assert all(atom.is_aromatic for atom in atoms)
    assert all(bond.is_aromatic for bond in bonds)
    expected = (False, tuple(atom._uid for atom in atoms), (True,))
    assert observed and all(call == expected for call in observed)
    clean_calls = len(observed)
    assert all(atom.is_aromatic for atom in atoms)
    assert len(observed) == clean_calls
    mol.atoms[0].z = 2.
    assert mol.atoms[-1].is_aromatic is False
    assert not any(atom.is_aromatic for atom in atoms)
    assert len(observed) > clean_calls and all(call == expected for call in observed)
    clean_calls = len(observed)
    assert not any(atom.is_aromatic for atom in atoms)
    assert len(observed) == clean_calls
    assert tuple(bond.bond_order for bond in bonds) == (1.,) * 6


@pytest.mark.parametrize('entry', ('explicit', 'lazy'))
def test_actual_rule_none_keeps_facts_and_reports_existing_unresolved_scope(entry, monkeypatch):
    mol = _benzene()
    if entry == 'lazy':
        mol.determine_rings_aromatic()
        mol.atoms[0].z = .25
    else:
        chemistry._prepare_aromatic_representation(mol)
    capture = chemistry._bonding_snapshot
    observed = []

    def nonspatial_snapshot(owner, **kwargs):
        # Only the test selects the existing nonspatial branch; production's
        # accepted snapshot still supplies True for every ring.
        snapshot = capture(owner, **kwargs)
        return replace(snapshot, ring_has_3d=(False,) * len(snapshot.rings))

    def observe(snapshot, *, kekule):
        assert all(value is None for value in snapshot.implicit_hydrogens)
        result = _evaluate_aromatic_assignment(snapshot, kekule=kekule)
        observed.append((kekule, result))
        return result

    before = factual_state(mol)
    monkeypatch.setattr(chemistry, '_bonding_snapshot', nonspatial_snapshot)
    monkeypatch.setattr(chemistry, '_calculate_aromatic_assignment', observe)
    if entry == 'explicit':
        assert mol.determine_rings_aromatic() is None
    else:
        chemistry._prepare_aromatic_representation(mol)
    assert observed == [(entry == 'explicit', None)]
    assert factual_state(mol) == before
    assert mol._atom_table._group_status[:6, GROUP_INDEX['AROMATIC_REPRESENTATION']].tolist() == [
        _FactStatus.UNRESOLVED] * 6
    with pytest.raises(RuntimeError) as caught:
        chemistry._require_aromatic_representation(mol)
    assert 'AROMATIC.NO_CURRENT_ASSIGNMENT' in str(caught.value)
    assert 'status=UNRESOLVED' in str(caught.value)
    assert 'NONCYCLIC_SOURCE' not in str(caught.value)
    assert len(observed) == 1 and factual_state(mol) == before


@pytest.mark.parametrize('phase', ('rule', 'late_commit'))
def test_public_operation_preserves_original_exception_and_rolls_back_exact_state(phase, monkeypatch):
    mol = _benzene()
    before = _state(mol)
    atoms, bonds, index = tuple(mol.atoms), tuple(mol.bonds), mol._topology_index
    failure = (RuntimeError('original aromatic rule failure') if phase == 'rule'
               else KeyboardInterrupt('failure after the actual coupled commit'))
    commit = chemistry._commit_aromatic_assignment
    committed = []

    def fail_rule(*args, **kwargs):
        raise failure

    def fail_after_commit(owner, proposal, **kwargs):
        commit(owner, proposal, **kwargs)
        assert owner._atom_table.column('is_aromatic').tolist() == [1.] * 6
        assert tuple(bond.bond_order for bond in owner.bonds) == (2., 1., 2., 1., 2., 1.)
        committed.append(True)
        raise failure

    with monkeypatch.context() as patch:
        if phase == 'rule':
            patch.setattr(implementation, '_determine_ring_aromaticity', fail_rule)
        else:
            patch.setattr(chemistry, '_commit_aromatic_assignment', fail_after_commit)
        with pytest.raises(type(failure)) as caught:
            mol.determine_rings_aromatic()
    assert caught.value is failure
    assert committed == ([True] if phase == 'late_commit' else [])
    assert _state(mol) == before
    assert tuple(mol.atoms) == atoms and tuple(mol.bonds) == bonds
    assert mol._topology_index is index
    assert mol.determine_rings_aromatic() is None
    assert all(atom.is_aromatic for atom in mol.atoms)
