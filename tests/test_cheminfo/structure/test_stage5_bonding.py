"""Stage5 pure rules, named preparation and immutable Ring contracts.

Missing-H state stays absent/UNRESOLVED; public display and strict consumer
checks live in test_stage5_hydrogen_defaults.py. Historical node names remain.
"""

from dataclasses import FrozenInstanceError, replace
from copy import copy
import math
import pickle

import numpy as np
import pytest

from hotpot import Atom, Molecule
from hotpot.cheminfo.operations import chemistry
from hotpot.cheminfo.structure._commit import _commit_aromatic_assignment
from hotpot.cheminfo.structure._dependencies import GROUP_INDEX, _FactStatus, _dependency_state
from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES, FIELD_INDEX
from hotpot.cheminfo.rules.chemistry.bonding.kekule import _calculate_aromatic_assignment
from tests.test_cheminfo._stage5_fixtures import factual_state, supplied_aromatic_flags


def _cycle(size=6):
    mol = Molecule()
    for row in range(size):
        angle = row * 2 * math.pi / size
        mol.create_atom(atomic_number=6, coordinates=(math.cos(angle), math.sin(angle), 0.))
    for row in range(size):
        mol.add_bond(row, (row + 1) % size, 1.)
    return mol


def _state(mol):
    status, seeds = _dependency_state(mol._atom_table)
    return factual_state(mol), status.tobytes(), seeds


def test_element_and_bond_writers_never_run_a_chemistry_rule(monkeypatch):
    mol = _cycle()
    def forbidden(*args, **kwargs):
        raise AssertionError('factual writer executed a chemistry rule')
    for name in ('_calculate_valence', '_calculate_implicit_hydrogens', '_calculate_aromatic_assignment'):
        monkeypatch.setattr(chemistry, name, forbidden)
    mol.atoms[0].atomic_number = 7
    mol.updata_atoms(atoms=mol.atoms[1:3], atomic_number=(8, 6), formal_charge=(0, 1))
    mol.bonds[0].bond_order = 2.
    assert [atom.atomic_number for atom in mol.atoms[:3]] == [7, 8, 6]
    assert mol.bonds[0].bond_order == 2.


@pytest.mark.parametrize('field', ('n', 's', 'p', 'd', 'f', 'g', 'valence', 'electron_configuration', 'is_aromatic'))
def test_scalar_derived_writes_fail_before_any_fact_changes(field):
    mol = _cycle()
    before = _state(mol)
    with pytest.raises(AttributeError):
        setattr(mol.atoms[0], field, 1)
    assert _state(mol) == before


@pytest.mark.parametrize('kwargs', (
    {'attrs_array': np.zeros(19)}, {'update_electron_config': False},
    {'valence': 4}, {'is_aromatic': True}, {'partial_charge': 0.}, {'s': 2},
))
def test_removed_constructor_inputs_leave_no_member_and_burn_the_uid(kwargs):
    mol = Molecule()
    with pytest.raises(TypeError):
        Atom(mol, **kwargs)
    assert not mol.atoms and len(mol._atom_table) == 0
    assert mol._next_atom_uid == 1


def test_known_hydrogen_group_materializes_once_and_retries_only_after_new_input(monkeypatch):
    atom = Atom(atomic_number=6)
    calls = []
    calculate = chemistry._calculate_implicit_hydrogens
    def observe(*args):
        calls.append(args[0])
        return calculate(*args)
    monkeypatch.setattr(chemistry, '_calculate_implicit_hydrogens', observe)
    assert chemistry._require_implicit_hydrogens(atom.mol) == (4,)
    assert chemistry._require_implicit_hydrogens(atom.mol) == (4,)
    assert calls == [6]
    atom.atomic_number = 8
    assert chemistry._require_implicit_hydrogens(atom.mol) == (2,)
    assert calls == [6, 8]
    assert atom.mol._atom_table._sources[0, FIELD_INDEX['implicit_hydrogens']] == ATOM_SOURCE_CODES['implicit_hydrogens']


def test_unresolved_private_h_state_does_not_repeat_and_new_element_retries(monkeypatch):
    atom = Atom()
    calls = []
    calculate = chemistry._calculate_implicit_hydrogens
    def observe(*args):
        calls.append(args[0])
        return calculate(*args)
    monkeypatch.setattr(chemistry, '_calculate_implicit_hydrogens', observe)
    chemistry._prepare_hydrogens(atom.mol)
    chemistry._prepare_hydrogens(atom.mol)
    assert calls == [0]
    assert atom.mol._atom_table._group_status[0, GROUP_INDEX['IMPLICIT_H']] == _FactStatus.UNRESOLVED
    assert not int(atom.mol._atom_table._presence[0]) & (1 << FIELD_INDEX['implicit_hydrogens'])
    atom.atomic_number = 6
    assert chemistry._require_implicit_hydrogens(atom.mol) == (4,)
    assert calls == [0, 6]


def test_isolated_hydrogen_private_state_retries_after_neighbour_is_added():
    mol = Molecule()
    first = mol.create_atom(atomic_number=1)
    chemistry._prepare_hydrogens(mol)
    assert chemistry._require_implicit_hydrogens(mol) == (0,)
    assert mol._atom_table._group_status[0, GROUP_INDEX['IMPLICIT_H']] == _FactStatus.CURRENT
    assert int(mol._atom_table._presence[0]) & (1 << FIELD_INDEX['implicit_hydrogens'])
    assert mol._atom_table._sources[0, FIELD_INDEX['implicit_hydrogens']] == ATOM_SOURCE_CODES['implicit_hydrogens']
    second = mol.create_atom(atomic_number=1)
    mol.add_bond(first, second, 1.)
    assert chemistry._require_implicit_hydrogens(mol) == (1, 1)


def test_explicit_zero_hydrogens_and_formal_charge_are_independent_declarations():
    atom = Atom(atomic_number=6, implicit_hydrogens=0, formal_charge=-1)
    atom.atomic_number = 7
    atom.add_atom(6)
    chemistry._prepare_hydrogens(atom.mol, force=True)
    assert chemistry._require_implicit_hydrogens(atom.mol, rows=(0,)) == (0,)
    assert atom.formal_charge == -1
    assert atom.mol._atom_table._sources[0, FIELD_INDEX['implicit_hydrogens']] == ATOM_SOURCE_CODES['explicit']


def test_formal_charge_coverage_tracks_appended_member_and_whole_molecule_charge():
    mol = Molecule()
    mol.create_atom(atomic_number=6, formal_charge=0)
    chemistry._prepare_formal_charge_coverage(mol)
    index = GROUP_INDEX['FORMAL_CHARGE_COVERAGE']
    assert mol._atom_table._group_status[0, index] == _FactStatus.CURRENT
    second = mol.create_atom(atomic_number=8)
    chemistry._prepare_formal_charge_coverage(mol)
    assert mol._atom_table._group_status[:2, index].tolist() == [_FactStatus.UNRESOLVED] * 2
    second.formal_charge = 0
    chemistry._prepare_formal_charge_coverage(mol)
    assert mol._atom_table._group_status[:2, index].tolist() == [_FactStatus.CURRENT] * 2
    mol.charge = 1
    chemistry._prepare_formal_charge_coverage(mol)
    assert mol._atom_table._group_status[:2, index].tolist() == [_FactStatus.CONFLICT] * 2
    assert [atom.formal_charge for atom in mol.atoms] == [0, 0]
    second.formal_charge = 1
    chemistry._prepare_formal_charge_coverage(mol)
    assert mol._atom_table._group_status[:2, index].tolist() == [_FactStatus.CURRENT] * 2


def test_bonding_input_and_assignment_are_immutable_and_do_not_change_source():
    mol = _cycle()
    before = _state(mol)
    snapshot = chemistry._bonding_snapshot(mol)
    proposal = _calculate_aromatic_assignment(snapshot, kekule=True)
    assert _state(mol) == before
    with pytest.raises(FrozenInstanceError):
        snapshot.rings = ()
    with pytest.raises(TypeError):
        snapshot.coordinates[0][0] = 9.
    with pytest.raises(FrozenInstanceError):
        proposal.bond_orders = ()
    assert proposal.atom_aromatic == (True,) * 6
    assert proposal.bond_aromatic == (True,) * 6
    assert proposal.bond_orders == (2., 1., 2., 1., 2., 1.)


def test_aromatic_assignment_rolls_back_atom_bond_and_dependency_changes_together():
    mol = _cycle()
    snapshot = chemistry._bonding_snapshot(mol)
    proposal = _calculate_aromatic_assignment(snapshot, kekule=True)
    before = _state(mol)
    bad = replace(proposal, bond_kinds=proposal.bond_kinds[:-1] + ('invalid',))
    with pytest.raises(ValueError):
        _commit_aromatic_assignment(mol, bad, snapshot=snapshot)
    assert _state(mol) == before


@pytest.mark.parametrize('change', ('geometry', 'membership', 'bond_order'))
def test_aromatic_proposal_rejects_changed_inputs_before_any_commit(change):
    mol = _cycle()
    snapshot = chemistry._bonding_snapshot(mol)
    proposal = _calculate_aromatic_assignment(snapshot, kekule=True)
    if change == 'geometry':
        mol.atoms[0].z = 1.
    elif change == 'membership':
        mol.create_atom(atomic_number=6)
    else:
        mol.bonds[0].bond_order = 2.
    before = _state(mol)
    with pytest.raises(RuntimeError, match='changed'):
        _commit_aromatic_assignment(mol, proposal, snapshot=snapshot)
    assert _state(mol) == before


def test_inferred_false_aromaticity_reconsiders_new_geometry():
    mol = _cycle()
    original = mol.coordinates.copy()
    mol.atoms[0].z = 2.
    mol.calc_atom_valence(assign_aromatic=True)
    assert not any(atom.is_aromatic for atom in mol.atoms)
    mol.coordinates = original
    assert all(atom.is_aromatic for atom in mol.atoms)
    assert all(bond.is_aromatic for bond in mol.bonds)


def test_breaking_imported_aromatic_cycle_retains_facts_and_marks_conflict():
    mol = supplied_aromatic_flags(_cycle(), (True,) * 6)
    assert all(atom.is_aromatic for atom in mol.atoms)
    mol.remove_bond(mol.bonds[0])
    raw = mol._atom_table.column('is_aromatic').copy()
    with pytest.raises(RuntimeError, match='conflict'):
        _ = mol.atoms[3].is_aromatic
    np.testing.assert_array_equal(mol._atom_table.column('is_aromatic'), raw)
    assert mol._atom_table._group_status[:6, GROUP_INDEX['AROMATIC_REPRESENTATION']].tolist() == [_FactStatus.CONFLICT] * 6
    mol.determine_rings_aromatic()
    assert not any(atom.is_aromatic for atom in mol.atoms)


def test_ring_membership_is_fixed_while_chemical_and_geometry_reads_reject_changed_source():
    mol = _cycle()
    ring = mol.rings[0]
    atoms, bonds = ring.atoms, tuple(ring.bonds)
    for field, value in (('_atoms', ()), ('_bonds', ()), ('_view_representation', ()), ('is_aromatic', True)):
        with pytest.raises(AttributeError):
            setattr(ring, field, value)
    assert not hasattr(ring, 'determine_aromatic') and not hasattr(ring, 'kekulize')
    mol.determine_rings_aromatic()
    with pytest.raises(RuntimeError, match='representation source changed'):
        _ = ring.is_aromatic
    assert ring.atoms == atoms and tuple(ring.bonds) == bonds
    current = mol.rings[0]
    mol.atoms[0].x += .25
    with pytest.raises(RuntimeError, match='geometry source changed'):
        _ = current.geometry_cycle
    assert current.atoms == atoms and tuple(current.bonds) == bonds


def test_joint_ring_preserves_first_member_zero_and_checks_live_neighbour_reads():
    mol = _cycle()
    mol.add_bond(0, 3, 1.)
    rings = mol.rings
    joint = rings[0].joint_ring(aromatic=False)
    first = joint.atoms[0]
    assert first in joint and joint.index(first) == 0
    assert isinstance(joint.rings, tuple) and isinstance(joint.atoms, tuple) and isinstance(joint.bonds, tuple)
    assert joint.atom_neigh_atom(first)
    with pytest.raises(AttributeError):
        joint._atoms = ()
    restored = pickle.loads(pickle.dumps(joint))
    assert restored.index(restored.atoms[0]) == 0
    assert restored.atom_neigh_bond(restored.atoms[0])
    mol.create_atom(atomic_number=6)
    assert first in joint and joint.index(first) == 0
    for call in (lambda: joint.atom_neigh_atom(first), lambda: joint.atom_neigh_bond(first),
                 lambda: joint.bond_neigh_bond(joint.bonds[0]), lambda: joint.index_bond(joint.bonds[0]),
                 lambda: joint.check_kekulize(), lambda: rings[0].joint_ring(aromatic=False)):
        with pytest.raises(RuntimeError, match='topology source changed'):
            call()


def test_fresh_lazy_working_copy_prepares_required_groups_before_source_capture_and_commit(monkeypatch):
    from hotpot.cheminfo.forcefields import working_copy
    mol = Molecule()
    mol.create_atom(atomic_number=6)
    hfield = FIELD_INDEX['implicit_hydrogens']
    assert not int(mol._atom_table._presence[0]) & (1 << hfield)
    capture = working_copy._capture_source
    observations = []
    def observe(owner, domains=()):
        if domains == working_copy._FF_SOURCE_DOMAINS:
            observations.append(owner._atom_table._group_status[0, 1:3].tolist())
            assert not any(owner._atom_table._dirty_seeds[1:3])
        return capture(owner, domains)
    monkeypatch.setattr(working_copy, '_capture_source', observe)
    source = working_copy._capture_working_source(mol)
    assert observations == [[_FactStatus.CURRENT, _FactStatus.CURRENT]]
    assert int(mol._atom_table._presence[0]) & (1 << hfield)
    assert mol._atom_table._columns[hfield][0] == 4
    working = working_copy._hydrogenated_working_copy(mol, add_hydrogens=False, source=source)
    working.coordinates = ((1., 2., 3.),)
    working_copy._commit_working_copy(mol, working, source=source)
    assert mol.coordinates.tolist() == [[1., 2., 3.]]
    assert chemistry._require_implicit_hydrogens(mol) == (4,)


def test_atomic_number_only_projection_does_not_prepare_chemistry(monkeypatch):
    from hotpot.cheminfo.AImodels.data_extract import extract_atom_attrs
    from hotpot.cheminfo._native.buffers import NativeMoleculeField, pack_molecule
    mol = _cycle()
    before = _state(mol)
    def forbidden(*args, **kwargs):
        raise AssertionError('field-selective factual export prepared chemistry')
    monkeypatch.setattr(chemistry, '_prepare_hydrogens', forbidden)
    monkeypatch.setattr(chemistry, '_prepare_aromatic_representation', forbidden)
    assert extract_atom_attrs(mol, atomic_number_only=True)['x'].tolist() == [6] * 6
    assert pack_molecule(mol, frozenset({NativeMoleculeField.ATOMIC_NUMBERS})).atomic_numbers.tolist() == [6] * 6
    assert _state(mol) == before


def test_strict_model_and_native_hydrogen_consumers_prepare_known_values_once():
    from hotpot.cheminfo.AImodels.data_extract import extract_atom_attrs
    from hotpot.cheminfo._native.buffers import NativeMoleculeField, pack_molecule
    mol = _cycle()
    mol.determine_rings_aromatic()
    features = extract_atom_attrs(mol)
    assert features['x'].shape == (6, 16)
    assert features['x_names'] == Atom._attrs_enumerator[:15] + ('is_metal',)
    assert features['x'][:, 1:7].tolist() == [[2., 2., 2., 0., 0., 0.]] * 6
    assert features['x'][:, 9].tolist() == [1.] * 6
    assert features['x'][:, 13:15].tolist() == [[4., 1.]] * 6
    buffers = pack_molecule(mol, frozenset({NativeMoleculeField.ATOM_TRANSPORT,
        NativeMoleculeField.ATOM_AROMATIC, NativeMoleculeField.BOND_AROMATIC}))
    assert buffers.atom_transport.current_implicit_hydrogens.tolist() == [1] * 6
    assert buffers.atom_aromatic.tolist() == [1] * 6
    assert buffers.bond_aromatic.tolist() == [1] * 6


def test_batch_hydrogen_addition_prepares_once_for_its_heavy_atom_selection(monkeypatch):
    mol = _cycle()
    calls = []
    prepare = chemistry._prepare_hydrogens
    def observe(owner, *, rows=None, force=False):
        calls.append(tuple(rows))
        return prepare(owner, rows=rows, force=force)
    monkeypatch.setattr(chemistry, '_prepare_hydrogens', observe)
    mol.add_hydrogens(rng=np.random.default_rng(5))
    assert calls == [tuple(range(6))]
    assert len(mol.hydrogens) == 12


def test_kekule_bond_aromatic_flags_survive_all_factual_rebuild_paths():
    mol = _cycle()
    mol.determine_rings_aromatic()
    assert all(bond.bond_kind.value in ('single', 'double') for bond in mol.bonds)
    record, metadata = mol._capture_factual_record()
    rebuilt = Molecule._build_molecule(record, bond_metadata=metadata)
    combined = Molecule()
    combined.add_component(mol)
    for candidate in (rebuilt, copy(mol), pickle.loads(pickle.dumps(mol)), combined, *mol.components):
        assert tuple(bond._is_aromatic for bond in candidate.bonds) == (True,) * 6
        assert tuple(bond.bond_order for bond in candidate.bonds) == (2., 1., 2., 1., 2., 1.)


def test_working_source_identity_covers_independent_bond_aromatic_flags():
    from hotpot.cheminfo.structure._commit import _commit_bond_facts
    from hotpot.cheminfo.structure._identity import _capture_source, _check_source
    mol = _cycle()
    mol.determine_rings_aromatic()
    source = _capture_source(mol, ('forcefield_inputs',))
    bond = mol.bonds[0]
    _commit_bond_facts(mol, bond, _capture_source(mol), raw_values=bond._attrs,
        kind=bond.bond_kind, direction=bond.bond_direction, source_namespace=bond.bond_source,
        metadata=bond.bond_source_metadata, aromatic=False)
    with pytest.raises(RuntimeError, match='forcefield_inputs'):
        _check_source(mol, source)


def test_known_numeric_hydrogen_projection_keeps_raw_float_domain_and_model_order():
    from hotpot.cheminfo.AImodels.data_extract import extract_atom_attrs
    mol = Molecule()
    values = np.zeros(19)
    values[0], values[14], values[15] = 6., 65537.25, 2. ** 70
    atom = mol._create_atom_from_array(values)
    projected = chemistry._atom_projection(mol, fields=('implicit_hydrogens', 'id', 'implicit_hydrogens'))
    np.testing.assert_array_equal(projected, [[65537.25, 2. ** 70, 65537.25]])
    assert atom.attrs[14] == 65537.25 and mol.atom_matrix[0, 14] == 65537.25
    assert extract_atom_attrs(mol)['x'][0, 14] == 65537.25


def test_scalar_representation_reads_coalesce_dirty_disconnected_components_once(monkeypatch):
    mol = Molecule()
    for _ in range(8):
        mol.add_component(_cycle())
    mol.calc_atom_valence(assign_aromatic=True)
    mol.updata_atoms(atomic_number=6)
    captures = []
    snapshot = chemistry._bonding_snapshot
    def observe(owner, **kwargs):
        captures.append(len(owner.atoms))
        return snapshot(owner, **kwargs)
    monkeypatch.setattr(chemistry, '_bonding_snapshot', observe)
    assert all(atom.is_aromatic for atom in mol.atoms)
    assert captures == [48]


def test_partial_hydrogen_requests_coalesce_pending_rows_and_bind_source_once(monkeypatch):
    mol = Molecule()
    for _ in range(32):
        mol.create_atom(atomic_number=6)
    calls, captures = [], []
    calculate, capture = chemistry._calculate_implicit_hydrogens, chemistry._capture_source
    def observe_rule(*args):
        calls.append(args[0])
        return calculate(*args)
    def observe_source(owner, domains=()):
        if domains:
            captures.append(domains)
        return capture(owner, domains)
    monkeypatch.setattr(chemistry, '_calculate_implicit_hydrogens', observe_rule)
    monkeypatch.setattr(chemistry, '_capture_source', observe_source)
    assert tuple(chemistry._require_implicit_hydrogens(mol, rows=(row,))[0] for row in range(32)) == (4,) * 32
    assert calls == [6] * 32
    assert captures == [('bonding_inputs',)]


def test_later_hydrogen_rule_source_change_rejects_the_complete_proposal(monkeypatch):
    mol = Molecule()
    for _ in range(2):
        mol.create_atom(atomic_number=6)
    before = _state(mol)
    calls = []
    calculate = chemistry._calculate_implicit_hydrogens
    def alter_earlier_input(*args):
        calls.append(args[0])
        if len(calls) == 2:
            mol.atoms[0].atomic_number = 8
        return calculate(*args)
    monkeypatch.setattr(chemistry, '_calculate_implicit_hydrogens', alter_earlier_input)
    with pytest.raises(RuntimeError, match='changed'):
        chemistry._prepare_hydrogens(mol)
    assert calls == [6, 6]
    assert _state(mol) == before


@pytest.mark.parametrize('inferred', (False, True))
def test_coalescing_preserves_default_false_scope_and_separates_conflicting_components(inferred):
    valid = _cycle()
    if inferred:
        valid.calc_atom_valence(assign_aromatic=True)
    invalid = Molecule()
    invalid.create_atom(atomic_number=6)
    invalid = supplied_aromatic_flags(invalid, (True,))
    mol = Molecule()
    mol.add_component(valid)
    mol.add_component(invalid)
    assert mol.atoms[0].is_aromatic is inferred
    with pytest.raises(RuntimeError, match='conflict'):
        _ = mol.atoms[6].is_aromatic
    assert mol._atom_table.column('is_aromatic')[:6].tolist() == [float(inferred)] * 6


def test_bonding_operations_preserve_opaque_annotations_without_hashing_them_as_inputs():
    mol = _cycle()
    marker = object()
    annotations = tuple(bond._bond_source_metadata for bond in mol.bonds)
    for metadata in annotations:
        metadata['opaque_annotation'] = marker
    mol.determine_rings_aromatic()
    assert chemistry._require_implicit_hydrogens(mol) == (1,) * 6
    assert all(bond._bond_source_metadata is metadata and metadata['opaque_annotation'] is marker
               for bond, metadata in zip(mol.bonds, annotations))


@pytest.mark.parametrize('writer', ('scalar', 'batch'))
def test_neighbour_element_change_refreshes_inferred_h_before_remote_first_read(writer, monkeypatch):
    mol = Molecule()
    for number in (7, 6, 6, 6):
        mol.create_atom(atomic_number=number)
    mol.add_bond(0, 1, 1.)
    mol.add_bond(2, 3, 1.)
    initial = tuple(chemistry._atom_implicit_proposal(atom) for atom in mol.atoms)
    assert chemistry._require_implicit_hydrogens(mol) == initial
    calls = []
    calculate = chemistry._calculate_implicit_hydrogens
    def observe(*args):
        calls.append((args[0], args[1]))
        return calculate(*args)
    monkeypatch.setattr(chemistry, '_calculate_implicit_hydrogens', observe)
    if writer == 'scalar':
        mol.atoms[1].atomic_number = 8
    else:
        mol.updata_atoms(atoms=(mol.atoms[1],), atomic_number=(8,))
    expected = tuple(chemistry._atom_implicit_proposal(atom) for atom in mol.atoms)
    calls.clear()
    assert chemistry._require_implicit_hydrogens(mol, rows=(0,)) == (expected[0],)
    assert chemistry._require_implicit_hydrogens(mol) == expected
    assert chemistry._require_implicit_hydrogens(mol, rows=(0, 2)) == (expected[0], expected[2])
    assert calls == [(7, (8,)), (8, (7,))]


@pytest.mark.parametrize('case', ('missing', 'extra_ligand_bond', 'wrong_component'))
def test_aromatic_assignment_rejects_incomplete_or_out_of_scope_bond_coverage(case):
    mol = _cycle()
    if case == 'extra_ligand_bond':
        zinc = mol.create_atom(atomic_number=30)
        ligand_bond = mol.add_bond(mol.atoms[0], zinc, 1.)
    elif case == 'wrong_component':
        mol.add_component(_cycle())
    snapshot = chemistry._bonding_snapshot(mol)
    proposal = _calculate_aromatic_assignment(snapshot, kekule=True)
    if case == 'missing':
        bad = replace(proposal, bond_uids=proposal.bond_uids[:-1],
            bond_aromatic=proposal.bond_aromatic[:-1], bond_orders=proposal.bond_orders[:-1],
            bond_kinds=proposal.bond_kinds[:-1])
    elif case == 'extra_ligand_bond':
        bad = replace(proposal, bond_uids=proposal.bond_uids + (ligand_bond._uid,),
            bond_aromatic=proposal.bond_aromatic + (True,),
            bond_orders=proposal.bond_orders + (2.,), bond_kinds=proposal.bond_kinds + ('double',))
    else:
        bad = replace(proposal, bond_uids=proposal.bond_uids[6:],
            bond_aromatic=proposal.bond_aromatic[6:], bond_orders=proposal.bond_orders[6:],
            bond_kinds=proposal.bond_kinds[6:], covered_rows=tuple(range(6)))
    before = _state(mol)
    with pytest.raises(ValueError, match='coverage'):
        _commit_aromatic_assignment(mol, bad, snapshot=snapshot)
    assert _state(mol) == before


@pytest.mark.parametrize('change', ('element', 'bond_order'))
def test_stale_ring_export_rejects_changed_representation_and_retains_members(change):
    mol = _cycle()
    ring = mol.rings[0]
    atoms, bonds = ring.atoms, tuple(ring.bonds)
    if change == 'element':
        mol.atoms[0].atomic_number = 7
    else:
        mol.bonds[0].bond_order = 2.
    with pytest.raises(RuntimeError, match='representation source changed'):
        _ = ring.is_aromatic
    before = _state(mol)
    with pytest.raises(RuntimeError, match='representation source changed'):
        ring.to_mol()
    assert _state(mol) == before
    assert ring.atoms == atoms and tuple(ring.bonds) == bonds


@pytest.mark.parametrize('writer', ('scalar', 'batch'))
@pytest.mark.parametrize('producer', ('explicit', 'openbabel'))
def test_neighbour_element_change_preserves_declared_h_and_source(writer, producer, monkeypatch):
    from hotpot.cheminfo.structure._commit import _commit_atom_facts
    from hotpot.cheminfo.structure._identity import _capture_source
    from hotpot.cheminfo.structure._mutation import _normalize_atom_fields
    mol = Molecule()
    for number in (7, 6, 6, 6):
        mol.create_atom(atomic_number=number)
    mol.add_bond(0, 1, 1.)
    mol.add_bond(2, 3, 1.)
    declared = mol.atoms[0]
    update = _normalize_atom_fields(_capture_source(mol), (0,), (declared._uid,),
        {'implicit_hydrogens': (0,)}, producer=ATOM_SOURCE_CODES[producer])
    _commit_atom_facts(mol, update)
    assert chemistry._require_implicit_hydrogens(mol) == (0, 3, 3, 3)
    calls = []
    calculate = chemistry._calculate_implicit_hydrogens
    def observe(*args):
        calls.append(args[0])
        return calculate(*args)
    monkeypatch.setattr(chemistry, '_calculate_implicit_hydrogens', observe)
    if writer == 'scalar':
        mol.atoms[1].atomic_number = 8
    else:
        mol.updata_atoms(atoms=(mol.atoms[1],), atomic_number=(8,))
    assert chemistry._require_implicit_hydrogens(mol, rows=(0,)) == (0,)
    assert chemistry._require_implicit_hydrogens(mol) == (0, 1, 3, 3)
    assert calls == [8]
    assert mol._atom_table._sources[0, FIELD_INDEX['implicit_hydrogens']] == ATOM_SOURCE_CODES[producer]


def test_unresolved_h_retries_once_when_neighbour_element_changes(monkeypatch):
    mol = Molecule()
    mol.create_atom()
    mol.create_atom(atomic_number=6)
    mol.add_bond(0, 1, 1.)
    chemistry._prepare_hydrogens(mol)
    calls = []
    calculate = chemistry._calculate_implicit_hydrogens
    def observe(*args):
        calls.append((args[0], args[1]))
        return calculate(*args)
    monkeypatch.setattr(chemistry, '_calculate_implicit_hydrogens', observe)
    mol.atoms[1].atomic_number = 8
    chemistry._prepare_hydrogens(mol, rows=(0,))
    chemistry._prepare_hydrogens(mol, rows=(0,))
    assert calls == [(0, (8,)), (8, (0,))]
    assert mol._atom_table._group_status[0, GROUP_INDEX['IMPLICIT_H']] == _FactStatus.UNRESOLVED


@pytest.mark.parametrize('ligand_skeleton', (True, False))
def test_aromatic_assignment_covers_the_captured_bond_selection(ligand_skeleton):
    mol = _cycle()
    zinc = mol.create_atom(atomic_number=30)
    ligand_bond = mol.add_bond(mol.atoms[0], zinc, 1.)
    snapshot = chemistry._bonding_snapshot(mol, ligand_skeleton=ligand_skeleton)
    proposal = _calculate_aromatic_assignment(snapshot, kekule=True)
    assert (ligand_bond._uid not in proposal.bond_uids) is ligand_skeleton
    _commit_aromatic_assignment(mol, proposal, snapshot=snapshot)
    assert tuple(bond._is_aromatic for bond in mol.bonds) == (True,) * 6 + (False,)
    assert tuple(bond.bond_order for bond in mol.bonds) == (2., 1., 2., 1., 2., 1., 1.)
    assert mol._atom_table._group_status[:7, GROUP_INDEX['AROMATIC_REPRESENTATION']].tolist() == [_FactStatus.CURRENT] * 7


def test_aromatic_assignment_commits_only_its_covered_component():
    mol = _cycle()
    mol.add_component(_cycle())
    snapshot = chemistry._bonding_snapshot(mol)
    proposal = _calculate_aromatic_assignment(snapshot, kekule=True)
    partial = replace(proposal, bond_uids=proposal.bond_uids[:6],
        bond_aromatic=proposal.bond_aromatic[:6], bond_orders=proposal.bond_orders[:6],
        bond_kinds=proposal.bond_kinds[:6], covered_rows=tuple(range(6)))
    other_status = mol._atom_table._group_status[6:12].copy()
    _commit_aromatic_assignment(mol, partial, snapshot=snapshot)
    assert mol._atom_table.column('is_aromatic').tolist() == [1.] * 6 + [0.] * 6
    assert tuple(bond._is_aromatic for bond in mol.bonds) == (True,) * 6 + (False,) * 6
    np.testing.assert_array_equal(mol._atom_table._group_status[6:12], other_status)
    assert mol._atom_table._group_status[:6, GROUP_INDEX['AROMATIC_REPRESENTATION']].tolist() == [_FactStatus.CURRENT] * 6


@pytest.mark.parametrize('change', ('source', 'snapshot_membership', 'proposal_membership'))
def test_aromatic_assignment_matches_its_captured_source_and_atom_membership(change):
    mol = _cycle()
    snapshot = chemistry._bonding_snapshot(mol)
    proposal = _calculate_aromatic_assignment(snapshot, kekule=True)
    if change == 'source':
        proposal = replace(proposal, source=replace(snapshot.source))
        error, message = ValueError, 'source must match'
    elif change == 'snapshot_membership':
        snapshot = replace(snapshot, atom_uids=tuple(reversed(snapshot.atom_uids)))
        error, message = RuntimeError, 'membership changed'
    else:
        proposal = replace(proposal, atom_uids=tuple(reversed(proposal.atom_uids)))
        error, message = RuntimeError, 'membership changed'
    before = _state(mol)
    with pytest.raises(error, match=message):
        _commit_aromatic_assignment(mol, proposal, snapshot=snapshot)
    assert _state(mol) == before


def test_current_ring_export_and_topology_only_angle_export_keep_their_source_policies():
    mol = _cycle()
    ring = mol.rings[0]
    copied = ring.to_mol()
    assert len(copied.atoms) == len(copied.bonds) == 6
    assert tuple(atom.atomic_number for atom in copied.atoms) == (6,) * 6
    angle = mol.angles[0]
    angle.atoms[0].atomic_number = 7
    copied_angle = angle.to_mol()
    assert tuple(atom.atomic_number for atom in copied_angle.atoms) == tuple(atom.atomic_number for atom in angle.atoms)



def test_bond_order_perception_commits_the_captured_ligand_selection(monkeypatch):
    from hotpot.cheminfo import obconvert
    mol = _cycle()
    zinc = mol.create_atom(atomic_number=30, coordinates=(4., 0., 0.))
    ligand_bond = mol.add_bond(mol.atoms[0], zinc, 1.)
    ligand_values = ligand_bond._attrs.copy()
    observed = []
    perceive = obconvert._perceive_bond_orders
    def observe(owner, bonds):
        flags, entries = perceive(owner, bonds)
        observed.append((tuple(bond._uid for bond in bonds), flags, entries))
        return flags, entries
    monkeypatch.setattr(obconvert, '_perceive_bond_orders', observe)
    mol.assign_bond_order()
    assert len(observed) == 1
    selected, flags, entries = observed[0]
    assert selected == tuple(bond._uid for bond in mol.bonds[:6])
    assert mol._atom_table.column('is_aromatic').tolist() == list(flags)
    for ends, order, kind, direction, source, metadata, aromatic in entries:
        bond = mol.bond(*ends)
        assert (bond.bond_order, bond.bond_kind.value, bond.bond_direction,
                bond.bond_source, bond._is_aromatic) == (order, kind, direction, source, aromatic)
    np.testing.assert_array_equal(ligand_bond._attrs, ligand_values)
    assert not ligand_bond._is_aromatic


def test_assembly_flag_restore_commits_the_captured_ligand_selection():
    mol = _cycle()
    zinc = mol.create_atom(atomic_number=30)
    ligand_bond = mol.add_bond(mol.atoms[0], zinc, 1.)
    chemistry._restore_assembly_aromaticity(mol, tuple((atom, True) for atom in mol.atoms[:6]))
    assert mol._atom_table.column('is_aromatic').tolist() == [1.] * 6 + [0.]
    assert tuple(bond._is_aromatic for bond in mol.bonds) == (True,) * 6 + (False,)
    assert tuple(bond.bond_order for bond in mol.bonds) == (1.,) * 7
    assert not ligand_bond._is_aromatic
