"""Private member identities stay distinct from rows and source labels."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import pickle

import numpy as np
import pytest

from hotpot import Atom, Bond, Molecule
from hotpot.cheminfo.structure._fields import ATOM_SOURCE_CODES, FACT_FIELDS, FIELD_INDEX
from hotpot.cheminfo.structure._results import _import_partial_charges
from hotpot.cheminfo.operations.conversion import from_rdmol, to_rdmol
from hotpot.cheminfo.stereo import (
    AtomTransportRecord, BondTransportRecord, HydrogenTransportState,
    StereoAnchor, StereoAnchorKind, StereoPayload, StereoRecord,
    get_stereo_payload, set_stereo_payload,
)
from rdkit import Chem


def test_atom_uids_survive_row_compaction_and_source_id_changes():
    mol = Molecule()
    first, removed, last = [mol.create_atom(atomic_number=6, id=81) for _ in range(3)]
    original = (first._uid, removed._uid, last._uid)
    mol.remove_atom(removed)
    mol.refresh_atom_id()
    added = mol.create_atom(atomic_number=8, id=81)
    assert (first._uid, removed._uid, last._uid) == original
    assert added._uid > max(original)
    assert (first.idx, last.idx, added.idx) == (0, 1, 2)
    assert removed.mol is mol
    assert removed.bonds == [] and removed.neighbours == []
    with pytest.raises(KeyError):
        _ = removed.idx
    with pytest.raises(AttributeError):
        first._uid = 500


@pytest.mark.parametrize("detached", [False, True])
@pytest.mark.parametrize("foreign", [False, True])
def test_atom_molecule_owner_is_read_only_without_changing_identity_or_topology(detached, foreign):
    mol = Molecule()
    atom = mol.create_atom(atomic_number=6)
    neighbour = mol.create_atom(atomic_number=8)
    mol.add_bond(atom, neighbour)
    if detached:
        mol.remove_atom(atom)
    target = Molecule() if foreign else mol
    original_uid, original_owner = atom._uid, atom._uid_owner
    original_atoms = tuple(mol.atoms)
    original_bonds = tuple(mol.bonds)
    original_topology = dict(mol._topology_index)
    target_atoms = tuple(target.atoms)
    target_topology = dict(target._topology_index)

    with pytest.raises(AttributeError):
        atom.mol = target

    assert atom.mol is atom.molecule is mol
    assert atom._uid == original_uid and atom._uid_owner is original_owner
    assert tuple(mol.atoms) == original_atoms
    assert tuple(mol.bonds) == original_bonds
    assert mol._topology_index == original_topology
    assert tuple(target.atoms) == target_atoms
    assert target._topology_index == target_topology


def test_direct_selected_and_registered_bonds_share_one_non_reusing_allocator():
    mol = Molecule()
    for number in (30, 7, 6, 8):
        mol.create_atom(atomic_number=number)
    direct = Bond(mol.atoms[0], mol.atoms[1], bond_order=1)
    registered = mol.add_bond(1, 2, bond_order=1)
    assert registered._uid == direct._uid + 1
    assert mol._next_bond_uid == registered._uid + 1
    from hotpot.cheminfo.structure._selection import _capture_topology_selection, _selected_bonds
    selection = _capture_topology_selection(mol, excluded_bonds=(registered,))
    assert _selected_bonds(mol, selection) == ()
    assert registered._uid_owner is mol._owner_token
    hidden_uid = registered._uid
    assert mol.bonds == [registered]
    assert registered._uid == hidden_uid
    mol.remove_bond(registered)
    added = mol.add_bond(2, 3, bond_order=1)
    assert added._uid > hidden_uid
    with pytest.raises(AttributeError):
        added._uid = direct._uid


def test_foreign_insertion_copies_raw_facts_without_changing_original_owner():
    values = np.arange(19, dtype=np.float64) + 0.25
    values[0] = 6
    values[9] = 0
    source = Molecule()._create_atom_from_array(values)
    source.isotope = 13
    source.frac_valence = 3.75
    original_mol, original_uid = source.mol, source._uid
    source_selection = original_mol._partial_charge_result
    target = Molecule()
    target.create_atom(atomic_number=8)
    added = target.add_atom(source)
    assert added is not source
    assert source.mol is original_mol and source._uid == original_uid
    assert original_mol.atoms == [source]
    assert added.mol is target and added._uid_owner is target._owner_token
    assert added._uid == 1
    expected = values.copy()
    expected[1:7] = (2, 2, 2, 0, 0, 0)
    expected[13] = 4
    expected[FIELD_INDEX['partial_charge']] = np.nan
    np.testing.assert_array_equal(added.attrs, expected)
    assert added.partial_charge is None and target._partial_charge_result is None
    assert original_mol._partial_charge_result is source_selection
    assert source_selection.payload.values == (values[FIELD_INDEX['partial_charge']],)
    assert source_selection.payload.provenance == (ATOM_SOURCE_CODES['legacy_array'],)
    assert source_selection.payload.method == 'supplied' and source_selection.invalidated_by
    with pytest.raises(RuntimeError, match='charge_representation'):
        _ = source.partial_charge
    indices = [FIELD_INDEX[field] for field in FACT_FIELDS]
    np.testing.assert_array_equal(added.attrs[indices], values[indices])
    assert not np.shares_memory(added.attrs, source.attrs)
    assert (added.isotope, added.frac_valence) == (13, 3.75)
    before = tuple(target.atoms)
    with pytest.raises(ValueError, match="already owned"):
        target.add_atom(added)
    assert tuple(target.atoms) == before


def test_replacement_allocates_local_identity_and_retains_detached_original():
    mol = Molecule()
    first = mol.create_atom(atomic_number=6)
    old = mol.create_atom(atomic_number=6)
    bond = mol.add_bond(first, old)
    foreign = Atom(atomic_number=8)
    foreign_owner = foreign.mol
    uid = old._uid
    assert mol.replace_atom(1, foreign) is old
    replacement = mol.atoms[1]
    assert replacement is not foreign and foreign.mol is foreign_owner
    assert replacement._uid > uid
    assert old._uid == uid and old.mol is mol
    assert old.bonds == [] and old.neighbours == []
    assert bond.atom1 is first and bond.atom2 is replacement
    assert len({atom._uid for atom in mol.atoms}) == 2


def test_copy_components_and_merge_bind_fresh_owner_and_destination_uids():
    source = Molecule()
    for number in (30, 7, 6, 8):
        source.create_atom(atomic_number=number)
    hidden = source.add_bond(0, 1)
    source.add_bond(2, 3)
    clone = source.copy()
    assert clone._owner_token is not source._owner_token
    assert all(atom._uid_owner is clone._owner_token for atom in clone.atoms)
    assert all(bond._uid_owner is clone._owner_token
               for bond in clone._bonds)
    for component in source.components:
        assert component._owner_token is not source._owner_token
        assert all(atom._uid_owner is component._owner_token for atom in component.atoms)
    target = Molecule()
    target.create_atom(atomic_number=1)
    target.add_component(source)
    atoms = tuple(target.atoms)
    bonds = tuple(target._bonds)
    assert len({atom._uid for atom in atoms}) == len(atoms)
    assert len({bond._uid for bond in bonds}) == len(bonds)
    assert all(atom._uid_owner is target._owner_token for atom in atoms)
    assert all(bond._uid_owner is target._owner_token for bond in bonds)
    assert all(bond.atom1 in atoms and bond.atom2 in atoms for bond in bonds)
    assert all(atom.mol is source for atom in source.atoms)


def test_factual_copy_and_component_preserve_supplied_electrons_and_sparse_inputs():
    """Keep the frozen regression name; current contracts are asserted below."""
    raw = np.arange(19, dtype=np.float64) + 0.25
    raw[0] = 6
    raw[9] = 0
    source = Molecule()._create_atom_from_array(raw)
    source.isotope, source.frac_valence = 13, 3.75
    original_selection = source.mol._partial_charge_result
    assert original_selection.invalidated_by == ('charge_representation',)
    clone = source.mol.copy()
    component = source.mol.components[0]
    expected = raw.copy()
    expected[1:7] = (2, 2, 2, 0, 0, 0)
    expected[13] = 4
    expected[FIELD_INDEX['partial_charge']] = np.nan
    for copied in (clone.atoms[0], component.atoms[0]):
        np.testing.assert_array_equal(copied.attrs, expected)
        indices = [FIELD_INDEX[field] for field in FACT_FIELDS]
        np.testing.assert_array_equal(copied.attrs[indices], raw[indices])
        assert (copied.isotope, copied.frac_valence) == (13, 3.75)
        assert not np.shares_memory(copied.attrs, source.attrs)
        selection = copied.mol._partial_charge_result
        assert selection.source.owner is copied.mol._owner_token
        assert selection.payload.atom_uids == (copied._uid,)
        assert selection.payload.values == (raw[FIELD_INDEX['partial_charge']],)
        assert selection.payload.method == 'supplied' and selection.payload.method_version is None
        assert selection.payload.provenance == (ATOM_SOURCE_CODES['legacy_array'],)
        assert selection.invalidated_by == original_selection.invalidated_by
        with pytest.raises(RuntimeError, match='charge_representation'):
            _ = copied.partial_charge


def test_replacement_remaps_complete_endpoints_without_changing_bond_uids_or_direction():
    mol = Molecule()
    for number in (30, 7, 6):
        mol.create_atom(atomic_number=number)
    old = mol.atoms[1]
    metal = mol.add_bond(0, 1, bond_direction='atom1_to_atom2')
    covalent = mol.add_bond(1, 2)
    uids = (metal._uid, covalent._uid)
    assert mol.replace_atom(1, 8) is old
    new = mol.atoms[1]
    assert metal.atom1 is mol.atoms[0] and metal.atom2 is new
    assert covalent.atom1 is new and covalent.atom2 is mol.atoms[2]
    assert (metal._uid, covalent._uid) == uids
    assert metal.bond_direction == 'atom1_to_atom2'
    assert set(bond._uid for bond in mol.bonds) == set(uids)
    assert old.bonds == [] and old.neighbours == []


def test_foreign_insertion_remaps_atom_transport_without_losing_source_hydrogens():
    source = from_rdmol(Chem.MolFromSmiles('[13CH4]'), Molecule())
    target = Molecule()
    target.create_atom(atomic_number=8)
    imported = target.add_atom(source.atoms[0])
    original_fact = get_stereo_payload(source).atom_facts[0]
    fact = get_stereo_payload(target).atom_facts[0]
    assert fact.atom == imported.idx == 1
    assert fact.isotope == original_fact.isotope == 13
    assert fact.radical_electrons == original_fact.radical_electrons
    assert fact.current_hydrogens == original_fact.current_hydrogens
    assert fact.source_hydrogens == original_fact.source_hydrogens
    assert fact.source_namespace == original_fact.source_namespace
    assert to_rdmol(target).GetAtomWithIdx(1).GetIsotope() == 13
    assert to_rdmol(source).GetAtomWithIdx(0).GetIsotope() == 13


def test_replacement_installs_new_atom_transport_and_removes_old_source_facts():
    target = from_rdmol(Chem.MolFromSmiles('[12CH4]'), Molecule())
    source = from_rdmol(Chem.MolFromSmiles('[13CH4]'), Molecule())
    target.replace_atom(-1, source.atoms[0])
    assert to_rdmol(target).GetAtomWithIdx(0).GetIsotope() == 13
    assert len(get_stereo_payload(target).atom_facts) == 1
    target.replace_atom(0, 6)
    assert get_stereo_payload(target).atom_facts == ()
    assert to_rdmol(target).GetAtomWithIdx(0).GetIsotope() == 0


def test_failed_construction_does_not_make_allocated_identity_reusable():
    mol = Molecule()
    with pytest.raises(ValueError, match="[Cc]onflict"):
        Atom(mol, symbol="C", atomic_number=8)
    # A failed constructor now leaves no partial member and still consumes UID.
    assert mol.atoms == [] and mol._topology_index == {}
    failed_high_water = mol._next_atom_uid
    assert failed_high_water > 0
    atom = Atom(mol, atomic_number=8)
    assert atom._uid >= failed_high_water


def test_allocators_serialize_competing_requests_without_an_extra_member_index():
    mol = Molecule()
    with ThreadPoolExecutor(max_workers=8) as executor:
        allocated = list(executor.map(lambda _: mol._allocate_atom_uid(), range(1000)))
    assert sorted(allocated) == list(range(1000))
    assert mol._next_atom_uid == 1000 and mol._topology_index == {}
    assert Atom(mol)._uid == 1000


def test_pickle_recreates_lock_and_preserves_uid_owner_graph_and_high_water():
    mol = Molecule()
    first = mol.create_atom(atomic_number=6)
    second = mol.create_atom(atomic_number=8)
    mol.add_bond(first, second)
    removed = mol.create_atom(atomic_number=1)
    mol.remove_atom(removed)
    restored = pickle.loads(pickle.dumps(mol))
    assert restored._owner_token is not mol._owner_token
    assert restored._mutation_lock is not mol._mutation_lock
    assert [atom._uid for atom in restored.atoms] == [first._uid, second._uid]
    assert all(atom._uid_owner is restored._owner_token for atom in restored.atoms)
    assert restored.bonds[0]._uid_owner is restored._owner_token
    assert [atom.idx for atom in restored.atoms] == [0, 1]
    assert restored.atoms[0].neighbours == [restored.atoms[1]]
    assert restored.create_atom(atomic_number=7)._uid > removed._uid


@pytest.mark.parametrize("transport", [deepcopy, pytest.param(
    lambda value: pickle.loads(pickle.dumps(value)), id="pickle"
)])
@pytest.mark.parametrize("root", ["atom", "bond", "molecule"])
def test_current_transport_preserves_read_only_atom_owner_graph(transport, root):
    mol = Molecule()
    atom = mol.create_atom(atomic_number=6)
    neighbour = mol.create_atom(atomic_number=8)
    bond = mol.add_bond(atom, neighbour)
    restored = transport({"atom": atom, "bond": bond, "molecule": mol}[root])
    restored_mol = restored if root == "molecule" else restored.mol
    restored_atom, restored_neighbour = restored_mol.atoms
    restored_bond = restored_mol.bonds[0]

    assert restored_mol is not mol
    assert restored_mol._owner_token is not mol._owner_token
    assert restored_mol._mutation_lock is not mol._mutation_lock
    assert (restored_atom._uid, restored_neighbour._uid) == (atom._uid, neighbour._uid)
    assert restored_bond._uid == bond._uid
    assert all(member._uid_owner is restored_mol._owner_token
               for member in (restored_atom, restored_neighbour, restored_bond))
    assert restored_atom.mol is restored_neighbour.mol is restored_mol
    assert restored_bond.atom1 is restored_atom and restored_bond.atom2 is restored_neighbour
    assert restored_atom.neighbours == [restored_neighbour]
    assert (restored_atom.idx, restored_neighbour.idx) == (0, 1)
    with pytest.raises(AttributeError):
        restored_atom.mol = mol


@pytest.mark.parametrize('protocol', range(pickle.HIGHEST_PROTOCOL + 1))
def test_current_pickle_roots_round_trip_in_a_fresh_process(tmp_path, protocol):
    import os
    from pathlib import Path
    import subprocess
    import sys

    mol = Molecule()
    removed = mol.create_atom(atomic_number=1)
    mol.remove_atom(removed)
    first = mol.create_atom(atomic_number=6, id=43)
    second = mol.create_atom(atomic_number=8, id=87)
    bond = mol.add_bond(second, first, bond_order=1., constraint=True, id=91,
                        bond_source='fresh-process', bond_source_metadata={'raw': [1, b'2']})
    source_h = HydrogenTransportState(4, None, False)
    payload = StereoPayload(47, (), (
        AtomTransportRecord(0, 13, 0, None, source_h, 'first'),
        AtomTransportRecord(1, None, None, None, None, None),
        AtomTransportRecord(0, 13, 0, HydrogenTransportState(None, 0, False), source_h, ''),
        AtomTransportRecord(81, -(2**130), 2**140, None, None, 'unbound'),
    ), (BondTransportRecord((1, 0), 'raw display'),))
    set_stereo_payload(mol, payload)
    mol.conformer_add(mol.coordinates, energy=-2.)
    mol._conformers_index = -1
    mol.properties['owner'] = mol
    mol.properties['members'] = (first, bond)
    _import_partial_charges(mol, (0.0, None), producer=ATOM_SOURCE_CODES['explicit'])
    record, metadata = mol._capture_factual_record()
    artifact = tmp_path / 'current.pkl'
    artifact.write_bytes(pickle.dumps((mol, first, bond, record, metadata), protocol=protocol))
    origin = Path(sys.modules[Molecule.__module__].__file__).resolve()
    environment = os.environ.copy()
    environment.pop('PYTHONHOME', None)
    environment['PYTHONPATH'] = os.pathsep.join((str(origin.parents[2]),
                                               str(Path(__file__).resolve().parents[3])))
    script = '''
import pickle, sys
from dataclasses import fields
from pathlib import Path
from hotpot import Molecule
from hotpot.cheminfo.structure._builder import MoleculeRecord
from hotpot.cheminfo.stereo import get_stereo_payload
from tests.test_cheminfo._stage5_fixtures import immutable_snapshot
assert Path(sys.modules[Molecule.__module__].__file__).resolve() == Path(sys.argv[2])
mol, atom, bond, record, metadata = pickle.loads(Path(sys.argv[1]).read_bytes())
assert type(record) is MoleculeRecord and record.__reduce__()[1][1] == 5
assert atom is mol.atoms[0] and bond is mol.bonds[0]
assert bond.atom1 is mol.atoms[1] and bond.atom2 is atom
assert atom._uid == 1 and mol.atoms[1]._uid == 2 and bond._uid == 0
assert mol._next_atom_uid == 3 and mol._next_bond_uid == 1
assert mol.properties['owner'] is mol
assert mol.properties['members'][0] is atom and mol.properties['members'][1] is bond
assert bond.constraint and bond.id == 91
assert bond.bond_source_metadata == {'raw': [1, b'2']}
assert mol.conformers_number == 1 and mol._conformers_index == -1
captured, captured_metadata = mol._capture_factual_record()
assert len(fields(record)) == 24
assert get_stereo_payload(mol) == record.stereo_payload
assert mol._stereo_source[0][:3] == ((1, 0, record.stereo_payload.atom_facts[0].source_hydrogens),
                                     (2, 0, None),
                                     (1, 1, record.stereo_payload.atom_facts[2].source_hydrogens))
assert mol._stereo_source[0][3] == record.stereo_payload.atom_facts[3]
assert bond._display_facts == ((2, 1, 'raw display'),)
assert '_stereo_payload' not in mol.__dict__
assert all('isotope' not in member.__dict__ for member in mol.atoms)
assert atom.isotope == 13 and mol.atoms[1].isotope is None
payload, marker = record.partial_charge_result
assert payload.atom_uids == (0, 1) and payload.values == (0.0, None)
assert payload.method == 'supplied' and payload.method_version is None
assert payload.parameters == () and payload.units == 'e' and payload.provenance == (1, 1)
assert marker == ()
selection = mol._partial_charge_result
assert selection.source.owner is mol._owner_token
assert selection.payload.atom_uids == (1, 2) and selection.payload.values == payload.values
assert selection.payload.provenance == payload.provenance and selection.invalidated_by == ()
assert atom.partial_charge == 0.0 and mol.atoms[1].partial_charge is None
assert mol._atom_table._columns[8] is None
assert not (mol._atom_table.presence() & (1 << 8)).any()
for field in fields(record):
    assert immutable_snapshot(getattr(captured, field.name)) == immutable_snapshot(getattr(record, field.name)), field.name
assert immutable_snapshot(captured_metadata) == immutable_snapshot(metadata)
assert all(member._uid_owner is mol._owner_token for member in (*mol.atoms, *mol.bonds))
print('current roots restored')
'''
    result = subprocess.run([sys.executable, '-B', '-c', script, str(artifact), str(origin)],
                            cwd=tmp_path, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert result.stdout.strip() == 'current roots restored'


def test_whole_transport_preserves_opaque_unbound_facts_and_selection_filters_them():
    mol = Molecule()
    for number in (6, 8, 7):
        mol.create_atom(atomic_number=number)
    mol.add_bond(0, 1)
    payload = StereoPayload(47, (StereoRecord(
        StereoAnchor(StereoAnchorKind.ATOM, (81,)), (), 'opaque', None, '', ''),), (
        AtomTransportRecord(1, 2**130, -(2**140), HydrogenTransportState(None, 0, False),
                            HydrogenTransportState(2**150, None, None), ''),
        AtomTransportRecord(81, None, None, None, None, None),
        AtomTransportRecord(1, 2**130, None, None, None, None),
    ), (BondTransportRecord((1, 0), ''), BondTransportRecord((0, 2), 'missing edge')))
    set_stereo_payload(mol, payload)
    record, _ = mol._capture_factual_record()
    empty = Molecule()
    empty.add_component(mol)
    for transported in (Molecule._build_molecule(record), mol.copy(), empty):
        assert get_stereo_payload(transported) == payload
        assert '_stereo_payload' not in transported.__dict__
        assert all('isotope' not in atom.__dict__ for atom in transported.atoms)
        assert transported.atoms[1].isotope == 2**130
    reordered, _ = mol._capture_factual_record(rows=(1, 0, 2))
    assert reordered.stereo_payload.records == ()
    assert reordered.stereo_payload.atom_facts == tuple(
        replace(fact, atom=0) for fact in (payload.atom_facts[0], payload.atom_facts[2]))
    assert reordered.stereo_payload.bond_facts == (BondTransportRecord((0, 1), ''),)
    component = next(value for value in mol.components if len(value.atoms) == 2)
    selected = get_stereo_payload(component)
    assert selected.records == ()
    assert selected.atom_facts == (payload.atom_facts[0], payload.atom_facts[2])
    assert selected.bond_facts == (payload.bond_facts[0],)


def test_detached_atom_transfer_preserves_current_occurrences_without_source_history():
    source = Molecule()
    atom = source.create_atom(atomic_number=6)
    source.create_atom(atomic_number=8)
    current = HydrogenTransportState(-(2**90), 0, False)
    history = HydrogenTransportState(4, None, True)
    payload = StereoPayload(1, (), (
        AtomTransportRecord(0, 13, -(2**100), current, history, ''),
        AtomTransportRecord(0, 13, None, None, history, None),
    ), ())
    set_stereo_payload(source, payload)
    source.remove_atom(atom)
    assert atom.isotope == 13
    assert atom._detached_table._stereo_facts[0] == (
        (13, -(2**100), True, -(2**90), 0, False, ''),
        (13, None, False, None, None, None, None),
    )
    target = Molecule()
    target.create_atom(atomic_number=7)
    added = target.add_atom(atom)
    assert added.isotope == 13 and 'isotope' not in added.__dict__
    assert get_stereo_payload(target).atom_facts == tuple(
        replace(fact, atom=1, source_hydrogens=None) for fact in payload.atom_facts)


def _molecule_with_bond_displays():
    mol = Molecule()
    for number in (6, 7, 8):
        mol.create_atom(atomic_number=number)
    mol.add_bond(0, 1)
    mol.add_bond(1, 2)
    payload = StereoPayload(47, (), (), (
        BondTransportRecord((1, 0), 'ENDUPRIGHT'),
        BondTransportRecord((0, 1), ''),
        BondTransportRecord((1, 0), None),
        BondTransportRecord((2, 1), 'unknown display'),
    ))
    set_stereo_payload(mol, payload)
    return mol, payload


@pytest.mark.parametrize('removed_rows', [(0,), (0, 1)])
def test_removed_bonds_retain_exact_detached_display_owners(removed_rows):
    mol, payload = _molecule_with_bond_displays()
    bonds = tuple(mol.bonds)
    displays = tuple(bond._display_facts for bond in bonds)
    removed = tuple(bonds[row] for row in removed_rows)
    mol.remove_bonds(removed)
    assert tuple(mol.bonds) == tuple(bond for row, bond in enumerate(bonds)
                                     if row not in removed_rows)
    assert get_stereo_payload(mol) == replace(
        payload, bond_facts=payload.bond_facts[3:] if removed_rows == (0,) else ())
    for row in removed_rows:
        assert bonds[row]._display_facts is displays[row]
        assert bonds[row]._display_facts == (
            ((1, 0, 'ENDUPRIGHT'), (0, 1, ''), (1, 0, None)) if row == 0
            else ((2, 1, 'unknown display'),))
        assert bonds[row]._constraint_values is not mol._constraints.bond_values
        assert bonds[row].mol is mol


@pytest.mark.parametrize('removed_rows', [(0,), (0, 1)])
def test_removed_bond_display_owners_and_transport_survive_outer_rollback(removed_rows):
    from hotpot.cheminfo.structure._commit import _FactTransaction

    mol, payload = _molecule_with_bond_displays()
    bonds = tuple(mol.bonds)
    displays = tuple(bond._display_facts for bond in bonds)
    source = mol._stereo_source
    index = mol._topology_index
    constraint_values = mol._constraints.bond_values
    with pytest.raises(RuntimeError, match='abort removed displays'):
        with _FactTransaction(mol, producer=ATOM_SOURCE_CODES['explicit']):
            mol.remove_bonds(tuple(bonds[row] for row in removed_rows))
            assert get_stereo_payload(mol) == replace(
                payload, bond_facts=payload.bond_facts[3:] if removed_rows == (0,) else ())
            assert all(bonds[row]._display_facts is displays[row] for row in removed_rows)
            raise RuntimeError('abort removed displays')
    assert tuple(mol.bonds) == bonds
    assert mol._topology_index is index
    assert mol._stereo_source is source
    assert get_stereo_payload(mol) == payload
    assert all(bond._display_facts is display for bond, display in zip(bonds, displays))
    assert mol._constraints.bond_values is constraint_values
    assert all(bond._constraint_values is constraint_values for bond in bonds)
