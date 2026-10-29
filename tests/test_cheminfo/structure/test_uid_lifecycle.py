"""Private member identities stay distinct from rows and source labels."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import pickle

import numpy as np
import pytest

from hotpot import Atom, Bond, Molecule
from hotpot.cheminfo.rdconvert import from_rdmol, to_rdmol
from hotpot.cheminfo.stereo import get_stereo_payload
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


def test_direct_hidden_and_registered_bonds_share_one_non_reusing_allocator():
    mol = Molecule()
    for number in (30, 7, 6, 8):
        mol.create_atom(atomic_number=number)
    direct = Bond(mol.atoms[0], mol.atoms[1], bond_order=1)
    registered = mol.add_bond(1, 2, bond_order=1)
    assert registered._uid == direct._uid + 1
    assert mol._next_bond_uid == registered._uid + 1
    mol.hide_bonds(registered, clear_conformers=False)
    assert registered._uid_owner is mol._owner_token
    hidden_uid = registered._uid
    mol.recover_hided_covalent_bonds(clear_conformers=False)
    assert registered._uid == hidden_uid
    mol.remove_bond(registered)
    added = mol.add_bond(2, 3, bond_order=1)
    assert added._uid > hidden_uid
    with pytest.raises(AttributeError):
        added._uid = direct._uid


def test_foreign_insertion_copies_raw_facts_without_changing_original_owner():
    values = np.arange(19, dtype=np.float64) + 0.25
    values[0] = 6
    source = Atom(attrs_array=values, update_electron_config=False)
    source.isotope = 13
    source.frac_valence = 3.75
    original_mol, original_uid = source.mol, source._uid
    target = Molecule()
    target.create_atom(atomic_number=8)
    added = target.add_atom(source)
    assert added is not source
    assert source.mol is original_mol and source._uid == original_uid
    assert original_mol.atoms == [source]
    assert added.mol is target and added._uid_owner is target._owner_token
    assert added._uid == 1
    np.testing.assert_array_equal(added.attrs, values)
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
    source.hide_bonds(hidden, clear_conformers=False)
    clone = source.copy()
    assert clone._owner_token is not source._owner_token
    assert all(atom._uid_owner is clone._owner_token for atom in clone.atoms)
    assert all(bond._uid_owner is clone._owner_token
               for bond in clone._bonds + clone._hided_bonds)
    for component in source.components:
        assert component._owner_token is not source._owner_token
        assert all(atom._uid_owner is component._owner_token for atom in component.atoms)
    target = Molecule()
    target.create_atom(atomic_number=1)
    target.add_component(source)
    atoms = tuple(target.atoms)
    bonds = tuple(target._bonds + target._hided_bonds)
    assert len({atom._uid for atom in atoms}) == len(atoms)
    assert len({bond._uid for bond in bonds}) == len(bonds)
    assert all(atom._uid_owner is target._owner_token for atom in atoms)
    assert all(bond._uid_owner is target._owner_token for bond in bonds)
    assert all(bond.atom1 in atoms and bond.atom2 in atoms for bond in bonds)
    assert all(atom.mol is source for atom in source.atoms)


def test_factual_copy_and_component_preserve_supplied_electrons_and_sparse_inputs():
    raw = np.arange(19, dtype=np.float64) + 0.25
    raw[0] = 6
    source = Atom(attrs_array=raw, update_electron_config=False)
    source.isotope, source.frac_valence = 13, 3.75
    clone = source.mol.copy()
    component = source.mol.components[0]
    for copied in (clone.atoms[0], component.atoms[0]):
        np.testing.assert_array_equal(copied.attrs, raw)
        assert (copied.isotope, copied.frac_valence) == (13, 3.75)
        assert not np.shares_memory(copied.attrs, source.attrs)


def test_replacement_remaps_hidden_endpoints_without_changing_bond_uids_or_direction():
    mol = Molecule()
    for number in (30, 7, 6):
        mol.create_atom(atomic_number=number)
    old = mol.atoms[1]
    metal = mol.add_bond(0, 1, bond_direction='atom1_to_atom2')
    covalent = mol.add_bond(1, 2)
    mol.hide_bonds(metal, covalent, clear_conformers=False)
    uids = (metal._uid, covalent._uid)
    assert mol.replace_atom(1, 8) is old
    new = mol.atoms[1]
    assert metal.atom1 is mol.atoms[0] and metal.atom2 is new
    assert covalent.atom1 is new and covalent.atom2 is mol.atoms[2]
    assert (metal._uid, covalent._uid) == uids
    assert metal.bond_direction == 'atom1_to_atom2'
    mol.recover_hided_metal_ligand_bonds(clear_conformers=False)
    mol.recover_hided_covalent_bonds(clear_conformers=False)
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
    with pytest.raises(ValueError, match="length"):
        Atom(mol, attrs_array=np.zeros(2))
    # Legacy failed-constructor registration is characterized here; the table
    # builder cutover must separately make construction failure atomic.
    failed_uid = mol._atoms[-1]._uid
    atom = Atom(mol, atomic_number=8)
    assert atom._uid > failed_uid


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
