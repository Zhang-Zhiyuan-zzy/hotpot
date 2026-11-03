from copy import copy

from hotpot import read_mol
from hotpot.cheminfo.structure._selection import _capture_topology_selection, _selected_bonds


def _bond_indices(molecule):
    return tuple((bond.a1idx, bond.a2idx) for bond in molecule.bonds)


def test_components_keep_atoms_in_index_order():
    molecule = read_mol("CC.N.O", "smi")
    molecule.refresh_atom_id()

    component_ids = [tuple(atom.id for atom in part.atoms) for part in molecule.components]

    assert component_ids == [(0, 1), (2,), (3,)]


def test_ligand_selection_preserves_complete_order_and_is_repeatable():
    molecule = read_mol("[Zn](N)(N)", "smi")
    first = copy(molecule)
    second = copy(molecule)

    for candidate in (first, second):
        original = _bond_indices(candidate)
        first_selection = _capture_topology_selection(candidate, ligand_skeleton=True)
        second_selection = _capture_topology_selection(candidate, ligand_skeleton=True)
        assert first_selection.bond_mask == second_selection.bond_mask == bytes(2)
        assert _selected_bonds(candidate, first_selection) == ()
        assert _bond_indices(candidate) == original

    assert _bond_indices(first) == _bond_indices(second)
