from copy import copy

from hotpot import read_mol


def _bond_indices(molecule):
    return tuple((bond.a1idx, bond.a2idx) for bond in molecule.bonds)


def test_components_keep_atoms_in_index_order():
    molecule = read_mol("CC.N.O", "smi")
    molecule.refresh_atom_id()

    component_ids = [tuple(atom.id for atom in part.atoms) for part in molecule.components]

    assert component_ids == [(0, 1), (2,), (3,)]


def test_hidden_bond_recovery_is_order_preserving_and_repeatable():
    molecule = read_mol("[Zn](N)(N)", "smi")
    first = copy(molecule)
    second = copy(molecule)

    for candidate in (first, second):
        original = _bond_indices(candidate)
        candidate.hide_metal_ligand_bonds(clear_conformers=False)
        candidate.recover_hided_metal_ligand_bonds(clear_conformers=False)
        assert _bond_indices(candidate) == original

    assert _bond_indices(first) == _bond_indices(second)
