import pytest

from hotpot.cheminfo.core import BondKind, Molecule, ObjNotInMolecule


def _complex_with_a_ligand_ring():
    molecule = Molecule()
    for atomic_number in (30, 7, 6, 6, 6):
        molecule.create_atom(atomic_number=atomic_number)

    metal_bond = molecule._add_bond(
        0,
        1,
        bond_order=1.0,
        constraint=True,
        id=17,
        bond_kind=BondKind.DATIVE,
        bond_direction="atom1_to_atom2",
        bond_source="test",
        bond_source_metadata={"label": "coordination"},
    )
    ring_bond = molecule._add_bond(1, 2, bond_order=1.0, id=23)
    molecule._add_bond(2, 3, bond_order=1.0)
    molecule._add_bond(3, 4, bond_order=1.0)
    molecule._add_bond(4, 1, bond_order=1.0)
    molecule._update_graph(clear_conformers=False)
    return molecule, metal_bond, ring_bond


def test_restore_bonds_reuses_objects_and_updates_topology_once(monkeypatch):
    molecule, metal_bond, ring_bond = _complex_with_a_ligand_ring()
    metal_attrs = metal_bond.attr_dict
    ring_attrs = ring_bond.attr_dict

    molecule.hide_bonds(metal_bond, ring_bond, clear_conformers=False)
    assert molecule.rings == []

    update_calls = []
    update_graph = molecule._update_graph

    def record_update(clear_conformers=True):
        update_calls.append(clear_conformers)
        update_graph(clear_conformers)

    monkeypatch.setattr(molecule, "_update_graph", record_update)
    molecule.restore_bonds(
        metal_bond,
        ring_bond,
        clear_conformers=False,
    )

    assert update_calls == [False]
    assert any(bond is metal_bond for bond in molecule.bonds)
    assert any(bond is ring_bond for bond in molecule.bonds)
    assert metal_bond.attr_dict == metal_attrs
    assert ring_bond.attr_dict == ring_attrs
    assert molecule.graph.edges[0, 1]["bond"] is metal_bond
    assert molecule.graph.edges[1, 2]["bond"] is ring_bond
    assert molecule.atoms[1] in molecule.atoms[0].neighbours
    assert molecule.atoms[2] in molecule.atoms[1].neighbours
    assert len(molecule.rings) == 1
    assert molecule._hided_metal_bonds == []
    assert molecule._hided_covalent_bonds == []


def test_restore_bonds_can_restore_one_hidden_metal_bond_at_a_time():
    molecule = Molecule()
    for atomic_number in (30, 7, 8):
        molecule.create_atom(atomic_number=atomic_number)
    first_bond = molecule._add_bond(0, 1, bond_order=1.0)
    second_bond = molecule._add_bond(0, 2, bond_order=1.0)
    molecule._update_graph(clear_conformers=False)

    molecule.hide_metal_ligand_bonds(clear_conformers=False)
    molecule.restore_bonds(first_bond, clear_conformers=False)

    assert tuple(molecule.bonds) == (first_bond,)
    assert molecule._hided_metal_bonds == [second_bond]
    assert set(molecule.graph.edges) == {(0, 1)}
    assert molecule.atoms[0].neighbours == [molecule.atoms[1]]

    molecule.recover_hided_metal_ligand_bonds(clear_conformers=False)

    assert tuple(molecule.bonds) == (first_bond, second_bond)
    assert molecule._hided_metal_bonds == []
    assert set(molecule.graph.edges) == {(0, 1), (0, 2)}


@pytest.mark.parametrize("invalid_selection", ["active", "duplicate"])
def test_restore_bonds_validates_the_whole_selection_before_mutating(
        invalid_selection,
        monkeypatch,
):
    molecule, metal_bond, ring_bond = _complex_with_a_ligand_ring()
    active_bond = molecule.bonds[-1]
    molecule.hide_bonds(metal_bond, ring_bond, clear_conformers=False)
    original_bonds = tuple(molecule.bonds)
    original_hidden_metal_bonds = tuple(molecule._hided_metal_bonds)
    original_hidden_covalent_bonds = tuple(molecule._hided_covalent_bonds)
    update_calls = []
    monkeypatch.setattr(
        molecule,
        "_update_graph",
        lambda clear_conformers=True: update_calls.append(clear_conformers),
    )

    if invalid_selection == "active":
        selected_bonds = (metal_bond, active_bond)
        expected_error = ObjNotInMolecule
    else:
        selected_bonds = (metal_bond, metal_bond)
        expected_error = ValueError

    with pytest.raises(expected_error):
        molecule.restore_bonds(*selected_bonds, clear_conformers=False)

    assert tuple(molecule.bonds) == original_bonds
    assert tuple(molecule._hided_metal_bonds) == original_hidden_metal_bonds
    assert tuple(molecule._hided_covalent_bonds) == original_hidden_covalent_bonds
    assert update_calls == []
