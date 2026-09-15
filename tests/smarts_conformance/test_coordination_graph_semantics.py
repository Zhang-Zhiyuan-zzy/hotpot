"""L2 coordination tests on manually constructed Hotpot graphs.

These cases deliberately bypass ``read_mol`` so that graph-query semantics are
tested independently from Open Babel's target perception.
"""

import pytest

from .coordination_cases import (
    matching_atom_sets,
    matching_indices,
    metal_star,
    tertiary_amine,
)


pytestmark = pytest.mark.smarts_core


@pytest.mark.parametrize(
    ("coordinate_order", "degree", "connectivity", "valence"),
    [
        (None, 3, 3, 3),
        (0.0, 4, 4, 3),
        (1.0, 4, 4, 4),
    ],
)
def test_d_x_v_are_independent_graph_predicates(
    coordinate_order, degree, connectivity, valence
):
    molecule = tertiary_amine(coordinate_order)
    donor = molecule.atoms[0]

    assert len(donor.neighbours) == degree
    assert donor.implicit_hydrogens == 0
    assert donor.sum_bond_orders == valence
    assert matching_indices(molecule, f"[N;D{degree};X{connectivity};v{valence}]") == {
        0
    }


def test_zero_order_coordination_is_an_edge_but_not_a_single_bond():
    molecule = tertiary_amine(0.0)
    coordinate_bond = molecule.bond(0, 4)

    assert coordinate_bond.is_metal_ligand_bond
    assert coordinate_bond.bond_order == 0.0
    assert matching_atom_sets(molecule, "[N]-[M]") == set()
    assert matching_atom_sets(molecule, "[N]~[M]") == {frozenset((0, 4))}


@pytest.mark.parametrize("connectivity", range(1, 9))
def test_metal_d_x_v_connectivity_truth_table(connectivity):
    molecule = metal_star(connectivity)
    metal = molecule.atoms[0]
    near_value = connectivity + 1 if connectivity < 8 else connectivity - 1

    assert len(metal.neighbours) == connectivity
    assert metal.implicit_hydrogens == 0
    assert metal.sum_bond_orders == connectivity
    assert matching_indices(
        molecule,
        f"[M;D{connectivity};X{connectivity};v{connectivity}]",
    ) == {0}
    assert matching_indices(molecule, f"[M;X{near_value}]") == set()
    assert all(bond.is_metal_ligand_bond for bond in molecule.bonds)


def test_zero_order_metal_star_separates_x_from_v():
    molecule = metal_star(6, bond_order=0.0)
    metal = molecule.atoms[0]

    assert len(metal.neighbours) == 6
    assert metal.sum_bond_orders == 0
    assert matching_indices(molecule, "[M;D6;X6;v0]") == {0}
    assert len(matching_atom_sets(molecule, "[M]~[N]")) == 6
    assert matching_atom_sets(molecule, "[M]-[N]") == set()
