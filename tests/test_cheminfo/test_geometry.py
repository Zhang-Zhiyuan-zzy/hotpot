import numpy as np
import pytest

from hotpot.cheminfo import geometry as geo
from hotpot.cheminfo.core import Molecule


def _molecule(coordinates, bonds=(), atomic_numbers=None):
    molecule = Molecule()
    if atomic_numbers is None:
        atomic_numbers = [6] * len(coordinates)
    for atomic_number, position in zip(atomic_numbers, coordinates):
        molecule.create_atom(
            atomic_number=atomic_number,
            coordinates=position,
        )
    for first, second in bonds:
        molecule.add_bond(first, second, bond_order=1.0)
    molecule.refresh_atom_id()
    return molecule


def _square_with_probe(probe_x=0.0):
    return _molecule(
        (
            (-1.0, -1.0, 0.0),
            (1.0, -1.0, 0.0),
            (1.0, 1.0, 0.0),
            (-1.0, 1.0, 0.0),
            (probe_x, 0.0, -1.0),
            (probe_x, 0.0, 1.0),
        ),
        ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5)),
    )


def test_cycle_planes_include_closing_edge_and_accept_square():
    cycle = geo.CyclePlanes(
        (-1.0, -1.0, 0.0),
        (1.0, -1.0, 0.0),
        (1.0, 1.0, 0.0),
        (-1.0, 1.0, 0.0),
    )

    assert len(cycle.planes) == 4
    assert cycle.is_line_intersect_the_cycle(
        geo.Line((0.0, 0.0, -1.0), (0.0, 0.0, 1.0))
    )


def test_cycle_planes_reject_point_outside_closing_hexagon_edge():
    points = tuple(
        (2.0 * np.cos(i * np.pi / 3.0), 2.0 * np.sin(i * np.pi / 3.0), 0.0)
        for i in range(6)
    )
    cycle = geo.CyclePlanes(*points)

    assert not cycle.is_line_intersect_the_cycle(
        geo.Line((1.8, -0.5, -1.0), (1.8, -0.5, 1.0))
    )


def test_bond_ring_intersection_returns_stable_detail():
    molecule = _square_with_probe()
    ring = molecule.rings[0]
    probe = molecule.bond(4, 5)

    assert geo.bond_intersects_ring(ring, probe)
    assert geo.find_bond_ring_intersections(molecule) == ((ring, probe),)
    assert geo.has_bond_ring_intersection(molecule)
    assert all(not geo.bond_intersects_ring(ring, edge) for edge in ring.bonds)


def test_ligand_ring_scope_ignores_a_chelate_cycle():
    molecule = _molecule(
        (
            (0.0, -1.0, 0.0),
            (-1.0, 0.0, 0.0),
            (-1.0, 1.5, 0.0),
            (1.0, 1.5, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.5, -1.0),
            (0.0, 0.5, 1.0),
        ),
        ((0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (5, 6)),
        (30, 7, 6, 6, 7, 8, 8),
    )

    assert geo.find_bond_ring_intersections(
        molecule,
        ring_scope="full_graph",
    )
    assert geo.find_bond_ring_intersections(
        molecule,
        ring_scope="ligand_skeleton",
    ) == ()


def test_metal_ligand_bond_can_cross_a_real_ligand_ring():
    ring_points = tuple(
        (2.0 * np.cos(i * np.pi / 3.0), 2.0 * np.sin(i * np.pi / 3.0), 0.0)
        for i in range(6)
    )
    molecule = _molecule(
        ring_points + ((0.0, 0.0, -2.0), (0.0, 0.0, 2.0)),
        tuple((i, (i + 1) % 6) for i in range(6)) + ((6, 7),),
        (6, 6, 6, 6, 6, 6, 30, 7),
    )

    intersections = geo.find_bond_ring_intersections(
        molecule,
        ring_scope="ligand_skeleton",
    )

    assert len(intersections) == 1
    assert intersections[0][1] is molecule.bond(6, 7)


def test_closest_ring_edge_uses_finite_segment_minimum():
    molecule = _square_with_probe(probe_x=1.2)
    ring = molecule.rings[0]

    closest = geo.closest_ring_edge_to_bond(ring, molecule.bond(4, 5))

    assert {closest.a1idx, closest.a2idx} == {1, 2}


@pytest.mark.parametrize(
    ("distance", "expected"),
    ((0.0009, True), (0.0010, True), (0.0011, False)),
)
def test_overlap_boundary_is_inclusive(distance, expected):
    molecule = _molecule(((0.0, 0.0, 0.0), (distance, 0.0, 0.0)))
    assert geo.has_overlapping_atoms(molecule, tolerance=0.001) is expected


@pytest.mark.parametrize(
    ("distance", "expected"),
    ((0.4999, True), (0.5000, False), (0.5001, False)),
)
def test_too_close_boundary_is_strict(distance, expected):
    molecule = _molecule(((0.0, 0.0, 0.0), (distance, 0.0, 0.0)))
    assert geo.has_too_close_atoms(
        molecule,
        minimum_distance=0.5,
    ) is expected


def test_too_close_pair_scopes_and_overlap_partition():
    molecule = _molecule(
        ((0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (0.6, 0.0, 0.0)),
        ((0, 1),),
    )

    all_pairs = geo.find_too_close_atom_pairs(molecule)
    bonded = geo.find_too_close_atom_pairs(molecule, pair_scope="bonded")
    nonbonded = geo.find_too_close_atom_pairs(molecule, pair_scope="nonbonded")

    assert {issue.atom_indices for issue in all_pairs} == {(0, 1), (1, 2)}
    assert {issue.atom_indices for issue in bonded} == {(0, 1)}
    assert {issue.atom_indices for issue in nonbonded} == {(1, 2)}

    molecule.atoms[1].coordinates = molecule.atoms[0].coordinates
    assert geo.has_too_close_atoms(molecule)
    assert not geo.find_too_close_atom_pairs(
        _molecule(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))),
        include_overlaps=False,
    )


def test_too_close_uses_element_covalent_radii():
    molecule = _molecule(
        ((0.0, 0.0, 0.0), (0.8, 0.0, 0.0)),
        atomic_numbers=(6, 6),
    )

    issue, = geo.find_too_close_atom_pairs(
        molecule,
        minimum_distance=0.5,
        covalent_radius_scale=0.55,
    )

    assert issue.threshold == pytest.approx(0.55 * (0.76 + 0.76))


def test_unknown_pair_and_ring_scopes_fail_explicitly():
    molecule = _molecule(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
    with pytest.raises(ValueError, match="atom-pair scope"):
        geo.find_too_close_atom_pairs(molecule, pair_scope="invalid")
    with pytest.raises(ValueError, match="ring scope"):
        geo.find_bond_ring_intersections(molecule, ring_scope="invalid")
