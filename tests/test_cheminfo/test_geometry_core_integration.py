"""Integration tests for Core facades over factual geometry APIs."""

import math

import pytest

from hotpot.cheminfo import geometry
from hotpot.cheminfo.core import Molecule, NotInSameMolecule


def _molecule(coordinates, bonds=()) -> Molecule:
    mol = Molecule()
    for coordinates3 in coordinates:
        mol.create_atom(atomic_number=6, coordinates=coordinates3)
    for first_index, second_index in bonds:
        mol.add_bond(first_index, second_index, bond_order=1.0)
    return mol


def _square_with_probe() -> Molecule:
    return _molecule(
        (
            (-1.0, -1.0, 0.0),
            (1.0, -1.0, 0.0),
            (1.0, 1.0, 0.0),
            (-1.0, 1.0, 0.0),
            (0.0, 0.0, -1.0),
            (0.0, 0.0, 1.0),
        ),
        ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5)),
    )


def test_molecule_facades_preserve_tristate_and_scan_coverage() -> None:
    mol = _square_with_probe()

    assert mol.bond_ring_piercing_state() is geometry.PiercingState.PIERCES

    report = mol.bond_ring_relations()
    assert report.ring_scope == "full_graph"
    assert report.max_ring_size == 8
    assert report.scan_complete
    assert report.piercing_pair_count == 1
    assert report.piercings[0].target.bond.bond is mol.bond(4, 5)


def test_disorder_policy_consumes_factual_atom_pair_distances() -> None:
    mol = _molecule(((0.0, 0.0, 0.0), (0.49, 0.0, 0.0)))

    assert mol.is_disorder

    mol.atoms[1].coordinates = (0.5, 0.0, 0.0)
    assert not mol.is_disorder


def test_bond_facades_use_finite_segments() -> None:
    mol = _molecule(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, -1.0, 0.0),
            (2.0, 1.0, 0.0),
        ),
        ((0, 1), (2, 3)),
    )
    first, second = mol.bonds

    assert isinstance(first.bond_segment, geometry.Segment)
    assert first.bond_segment.start.coordinates == (0.0, 0.0, 0.0)
    assert first.bond_segment.end.coordinates == (1.0, 0.0, 0.0)
    assert first.bond_segment_distance(second) == pytest.approx(1.0)


def test_ring_facades_preserve_sources_and_same_molecule_validation() -> None:
    mol = _square_with_probe()
    ring = mol.rings[0]
    probe = mol.bond(4, 5)

    assert isinstance(ring.geometry_cycle, geometry.Cycle)
    finding = ring.relation_to_bond(probe)
    assert finding.target.ring.ring is ring
    assert finding.target.bond.bond is probe
    assert finding.relation.state is geometry.PiercingState.PIERCES

    closest = ring.closest_edge_to_bond(probe)
    assert isinstance(closest, geometry.RingEdgeDistance)
    assert closest.bond is ring.bonds[closest.measurement.edge_index]
    assert closest.measurement.distance == pytest.approx(1.0)

    foreign = _molecule(((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)), ((0, 1),)).bonds[0]
    with pytest.raises(NotInSameMolecule):
        ring.relation_to_bond(foreign)
    with pytest.raises(NotInSameMolecule):
        ring.closest_edge_to_bond(foreign)


def test_aromaticity_keeps_core_chemical_planarity_tolerance() -> None:
    mol = _molecule(
        tuple(
            (
                math.cos(index * math.pi / 3.0),
                math.sin(index * math.pi / 3.0),
                0.08 if index == 0 else 0.0,
            )
            for index in range(6)
        ),
        tuple((index, (index + 1) % 6) for index in range(6)),
    )
    ring = mol.rings[0]

    measurement = geometry.measure_planarity(ring.geometry_cycle)
    assert measurement.kind is geometry.PlanarityKind.NONPLANAR
    assert (
        measurement.maximum_deviation / measurement.length_scale
        < 0.03
    )
    assert ring.determine_aromatic()
