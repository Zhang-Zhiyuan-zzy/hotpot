"""Public query contracts for the Stage9A semantic extraction."""

import math
from importlib import import_module

import numpy as np
import pytest

from hotpot import Atom, Molecule
NotInSameMolecule = import_module(Molecule.__module__).NotInSameMolecule
from hotpot.cheminfo.operations import chemistry
from hotpot.cheminfo.structure._fields import FIELD_INDEX
from tests.test_cheminfo._stage5_fixtures import factual_state, supplied_aromatic_flags


def _molecule(coordinates, edges=(), numbers=None):
    mol = Molecule()
    numbers = (6,) * len(coordinates) if numbers is None else numbers
    for number, row in zip(numbers, coordinates):
        mol.create_atom(atomic_number=number, coordinates=row)
    for first, second in edges:
        mol.add_bond(first, second, 1.)
    return mol


def _cycle(coordinates):
    return _molecule(coordinates, tuple((row, (row + 1) % len(coordinates))
                                       for row in range(len(coordinates))))


def _regular_cycle(size):
    return _cycle(tuple((math.cos(2 * math.pi * row / size),
                         math.sin(2 * math.pi * row / size), 0.)
                        for row in range(size)))


@pytest.mark.parametrize('coordinates, expected', (
    (((0., 0., 0.),) * 3, False),
    (((2., 3., 4.),) * 3, False),
    (((0., 0., 0.), (0., 0., 0.), (1.e-12, 0., 0.)), True),
    (((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)), True),
))
def test_ring_and_molecule_coordinate_predicates_use_distinct_rows(coordinates, expected):
    mol = _cycle(coordinates)
    ring = mol.rings[0]
    before = factual_state(mol)
    assert mol.has_3d is expected
    assert ring.has_3d is expected
    assert factual_state(mol) == before


def test_molecule_coordinate_predicate_empty_singleton_and_coincident_boundary():
    mol = Molecule()
    assert mol.has_3d is False
    mol.create_atom(atomic_number=6, coordinates=(3., 4., 5.))
    assert mol.has_3d is False
    mol.create_atom(atomic_number=6, coordinates=(3., 4., 5.))
    assert mol.has_3d is False
    mol.atoms[1].z += 1.e-12
    assert mol.has_3d is True


@pytest.mark.parametrize('distance, expected', ((.499, True), (.5, False), (.501, False)))
def test_disorder_threshold_is_strict_for_ring_and_molecule(distance, expected):
    mol = _cycle(((0., 0., 0.), (distance, 0., 0.), (0., 2., 0.)))
    assert mol.is_disorder is expected
    assert mol.rings[0].is_disorder is expected


def test_ring_coordinate_predicate_rejects_stale_geometry():
    mol = _regular_cycle(6)
    ring = mol.rings[0]
    mol.atoms[0].z = .2
    for read in (lambda: ring.has_3d, lambda: ring.geometry_cycle):
        with pytest.raises(RuntimeError, match='geometry source changed'):
            read()
    assert mol.rings[0].has_3d is True


@pytest.mark.parametrize('radius', (0., 1., 2.5))
def test_sphere_sampling_consumes_exactly_two_caller_draws(radius):
    rng = np.random.default_rng(5917)
    control = np.random.default_rng(5917)
    control.random(2)
    point = Atom.random_point_on_sphere(radius, rng)
    assert isinstance(point, tuple) and len(point) == 3
    assert np.linalg.norm(point) == pytest.approx(radius, abs=1.e-14)
    np.testing.assert_array_equal(rng.random(8), control.random(8))


def test_sphere_sampling_preserves_global_rng_route():
    state = np.random.get_state()
    try:
        np.random.seed(119)
        point = Atom.random_point_on_sphere(3.)
        observed_next = np.random.rand(8)
        np.random.seed(119)
        np.random.rand(2)
        np.testing.assert_array_equal(observed_next, np.random.rand(8))
        assert np.linalg.norm(point) == pytest.approx(3.)
    finally:
        np.random.set_state(state)


@pytest.mark.parametrize('draws, expected', (
    ((.5, 0.), (2., 0., 0.)),
    ((.5, .25), (0., 2., 0.)),
    ((1., .75), (0., 0., 2.)),
    ((0., .5), (0., 0., -2.)),
))
def test_sphere_sampling_axes_and_poles(draws, expected):
    class Draws:
        def __init__(self):
            self.values = iter(draws)
            self.count = 0

        def random(self):
            self.count += 1
            return next(self.values)

    rng = Draws()
    np.testing.assert_allclose(Atom.random_point_on_sphere(2., rng), expected, atol=1.e-14)
    assert rng.count == 2


def test_organic_metal_and_explicit_h_short_circuit_implicit_preparation(monkeypatch):
    metal = _molecule(((0., 0., 0.),), numbers=(30,))
    carbon_hydrogen = _molecule(((0., 0., 0.), (1., 0., 0.)), ((0, 1),), (6, 1))
    def forbidden(mol):
        raise AssertionError('unneeded implicit-H preparation')
    monkeypatch.setattr(chemistry, '_require_implicit_hydrogens', forbidden)
    assert metal.is_organic is False
    assert carbon_hydrogen.is_organic is True


def test_organic_unresolved_h_does_not_skip_the_existing_failure():
    atom = Atom()
    with pytest.raises(ValueError, match='resolved implicit-H'):
        _ = atom.mol.is_organic
    hfield = FIELD_INDEX['implicit_hydrogens']
    assert not int(atom.mol._atom_table._presence[0]) & (1 << hfield)


def test_full_halogenated_query_changes_only_its_working_copy():
    mol = _molecule(((0., 0., 0.), (1., 0., 0.), (-1., 0., 0.),
                     (0., 1., 0.), (0., -1., 0.)),
                    ((0, 1), (0, 2), (0, 3), (0, 4)), (6, 9, 17, 35, 53))
    assert mol.is_organic is False
    before = factual_state(mol)
    assert mol.is_full_halogenated is True
    assert tuple(atom.atomic_number for atom in mol.atoms) == (6, 9, 17, 35, 53)
    assert factual_state(mol) == before


def test_full_halogenated_organic_short_circuit_avoids_copy(monkeypatch):
    mol = _molecule(((0., 0., 0.), (1., 0., 0.)), ((0, 1),), (6, 1))
    def forbidden():
        raise AssertionError('organic query copied its source')
    monkeypatch.setattr(mol, 'copy', forbidden)
    assert mol.is_full_halogenated is False


def test_spectral_similarity_retains_source_facts_and_rejects_unknown_method():
    first = _molecule(((0., 0., 0.), (1.5, 0., 0.)), ((0, 1),))
    second = first.copy()
    before = factual_state(first), factual_state(second)
    assert first.similarity(second) == pytest.approx(1., abs=1.e-14)
    assert (factual_state(first), factual_state(second)) == before
    with pytest.raises(NotImplementedError):
        first.similarity(second, method='unimplemented')


def test_constraints_filter_metals_and_false_remains_a_noop():
    mol = _molecule(((0., 0., 0.), (1., 0., 0.), (2., 1., 0.), (3., 1., 0.)),
                    ((0, 1), (1, 2), (2, 3)), (6, 6, 7, 30))
    mol.constraint_bonds_angles(False)
    assert not any(bond.constraint for bond in mol.bonds)
    assert not any(angle.constraint for angle in mol.angles)
    mol.constraint_bonds_angles()
    assert tuple(bond.constraint for bond in mol.bonds) == (True, True, False)
    assert sorted((tuple(angle.atoms_indices), angle.constraint) for angle in mol.angles) == [
        ((0, 1, 2), True), ((1, 2, 3), False)]
    before = factual_state(mol)
    mol.constraint_bonds_angles(False)
    assert factual_state(mol) == before


def test_joint_ring_preserves_ring_order_and_declared_errors():
    mol = _molecule(((0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.)),
                    ((0, 1), (1, 2), (2, 3), (3, 0), (0, 2)))
    rings = mol.rings
    assert len(rings) == 2
    assert rings[0].joint_with(rings[1]) is True
    joint = rings[0].joint_ring(aromatic=False)
    assert joint.rings == tuple(rings)
    assert tuple(atom.idx for atom in joint.atoms) == tuple(dict.fromkeys(
        atom.idx for ring in rings for atom in ring.atoms))
    with pytest.raises(TypeError, match='expected Ring'):
        rings[0].joint_with(mol.atoms[0])
    with pytest.raises(ArithmeticError, match='incorrect Ring'):
        rings[0].joint_with(rings[0])
    assert rings[0].joint_with(_regular_cycle(3).rings[0]) is False


def test_joint_ring_rejects_changed_topology():
    mol = _regular_cycle(6)
    ring = mol.rings[0]
    mol.create_atom(atomic_number=6)
    with pytest.raises(RuntimeError, match='topology source changed'):
        ring.joint_ring(aromatic=False)


def test_small_rings_boundary_and_scope_errors():
    assert len(_regular_cycle(8).rings_small) == 1
    assert _regular_cycle(9).rings_small == []
    mol = _regular_cycle(6)
    assert mol.rings_for_scope('full_graph') == mol.rings
    assert mol.rings_for_scope('ligand_skeleton') == mol.rings
    assert mol.rings_for_scope('full_graph', max_size=5) == []
    with pytest.raises(ValueError, match="Unsupported ring scope: 'unknown'"):
        mol.rings_for_scope('unknown')


def test_aromatic_joint_rings_keep_reverse_component_selection_order():
    coordinates = []
    edges = []
    for center in (0., 5.):
        offset = len(coordinates)
        coordinates.extend((center + math.cos(row * math.pi / 3), math.sin(row * math.pi / 3), 0.)
                           for row in range(6))
        edges.extend((offset + row, offset + (row + 1) % 6) for row in range(6))
    mol = supplied_aromatic_flags(_molecule(coordinates, edges), (True,) * 12)
    rings = mol.rings
    joint = mol.aromatic_joint_rings
    assert len(joint) == 2
    assert joint[0].rings == (rings[-1],)
    assert joint[1].rings == (rings[0],)


def test_bond_segment_distance_uses_finite_endpoints():
    mol = _molecule(((0., 0., 0.), (1., 0., 0.), (2., -1., 0.), (2., 1., 0.)),
                    ((0, 1), (2, 3)))
    first, second = mol.bonds
    assert first.bond_segment_distance(second) == pytest.approx(1.)
    assert second.bond_segment_distance(first) == pytest.approx(1.)


def test_ring_geometry_queries_preserve_bond_mapping_and_foreign_error():
    mol = _molecule(((-1., -1., 0.), (1., -1., 0.), (1., 1., 0.), (-1., 1., 0.),
                     (0., 0., -1.), (0., 0., 1.)),
                    ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5)))
    ring = mol.rings[0]
    probe = mol.bond(4, 5)
    before = factual_state(mol)
    assert tuple(point.coordinates for point in ring.geometry_cycle.vertices) == tuple(
        tuple(atom.coordinates) for atom in ring.atoms)
    relation = ring.relation_to_bond(probe)
    assert relation.target.ring.ring is ring
    assert relation.target.bond.bond is probe
    nearest = ring.closest_edge_to_bond(probe)
    assert nearest.bond is ring.bonds[nearest.measurement.edge_index]
    assert nearest.measurement.distance == pytest.approx(1.)
    assert factual_state(mol) == before
    foreign = _molecule(((0., 0., 0.), (0., 0., 1.)), ((0, 1),)).bonds[0]
    for read in (ring.relation_to_bond, ring.closest_edge_to_bond):
        with pytest.raises(NotInSameMolecule):
            read(foreign)


def test_coordination_bond_query_retains_original_member_order():
    mol = _molecule(((0., 0., 0.), (1., 0., 0.), (2., 0., 0.), (3., 0., 0.)),
                    ((0, 1), (1, 2), (2, 3)), (6, 7, 30, 8))
    before = factual_state(mol)
    assert mol.c_bonds == mol.bonds[1:]
    assert all(selected is original for selected, original in zip(mol.c_bonds, mol.bonds[1:]))
    assert factual_state(mol) == before
