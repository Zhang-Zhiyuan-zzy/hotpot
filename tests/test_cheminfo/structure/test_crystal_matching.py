"""Independent complete overlap, chemistry and full placed-action witnesses."""

import builtins
from dataclasses import fields, replace
from fractions import Fraction
import warnings

import numpy as np
import pytest

from hotpot import Crystal, Molecule, SpaceGroup, SymmUnit
from hotpot.cheminfo.structure._fields import ATOM_FIELDS, FIELD_INDEX
from hotpot.cheminfo.rules.mathematics.periodic import _lattice_images
from hotpot.cheminfo.operations.crystal_matching import _same_molecule_data
from hotpot.cheminfo import stereo


I = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def molecule(points, numbers=None, edges=()):
    value = Molecule()
    numbers = numbers if numbers is not None else (6,) * len(points)
    for number, point in zip(numbers, points):
        value.create_atom(atomic_number=number, coordinates=point, formal_charge=0, implicit_hydrogens=0)
    for first, second in edges:
        value.add_bond(first, second, bond_order=1)
    return value


def crystal(points, numbers=None, edges=(), number=1, **kwargs):
    return Crystal(10, 10, 10, 90, 90, 90, molecule=molecule(points, numbers, edges),
                   space_group=SpaceGroup(number), **kwargs)


def raw(value):
    record, metadata = value.molecule._capture_factual_record()
    return tuple(getattr(record, item.name) for item in fields(record)), metadata


def test_symmetric_triatom_has_one_complete_inversion_map_without_implicit_merge():
    value = crystal(((0., 0., 0.), (1., 0., 0.), (-1., 0., 0.)), (6, 8, 8), ((0, 1), (0, 2)), 2)
    before = raw(value)
    candidates = value.same_molecule_candidates(tolerance=1e-9)
    assert len(candidates) == 1
    operation, pairs, distance = candidates[0]
    assert operation.xyz == '-x,-y,-z' and distance < 1e-12
    assert tuple((first[0], second[0]) for first, second in pairs) == ((0, 0), (1, 2), (2, 1))
    assert len(value.molecule.atoms) == 3 and raw(value) == before
    assert set(value.molecular_symmetry(tolerance=1e-9)) == {SpaceGroup(2).identity(), operation}


def test_one_coincident_centre_does_not_identify_a_whole_asymmetric_unit():
    value = crystal(((0., 0., 0.), (1., 0., 0.), (-2., 0., 0.)), (6, 8, 8), ((0, 1), (0, 2)), 2)
    assert value.same_molecule_candidates(tolerance=1e-8) == ()
    assert value.molecular_symmetry(tolerance=1e-8) == (value.space_group.identity(),)


def test_isomorphic_but_spatially_distinct_inversion_copies_do_not_become_same_molecule():
    value = crystal(((1., 1., 1.), (2., 1., 1.)), (6, 8), ((0, 1),), 2)
    assert value.same_molecule_candidates(tolerance=1e-8) == ()


def test_identical_coordinates_with_a_different_graph_have_no_complete_same_map():
    points = ((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
    value = crystal(points, edges=((0, 1), (1, 2), (3, 4), (3, 5), (4, 5)))
    assert value.same_molecule_candidates(tolerance=1e-8) == ()


def test_coordinate_permutation_and_graph_map_must_be_the_same_bijection():
    points = ((0., 0., 0.), (1., 0., 0.), (2., 0., 0.)) * 2
    value = crystal(points, edges=((0, 1), (1, 2), (3, 4), (3, 5)))
    assert value.same_molecule_candidates(tolerance=1e-8) == ()


def test_ambiguous_complete_maps_remain_separate_values():
    points = ((0., 0., 0.), (1., 0., 0.), (1., 0., 0.)) * 2
    value = crystal(points, (6, 8, 8) * 2, ((0, 1), (0, 2), (3, 4), (3, 5)))
    matches = value.same_molecule_candidates(tolerance=1e-8)
    assert len(matches) == 2
    assert {tuple(second[0] for _, second in pairs) for _, pairs, _ in matches} == {(3, 4, 5), (3, 5, 4)}


@pytest.mark.parametrize('kind', ('element', 'isotope', 'charge', 'occupancy', 'disorder', 'direction'))
def test_incompatible_declared_site_and_bond_facts_reject_the_complete_relation(kind):
    value = crystal(((1., 1., 1.), (2., 1., 1.)) * 2, (6, 8) * 2, ((0, 1), (2, 3)))
    mol = value.molecule
    if kind == 'element':
        mol.atoms[2].atomic_number = 7
    elif kind == 'isotope':
        facts = (stereo.AtomTransportRecord(0, 12, 0, None, None, 'fixture'),
                 stereo.AtomTransportRecord(2, 13, 0, None, None, 'fixture'))
        stereo.set_stereo_payload(mol, stereo.StereoPayload(1, (), facts, ()))
    elif kind == 'charge':
        mol.atoms[2].formal_charge = 1
    elif kind == 'occupancy':
        value.set_site_relations({**value.site_relations, 'occupancies': (1., 1., .5, 1.)})
    elif kind == 'disorder':
        value.set_site_relations({**value.site_relations, 'disorder': (('A', 1), None, ('A', 2), None)})
    else:
        for edge, direction in zip(mol.bonds, ('1->2', '2->1')):
            edge._set_bond_metadata(bond_kind='dative', bond_direction=direction)
    before = raw(value)
    assert value.same_molecule_candidates(tolerance=1e-8) == ()
    assert raw(value) == before


def test_common_whole_unit_lift_succeeds_for_crossing_bond_and_retains_physical_keys():
    value = crystal(((9., 1., 1.), (1., 1., 1.), (-1., 1., 1.), (1., 1., 1.)),
                    (6, 8, 6, 8), ((0, 1), (2, 3)), bond_images=((1, 0, 0), (0, 0, 0)))
    matches = value.same_molecule_candidates(tolerance=1e-8)
    assert len(matches) == 1
    operation, pairs, distance = matches[0]
    assert operation.t == (-1, 0, 0)
    assert distance < 1e-10
    # The same placed whole unit has x=9,11; neither atom is independently wrapped.
    assert pairs[0][0][2] == (0, 0, 0) and pairs[1][0][2] == (1, 0, 0)
    assert pairs[0][1][2] == (1, 0, 0) and pairs[1][1][2] == (1, 0, 0)


def test_independent_minimum_images_cannot_hide_a_whole_unit_shape_difference():
    value = crystal(((9., 1., 1.), (1., 1., 1.), (9., 1., 1.), (1., 1., 1.)),
                    (6, 8, 6, 8), ((0, 1), (2, 3)), bond_images=((1, 0, 0), (0, 0, 0)))
    assert value.same_molecule_candidates(tolerance=1e-8) == ()


def test_numeric_matching_uses_the_caller_tolerance_instead_of_generation_default():
    value = crystal(((1., 1., 1.), (2., 1., 1.), (1.000001, 1., 1.), (2.000001, 1., 1.)),
                    (6, 8, 6, 8), ((0, 1), (2, 3)))
    assert len(value.same_molecule_candidates(tolerance=2e-6)) == 1
    assert value.same_molecule_candidates(tolerance=.5e-6) == ()


def test_generated_current_keys_do_not_reuse_historical_source_rows():
    value = crystal(((1., 1., 1.),), number=2).unit_cell()
    assert value.space_group == SpaceGroup(1)
    # Deliberately move current row1 onto row0; query maps current rows0,1.
    value.molecule.atoms[1].coordinates = value.molecule.atoms[0].coordinates
    before = raw(value)
    matches = value.same_molecule_candidates(tolerance=1e-8)
    assert len(matches) == 1
    assert matches[0][1][0][0][0] == 0 and matches[0][1][0][1][0] == 1
    assert all(key[1].context == SpaceGroup(1) for pair in matches[0][1] for key in pair)
    assert raw(value) == before


def test_group_supported_molecular_symmetry_retains_full_nonzero_origin_translation():
    value = crystal(((1., 2., 3.), (3., 2., 3.)), edges=((0, 1),), number=2)
    # Inversion about(2,2,3) is not in this10Angstrom lattice's group.
    assert value.molecular_symmetry(tolerance=1e-8) == (value.space_group.identity(),)
    value = crystal(((4., 5., 5.), (6., 5., 5.)), edges=((0, 1),), number=2)
    operations = value.molecular_symmetry(tolerance=1e-8)
    assert len(operations) == 2
    inversion = next(op for op in operations if op.R != I)
    assert inversion.t == (1, 1, 1)
    assert inversion @ inversion == value.space_group.identity()


def test_finite_molecule_has_no_nonzero_pure_translation_stabilizer_even_at_loose_tolerance():
    value = crystal(((0., 0., 0.),))
    assert value.molecular_symmetry(tolerance=11.) == (value.space_group.identity(),)


def test_multiple_units_require_an_explicit_finite_unit_for_symmetry():
    value = crystal(((1., 1., 1.), (9., 9., 9.)), number=2)
    with pytest.raises(ValueError, match='choose one complete finite'):
        value.molecular_symmetry(tolerance=1e-8)
    selected = value.unit_molecules()[0]
    assert value.molecular_symmetry(selected, tolerance=1e-8) == (value.space_group.identity(),)
    value.molecule.atoms[0].coordinates = (2., 1., 1.)
    with pytest.raises(ValueError, match='no longer maps current'):
        value.molecular_symmetry(selected, tolerance=1e-8)


def test_detached_explicit_molecule_is_not_freely_aligned_into_crystal():
    value = crystal(((1., 1., 1.),))
    with pytest.raises(ValueError, match='explicit cell frame'):
        value.molecular_symmetry(molecule(((1., 1., 1.),)), tolerance=1e-8)


def test_unknown_or_periodic_component_does_not_block_independent_finite_matches():
    value = crystal(((1., 1., 1.), (1., 1., 1.), (2., 2., 2.), (3., 2., 2.)),
                    edges=((2, 3),), bond_images=(None,))
    before = raw(value)
    with pytest.warns(RuntimeWarning, match='CRYSTAL-MOLECULAR-LIFT-UNAVAILABLE'):
        result = value.same_molecule_candidates(tolerance=1e-8)
    assert len(result) == 1 and raw(value) == before


def test_query_import_and_actions_do_not_call_forbidden_toolkits(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in ('rdkit', 'openbabel', 'ccdc', 'pybel'):
            raise AssertionError('forbidden toolkit ' + name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    value = crystal(((4., 5., 5.), (6., 5., 5.)), edges=((0, 1),), number=2)
    assert len(value.same_molecule_candidates(tolerance=1e-8)) == 1
    assert len(value.molecular_symmetry(tolerance=1e-8)) == 2


def test_skew_metric_image_search_reaches_beyond_component_rounding_and27stencil():
    matrix = np.array(((1., 0., 0.), (10., .1, 0.), (0., 0., 2.)))
    delta = np.array((0., .49, 0.))
    images = _lattice_images(delta, matrix, .12)
    assert (-5, 0, 0) in images
    assert all(abs(image[0]) > 1 for image in images)
    assert _lattice_images((0., 0., 0.), np.eye(3), 1.) == (
        (-1, 0, 0), (0, -1, 0), (0, 0, -1), (0, 0, 0),
        (0, 0, 1), (0, 1, 0), (1, 0, 0))


@pytest.mark.parametrize('same_orientation', (False, True))
def test_full_geometric_overlap_uses_actual_stereo_orientation(same_orientation):
    points = ((0., 0., 0.), (1., 1., 1.), (1., -1., -1.), (-1., 1., -1.), (-1., -1., 1.))
    value = crystal(points * 2, (6, 1, 7, 8, 9) * 2,
                    tuple((centre, centre + offset) for centre in (0, 5) for offset in range(1, 5)))
    def record(centre, carriers, descriptor):
        return stereo.StereoRecord(stereo.StereoAnchor(stereo.StereoAnchorKind.ATOM, (centre,)),
            tuple(stereo.StereoCarrier(stereo.StereoCarrierKind.ATOM, row) for row in carriers),
            descriptor, None, 'hotpot.stereo/2', 'fixture')
    records = (record(0, (1, 2, 3, 4), 'hotpot.stereo/2/TH/even'),
               record(5, (7, 6, 8, 9) if same_orientation else (6, 7, 8, 9), 'hotpot.stereo/2/TH/odd'))
    stereo.set_stereo_payload(value.molecule, stereo.StereoPayload(1, records, (), ()))
    before = raw(value)
    matches = value.same_molecule_candidates(tolerance=1e-8)
    assert len(matches) == int(same_orientation)
    assert raw(value) == before


def test_supported_improper_square_planar_action_uses_the_same_chemical_map():
    value = crystal(((0., 0., 0.), (1., 0., 0.), (-1., 0., 0.), (0., 1., 0.), (0., -1., 0.)),
                    (78, 17, 17, 17, 17), tuple((0, row) for row in range(1, 5)), 2)
    record = stereo.StereoRecord(stereo.StereoAnchor(stereo.StereoAnchorKind.ATOM, (0,)),
        tuple(stereo.StereoCarrier(stereo.StereoCarrierKind.ATOM, row) for row in range(1, 5)),
        'hotpot.stereo/2/SP/1', None, 'hotpot.stereo/2', 'fixture')
    stereo.set_stereo_payload(value.molecule, stereo.StereoPayload(1, (record,), (), ()))
    assert len(value.same_molecule_candidates(tolerance=1e-8)) == 1
    assert len(value.molecular_symmetry(tolerance=1e-8)) == 2


def test_opaque_improper_meaning_stays_unavailable_while_identity_observation_survives():
    value = crystal(((0., 0., 0.),), number=2)
    record = stereo.StereoRecord(stereo.StereoAnchor(stereo.StereoAnchorKind.ATOM, (0,)), (),
                                'UNEXPLAINED', None, 'native-opaque', 'UNEXPLAINED')
    stereo.set_stereo_payload(value.molecule, stereo.StereoPayload(1, (record,), (), ()))
    before = raw(value)
    with pytest.warns(RuntimeWarning, match='CRYSTAL-UNSUPPORTED-PROJECTION'):
        operations = value.molecular_symmetry(tolerance=1e-8)
    assert operations == (value.space_group.identity(),)
    assert raw(value) == before


def test_approximate_nonclosed_matches_are_retained_without_fabricated_subgroup():
    delta = 1e-4
    value = crystal(((1., 0., 0.), (0., 1. + delta, 0.), (-1. - 2 * delta, 0., 0.), (0., -1. - delta, 0.)),
                    edges=((0, 1), (1, 2), (2, 3), (3, 0)), number=75)
    before = raw(value)
    with pytest.warns(RuntimeWarning, match='CRYSTAL-STABILIZER-NONCLOSURE'):
        operations = value.molecular_symmetry(tolerance=1.5 * delta)
    assert len(operations) == 3
    assert value.space_group.identity() in operations
    assert any(first @ second not in operations for first in operations for second in operations)
    assert raw(value) == before


def test_absent_required_element_facts_stay_unresolved_without_a_native_none_target():
    value = crystal(((0., 0., 0.),), number=2)
    record = value.molecule._capture_factual_record()[0]
    from hotpot.cheminfo.structure._fields import FIELD_PRESENCE_BITS
    masks = np.frombuffer(record.atom_presence, dtype='<u4').copy()
    masks &= np.uint32(~FIELD_PRESENCE_BITS[FIELD_INDEX['atomic_number']] & ((1 << len(ATOM_FIELDS)) - 1))
    value = Molecule._build_molecule(replace(record, atom_presence=masks.tobytes())).crystal
    before = raw(value)
    with pytest.warns(RuntimeWarning, match='required element facts are absent'):
        assert value.molecular_symmetry(tolerance=1e-8) == ()
    with pytest.warns(RuntimeWarning, match='required element facts are absent'):
        assert value.same_molecule_candidates(tolerance=1e-8) == ()
    assert raw(value) == before


def test_native_undirected_packing_preserves_actual_reversed_dative_endpoint_meaning():
    value = crystal(((1., 1., 1.), (2., 1., 1.)) * 2, (6, 8) * 2, ((0, 1), (3, 2)))
    for bond, direction in zip(value.molecule.bonds, ('1->2', '2->1')):
        bond._set_bond_metadata(bond_kind='dative', bond_direction=direction)
    before = raw(value)
    result = value.same_molecule_candidates(tolerance=1e-8)
    assert len(result) == 1
    assert tuple((first[0], second[0]) for first, second in result[0][1]) == ((0, 2), (1, 3))
    assert raw(value) == before
