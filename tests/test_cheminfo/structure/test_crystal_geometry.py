"""Independent periodic metric, contact, surface and finite-link observations."""

from dataclasses import fields
import warnings

import numpy as np
import pytest

from hotpot import Crystal, Molecule, SpaceGroup
from hotpot.cheminfo import stereo
from hotpot.cheminfo.operations import crystal_geometry as operations


ZERO = (0, 0, 0)


def crystal(points, numbers=None, edges=(), *, matrix=None, number=1, images=None):
    value = Molecule()
    numbers = (6,) * len(points) if numbers is None else numbers
    for atomic_number, point in zip(numbers, points):
        facts = dict(coordinates=point, formal_charge=0, implicit_hydrogens=0)
        if atomic_number is not None:
            facts['atomic_number'] = atomic_number
        value.create_atom(**facts)
    for first, second in edges:
        value.add_bond(first, second, bond_order=1)
    matrix = np.eye(3) * 10 if matrix is None else matrix
    return Crystal.from_matrix(matrix, molecule=value, space_group=SpaceGroup(number), bond_images=images)


def raw(value):
    record, metadata = value.molecule._capture_factual_record()
    return tuple(getattr(record, field.name) for field in fields(record)), metadata


def query(value, *, coincidence=1e-8, cutoff=.5, scope='full_graph', max_size=None):
    before = raw(value)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        findings = value.geometry_findings(coincidence_tolerance=coincidence, close_contact_cutoff=cutoff,
                                          ring_scope=scope, max_ring_size=max_size)
    assert raw(value) == before
    assert len(caught) == len(findings)
    assert all(str(item.message).startswith(finding[0] + ':') for item, finding in zip(caught, findings))
    return findings


def selected(findings, code):
    return tuple(item for item in findings if item[0] == code)


def set_descriptors(value, descriptors):
    records = tuple(stereo.StereoRecord(stereo.StereoAnchor(stereo.StereoAnchorKind.ATOM, (row,)),
                    (), descriptor, None, 'fixture-current', descriptor)
                    for row, descriptor in enumerate(descriptors))
    stereo.set_stereo_payload(value.molecule, stereo.StereoPayload(1, records, (), ()))


def test_isolated_well_separated_content_is_silent_and_does_not_mutate():
    value = crystal(((1., 1., 1.), (4., 4., 4.)), (6, 8))
    before = raw(value)
    assert query(value) == ()
    assert raw(value) == before


def test_close_distinct_sites_report_real_distance_and_cannot_merge():
    value = crystal(((1., 1., 1.), (1.125, 1., 1.)), (6, 8))
    before = raw(value)
    findings = query(value)
    crowd = selected(findings, 'CRYSTAL-CROWDING')
    assert len(crowd) == 1
    assert dict(crowd[0][2])['distance'] == .125
    assert dict(crowd[0][2])['relation'] == 'no_complete_coincident_relation'
    assert len(value.molecule.atoms) == 2 and raw(value) == before


def test_local_coincidence_is_observed_without_a_complete_chemical_relation():
    value = crystal(((1., 1., 1.), (1., 1., 1.)), (6, 8))
    findings = query(value)
    assert len(selected(findings, 'CRYSTAL-COINCIDENT-IMAGES')) == 1
    assert len(selected(findings, 'CRYSTAL-CROWDING')) == 1
    assert not selected(findings, 'CRYSTAL-POSSIBLE-SAME-MOLECULE')


def test_complete_coincident_units_are_a_relation_without_false_independence_or_deletion():
    value = crystal(((1., 1., 1.), (1.125, 1., 1.)) * 2, (6, 8) * 2, ((0, 1), (2, 3)))
    before = raw(value)
    findings = query(value)
    assert len(selected(findings, 'CRYSTAL-POSSIBLE-SAME-MOLECULE')) == 1
    assert not selected(findings, 'CRYSTAL-COINCIDENT-IMAGES')
    assert not selected(findings, 'CRYSTAL-CROWDING')
    assert len(value.molecule.atoms) == 4 and raw(value) == before


def test_distinct_orbit_copies_are_still_crowding():
    value = crystal(((.125, 0., 0.),), number=2)
    findings = query(value)
    crowd = selected(findings, 'CRYSTAL-CROWDING')
    assert len(crowd) == 1
    assert dict(crowd[0][2])['distance'] == .25
    assert not selected(findings, 'CRYSTAL-POSSIBLE-SAME-MOLECULE')


def test_incomplete_chemical_match_preserves_proximity_and_its_reason():
    value = crystal(((1., 1., 1.), (1., 1., 1.)), (None, 6))
    before = raw(value)
    findings = query(value)
    crowd = selected(findings, 'CRYSTAL-CROWDING')
    assert len(crowd) == 1
    data = dict(crowd[0][2])
    assert data['relation'] == 'unresolved' and 'element' in data['reason']
    assert raw(value) == before


def test_unsupported_complete_map_reason_is_retained_with_contact():
    value = crystal(((1., 1., 1.), (1., 1., 1.)))
    set_descriptors(value, ('hotpot.stereo/2/FUTURE/1',) * 2)
    findings = query(value)
    unsupported = selected(findings, 'CRYSTAL-UNSUPPORTED-PROJECTION')
    assert len(unsupported) == 1
    reason = dict(unsupported[0][2])['reason']
    assert reason == 'CRYSTAL-UNSUPPORTED-PROJECTION: identified stereo has an unsupported Level-2 descriptor'
    assert not selected(findings, 'CRYSTAL-POSSIBLE-SAME-MOLECULE')
    contacts = (*selected(findings, 'CRYSTAL-COINCIDENT-IMAGES'), *selected(findings, 'CRYSTAL-CROWDING'))
    assert len(contacts) == 2
    for finding in contacts:
        data = dict(finding[2])
        assert data['relation'] == 'unresolved' and data['reason'] == reason
        assert data['distance'] == 0.


def test_known_incompatible_opaque_meanings_remain_a_known_no_match():
    value = crystal(((1., 1., 1.), (1., 1., 1.)))
    set_descriptors(value, ('OPAQUE-A', 'OPAQUE-B'))
    findings = query(value)
    assert not selected(findings, 'CRYSTAL-POSSIBLE-SAME-MOLECULE')
    assert not selected(findings, 'CRYSTAL-UNSUPPORTED-PROJECTION')
    contacts = (*selected(findings, 'CRYSTAL-COINCIDENT-IMAGES'), *selected(findings, 'CRYSTAL-CROWDING'))
    assert len(contacts) == 2
    assert all(dict(item[2])['relation'] == 'no_complete_coincident_relation' for item in contacts)
    assert all('reason' not in dict(item[2]) for item in contacts)


@pytest.mark.parametrize('unsupported', (False, True))
def test_complete_relation_uses_one_common_shift_with_different_member_lifts(unsupported):
    points = ((9.875, 0., 0.), (.125, 0., 0.), (-.125, 0., 0.), (.125, 0., 0.))
    value = crystal(points, (6, 8, 6, 8), ((0, 1), (2, 3)), images=((1, 0, 0), ZERO))
    if unsupported:
        set_descriptors(value, ('hotpot.stereo/2/FUTURE/1',) * 4)
    findings = query(value)
    assert len(selected(findings, 'CRYSTAL-POSSIBLE-SAME-MOLECULE')) == (0 if unsupported else 1)
    contacts = (*selected(findings, 'CRYSTAL-COINCIDENT-IMAGES'), *selected(findings, 'CRYSTAL-CROWDING'))
    if unsupported:
        assert len(selected(findings, 'CRYSTAL-COINCIDENT-IMAGES')) == 2
        assert len(selected(findings, 'CRYSTAL-CROWDING')) == 4
        reason = dict(selected(findings, 'CRYSTAL-UNSUPPORTED-PROJECTION')[0][2])['reason']
        assert all(dict(item[2])['relation'] == 'unresolved' and dict(item[2])['reason'] == reason
                   for item in contacts)
    else:
        assert not contacts
    farther = query(value, cutoff=10.25)
    distinct_copy = [item for item in selected(farther, 'CRYSTAL-CROWDING')
                     if tuple(key[0] for key in item[1]) == (0, 2) and item[1][1][2] == ZERO]
    assert len(distinct_copy) == 1
    data = dict(distinct_copy[0][2])
    assert data['relation'] == 'no_complete_coincident_relation' and 'reason' not in data
    assert data['distance'] == 10.


def test_supported_complete_relation_takes_precedence_over_unavailable_alternate_map():
    from hotpot.cheminfo.operations.crystal_matching import _same_molecule_data

    value = crystal(((1., 1., 1.), (1., 1., 1.)))
    candidates, unavailable, _, units, placements = _same_molecule_data(value, 1e-8)
    assert len(candidates) == 1 and not unavailable
    first, second = candidates[0][1][0]
    unavailable = (('CRYSTAL-UNSUPPORTED-PROJECTION', ((first,), (second,)), 'unsupported alternate map'),)
    _, relation = operations._relation_context(candidates, units, placements, unavailable, value.space_group)
    assert relation(first, second) == ('complete_coincident_unit', None)
    assert relation(second, first) == ('complete_coincident_unit', None)


def test_actual_bond_image_is_excluded_but_bonded_atom_coincidence_is_retained():
    value = crystal(((9.875, 0., 0.), (.125, 0., 0.)), edges=((0, 1),), images=((1, 0, 0),))
    assert not selected(query(value), 'CRYSTAL-CROWDING')
    contacts = [item for item in selected(query(value, cutoff=10.), 'CRYSTAL-CROWDING')
                if tuple(key[0] for key in item[1]) == (0, 1)]
    assert any(item[1][1][2] == ZERO and dict(item[2])['distance'] == 9.75 for item in contacts)
    assert not any(item[1][1][2] == (1, 0, 0) for item in contacts)
    coincident = crystal(((0., 0., 0.), (0., 0., 0.)), edges=((0, 1),))
    assert len(selected(query(coincident), 'CRYSTAL-COINCIDENT-IMAGES')) == 1


def test_unwrapped_inputs_keep_the_actual_translation_instead_of_one_cell_rounding():
    value = crystal(((19.875, 0., 0.), (.125, 0., 0.)), (6, 8))
    crowd = selected(query(value), 'CRYSTAL-CROWDING')
    assert len(crowd) == 1
    assert crowd[0][1][0][2] == ZERO and crowd[0][1][1][2] == (2, 0, 0)
    assert dict(crowd[0][2])['distance'] == .25


def test_skew_metric_finds_images_outside_the_fractional_27_stencil():
    basis = np.array(((1., 0., 0.), (10., 1., 0.), (0., 0., 10.)))
    value = crystal(((0., 0., 0.), tuple(np.array((.49, .49, 0.)) @ basis)), (6, 8), matrix=basis)
    crowd = selected(query(value, cutoff=.7), 'CRYSTAL-CROWDING')
    assert any(item[1][1][2] == (-5, 0, 0) for item in crowd)
    assert any(abs(dict(item[2])['distance'] - np.hypot(.39, .49)) < 1e-12 for item in crowd)


def test_self_images_are_not_erased_by_common_orbit_membership():
    value = crystal(((0., 0., 0.),), matrix=np.eye(3) * .2)
    crowd = selected(query(value, cutoff=.3), 'CRYSTAL-CROWDING')
    assert len(crowd) == 9
    assert all(item[1][0][0] == item[1][1][0] == 0 and item[1][1][2] != ZERO for item in crowd)


def boundary_ring():
    points = ((9.5, 4., 0.), (.5, 4., 0.), (.5, 6., 0.), (9.5, 6., 0.), (0., 5., -1.), (0., 5., 1.))
    return crystal(points, edges=((0, 1), (1, 2), (2, 3), (3, 0), (4, 5)),
                   images=((1, 0, 0), ZERO, (-1, 0, 0), ZERO, ZERO))


def test_crossing_cell_ring_is_whole_and_nearby_bond_pierces_its_lift():
    value = boundary_ring()
    before = raw(value)
    findings = query(value)
    surface = selected(findings, 'CRYSTAL-PERIODIC-BOND-RING')
    assert len(surface) == 1
    ring, bond = surface[0][1]
    assert tuple(key[0] for key in ring) == (0, 1, 2, 3)
    assert ring[1][2] == ring[2][2] == (1, 0, 0)
    assert bond[0][2] == bond[1][2] == (1, 0, 0)
    assert dict(surface[0][2])['state'] == 'pierces'
    assert dict(surface[0][2])['intersection_points'] == ((10., 5., 0.),)
    assert raw(value) == before


@pytest.mark.parametrize('bridge', (False, True))
def test_hopf_cycles_report_link_and_surface_interpenetration_without_repair(bridge):
    first = ((-2., -2., 0.), (2., -2., 0.), (2., 2., 0.), (-2., 2., 0.))
    second = ((1., 0., -1.), (3., 0., -1.), (3., 0., 1.), (1., 0., 1.))
    edges = tuple((start + index, start + (index + 1) % 4) for start in (0, 4) for index in range(4))
    if bridge:
        edges += ((0, 4),)
    value = crystal((*first, *second), edges=edges)
    before = raw(value)
    findings = query(value)
    links = selected(findings, 'CRYSTAL-CLOSED-LOOP-LINK')
    assert len(links) == 1 and dict(links[0][2])['linking_number'] == -1
    assert len(selected(findings, 'CRYSTAL-PERIODIC-BOND-RING')) == 2
    assert len(selected(findings, 'CRYSTAL-INTERPENETRATION')) == 2
    with pytest.warns(RuntimeWarning, match='CRYSTAL-CLOSED-LOOP-LINK:'):
        assert value.linking_number(tuple((row, ZERO) for row in range(4)),
                                    tuple((row, ZERO) for row in range(4, 8)), tolerance=1e-8) == -1
    assert raw(value) == before


def test_equal_source_row_at_another_image_is_not_an_incident_vertex():
    points = ((-2., -2., 0.), (2., -2., 0.), (2., 2., 0.), (-2., 2., 0.), (2., 2., -8.))
    edges = ((0, 1), (1, 2), (2, 3), (3, 0), (0, 4))
    value = crystal(points, edges=edges, matrix=np.diag((20., 20., 4.)))
    findings = query(value)
    matches = [item for item in selected(findings, 'CRYSTAL-PERIODIC-BOND-RING')
               if tuple(key[0] for key in item[1][1]) == (0, 4) and item[1][1][0][2] == (0, 0, 1)]
    assert len(matches) == 1
    assert dict(matches[0][2])['state'] == 'pierces'
    assert dict(matches[0][2])['intersection_points'] == ((0., 0., 0.),)


def test_symmetry_generated_bond_pierces_identity_ring_and_conversely():
    points = ((2., 2., 3.), (4., 2., 3.), (4., 4., 3.), (2., 4., 3.),
              (-3., -3., -4.), (-3., -3., -2.))
    value = crystal(points, edges=((0, 1), (1, 2), (2, 3), (3, 0), (4, 5)), matrix=np.eye(3) * 20, number=2)
    findings = query(value)
    matches = [item for item in selected(findings, 'CRYSTAL-PERIODIC-BOND-RING')
               if tuple(key[0] for key in item[1][1]) == (4, 5) and dict(item[2])['state'] == 'pierces']
    assert len(matches) == 2
    assert all(item[1][0][0][1] != item[1][1][0][1] for item in matches)
    intersections = sorted(dict(item[2])['intersection_points'][0] for item in matches)
    np.testing.assert_allclose(intersections, ((-3., -3., -3.), (3., 3., 3.)), rtol=0., atol=2e-14)


def test_auto_omits_shared_actual_vertex_but_explicit_link_reports_touch():
    points = ((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.), (-1., 0., 1.))
    edges = ((0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0))
    value = crystal(points, edges=edges)
    findings = query(value)
    assert not selected(findings, 'CRYSTAL-CLOSED-LOOP-LINK')
    assert not selected(findings, 'CRYSTAL-LINKING-UNAVAILABLE')
    before = raw(value)
    with pytest.warns(RuntimeWarning, match='CRYSTAL-LINKING-UNAVAILABLE:.*intercurve_edges_within_tolerance'):
        assert value.linking_number(tuple((row, ZERO) for row in (0, 1, 2)),
                                    tuple((row, ZERO) for row in (0, 3, 4)), tolerance=1e-8) is None
    assert raw(value) == before


def test_zero_voltage_ring_survives_inside_nonfinite_component():
    points = ((-2., -2., 0.), (2., -2., 0.), (2., 2., 0.), (-2., 2., 0.),
              (-3., -2., 1.), (-3., -3., 1.), (0., 0., -1.), (0., 0., 1.))
    edges = ((0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (4, 5), (5, 0), (6, 7))
    value = crystal(points, edges=edges, images=(ZERO,) * 6 + ((1, 0, 0), ZERO))
    findings = query(value)
    assert selected(findings, 'CRYSTAL-MOLECULAR-LIFT-UNAVAILABLE')
    assert any(dict(item[2])['reason'] == 'nonzero_cycle_voltage'
               for item in selected(findings, 'CRYSTAL-LINKING-UNAVAILABLE'))
    assert any(tuple(key[0] for key in item[1][0]) == (0, 1, 2, 3)
               and tuple(key[0] for key in item[1][1]) == (6, 7)
               and dict(item[2])['state'] == 'pierces'
               for item in selected(findings, 'CRYSTAL-PERIODIC-BOND-RING'))


@pytest.mark.parametrize('kind,reason', [('unknown', 'unknown_bond_image'), ('periodic', 'nonzero_cycle_voltage')])
def test_unavailable_ring_does_not_suppress_independent_contact(kind, reason):
    images = ((1, 0, 0), ZERO, (-1, 0, 0), ZERO, ZERO)
    if kind == 'unknown':
        images = (None, *images[1:])
    else:
        images = (*images[:3], (1, 0, 0), ZERO)
    value = boundary_ring()
    points = [tuple(atom.coordinates) for atom in value.molecule.atoms] + [(4., 4., 4.), (4.125, 4., 4.)]
    value = crystal(points, edges=((0, 1), (1, 2), (2, 3), (3, 0), (4, 5)), images=images,
                    numbers=(6, 6, 6, 6, 6, 6, 6, 8))
    findings = query(value)
    assert any(dict(item[2]).get('reason') == reason for item in selected(findings, 'CRYSTAL-LINKING-UNAVAILABLE'))
    assert selected(findings, 'CRYSTAL-CROWDING')
    assert not selected(findings, 'CRYSTAL-PERIODIC-BOND-RING')


def test_scope_and_ring_size_reuse_existing_selector_without_guessing():
    value = boundary_ring()
    assert selected(query(value, scope='ligand_skeleton'), 'CRYSTAL-PERIODIC-BOND-RING')
    assert not selected(query(value, max_size=3), 'CRYSTAL-PERIODIC-BOND-RING')
    with pytest.raises(ValueError, match='ring_scope'):
        query(value, scope='new_scope')


def test_nonplanar_surface_budget_retains_indeterminacy():
    angles = np.arange(9) * (2 * np.pi / 9)
    points = tuple((float(2 * np.cos(angle)), float(2 * np.sin(angle)), float(.1 * np.sin(2 * angle)))
                   for angle in angles) + ((0., 0., -1.), (0., 0., 1.))
    edges = tuple((index, (index + 1) % 9) for index in range(9)) + ((9, 10),)
    value = crystal(points, edges=edges)
    surface = selected(query(value), 'CRYSTAL-PERIODIC-BOND-RING')
    assert surface
    data = dict(surface[0][2])
    assert data['state'] == 'undetermined'
    assert data['indeterminacy_causes']
    assert dict(data['surface_evidence'])['enumeration_complete'] is False


def test_code_break_receives_the_complete_same_findings_and_never_mutates(monkeypatch):
    value = boundary_ring()
    expected = query(value)
    before = raw(value)
    seen = []
    emit = operations._emit_findings
    def observe(findings):
        seen.append(findings)
        emit(findings)
    monkeypatch.setattr(operations, '_emit_findings', observe)
    with warnings.catch_warnings():
        warnings.filterwarnings('error', message='CRYSTAL-PERIODIC-BOND-RING:')
        with pytest.raises(RuntimeWarning, match='CRYSTAL-PERIODIC-BOND-RING'):
            value.geometry_findings(coincidence_tolerance=1e-8, close_contact_cutoff=.5)
    assert seen == [expected] and raw(value) == before
    assert query(value) == expected


def test_finding_contents_are_immutable_values_and_toolkits_are_never_loaded(monkeypatch):
    import builtins
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in ('openbabel', 'rdkit', 'ccdc', 'gemmi'):
            raise AssertionError('forbidden Crystal geometry toolkit: ' + name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    findings = query(boundary_ring())
    def immutable(value):
        if isinstance(value, tuple):
            for item in value:
                immutable(item)
        else:
            assert value is None or isinstance(value, (str, bool, int, float, np.generic)) or hasattr(value, 'context')
    immutable(findings)


def test_identity_placement_keeps_exact_source_contact_boundary_without_cell_roundtrip():
    value = crystal(((.25, 0., 0.), (.375, 0., 0.)), (6, 8))
    findings = query(value, coincidence=.125, cutoff=.125)
    observed = selected(findings, 'CRYSTAL-COINCIDENT-IMAGES')
    assert len(observed) == 1 and dict(observed[0][2])['distance'] == .125
    assert not selected(findings, 'CRYSTAL-CROWDING')
