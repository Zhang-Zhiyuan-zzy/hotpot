"""Actual Bond/image closure and public finite-loop control witnesses."""

from dataclasses import fields
import warnings

import numpy as np
import pytest

from hotpot import Crystal, Molecule, SpaceGroup


FIRST = ((-2., -2., 0.), (2., -2., 0.), (2., 2., 0.), (-2., 2., 0.))
SECOND = ((1., 0., -1.), (3., 0., -1.), (3., 0., 1.), (1., 0., 1.))
ZERO = (0, 0, 0)


def crystal(first=FIRST, second=SECOND, *, boundary=False, group=True, omit=None, unknown=None):
    value = Molecule()
    points = np.array((*first, *second))
    shifts = np.floor(points / 10).astype(int) if boundary else np.zeros(points.shape, dtype=int)
    for point in points - shifts * 10:
        value.create_atom(atomic_number=6, coordinates=point, formal_charge=0, implicit_hydrogens=0)
    images = []
    for start, count in ((0, len(first)), (len(first), len(second))):
        for local in range(count):
            a, b = start + local, start + (local + 1) % count
            if (a, b) == omit:
                continue
            value.add_bond(a, b, bond_order=1)
            images.append(None if (a, b) == unknown else tuple(map(int, shifts[b] - shifts[a])))
    result = Crystal(10, 10, 10, 90, 90, 90, molecule=value,
                     space_group=SpaceGroup(1) if group else None, bond_images=images)
    first_keys = tuple((row, tuple(map(int, image))) for row, image in enumerate(shifts[:len(first)]))
    second_keys = tuple((row + len(first), tuple(map(int, image))) for row, image in enumerate(shifts[len(first):]))
    return result, first_keys, second_keys


def raw(value):
    record, metadata = value.molecule._capture_factual_record()
    return tuple(getattr(record, field.name) for field in fields(record)), metadata


@pytest.mark.parametrize('boundary,group', [(False, True), (True, True), (False, False), (True, False)])
def test_actual_finite_boundary_hopf_retains_integer_and_source(boundary, group):
    value, first, second = crystal(boundary=boundary, group=group)
    before = raw(value)
    with pytest.warns(RuntimeWarning, match='CRYSTAL-CLOSED-LOOP-LINK'):
        assert value.linking_number(first, second, tolerance=1e-10) == -1
    assert raw(value) == before
    with pytest.warns(RuntimeWarning, match='CRYSTAL-CLOSED-LOOP-LINK'):
        assert value.linking_number(first[::-1], second, tolerance=1e-10) == 1
    assert raw(value) == before


def test_separated_unlinked_zero_is_silent_and_still_not_an_unlink_certificate():
    second = tuple((x + 10, y, z) for x, y, z in SECOND)
    value, first, second = crystal(second=second)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert value.linking_number(first, second, tolerance=1e-10) == 0
    assert not caught


@pytest.mark.parametrize('kind,reason', [
    ('short', 'fewer_than_three_vertices'), ('open', 'open_cycle'),
    ('unknown', 'unknown_bond_image'), ('voltage', 'nonzero_cycle_voltage'),
    ('intermediate', 'cycle_edge_image_mismatch'), ('touch', 'intercurve_edges_within_tolerance'),
])
def test_unavailable_is_local_and_never_fabricates_zero(kind, reason):
    kwargs = {'omit': (3, 0)} if kind == 'open' else {'unknown': (1, 2)} if kind == 'unknown' else {}
    if kind == 'touch':
        kwargs['second'] = ((2., 2., 0.), (4., 2., 1.), (4., 4., 1.), (2., 4., 1.))
    value, first, second = crystal(**kwargs)
    if kind == 'short':
        first = first[:2]
    elif kind == 'voltage':
        mol = value.molecule
        mol.create_crystal(10, 10, 10, 90, 90, 90, space_group=SpaceGroup(1),
                           bond_images=(*value.bond_images[:3], (1, 0, 0), *value.bond_images[4:]))
        value = mol.crystal
    elif kind == 'intermediate':
        first = (first[0], (first[1][0], (1, 0, 0)), *first[2:])
    before = raw(value)
    with pytest.warns(RuntimeWarning, match='CRYSTAL-LINKING-UNAVAILABLE.*' + reason):
        assert value.linking_number(first, second, tolerance=1e-10) is None
    assert raw(value) == before


def test_explicit_code_break_does_not_change_facts_or_a_later_observation():
    value, first, second = crystal(boundary=True)
    before = raw(value)
    with warnings.catch_warnings():
        warnings.filterwarnings('error', message='CRYSTAL-CLOSED-LOOP-LINK:')
        with pytest.raises(RuntimeWarning, match='CRYSTAL-CLOSED-LOOP-LINK'):
            value.linking_number(first, second, tolerance=1e-10)
    assert raw(value) == before
    with pytest.warns(RuntimeWarning, match='CRYSTAL-CLOSED-LOOP-LINK'):
        assert value.linking_number(first, second, tolerance=1e-10) == -1


def test_invalid_current_row_is_integrity_failure_and_not_a_link_observation():
    value, first, second = crystal()
    with pytest.raises(ValueError, match='CRYSTAL-IMAGE-MAP-INTEGRITY'):
        value.linking_number(((999, ZERO), *first[1:]), second, tolerance=1e-10)
