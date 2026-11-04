"""Independent analytic witnesses for the finite polygonal linking candidate.

The first square is CCW in z=0. Its spanning disk has normal +z. The
second square crosses its interior downward at x=1; its upward crossing
at x=3 is outside. Thus the oriented intersection number, and linking
under (p-q).(dp x dq)/(4*pi), is -1. Reversing one curve gives +1.
Zero fixtures below are separated by a plane; no zero result is used as
an unlink certificate for arbitrary curves.
"""

from copy import deepcopy
import json
from pathlib import Path
from math import cos, inf, nextafter, sin

import pytest

from hotpot.cheminfo.rules.mathematics.geometry.linking import _polygon_linking


FIRST = ((-2., -2., 0.), (2., -2., 0.), (2., 2., 0.), (-2., 2., 0.))
SECOND = ((1., 0., -1.), (3., 0., -1.), (3., 0., 1.), (1., 0., 1.))
SQUARE = ((0., 0., 0.), (2., 0., 0.), (2., 2., 0.), (0., 2., 0.))


def transform(points, function):
    return tuple(tuple(function(*point)) for point in points)


def shift(points, offset):
    return transform(points, lambda x, y, z: (x + offset[0], y + offset[1], z + offset[2]))


def subdivide(points):
    result = []
    for first, last in zip(points, points[1:] + points[:1]):
        result.extend((first, tuple((a + b) / 2 for a, b in zip(first, last))))
    return tuple(result)


def assert_resolved(first, second, expected, tolerance=1e-10):
    value, measurements = _polygon_linking(first, second, tolerance=tolerance)
    assert value == expected
    assert type(value) is int
    assert type(measurements) is tuple
    data = dict(measurements)
    assert data['gauss_lower'] <= expected <= data['gauss_upper']
    assert abs(data['gauss_estimate'] - expected) <= data['gauss_error_bound']
    assert data['gauss_upper'] - data['gauss_lower'] < 1e-12
    return measurements


def assert_unavailable(first, second, reason, tolerance=1e-10):
    value, measurements = _polygon_linking(first, second, tolerance=tolerance)
    assert value is None
    assert type(measurements) is tuple
    assert dict(measurements)['reason'] == reason
    return dict(measurements)


@pytest.mark.parametrize('reverse_first,reverse_second,swap,expected', [
    (False, False, False, -1), (True, False, False, 1),
    (False, True, False, 1), (True, True, False, -1),
    (False, False, True, -1), (True, False, True, 1),
    (False, True, True, 1), (True, True, True, -1),
])
def test_spanning_disk_hopf_orientation(reverse_first, reverse_second, swap, expected):
    first = FIRST[::-1] if reverse_first else FIRST
    second = SECOND[::-1] if reverse_second else SECOND
    assert_resolved(*( (second, first) if swap else (first, second) ), expected)


@pytest.mark.parametrize('first_start,second_start', [(0, 1), (1, 0), (2, 3), (3, 2)])
def test_cyclic_start_does_not_change_linking(first_start, second_start):
    assert_resolved(FIRST[first_start:] + FIRST[:first_start],
                    SECOND[second_start:] + SECOND[:second_start], -1)


@pytest.mark.parametrize('first,second', [
    (subdivide(FIRST), SECOND), (FIRST, subdivide(SECOND)),
    (subdivide(FIRST), subdivide(SECOND)),
])
def test_forward_collinear_subdivision(first, second):
    assert_resolved(first, second, -1)


def test_proper_rigid_transform():
    c1, s1, c2, s2 = cos(.37), sin(.37), cos(.63), sin(.63)
    def move(x, y, z):
        a, b = c1*x - s1*y, s1*x + c1*y
        return a + 10., c2*b - s2*z - 7., s2*b + c2*z + 4.
    assert_resolved(transform(FIRST, move), transform(SECOND, move), -1)


def test_improper_reflection_changes_orientation_sign():
    reflect = lambda x, y, z: (-x, y, z)
    assert_resolved(transform(FIRST, reflect), transform(SECOND, reflect), 1)


@pytest.mark.parametrize('scale,tolerance', [(1e-150, 1e-160), (1e150, 1e140)])
def test_uniform_scale_extremes_remain_supported(scale, tolerance):
    scale_point = lambda x, y, z: (scale*x, scale*y, scale*z)
    assert_resolved(transform(FIRST, scale_point), transform(SECOND, scale_point), -1, tolerance)


@pytest.mark.parametrize('second', [
    shift(SECOND, (10., 0., 0.)), shift(FIRST, (10., 0., 0.)),
    shift(FIRST, (0., 0., 3.)),
])
def test_separated_unlinked_zero_is_resolved_integer(second):
    assert_resolved(FIRST, second, 0)


@pytest.mark.parametrize('gap,available', [
    (nextafter(.125, -inf), False), (.125, False),
    (nextafter(.125, inf), True),
])
def test_exact_intercurve_tolerance_boundary(gap, available):
    second = shift(SQUARE, (0., 0., gap))
    if available:
        assert_resolved(SQUARE, second, 0, .125)
    else:
        assert_unavailable(SQUARE, second, 'intercurve_edges_within_tolerance', .125)


def test_intercurve_vertex_touch():
    touching = ((2., 2., 0.), (4., 2., 1.), (4., 4., 1.), (2., 4., 1.))
    assert_unavailable(FIRST, touching, 'intercurve_edges_within_tolerance')


def test_transverse_intercurve_segment_intersection():
    crossing = ((2., 0., -1.), (3., 0., -1.), (3., 0., 1.), (2., 0., 1.))
    assert_unavailable(FIRST, crossing, 'intercurve_edges_within_tolerance')


def test_coplanar_overlapping_segment_intersection():
    overlap = ((0., -2., 0.), (4., -2., 0.), (4., -4., 0.), (0., -4., 0.))
    assert_unavailable(FIRST, overlap, 'intercurve_edges_within_tolerance')


def test_nonadjacent_self_intersection():
    bowtie = ((-2., -2., 0.), (2., 2., 0.), (-2., 2., 0.), (2., -2., 0.))
    assert_unavailable(bowtie, shift(SECOND, (10., 0., 0.)), 'nonadjacent_edges_within_tolerance')


def test_nonadjacent_self_touch():
    touching = ((0., 0., 0.), (2., 0., 0.), (2., 2., 0.),
                (1., 0., 0.), (0., 2., 0.))
    assert_unavailable(touching, shift(SECOND, (10., 0., 0.)), 'nonadjacent_edges_within_tolerance')


@pytest.mark.parametrize('first,reason', [
    (FIRST[:2], 'fewer_than_three_vertices'),
    ((FIRST[0], FIRST[0], *FIRST[1:]), 'edge_length_within_tolerance'),
    ((*FIRST, FIRST[0]), 'edge_length_within_tolerance'),
    (((0., 0., 0.), (2., 0., 0.), (1., 0., 0.), (1., 2., 0.), (0., 2., 0.)), 'adjacent_edges_overlap'),
])
def test_degenerate_curves_are_unavailable(first, reason):
    assert_unavailable(first, shift(SECOND, (10., 0., 0.)), reason)


def test_edge_exactly_tolerance_is_unavailable():
    short = ((0., 0., 0.), (.125, 0., 0.), (1., 2., 0.), (0., 2., 0.))
    assert_unavailable(short, shift(SECOND, (10., 0., 0.)), 'edge_length_within_tolerance', .125)


def test_second_curve_failure_retains_curve_index():
    data = assert_unavailable(FIRST, SECOND[:2], 'fewer_than_three_vertices')
    assert data['curve_index'] == 1


@pytest.mark.parametrize('scale', [1e40, 1e200])
def test_extreme_anisotropic_hopf_remains_numerically_unresolved(scale):
    # Positive x stretching preserves the analytic -1 topology and separation.
    # The intentionally fixed interval precision cannot certify its angle sum.
    stretch = lambda x, y, z: (scale*x, y, z)
    data = assert_unavailable(transform(FIRST, stretch), transform(SECOND, stretch),
                              'numerical_interval_does_not_resolve_one_integer')
    assert data['gauss_lower'] <= -1 <= data['gauss_upper']
    assert data['gauss_lower'] < data['gauss_upper']


def test_input_nonmutation_and_recursive_measurement_immutability():
    first, second = [list(point) for point in FIRST], [list(point) for point in SECOND]
    before = deepcopy((first, second))
    measurements = assert_resolved(first, second, -1)
    assert (first, second) == before
    def immutable(value):
        assert type(value) in (tuple, str, int, float)
        if type(value) is tuple:
            for item in value:
                immutable(item)
    immutable(measurements)
    unavailable = assert_unavailable(FIRST, SECOND[:2], 'fewer_than_three_vertices')
    immutable(tuple(unavailable.items()))


def test_no_warning_is_emitted_by_pure_helper(recwarn):
    assert_resolved(FIRST, SECOND, -1)
    assert_unavailable(FIRST, SECOND[:2], 'fewer_than_three_vertices')
    assert len(recwarn) == 0


def whitehead_fixture():
    return json.loads((Path(__file__).parent / 'fixtures' / 'crystal_whitehead_braid.json').read_text())


def test_whitehead_fixture_tracks_published_braid_and_zero_expected_crossing_sum():
    # arXiv:math/9811128v2, printed p.17: (sigma1 sigma2^-1)^2 sigma2^-1.
    data = whitehead_fixture()
    assert data['braid_word'] == [1, -2, 1, -2, -2]
    assert data['strand_cycles'] == [[0, 2], [1]]
    mixed = [crossing['sign'] for crossing in data['crossings']
             if len(set(crossing['component_indices'])) == 2]
    assert mixed == [1, 1, -1, -1]
    assert sum(mixed) / 2 == data['expected_linking'] == 0
    assert [len(points) for points in data['coordinates']] == [30, 15]
    assert all(points[0] != points[-1] for points in data['coordinates'])


@pytest.mark.parametrize('variant', ['original', 'reverse_first', 'swap', 'cyclic_shift', 'subdivide', 'rigid'])
def test_whitehead_zero_does_not_certify_unlinking(variant):
    first, second = (tuple(tuple(point) for point in points)
                     for points in whitehead_fixture()['coordinates'])
    if variant == 'reverse_first':
        first = first[::-1]
    elif variant == 'swap':
        first, second = second, first
    elif variant == 'cyclic_shift':
        first, second = first[7:] + first[:7], second[4:] + second[:4]
    elif variant == 'subdivide':
        first, second = subdivide(first), subdivide(second)
    elif variant == 'rigid':
        proper = lambda x, y, z: (z + 11, x - 7, y + 3)
        first, second = transform(first, proper), transform(second, proper)
    assert_resolved(first, second, 0)
