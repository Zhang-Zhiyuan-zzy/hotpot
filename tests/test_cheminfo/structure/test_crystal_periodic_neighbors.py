"""Frozen analytic cases plus an independent finite-cube brute-force oracle.

The oracle enumerates explicit image cubes and uses Cartesian vector norms. It
has no fractional wrapping, inverse-column search or spatial bins. Cube extents
are deliberately larger than analytic bounds for each fixed test basis.
"""

from copy import deepcopy
from itertools import product
from math import inf, nextafter

import numpy as np
import pytest

from hotpot.cheminfo.rules.mathematics.periodic import _periodic_pairs


IDENTITY = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
ORTHOGONAL = ((4., 0., 0.), (0., 5., 0.), (0., 0., 6.))
SKEW = ((1., 0., 0.), (8., 1., 0.), (0., 0., 3.))


def oracle(coordinates, matrix, cutoff, extent, include_self_images=False):
    points, basis = np.asarray(coordinates, dtype=float), np.asarray(matrix, dtype=float)
    images = np.array(tuple(product(range(-extent, extent + 1), repeat=3)), dtype=np.int64)
    translations = images @ basis
    rows = []
    for first in range(len(points)):
        for second in range(first if include_self_images else first + 1, len(points)):
            distances = np.linalg.norm(points[second] - points[first] + translations, axis=1)
            for index in np.flatnonzero(distances <= cutoff):
                image = tuple(map(int, images[index]))
                if first == second and image <= (0, 0, 0):
                    continue
                # The fixed cases have a strict unused outer cube shell.
                assert max(map(abs, image)) < extent
                rows.append((first, second, image, float(distances[index])))
    return tuple(sorted(rows, key=lambda row: row[:3]))


def assert_rows(actual, expected):
    assert tuple(row[:3] for row in actual) == tuple(row[:3] for row in expected)
    assert [row[3] for row in actual] == pytest.approx([row[3] for row in expected], rel=2e-15, abs=1e-14)
    assert tuple(row[:3] for row in actual) == tuple(sorted(row[:3] for row in actual))
    assert len({row[:3] for row in actual}) == len(actual)
    assert type(actual) is tuple
    for first, second, image, distance in actual:
        assert type(first) is type(second) is int
        assert type(image) is tuple and len(image) == 3
        assert all(type(value) is int for value in image)
        assert type(distance) is float


@pytest.mark.parametrize('self_images', [False, True])
def test_skew_nearest_image_outside_fractional_27(self_images):
    # Fractional separation (1/2,1/2,0) has Cartesian x=4.5. Images
    # (-4,0,0), (-5,0,0), (3,-1,0), (4,-1,0) are all sqrt(1/2) away.
    coordinates = ((0., 0., 0.), (4.5, .5, 0.))
    rows = _periodic_pairs(coordinates, SKEW, .75, include_self_images=self_images)
    expected = ((0, 1, (-5, 0, 0), 2**-.5), (0, 1, (-4, 0, 0), 2**-.5),
                (0, 1, (3, -1, 0), 2**-.5), (0, 1, (4, -1, 0), 2**-.5))
    assert_rows(rows, expected)
    # ||inverse[:,0]||=sqrt(65), so every possible image is inside this cube.
    assert_rows(rows, oracle(coordinates, SKEW, .75, 12, self_images))


@pytest.mark.parametrize('self_images', [False, True])
@pytest.mark.parametrize('seed', [1701, 1702, 1703])
def test_random_skew_cartesian_oracle(seed, self_images):
    rng = np.random.default_rng(seed)
    basis = np.array(((4., 0., 0.), (5., 3., 0.), (2., 1., 5.)))
    # Dyadic fractions make fixture conversion to Cartesian exact here.
    fractional = rng.integers(0, 64, size=(14, 3)) / 64
    coordinates = fractional @ basis
    # Inverse-column widths at r=2 are all <2, and fractional span is <1.
    assert_rows(_periodic_pairs(coordinates, basis, 2., include_self_images=self_images),
                oracle(coordinates, basis, 2., 4, self_images))


@pytest.mark.parametrize('self_images', [False, True])
def test_unwrapped_coordinates_restore_actual_translation(self_images):
    fractions = np.array(((.125, .25, .375), (.875, .25, .375), (.25, .75, .5)))
    carries = np.array(((4, -3, 2), (-5, 1, -2), (0, 3, -4)))
    basis = np.array(ORTHOGONAL)
    coordinates = (fractions + carries) @ basis
    # Each carry difference is <=9; local images are <=1 in magnitude.
    rows = _periodic_pairs(coordinates, basis, 2., include_self_images=self_images)
    assert_rows(rows, oracle(coordinates, basis, 2., 12, self_images))
    assert (0, 1, (8, -4, 4)) in tuple(row[:3] for row in rows)


def test_independent_lattice_carries_transform_images_exactly():
    points = np.array(((0., 0., 0.), (.5, .25, 0.), (3.75, .25, 0.)))
    carries = ((4, -3, 2), (-5, 1, -2), (0, 3, -4))
    baseline = _periodic_pairs(points, ORTHOGONAL, 1.)
    moved = points + np.array(carries) @ np.array(ORTHOGONAL)
    expected = tuple(sorted(((i, j, tuple(n[k] + carries[i][k] - carries[j][k] for k in range(3)), d)
                             for i, j, n, d in baseline), key=lambda row: row[:3]))
    assert_rows(_periodic_pairs(moved, ORTHOGONAL, 1.), expected)


def test_python_integer_image_beyond_int64():
    carry = 2**100
    rows = _periodic_pairs(((0., 0., 0.), (float(carry), 0., 0.)), IDENTITY, 0.)
    assert rows == ((0, 1, (-carry, 0, 0), 0.0),)


@pytest.mark.parametrize('cutoff,expected', [(nextafter(1., -inf), ()),
                                           (1., ((0, 1, (0, 0, 0), 1.),)),
                                           (nextafter(1., inf), ((0, 1, (0, 0, 0), 1.),))])
def test_cutoff_boundary_is_not_widened(cutoff, expected):
    assert_rows(_periodic_pairs(((0., 0., 0.), (1., 0., 0.)), ORTHOGONAL, cutoff), expected)


@pytest.mark.parametrize('cutoff,available', [(nextafter(.625, -inf), False), (.625, True),
                                            (nextafter(.625, inf), True)])
def test_pythagorean_exact_boundary(cutoff, available):
    rows = _periodic_pairs(((0., 0., 0.), (.375, .5, 0.)), ORTHOGONAL, cutoff)
    assert_rows(rows, ((0, 1, (0, 0, 0), .625),) if available else ())


def test_distance_just_outside_cutoff_not_accepted_by_float_norm_rounding():
    # sqrt(1+2**-60)>1, although a binary64 norm can round it to exactly 1.
    assert _periodic_pairs(((0., 0., 0.), (1., 2**-30, 0.)), ORTHOGONAL, 1.) == ()


@pytest.mark.parametrize('self_images', [False, True])
def test_duplicate_rows_are_retained(self_images):
    coordinates = ((0., 0., 0.), (0., 0., 0.))
    actual = _periodic_pairs(coordinates, IDENTITY, 1., include_self_images=self_images)
    assert_rows(actual, oracle(coordinates, IDENTITY, 1., 2, self_images))
    assert len(actual) == (13 if self_images else 7)


def test_duplicate_periodic_coordinates_at_zero_cutoff():
    rows = _periodic_pairs(((.5, .25, .125), (8.5, -4.75, 6.125)), ORTHOGONAL, 0.,
                           include_self_images=True)
    assert rows == ((0, 1, (-2, 1, -1), 0.),)


def test_self_images_keep_one_plus_minus_orientation():
    rows = _periodic_pairs(((.25, .5, .75),), IDENTITY, 1., include_self_images=True)
    assert rows == ((0, 0, (0, 0, 1), 1.), (0, 0, (0, 1, 0), 1.), (0, 0, (1, 0, 0), 1.))


def test_skew_self_images_include_far_integer_short_vector():
    rows = _periodic_pairs(((0., 0., 0.),), SKEW, 1., include_self_images=True)
    assert rows == ((0, 0, (1, 0, 0), 1.), (0, 0, (8, -1, 0), 1.))
    assert_rows(rows, oracle(((0., 0., 0.),), SKEW, 1., 12, True))


@pytest.mark.parametrize('self_images', [False, True])
def test_dense_oracle(self_images):
    coordinates = np.array(tuple(product((0., .125, .25), repeat=3)))
    rows = _periodic_pairs(coordinates, ORTHOGONAL, .5, include_self_images=self_images)
    assert_rows(rows, oracle(coordinates, ORTHOGONAL, .5, 2, self_images))
    assert len(rows) == len(coordinates)*(len(coordinates)-1)//2


@pytest.mark.parametrize('self_images', [False, True])
def test_sparse_oracle(self_images):
    basis = np.eye(3)*64
    coordinates = np.array(tuple(product((1., 9., 17., 25.), repeat=3)))
    rows = _periodic_pairs(coordinates, basis, .5, include_self_images=self_images)
    assert_rows(rows, oracle(coordinates, basis, .5, 2, self_images))
    assert rows == ()


@pytest.mark.parametrize('scale', [2.**-500, 2.**500])
def test_uniform_scale_extremes(scale):
    coordinates = np.array(((0., 0., 0.), (.375, .5, 0.)))*scale
    rows = _periodic_pairs(coordinates, np.array(ORTHOGONAL)*scale, .625*scale)
    assert rows == ((0, 1, (0, 0, 0), .625*scale),)


def test_proper_rigid_transform_and_negative_basis_orientation():
    points = np.array(((0., 0., 0.), (4.5, .5, 0.), (1., .25, 1.)))
    baseline = _periodic_pairs(points, SKEW, .9, include_self_images=True)
    rotation = np.array(((0., 1., 0.), (0., 0., 1.), (1., 0., 0.)))
    moved = points @ rotation + np.array((11., -7., 4.))
    assert_rows(_periodic_pairs(moved, np.array(SKEW) @ rotation, .9, include_self_images=True), baseline)
    reflection = np.diag((-1., 1., 1.))
    assert_rows(_periodic_pairs(points @ reflection, np.array(SKEW) @ reflection, .9,
                                include_self_images=True), baseline)


def test_row_permutation_remaps_orientation_and_images():
    points = np.array(((0., 0., 0.), (4.5, .5, 0.), (1., .25, 1.)))
    baseline = _periodic_pairs(points, SKEW, .9, include_self_images=True)
    order = (2, 0, 1)
    expected = []
    for first, second, image, distance in baseline:
        i, j = order.index(first), order.index(second)
        if i > j:
            i, j, image = j, i, tuple(-value for value in image)
        expected.append((i, j, image, distance))
    assert_rows(_periodic_pairs(points[list(order)], SKEW, .9, include_self_images=True),
                tuple(sorted(expected, key=lambda row: row[:3])))


def test_empty_and_single_rows():
    assert _periodic_pairs((), ORTHOGONAL, 1.) == ()
    assert _periodic_pairs((), ORTHOGONAL, 1., include_self_images=True) == ()
    assert _periodic_pairs(((0., 0., 0.),), ORTHOGONAL, 1.) == ()
    assert _periodic_pairs(((0., 0., 0.),), ORTHOGONAL, 0., include_self_images=True) == ()


def test_input_nonmutation_and_immutable_silent_result(recwarn):
    points = [[0., 0., 0.], [4.5, .5, 0.]]
    matrix = [list(row) for row in SKEW]
    before = deepcopy((points, matrix))
    result = _periodic_pairs(points, matrix, .75)
    assert (points, matrix) == before
    assert_rows(result, oracle(points, matrix, .75, 12))
    assert not recwarn
