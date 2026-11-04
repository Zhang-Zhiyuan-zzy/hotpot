"""Independent analytic contracts for the production periodic arithmetic."""

from fractions import Fraction
from itertools import product
from math import prod

import numpy as np
import pytest


from hotpot.cheminfo.rules.mathematics import periodic

IDENTITY = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
QUARTER_TURN = ((0, -1, 0), (1, 0, 0), (0, 0, 1))
INVERSION = ((-1, 0, 0), (0, -1, 0), (0, 0, -1))


def _matvec(matrix, vector):
    return tuple(sum(int(value) * int(item) for value, item in zip(row, vector)) for row in matrix)


def _matmul(first, second):
    return tuple(tuple(sum(first[row][middle] * second[middle][column] for middle in range(3))
                       for column in range(3)) for row in range(3))


@pytest.mark.parametrize(
    ("value", "expected", "shift"),
    (
        (-1.2, .8, -2), (-.2, .8, -1), (.2, .2, 0), (1.2, .2, 1),
        (-5e-13, 0., 0), (5e-13, 0., 0),
        (1. - 5e-13, 0., 1), (1. + 5e-13, 0., 1),
        (-1. - 5e-13, 0., -1), (-1. + 5e-13, 0., -1),
        (-2e-12, 1. - 2e-12, -1), (2e-12, 2e-12, 0),
        (1. - 2e-12, 1. - 2e-12, 0), (1. + 2e-12, 2e-12, 1),
        (-1. - 2e-12, 1. - 2e-12, -2), (-1. + 2e-12, 2e-12, -1),
    ),
)
def test_wrap_positive_negative_and_near_integer_boundaries(value, expected, shift):
    coordinates = np.array(((value, value, value),))
    wrapped, shifts = periodic._wrap_fractional(coordinates)
    assert shifts == ((shift, shift, shift),)
    np.testing.assert_allclose(wrapped, expected, rtol=0., atol=3e-16)
    if expected == 0:
        np.testing.assert_array_equal(wrapped, 0.)
    assert np.all((0 <= wrapped) & (wrapped < 1))


def test_wrap_inclusive_absolute_tolerance_and_the_next_float_outside():
    outside = np.nextafter(1e-12, np.inf)
    coordinates = np.array(((-1e-12, 0., 1e-12), (-outside, outside, 0.)))
    wrapped, shifts = periodic._wrap_fractional(coordinates)
    np.testing.assert_array_equal(wrapped[0], (0., 0., 0.))
    assert shifts == ((0, 0, 0), (-1, 0, 0))
    assert wrapped[1, 0] == 1 - outside
    assert wrapped[1, 1] == outside


def test_wrap_does_not_use_a_relative_tolerance_for_large_integers():
    value = 100_000_000. + 1e-5
    wrapped, shifts = periodic._wrap_fractional(((value, -value, 0.),))
    assert shifts == ((100_000_000, -100_000_001, 0),)
    assert 5e-6 < wrapped[0, 0] < 2e-5
    assert wrapped[0, 1] > .9999


@pytest.mark.parametrize("dtype", (np.float32, np.float64, np.int64))
def test_wrap_independent_float64_array_and_exact_immutable_shifts(dtype):
    coordinates = np.array(((2, -3, 5), (1, 0, -1)), dtype=dtype)
    original = coordinates.copy()
    coordinates.flags.writeable = False
    wrapped, shifts = periodic._wrap_fractional(coordinates)
    assert wrapped.dtype == np.float64
    assert not np.shares_memory(wrapped, coordinates)
    assert all(type(item) is int for row in shifts for item in row)
    assert isinstance(shifts, tuple) and all(isinstance(row, tuple) for row in shifts)
    wrapped[0, 0] = .25
    np.testing.assert_array_equal(coordinates, original)


def test_wrap_reconstruction_idempotence_and_huge_exact_integer_shifts():
    coordinates = np.array(((2.25, -3.5, 8.75), (float(2**90), -float(2**100), .125)))
    wrapped, shifts = periodic._wrap_fractional(coordinates)
    np.testing.assert_array_equal(wrapped + np.asarray(shifts, dtype=float), coordinates)
    assert shifts[1] == (2**90, -(2**100), 0)
    again, second_shifts = periodic._wrap_fractional(wrapped)
    np.testing.assert_array_equal(again, wrapped)
    assert second_shifts == ((0, 0, 0), (0, 0, 0))


def test_wrap_empty_rows_preserve_shape_without_aliasing():
    coordinates = np.empty((0, 3))
    wrapped, shifts = periodic._wrap_fractional(coordinates)
    assert wrapped.shape == (0, 3) and wrapped.dtype == np.float64
    assert wrapped is not coordinates and shifts == ()


@pytest.mark.parametrize("rotation", (IDENTITY, QUARTER_TURN, INVERSION))
def test_wrapped_image_matches_independent_fractional_displacement(rotation):
    fractional = np.array(((1.25, -.5, 2.75), (-2.5, 3.25, -.75)))
    image = (3, -4, 7)
    transformed = fractional @ np.asarray(rotation).T + np.array((.25, -.5, .75))
    wrapped, shifts = periodic._wrap_fractional(transformed)
    output = periodic._image_after_wrap(rotation, image, shifts[0], shifts[1])
    expected = (fractional[1] + image - fractional[0]) @ np.asarray(rotation).T
    np.testing.assert_array_equal(wrapped[1] + output - wrapped[0], expected)
    reversed_image = periodic._image_after_wrap(rotation, tuple(-value for value in image), shifts[1], shifts[0])
    assert reversed_image == tuple(-value for value in output)


@pytest.mark.parametrize("rotation", (IDENTITY, QUARTER_TURN, INVERSION))
def test_image_inverse_closure_with_large_python_integer_arithmetic(rotation):
    image = (2**95 + 3, -(2**100) - 5, 2**91 - 7)
    first_shift = (2**80, -(2**79), 4)
    second_shift = (-(2**82), 2**78, -3)
    transformed = periodic._image_after_wrap(rotation, image, first_shift, second_shift)
    inverse = tuple(zip(*rotation))
    back_first = tuple(-value for value in _matvec(inverse, first_shift))
    back_second = tuple(-value for value in _matvec(inverse, second_shift))
    assert periodic._image_after_wrap(inverse, transformed, back_first, back_second) == image
    assert all(type(value) is int for value in transformed)


def test_image_composition_and_numpy_int64_inputs_do_not_overflow():
    maximum = np.iinfo(np.int64).max
    image = np.array((maximum, -maximum, maximum), dtype=np.int64)
    first_shift = np.array((-maximum, maximum, 0), dtype=np.int64)
    second_shift = np.array((maximum, 0, maximum), dtype=np.int64)
    intermediate = periodic._image_after_wrap(np.asarray(QUARTER_TURN, dtype=np.int64), image, first_shift, second_shift)
    later_first = (4, -5, 6)
    later_second = (-1, 2, -3)
    composed = periodic._image_after_wrap(INVERSION, intermediate, later_first, later_second)
    total_first = tuple(a + b for a, b in zip(_matvec(INVERSION, first_shift), later_first))
    total_second = tuple(a + b for a, b in zip(_matvec(INVERSION, second_shift), later_second))
    assert composed == periodic._image_after_wrap(_matmul(INVERSION, QUARTER_TURN), image, total_first, total_second)
    assert intermediate[0] == 3 * int(maximum)


@pytest.mark.parametrize(
    ("count", "endpoints", "images", "expected"),
    (
        (0, (), (), ()),
        (3, (), (), (((0,), ((0, 0, 0),), (), ()), ((1,), ((0, 0, 0),), (), ()), ((2,), ((0, 0, 0),), (), ()))),
        (5, ((0, 3), (4, 1)), ((1, -2, 3), (0, 2, 0)),
         (((0, 3), ((0, 0, 0), (1, -2, 3)), (), ()), ((1, 4), ((0, 0, 0), (0, -2, 0)), (), ()), ((2,), ((0, 0, 0),), (), ()))),
        (3, ((2, 0), (1, 2)), ((3, -1, 2), (5, 0, -4)),
         (((0, 1, 2), ((0, 0, 0), (-8, 1, 2), (-3, 1, -2)), (), ()),)),
        (3, ((0, 1), (0, 2), (1, 2)), ((1, 0, 0), (1, 0, 0), (0, 0, 0)),
         (((0, 1, 2), ((0, 0, 0), (1, 0, 0), (1, 0, 0)), (), ()),)),
        (3, ((0, 1), (0, 2), (1, 2)), ((1, 0, 0), (1, 0, 0), (1, 0, 0)),
         (((0, 1, 2), None, (), (2,)),)),
        (3, ((0, 1), (1, 2)), ((1, 0, 0), None),
         (((0, 1, 2), None, (1,), ()),)),
        (3, ((0, 1), (0, 2), (1, 2)), ((0, 0, 0), (0, 0, 0), None),
         (((0, 1, 2), None, (2,), ()),)),
        (4, ((0, 1), (1, 2), (1, 3), (2, 3)), (None, (0, 0, 0), (0, 0, 0), (1, 0, 0)),
         (((0, 1, 2, 3), None, (0,), (3,)),)),
    ),
)
def test_components_use_actual_directed_edges_and_report_unknown_or_nonzero_voltage(count, endpoints, images, expected):
    assert periodic._finite_component_lifts(count, endpoints, images) == expected


def test_finite_crossing_ring_closes_whole_unit_without_per_atom_wrapping():
    endpoints = ((0, 1), (0, 2), (1, 2))
    images = ((1, 0, 0), (1, 0, 0), (0, 0, 0))
    fractional = np.array(((.9, 0., 0.), (.1, 0., 0.), (.2, .1, 0.)))
    ((rows, lifts, unknown, inconsistent),) = periodic._finite_component_lifts(3, endpoints, images)
    whole = fractional + np.asarray(lifts)
    assert rows == (0, 1, 2) and unknown == inconsistent == ()
    np.testing.assert_allclose(whole[:, 0], (.9, 1.1, 1.2), rtol=0., atol=1e-15)
    for (first, second), image in zip(endpoints, images):
        np.testing.assert_allclose(whole[second] - whole[first], fractional[second] + image - fractional[first], atol=1e-15)
    reverse = tuple((second, first) for first, second in endpoints)
    reverse_images = tuple(tuple(-value for value in image) for image in images)
    assert periodic._finite_component_lifts(3, reverse, reverse_images) == ((rows, lifts, unknown, inconsistent),)


def test_unknown_bridge_does_not_invent_absolute_lifts_between_known_subgraphs():
    result = periodic._finite_component_lifts(4, ((0, 1), (1, 2), (2, 3)),
                                             ((2**90, -7, 0), None, (-(2**100), 4, 3)))
    assert result == (((0, 1, 2, 3), None, (1,), ()),)


def test_integer_lifts_do_not_overflow_and_inputs_stay_immutable():
    maximum = np.iinfo(np.int64).max
    endpoints = np.array(((0, 1), (1, 2)), dtype=np.int64)
    images = np.array(((maximum, -maximum, 0), (maximum, -maximum, 1)), dtype=np.int64)
    original_endpoints, original_images = endpoints.copy(), images.copy()
    endpoints.flags.writeable = images.flags.writeable = False
    result = periodic._finite_component_lifts(3, endpoints, images)
    assert result == (((0, 1, 2), ((0, 0, 0), (int(maximum), -int(maximum), 0), (2 * int(maximum), -2 * int(maximum), 1)), (), ()),)
    assert all(type(value) is int for lift in result[0][1] for value in lift)
    np.testing.assert_array_equal(endpoints, original_endpoints)
    np.testing.assert_array_equal(images, original_images)


@pytest.mark.parametrize("repeats", ((2, 3, 2), (3, 1, 2), (1, 4, 1), (1, 1, 1)))
@pytest.mark.parametrize("image", ((1, -2, 3), (-7, 8, -5), (2**95 + 3, -(2**90) - 7, 2**80 + 5)))
def test_supercell_edges_preserve_exact_displacements_and_stable_source_copy_order(repeats, image):
    endpoints = ((0, 2), (2, 1))
    images = (image, tuple(-value + offset for value, offset in zip(image, (-1, 2, -3))))
    copies = tuple(product(*(range(repeat) for repeat in repeats)))
    copy_count = prod(repeats)
    result = periodic._supercell_edges(3, endpoints, images, repeats)
    assert len(result) == len(endpoints) * copy_count
    fractional = ((Fraction(1, 7), Fraction(2, 9), Fraction(3, 11)),
                  (Fraction(3, 8), Fraction(1, 5), Fraction(4, 13)),
                  (Fraction(2, 3), Fraction(5, 7), Fraction(1, 6)))
    for output_index, (new_first, new_second, new_image, source_edge, copy_image) in enumerate(result):
        assert source_edge == output_index // copy_count
        assert copy_image == copies[output_index % copy_count]
        first, second = endpoints[source_edge]
        assert new_first == first * copy_count + output_index % copy_count
        assert new_second // copy_count == second
        assert 0 <= new_first < 3 * copy_count and 0 <= new_second < 3 * copy_count
        target = copies[new_second % copy_count]
        for axis in range(3):
            source_displacement = fractional[second][axis] + images[source_edge][axis] - fractional[first][axis]
            output_displacement = repeats[axis] * (
                (fractional[second][axis] + target[axis]) / repeats[axis] + new_image[axis]
                - (fractional[first][axis] + copy_image[axis]) / repeats[axis])
            assert output_displacement == source_displacement
        assert all(type(value) is int for value in (new_first, new_second, *new_image, source_edge, *copy_image))


def test_supercell_exact_directed_boundary_fixture():
    assert periodic._supercell_edges(2, ((0, 1),), ((1, 0, 0),), (2, 1, 1)) == (
        (0, 3, (0, 0, 0), 0, (0, 0, 0)),
        (1, 2, (1, 0, 0), 0, (1, 0, 0)),
    )
    assert periodic._supercell_edges(2, ((1, 0),), ((-1, 0, 0),), (2, 1, 1)) == (
        (2, 1, (-1, 0, 0), 0, (0, 0, 0)),
        (3, 0, (0, 0, 0), 0, (1, 0, 0)),
    )


def test_supercell_directed_reversal_has_inverse_boundary_carries():
    repeats = (2, 3, 2)
    image = (2**85 + 7, -(2**90) - 3, 5)
    forward = periodic._supercell_edges(2, ((0, 1),), (image,), repeats)
    reverse = periodic._supercell_edges(2, ((1, 0),), (tuple(-value for value in image),), repeats)
    assert {(second, first, tuple(-value for value in carry)) for first, second, carry, _, _ in forward} == {
        (first, second, carry) for first, second, carry, _, _ in reverse
    }


def test_supercell_numpy_int64_image_addition_is_python_exact_and_immutable():
    maximum = np.iinfo(np.int64).max
    images = np.array(((maximum, -maximum, 0),), dtype=np.int64)
    endpoints = np.array(((0, 1),), dtype=np.int64)
    original = images.copy()
    images.flags.writeable = endpoints.flags.writeable = False
    result = periodic._supercell_edges(2, endpoints, images, (2, 3, 1))
    last = result[-1]
    assert last[2] == ((1 + int(maximum)) // 2, (2 - int(maximum)) // 3, 0)
    np.testing.assert_array_equal(images, original)
    np.testing.assert_array_equal(endpoints, ((0, 1),))


@pytest.mark.parametrize("atom_count", (0, 1, 5))
def test_supercell_of_edgeless_content_does_not_invent_connections(atom_count):
    assert periodic._supercell_edges(atom_count, (), (), (2, 3, 2)) == ()
