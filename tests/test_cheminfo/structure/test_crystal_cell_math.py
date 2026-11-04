"""Analytic oracles for the standard oriented row-cell contract."""

from math import acos, degrees, sqrt

import numpy as np
import pytest

from hotpot.cheminfo.rules.mathematics.cell import (
    _cell_matrix, _check_metric, _matrix_to_params, _replace_parameter,
    _to_cartesian, _to_fractional, _validate_matrix,
)


SKEW = np.array(((2., 0., 0.), (1., 3., 0.), (2., -1., 4.)))
# A fixed proper Cartesian rotation; every row is non-axis-aligned.
ORIENTATION = np.array(((1., 2., -2.), (2., 1., 2.), (2., -2., -1.))) / 3
PARAMETERS = (2., sqrt(10), sqrt(21), degrees(acos(-1 / sqrt(210))),
              degrees(acos(2 / sqrt(21))), degrees(acos(1 / sqrt(10))))


def test_validate_matrix_independent_float64_and_positive_volume():
    source = SKEW.astype(np.float32)
    matrix = _validate_matrix(source)
    assert matrix.shape == (3, 3) and matrix.dtype == np.dtype(np.float64)
    assert not np.shares_memory(source, matrix)
    np.testing.assert_array_equal(matrix, SKEW)
    assert np.linalg.det(matrix) == pytest.approx(24.)
    matrix[0, 0] = 8.
    np.testing.assert_array_equal(source, SKEW)
    assert np.linalg.det(_validate_matrix(np.diag((1e-4, 2e-4, 3e-4)))) > 0


@pytest.mark.parametrize('matrix', [
    np.zeros((3, 3)), np.diag((-1., 2., 3.)),
    ((1., 2., 3.), (2., 4., 6.), (0., 0., 1.)),
    np.array(((1., 0., 0.), (0., 1., 0.), (0., 0., np.nan))),
    np.array(((1., 0., 0.), (0., 1., 0.), (0., 0., np.inf))),
    np.ones((2, 3)), (1., 2., 3.),
])
def test_invalid_matrix_is_named_local_error(matrix):
    with pytest.raises(ValueError, match='CRYSTAL-CONTEXT-INTEGRITY'):
        _validate_matrix(matrix)
    with pytest.raises(ValueError, match='CRYSTAL-CONTEXT-INTEGRITY'):
        _matrix_to_params(matrix)


@pytest.mark.parametrize('parameters', [
    (0., 3., 4., 90., 90., 90.), (-2., 3., 4., 90., 90., 90.),
    (2., 3., np.inf, 90., 90., 90.), (2., 3., 4., np.nan, 90., 90.),
    (2., 3., 4., 0., 90., 90.), (2., 3., 4., 180., 90., 90.),
    (2., 3., 4., -10., 90., 90.), (2., 3., 4., 90., 90., 181.),
    (2., 3., 4., 120., 120., 120.), (2., 3., 4., 30., 60., 90.),
    (2., 3., 4., 170., 170., 170.), (2., 3., 4., 10., 10., 90.),
    (2., 3., 4., 90., 90.),
])
def test_invalid_physical_parameters_reject_without_repair(parameters):
    with pytest.raises(ValueError, match='CRYSTAL-CONTEXT-INTEGRITY'):
        _cell_matrix(parameters)


def test_standard_orthogonal_cell():
    matrix = _cell_matrix((2., 3., 4., 90., 90., 90.))
    np.testing.assert_allclose(matrix, np.diag((2., 3., 4.)), atol=1e-14)
    np.testing.assert_allclose(_matrix_to_params(matrix), (2., 3., 4., 90., 90., 90.), atol=1e-12)
    assert np.linalg.det(matrix) == pytest.approx(24.)


def test_independent_oblique_standard_angles_and_canonical_basis():
    matrix = _cell_matrix(PARAMETERS)
    np.testing.assert_allclose(matrix, SKEW, rtol=0, atol=1e-13)
    actual = _matrix_to_params(SKEW)
    assert isinstance(actual, tuple) and len(actual) == 6
    assert all(type(value) is float for value in actual)
    np.testing.assert_allclose(actual, PARAMETERS, rtol=0, atol=1e-13)
    assert actual[3] > 90 and actual[5] < 90  # alpha and gamma are distinct.
    assert matrix[0, 0] > 0 and matrix[1, 1] > 0 and matrix[2, 2] > 0
    assert np.linalg.det(matrix) == pytest.approx(24.)


def test_rotated_full_basis_retains_orientation_and_standard_parameters():
    rotated = SKEW @ ORIENTATION
    np.testing.assert_allclose(ORIENTATION @ ORIENTATION.T, np.eye(3), atol=1e-15)
    assert np.linalg.det(ORIENTATION) == pytest.approx(1.)
    np.testing.assert_allclose(_validate_matrix(rotated), rotated, rtol=0, atol=0)
    np.testing.assert_allclose(_matrix_to_params(rotated), PARAMETERS, rtol=0, atol=1e-13)
    assert np.linalg.det(rotated) == pytest.approx(24.)


@pytest.mark.parametrize('shape', [(3,), (2, 3), (2, 2, 3), (0, 3)])
def test_independent_row_coordinate_conversion_and_output_shapes(shape):
    fractional = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) / 4 - 2
    inverse = np.array(((.5, 0., 0.), (-1 / 6, 1 / 3, 0.), (-7 / 24, 1 / 12, .25)))
    expected_cartesian = fractional @ SKEW @ ORIENTATION
    np.testing.assert_allclose(_to_cartesian(fractional, SKEW @ ORIENTATION), expected_cartesian,
                               rtol=0, atol=1e-13)
    restored = _to_fractional(expected_cartesian, SKEW @ ORIENTATION)
    assert restored.shape == fractional.shape
    np.testing.assert_allclose(restored, expected_cartesian @ ORIENTATION.T @ inverse, rtol=0, atol=1e-13)
    np.testing.assert_allclose(restored, fractional, rtol=0, atol=1e-13)
    assert not np.shares_memory(restored, expected_cartesian)
    cartesian = _to_cartesian(fractional, SKEW)
    assert not np.shares_memory(cartesian, fractional)
    if fractional.size:
        cartesian.flat[0] += 10
        np.testing.assert_allclose(fractional, np.arange(np.prod(shape)).reshape(shape) / 4 - 2)


def test_known_skew_fractional_point_and_cartesian_inverse():
    fractional = np.array((.25, -2., 1.5))
    cartesian = np.array((1.5, -7.5, 6.))
    np.testing.assert_allclose(_to_cartesian(fractional, SKEW), cartesian, rtol=0, atol=1e-14)
    np.testing.assert_allclose(_to_fractional(cartesian, SKEW), fractional, rtol=0, atol=1e-14)


@pytest.mark.parametrize('index,value,expected', [
    (0, 4., ((4., 0., 0.), (1., 3., 0.), (2., -1., 4.))),
    (1, 2 * sqrt(10), ((2., 0., 0.), (2., 6., 0.), (2., -1., 4.))),
    (2, 2 * sqrt(21), ((2., 0., 0.), (1., 3., 0.), (4., -2., 8.))),
    (3, 90., ((2., 0., 0.), (1., 3., 0.), (2., -2 / 3, sqrt(149) / 3))),
    (4, 90., ((2., 0., 0.), (1., 3., 0.), (0., -1 / 3, sqrt(188) / 3))),
    (5, 90., ((2., 0., 0.), (0., sqrt(10), 0.), (2., -1 / sqrt(10), 13 / sqrt(10)))),
])
def test_scalar_replacement_retains_independently_known_rigid_orientation(index, value, expected):
    source = SKEW @ ORIENTATION
    before = source.copy()
    updated = _replace_parameter(source, index, value)
    np.testing.assert_allclose(updated, np.asarray(expected) @ ORIENTATION, rtol=0, atol=1e-12)
    wanted = list(PARAMETERS)
    wanted[index] = value
    np.testing.assert_allclose(_matrix_to_params(updated), wanted, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(source, before)
    assert not np.shares_memory(updated, source)
    assert np.linalg.det(updated) > 0


def test_cartesian_and_fractional_preservation_are_distinct_analytic_transforms():
    old = SKEW @ ORIENTATION
    new = _replace_parameter(old, 0, 4.)
    fractional = np.array((.25, -2., 1.5))
    cartesian = np.array((1.5, -7.5, 6.)) @ ORIENTATION
    changed_fractional = _to_fractional(cartesian, new)
    np.testing.assert_allclose(_to_cartesian(changed_fractional, new), cartesian, rtol=0, atol=1e-13)
    changed_cartesian = _to_cartesian(fractional, new)
    np.testing.assert_allclose(changed_cartesian, np.array((2., -7.5, 6.)) @ ORIENTATION, rtol=0, atol=1e-13)
    np.testing.assert_allclose(_to_fractional(changed_cartesian, new), fractional, rtol=0, atol=1e-13)
    assert not np.allclose(changed_cartesian, cartesian)


def test_invalid_replacement_preserves_supplied_basis():
    matrix = SKEW @ ORIENTATION
    before = matrix.copy()
    with pytest.raises(ValueError, match='CRYSTAL-CONTEXT-INTEGRITY'):
        _replace_parameter(matrix, 3, 180.)
    np.testing.assert_array_equal(matrix, before)


QUARTER_TURN = ((0, -1, 0), (1, 0, 0), (0, 0, 1))


def test_compatible_group_metrics_include_rotated_and_oblique_cells():
    tetragonal = np.diag((2., 2., 3.)) @ ORIENTATION
    before = tetragonal.copy()
    rotation = np.asarray(QUARTER_TURN)
    assert _check_metric(tetragonal, (np.eye(3), rotation, -np.eye(3))) is None
    np.testing.assert_array_equal(tetragonal, before)
    np.testing.assert_array_equal(rotation, QUARTER_TURN)
    hexagonal = np.array(((2., 0., 0.), (-1., sqrt(3), 0.), (0., 0., 4.))) @ ORIENTATION
    third_turn = ((0, -1, 0), (1, -1, 0), (0, 0, 1))
    assert _check_metric(hexagonal, (third_turn,)) is None
    assert _check_metric(SKEW, (-np.eye(3),)) is None


@pytest.mark.parametrize('matrix', [np.diag((2., 3., 4.)), SKEW, SKEW @ ORIENTATION])
def test_incompatible_fractional_rotation_rejects_against_actual_metric(matrix):
    with pytest.raises(ValueError, match='CRYSTAL-CONTEXT-INTEGRITY.*metric'):
        _check_metric(matrix, (QUARTER_TURN,))


def test_metric_tolerance_uses_relative_and_absolute_1e_minus_10():
    assert _check_metric(np.diag((sqrt(1 + 1e-10), 1., 2.)), (QUARTER_TURN,)) is None
    with pytest.raises(ValueError, match='CRYSTAL-CONTEXT-INTEGRITY.*metric'):
        _check_metric(np.diag((sqrt(1 + 5e-10), 1., 2.)), (QUARTER_TURN,))
