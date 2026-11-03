"""Pure displacement arithmetic and operation-owned random scheduling."""

import numpy as np

from hotpot.cheminfo.operations.forcefields.coordinates import _perturbed_coordinates
from hotpot.cheminfo.rules.physics.forcefields.coordinates import (
    _copy_coordinates,
    _perturbed_coordinates as apply_displacement,
)


def test_explicit_displacement_clips_without_changing_caller_arrays():
    coordinates = np.arange(6).reshape(2, 3)
    displacement = np.array([[-3.0, -0.5, 0.0], [0.25, 0.5, 3.0]])
    original_coordinates = coordinates.copy()
    original_displacement = displacement.copy()

    result = apply_displacement(coordinates, displacement=displacement, sigma=0.5)

    np.testing.assert_array_equal(
        result, original_coordinates + [[-1.0, -0.5, 0.0], [0.25, 0.5, 1.0]],
    )
    np.testing.assert_array_equal(coordinates, original_coordinates)
    np.testing.assert_array_equal(displacement, original_displacement)
    assert not np.shares_memory(result, coordinates)
    copied = _copy_coordinates(coordinates)
    assert copied.dtype == np.dtype(float)
    copied[0, 0] = 99.0
    np.testing.assert_array_equal(coordinates, original_coordinates)


def test_operation_preserves_draw_shape_order_and_generator_state():
    coordinates = np.arange(12, dtype=float).reshape(4, 3)
    rng = np.random.default_rng(413)
    reference_rng = np.random.default_rng(413)
    expected_displacement = reference_rng.normal(0.0, 0.2, coordinates.shape)

    result = _perturbed_coordinates(coordinates, sigma=0.2, rng=rng)

    np.testing.assert_array_equal(
        result, coordinates + np.clip(expected_displacement, -0.4, 0.4),
    )
    np.testing.assert_array_equal(rng.normal(size=16), reference_rng.normal(size=16))
