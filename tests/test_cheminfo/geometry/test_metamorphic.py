from dataclasses import replace

import numpy as np
import pytest

from hotpot.cheminfo.geometry.object import Cycle, Segment
from hotpot.cheminfo.geometry.relation import determine_segment_cycle_relation
from hotpot.cheminfo.geometry.settings import DEFAULT_GEOMETRY_SETTINGS


def _transform_point(point, matrix, translation):
    return tuple(matrix @ np.asarray(point, dtype=float) + translation)


def test_piercing_state_is_invariant_under_rigid_motion():
    cycle_coordinates = [(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)]
    segment_coordinates = ((0.6, 0.8, -1), (0.6, 0.8, 1))
    angle = 0.73
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    translation = np.asarray((11.0, -7.0, 3.0))

    baseline = determine_segment_cycle_relation(
        Segment(*segment_coordinates), Cycle(cycle_coordinates)
    )
    transformed = determine_segment_cycle_relation(
        Segment(
            *(
                _transform_point(point, rotation, translation)
                for point in segment_coordinates
            )
        ),
        Cycle(
            _transform_point(point, rotation, translation)
            for point in cycle_coordinates
        ),
    )

    assert transformed.state is baseline.state
    assert transformed.features == baseline.features
    assert transformed.indeterminacy_causes == baseline.indeterminacy_causes


@pytest.mark.parametrize("scale", [1.0e-4, 1.0e4])
def test_state_is_scale_invariant_when_absolute_tolerance_scales(scale):
    cycle_coordinates = [(0, 0, 0), (2, 0, 0), (2, 2, 0.4), (0, 2, 0)]
    segment_coordinates = ((0.6, 0.8, -1), (0.6, 0.8, 1))
    baseline = determine_segment_cycle_relation(
        Segment(*segment_coordinates), Cycle(cycle_coordinates)
    )
    tolerance = replace(
        DEFAULT_GEOMETRY_SETTINGS.tolerance,
        absolute_length=(
            DEFAULT_GEOMETRY_SETTINGS.tolerance.absolute_length * scale
        ),
    )
    settings = replace(DEFAULT_GEOMETRY_SETTINGS, tolerance=tolerance)
    scaled = determine_segment_cycle_relation(
        Segment(
            *(tuple(scale * value for value in point) for point in segment_coordinates)
        ),
        Cycle(
            tuple(scale * value for value in point) for point in cycle_coordinates
        ),
        settings,
    )

    assert scaled.state is baseline.state
    assert scaled.features == baseline.features
    assert scaled.indeterminacy_causes == baseline.indeterminacy_causes
    assert scaled.surface_evidence == baseline.surface_evidence


def test_cycle_rotation_and_orientation_do_not_change_relation():
    coordinates = [(0, 0, 0), (2, 0, 0), (2, 2, 0.4), (0, 2, 0)]
    segment = Segment((0.6, 0.8, -1), (0.6, 0.8, 1))

    baseline = determine_segment_cycle_relation(segment, Cycle(coordinates))
    rotated = determine_segment_cycle_relation(
        segment, Cycle(coordinates[2:] + coordinates[:2])
    )
    reversed_cycle = determine_segment_cycle_relation(
        segment, Cycle(list(reversed(coordinates)))
    )

    assert rotated.state is baseline.state
    assert reversed_cycle.state is baseline.state
    assert rotated.surface_evidence == baseline.surface_evidence
    assert reversed_cycle.surface_evidence == baseline.surface_evidence
