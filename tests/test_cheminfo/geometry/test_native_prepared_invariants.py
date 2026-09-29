import numpy as np
import pytest

from hotpot.cheminfo.geometry import _geometry_native, native
from hotpot.cheminfo.geometry.settings import DEFAULT_GEOMETRY_SETTINGS


def _square_coordinates() -> np.ndarray:
    return np.asarray(
        (
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 2.0, 0.0),
            (0.0, 2.0, 0.0),
        ),
        dtype=np.float64,
    )


def test_prepared_planar_cycle_is_factory_constructed_and_read_only():
    with pytest.raises(TypeError):
        _geometry_native.PreparedPlanarCycle()

    prepared = _geometry_native.prepare_planar_cycle(
        _square_coordinates(),
        native._native_tolerances(DEFAULT_GEOMETRY_SETTINGS),
    )
    with pytest.raises(AttributeError):
        prepared.simplicity = _geometry_native.PolygonSimplicity.UNDETERMINED


def test_prepared_nonplanar_family_is_factory_constructed_and_read_only():
    with pytest.raises(TypeError):
        _geometry_native.PreparedNonplanarSurfaceFamily()

    prepared = _geometry_native.prepare_nonplanar_surface_family(
        _square_coordinates(),
        native._native_tolerances(DEFAULT_GEOMETRY_SETTINGS),
        _geometry_native.SurfaceEnumerationLimits(8, 132, 792, 1980),
    )
    with pytest.raises(AttributeError):
        prepared.enumeration_complete = False
