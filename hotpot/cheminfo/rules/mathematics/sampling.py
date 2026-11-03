"""Spherical coordinates calculated from two explicit uniform draws."""

import numpy as np


__all__ = ()


def _point_on_sphere(radius, polar_draw, azimuthal_draw):
    theta = np.arccos(2 * polar_draw - 1)
    phi = 2 * np.pi * azimuthal_draw
    return (np.sin(theta) * np.cos(phi) * radius,
            np.sin(theta) * np.sin(phi) * radius,
            np.cos(theta) * radius)
