"""Pure Cartesian-coordinate copy and displacement arithmetic."""

from __future__ import annotations

import numpy as np


__all__ = ()


def _copy_coordinates(coordinates: np.ndarray) -> np.ndarray:
    """Return an independent floating-point Cartesian-coordinate array."""
    return np.asarray(coordinates, dtype=float).copy()



def _perturbed_coordinates(
    coordinates: np.ndarray,
    *,
    displacement: np.ndarray,
    sigma: float,
) -> np.ndarray:
    displacement = np.clip(displacement, -2.0 * sigma, 2.0 * sigma)
    return np.asarray(coordinates, dtype=float) + displacement
