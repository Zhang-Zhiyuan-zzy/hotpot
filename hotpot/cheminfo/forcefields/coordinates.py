"""Cartesian-coordinate operations shared by force-field workflows."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from ..core import Molecule


__all__ = ("perturb",)


def _copy_coordinates(coordinates: np.ndarray) -> np.ndarray:
    """Return an independent floating-point Cartesian-coordinate array."""
    return np.asarray(coordinates, dtype=float).copy()


def _perturbed_coordinates(
    coordinates: np.ndarray,
    *,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    displacement = rng.normal(0.0, sigma, np.asarray(coordinates).shape)
    displacement = np.clip(displacement, -2.0 * sigma, 2.0 * sigma)
    return np.asarray(coordinates, dtype=float) + displacement


def perturb(
    mol: "Molecule",
    *,
    sigma: float = 0.5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Perturb current coordinates in place with a local random generator."""
    coordinates = _perturbed_coordinates(
        mol.coordinates,
        sigma=sigma,
        rng=np.random.default_rng(seed),
    )
    mol.coordinates = coordinates
    return coordinates
