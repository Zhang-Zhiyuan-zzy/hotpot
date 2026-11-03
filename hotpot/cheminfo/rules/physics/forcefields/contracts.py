"""Validated numerical policy contracts for force-field optimizers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


__all__ = ("OptimizationStoppingCriteria",)


@dataclass(frozen=True)
class OptimizationStoppingCriteria:
    """Opt-in numerical thresholds for ending one optimizer segment."""

    window: int = 5
    maximum_energy_change_kj_mol: float = 1.0e-4
    maximum_atom_displacement_angstrom: float = 1.0e-4
    maximum_rms_gradient_kj_mol_angstrom: float = 1.0
    maximum_gradient_kj_mol_angstrom: float = 5.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.window, int)
            or isinstance(self.window, bool)
            or self.window < 1
        ):
            raise ValueError("window must be a positive integer")
        thresholds = (
            self.maximum_energy_change_kj_mol,
            self.maximum_atom_displacement_angstrom,
            self.maximum_rms_gradient_kj_mol_angstrom,
            self.maximum_gradient_kj_mol_angstrom,
        )
        if not all(
            np.isfinite(threshold) and threshold >= 0.0
            for threshold in thresholds
        ):
            raise ValueError(
                "optimization stopping thresholds must be finite and non-negative"
            )
