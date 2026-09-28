"""Numerical settings for Open Babel wrapper rules."""

from __future__ import annotations


__all__ = (
    "TORSION_REPAIR_ANGLE_RADIANS",
    "TORSION_SINGULARITY_THRESHOLD",
)


# These are numerical guard thresholds, not chemical acceptance criteria.
TORSION_SINGULARITY_THRESHOLD = 1.0e-6
TORSION_REPAIR_ANGLE_RADIANS = 1.0e-3
