"""Coordinate predicates over immutable numerical rows."""

import numpy as np


__all__ = ()


def _has_distinct_rows(coordinates):
    if len(coordinates) < 2:
        return False
    reference = np.asarray(coordinates[0])
    return any(bool(np.any(np.asarray(row) != reference)) for row in coordinates[1:])
