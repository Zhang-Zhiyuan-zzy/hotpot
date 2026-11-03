"""Canonical semantic bond categories shared by facts and chemical rules."""

from enum import Enum


__all__ = ("BondKind",)


class BondKind(str, Enum):
    """Semantic bond categories stored separately from numeric bond order."""

    SINGLE = "single"
    DOUBLE = "double"
    TRIPLE = "triple"
    AROMATIC = "aromatic"
    ZERO = "zero"
    DATIVE = "dative"
    UNKNOWN = "unknown"
