"""Optional RingDecomposerLib oracle used by differential tests."""

from .oracle import (
    RDL_LIBRARY_ENV,
    RDLOracle,
    canonical_cycle,
    load_oracle_from_environment,
)

__all__ = (
    "RDL_LIBRARY_ENV",
    "RDLOracle",
    "canonical_cycle",
    "load_oracle_from_environment",
)
