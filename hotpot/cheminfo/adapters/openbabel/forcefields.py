"""External Open Babel force-field and coordinate-build invocation."""

from __future__ import annotations

from ...obWrappers.builder import build
from ...obWrappers.forcefield import optimize
from ...obWrappers.native import _native_module, _native_molecule_data
from ...obWrappers.operation import single_optimize, _single_optimize_data


__all__ = ("build", "single_optimize", "optimize")


def _seed_openbabel_random(seed: int) -> None:
    """Seed the version-matched native Open Babel random generators."""
    _native_module().seed_random(seed)
