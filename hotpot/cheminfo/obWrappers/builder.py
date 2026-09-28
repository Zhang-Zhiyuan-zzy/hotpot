"""Hotpot-molecule facade for the native Open Babel coordinate builder."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from .contracts import BuildReport
from .native import _native_module, _native_molecule_data
from .reports import _execution_report


if TYPE_CHECKING:
    from ..core import Molecule


__all__ = ("build",)


def build(
    mol: "Molecule",
    *,
    stereo_warnings: Optional[bool] = None,
) -> BuildReport:
    """Build ``mol`` coordinates entirely in the native Open Babel backend."""
    result = _native_module().build(
        _native_molecule_data(mol),
        stereo_warnings,
    )
    if result.succeeded:
        mol.coordinates = result.coordinates
    return BuildReport(
        succeeded=result.succeeded,
        rules=_execution_report(result.rules),
    )
