"""Rule-aware delegation to the native Open Babel coordinate builder."""

from __future__ import annotations

from typing import Optional

from openbabel import openbabel as ob

from .contracts import BuildReport
from .snapshot import (
    _apply_hybridization_changes,
    _build_plan,
    _execution_report,
    _restore_hybridization_changes,
)


__all__ = ("build",)


def _run_native_builder(
    builder: ob.OBBuilder,
    obmol: ob.OBMol,
    stereo_warnings: Optional[bool],
) -> bool:
    if stereo_warnings is None:
        return bool(builder.Build(obmol))
    return bool(builder.Build(obmol, stereo_warnings))


def build(
    obmol: ob.OBMol,
    *,
    builder: Optional[ob.OBBuilder] = None,
    stereo_warnings: Optional[bool] = None,
) -> BuildReport:
    """Build coordinates through registered rules and native ``OBBuilder``.

    ``stereo_warnings=None`` preserves the native overload and its default.
    Rule mutations are temporary and are never retained as molecular metadata.
    """
    native_builder = ob.OBBuilder() if builder is None else builder
    rules = _execution_report(_build_plan(obmol))
    if not rules.applied:
        return BuildReport(
            succeeded=_run_native_builder(
                native_builder,
                obmol,
                stereo_warnings,
            ),
            rules=rules,
        )

    _apply_hybridization_changes(obmol, rules)
    try:
        succeeded = _run_native_builder(
            native_builder,
            obmol,
            stereo_warnings,
        )
    finally:
        _restore_hybridization_changes(obmol, rules)
    return BuildReport(succeeded=succeeded, rules=rules)
