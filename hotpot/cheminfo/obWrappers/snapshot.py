"""Open Babel SWIG object snapshots and native-plan translation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from openbabel import openbabel as ob

from .contracts import (
    CoordinateChange,
    HybridizationChange,
    RuleApplication,
    RuleDescriptor,
    RuleExecutionReport,
    RuleStage,
)
from .native import _native_module


if TYPE_CHECKING:
    from . import _ob_rules


__all__ = ()


def _native_stage(stage: RuleStage) -> "_ob_rules.RuleStage":
    native = _native_module()
    if stage is RuleStage.PRE_BUILD:
        return native.RuleStage.PRE_BUILD
    return native.RuleStage.PRE_FORCEFIELD_SETUP


def _rule_stage(stage: "_ob_rules.RuleStage") -> RuleStage:
    native = _native_module()
    if stage == native.RuleStage.PRE_BUILD:
        return RuleStage.PRE_BUILD
    return RuleStage.PRE_FORCEFIELD_SETUP


def _atom_snapshots(obmol: ob.OBMol) -> list["_ob_rules.AtomSnapshot"]:
    native = _native_module()
    return [
        native.AtomSnapshot(
            atom.GetAtomicNum(),
            atom.GetFormalCharge(),
            atom.GetHyb(),
            atom.IsMetal(),
        )
        for atom in ob.OBMolAtomIter(obmol)
    ]


def _bond_snapshots(obmol: ob.OBMol) -> list["_ob_rules.BondSnapshot"]:
    native = _native_module()
    return [
        native.BondSnapshot(
            bond.GetBeginAtomIdx() - 1,
            bond.GetEndAtomIdx() - 1,
            bond.GetBondOrder(),
            bond.IsAromatic(),
        )
        for bond in ob.OBMolBondIter(obmol)
    ]


def _coordinate_snapshot(
    obmol: ob.OBMol,
) -> list[tuple[float, float, float]]:
    return [
        (atom.GetX(), atom.GetY(), atom.GetZ())
        for atom in ob.OBMolAtomIter(obmol)
    ]


def _build_plan(obmol: ob.OBMol) -> "_ob_rules.RulePlan":
    return _native_module().plan_build(
        _atom_snapshots(obmol),
        _bond_snapshots(obmol),
    )


def _optimization_plan(
    obmol: ob.OBMol,
    *,
    singularity_threshold: float,
    repair_angle_radians: float,
) -> "_ob_rules.RulePlan":
    return _native_module().plan_optimization(
        _atom_snapshots(obmol),
        _bond_snapshots(obmol),
        _coordinate_snapshot(obmol),
        singularity_threshold,
        repair_angle_radians,
    )


def _execution_report(plan: "_ob_rules.RulePlan") -> RuleExecutionReport:
    applications = []
    for native_application in plan.applications:
        descriptor = RuleDescriptor(
            rule_id=native_application.rule_id,
            version=native_application.version,
            stage=_rule_stage(native_application.stage),
            priority=native_application.priority,
        )
        applications.append(
            RuleApplication(
                descriptor=descriptor,
                atom_indices=tuple(native_application.atom_indices),
                metric_before=native_application.metric_before,
                hybridization_changes=tuple(
                    HybridizationChange(
                        atom_index=change.atom_index,
                        before=change.before,
                        after=change.after,
                    )
                    for change in native_application.hybridization_changes
                ),
                coordinate_changes=tuple(
                    CoordinateChange(
                        atom_index=change.atom_index,
                        before=tuple(change.before),
                        after=tuple(change.after),
                    )
                    for change in native_application.coordinate_changes
                ),
            )
        )
    return RuleExecutionReport(
        stage=_rule_stage(plan.stage),
        applications=tuple(applications),
    )


def _apply_hybridization_changes(
    obmol: ob.OBMol,
    report: RuleExecutionReport,
) -> None:
    for application in report.applications:
        for change in application.hybridization_changes:
            obmol.GetAtom(change.atom_index + 1).SetHyb(change.after)


def _restore_hybridization_changes(
    obmol: ob.OBMol,
    report: RuleExecutionReport,
) -> None:
    for application in reversed(report.applications):
        for change in reversed(application.hybridization_changes):
            obmol.GetAtom(change.atom_index + 1).SetHyb(change.before)


def _apply_coordinate_changes(
    obmol: ob.OBMol,
    report: RuleExecutionReport,
) -> None:
    for application in report.applications:
        for change in application.coordinate_changes:
            obmol.GetAtom(change.atom_index + 1).SetVector(*change.after)
