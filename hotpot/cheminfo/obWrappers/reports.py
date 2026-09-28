"""Translate native Open Babel values into stable Python contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .contracts import (
    CoordinateChange,
    HybridizationChange,
    RuleApplication,
    RuleDescriptor,
    RuleExecutionReport,
    RuleStage,
)


if TYPE_CHECKING:
    from . import _ob_native


__all__ = ()


def _rule_stage(native_stage: "_ob_native.RuleStage") -> RuleStage:
    return RuleStage[native_stage.name]


def _execution_report(
    native_plan: "_ob_native.RulePlan",
) -> RuleExecutionReport:
    applications = []
    for native_application in native_plan.applications:
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
        stage=_rule_stage(native_plan.stage),
        applications=tuple(applications),
    )
