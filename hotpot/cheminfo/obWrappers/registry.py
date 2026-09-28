"""Read-only access to the compiled Open Babel wrapper rule registry."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from .contracts import RuleDescriptor, RuleExecutionReport, RuleStage
from .native import _native_module, _native_molecule_data
from .reports import _execution_report, _rule_stage
from .settings import (
    TORSION_REPAIR_ANGLE_RADIANS,
    TORSION_SINGULARITY_THRESHOLD,
)


if TYPE_CHECKING:
    from ..core import Molecule


__all__ = ("available_rules", "inspect_rules")


def available_rules(
    stage: Optional[RuleStage] = None,
) -> tuple[RuleDescriptor, ...]:
    """Return registered rules in deterministic execution order."""
    native = _native_module()
    native_stage = None if stage is None else getattr(native.RuleStage, stage.name)
    return tuple(
        RuleDescriptor(
            rule_id=descriptor.rule_id,
            version=descriptor.version,
            stage=_rule_stage(descriptor.stage),
            priority=descriptor.priority,
        )
        for descriptor in native.available_rules(native_stage)
    )


def inspect_rules(
    mol: "Molecule",
    stage: RuleStage,
    *,
    singularity_threshold: float = TORSION_SINGULARITY_THRESHOLD,
    repair_angle_radians: float = TORSION_REPAIR_ANGLE_RADIANS,
) -> RuleExecutionReport:
    """Return native rule evidence without running a builder or force field."""
    native = _native_module()
    native_stage = getattr(native.RuleStage, stage.name)
    return _execution_report(
        native.inspect_rules(
            _native_molecule_data(mol),
            native_stage,
            singularity_threshold,
            repair_angle_radians,
        )
    )
