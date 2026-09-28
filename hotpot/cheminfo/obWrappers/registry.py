"""Read-only access to the compiled Open Babel wrapper rule registry."""

from __future__ import annotations

from typing import Optional

from .contracts import RuleDescriptor, RuleStage
from .native import _native_module
from .snapshot import _native_stage, _rule_stage


__all__ = ("available_rules",)


def available_rules(
    stage: Optional[RuleStage] = None,
) -> tuple[RuleDescriptor, ...]:
    """Return registered rules in deterministic execution order."""
    native_stage = None if stage is None else _native_stage(stage)
    return tuple(
        RuleDescriptor(
            rule_id=descriptor.rule_id,
            version=descriptor.version,
            stage=_rule_stage(descriptor.stage),
            priority=descriptor.priority,
        )
        for descriptor in _native_module().available_rules(native_stage)
    )
