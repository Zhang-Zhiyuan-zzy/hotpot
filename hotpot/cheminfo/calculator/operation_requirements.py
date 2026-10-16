"""Registered and normalized scientific operation requirements."""

from __future__ import annotations

import hashlib
import json
from typing import Optional, Tuple

from .contracts import (
    ScientificOperationRequirements,
    ScientificQuantity,
    StatePolicy,
)

__all__ = []


_ALL_QUANTITIES = frozenset(ScientificQuantity)

_STRICT_ANALYSIS = ScientificOperationRequirements(
    operation_id="STRICT_ANALYSIS",
    required_resolved_quantities=frozenset(),
    allowed_unresolved_quantities=_ALL_QUANTITIES,
)

_COORDINATION_SCREENING = ScientificOperationRequirements(
    operation_id="COORDINATION_SCREENING_V1",
    required_resolved_quantities=frozenset(
        {
            ScientificQuantity.NET_CHARGE,
            ScientificQuantity.OXIDATION_STATE,
        }
    ),
    allowed_unresolved_quantities=frozenset(
        {
            ScientificQuantity.LEWIS_FORMAL_CHARGE,
            ScientificQuantity.COORDINATION_COUNTING,
            ScientificQuantity.ELECTRONIC_STATE,
        }
    ),
)


def _profile_requirements(
    policy: StatePolicy,
) -> ScientificOperationRequirements:
    if policy is StatePolicy.STRICT:
        return _STRICT_ANALYSIS
    return _COORDINATION_SCREENING


def _requirements_identity(
    inputs: Tuple[ScientificOperationRequirements, ...],
    required: frozenset[ScientificQuantity],
    allowed_unresolved: frozenset[ScientificQuantity],
) -> str:
    payload = {
        "input_ids": tuple(item.operation_id for item in inputs),
        "required": tuple(sorted(item.value for item in required)),
        "allowed_unresolved": tuple(
            sorted(item.value for item in allowed_unresolved)
        ),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _effective_requirements(
    policy: StatePolicy,
    caller: Optional[ScientificOperationRequirements],
    backend: Optional[ScientificOperationRequirements] = None,
) -> ScientificOperationRequirements:
    profile = _profile_requirements(policy)
    inputs = tuple(
        item for item in (profile, caller, backend) if item is not None
    )
    required = frozenset(
        quantity
        for item in inputs
        for quantity in item.required_resolved_quantities
    )
    allowed_unresolved = frozenset(
        quantity
        for item in inputs
        for quantity in item.allowed_unresolved_quantities
        if quantity not in required
    )
    identity = _requirements_identity(inputs, required, allowed_unresolved)
    return ScientificOperationRequirements(
        operation_id=f"EFFECTIVE:{identity}",
        required_resolved_quantities=required,
        allowed_unresolved_quantities=allowed_unresolved,
    )
