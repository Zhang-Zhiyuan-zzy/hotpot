"""Versioned element oxidation-state possible domains and operational defaults."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from importlib import resources
from typing import Tuple

from hotpot.cheminfo.rules.chemistry.scientific_state.oxidation_state.contracts import (
    ElementOxidationStateDomain,
)

__all__ = [
    'element_oxidation_state_domain',
]


_REGISTRY_RESOURCE = "data/element_oxidation_states.json"


@lru_cache(maxsize=1)
def _load_registry() -> Tuple[ElementOxidationStateDomain, ...]:
    raw_bytes = resources.files(
        "hotpot.cheminfo.rules.chemistry.scientific_state"
    ).joinpath(_REGISTRY_RESOURCE).read_bytes()
    payload = json.loads(raw_bytes)
    content_digest = hashlib.sha256(raw_bytes).hexdigest()
    rows = tuple(
        ElementOxidationStateDomain(
            atomic_number=int(row["atomic_number"]),
            states=tuple(int(state["value"]) for state in row["states"]),
            source_identity=str(payload["registry_id"]),
            registry_version=str(payload["registry_version"]),
            content_digest=content_digest,
            common_states=tuple(row["common_states"]),
            important_states=tuple(row["important_states"]),
            default_state=row["default_state"],
            default_basis=row["default_basis"],
        )
        for row in payload["elements"]
    )
    if tuple(row.atomic_number for row in rows) != tuple(range(1, 119)):
        raise ValueError("Oxidation-state registry must contain ordered Z=1..118")
    return rows


def element_oxidation_state_domain(
    atomic_number: int,
) -> ElementOxidationStateDomain:
    """Return unranked possible states and source-attributed element defaults."""
    if not 1 <= atomic_number <= 118:
        raise ValueError("atomic_number must be between 1 and 118")
    return _load_registry()[atomic_number - 1]
