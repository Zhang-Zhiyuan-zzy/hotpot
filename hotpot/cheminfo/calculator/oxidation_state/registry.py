"""Versioned element oxidation-state candidate-domain registry."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Tuple

from .contracts import ElementOxidationStateDomain

__all__ = ["element_oxidation_state_domain"]


_REGISTRY_PATH = Path(__file__).with_name("data") / "element_oxidation_states.json"


@lru_cache(maxsize=1)
def _load_registry() -> Tuple[str, str, str, Tuple[Tuple[int, Tuple[int, ...]], ...]]:
    raw_bytes = _REGISTRY_PATH.read_bytes()
    payload = json.loads(raw_bytes)
    rows = tuple(
        (
            int(row["atomic_number"]),
            tuple(int(state["value"]) for state in row["states"]),
        )
        for row in payload["elements"]
    )
    if tuple(atomic_number for atomic_number, _ in rows) != tuple(range(1, 119)):
        raise ValueError("Oxidation-state registry must contain ordered Z=1..118")
    return (
        str(payload["registry_id"]),
        str(payload["registry_version"]),
        hashlib.sha256(raw_bytes).hexdigest(),
        rows,
    )


def element_oxidation_state_domain(
    atomic_number: int,
) -> ElementOxidationStateDomain:
    """Return the bounded, unranked registry domain for one element."""
    if not 1 <= atomic_number <= 118:
        raise ValueError("atomic_number must be between 1 and 118")
    source_identity, registry_version, content_digest, rows = _load_registry()
    registered_atomic_number, states = rows[atomic_number - 1]
    return ElementOxidationStateDomain(
        atomic_number=registered_atomic_number,
        states=states,
        source_identity=source_identity,
        registry_version=registry_version,
        content_digest=content_digest,
    )
