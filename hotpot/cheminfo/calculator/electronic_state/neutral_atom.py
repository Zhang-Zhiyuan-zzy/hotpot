"""Versioned isolated-neutral-atom reference lookup."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from typing import Dict, Optional

from .contracts import NeutralAtomReference

__all__ = ["neutral_atom_reference"]


_REGISTRY_RESOURCE = "neutral_atom_references.json"


def _registry_payload() -> dict:
    resource = files(__package__).joinpath("data", _REGISTRY_RESOURCE)
    return json.loads(resource.read_text(encoding="utf-8"))


def _content_digest(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_REGISTRY = _registry_payload()


def _validate_registry(payload: dict) -> None:
    records = payload["records"]
    atomic_numbers = tuple(record["atomic_number"] for record in records)
    if atomic_numbers != tuple(range(1, 119)):
        raise RuntimeError(
            "Neutral-atom registry must contain ordered unique Z=1..118 records"
        )
    for record in records:
        if record["available"] and not (
            record["electron_count"] == record["atomic_number"]
            and isinstance(record["ground_configuration"], str)
            and bool(record["ground_configuration"])
            and type(record["two_s"]) is int
            and record["two_s"] >= 0
        ):
            raise RuntimeError(
                "Available neutral-atom references require complete reviewed fields"
            )


_validate_registry(_REGISTRY)
_REGISTRY_DIGEST = _content_digest(_REGISTRY)
_RECORDS: Dict[int, dict] = {
    int(record["atomic_number"]): record for record in _REGISTRY["records"]
}


def neutral_atom_reference(
    atomic_number: int,
) -> Optional[NeutralAtomReference]:
    """Return a reviewed isolated-neutral-atom reference when available."""
    if not 1 <= atomic_number <= 118:
        raise ValueError("atomic_number must be between 1 and 118")

    record = _RECORDS[atomic_number]
    if not record["available"]:
        return None
    return NeutralAtomReference(
        atomic_number=atomic_number,
        electron_count=record["electron_count"],
        ground_configuration=record["ground_configuration"],
        two_s=record["two_s"],
        source_identity=_REGISTRY["source"]["citation"],
        registry_version=_REGISTRY["registry_version"],
        content_digest=_REGISTRY_DIGEST,
    )
