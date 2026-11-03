"""Immutable scientific backend identity and projection boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from hotpot.cheminfo.rules.chemistry.scientific_state.contracts import (
    ScientificOperationRequirements,
)

__all__ = [
    'BackendIdentity',
    'BackendContext',
]


@dataclass(frozen=True)
class BackendIdentity:
    """Identity of one exact backend and projection context."""

    digest: str


class BackendContext(Protocol):
    """Minimal backend facts consumed by state projection."""

    @property
    def identity(self) -> BackendIdentity:
        """Return the exact backend-context identity."""

    @property
    def scientific_operation_requirements(self) -> ScientificOperationRequirements:
        """Return scientific quantities required by the requested operation."""

    @property
    def projection_convention(self) -> str:
        """Return the named scientific-to-backend projection convention."""

    @property
    def consumes_uhf(self) -> bool:
        """Return whether this exact operation consumes a UHF value."""

    @property
    def method_supported(self) -> bool:
        """Return whether the exact method is supported by this context."""

    def supports_atomic_number(self, atomic_number: int) -> bool:
        """Return whether the exact method supports one element."""

    def effective_nuclear_charge(self, atomic_number: int) -> int:
        """Return the electron-count contribution used by this backend context."""
