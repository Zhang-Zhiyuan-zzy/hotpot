"""Value contracts for the private conformer-provider seam."""

from __future__ import annotations

from dataclasses import dataclass

from ...structure.conformer import ConformerBatch


__all__ = (
    "ConformerBatch",
    "ConformerSamplingError",
    "ConformerSamplingRequest",
)


class ConformerSamplingError(RuntimeError):
    """Raised when a coordinate provider cannot produce one conformer."""


@dataclass(frozen=True, kw_only=True)
class ConformerSamplingRequest:
    """Request one placement at the force-field builder seam."""

    seed: int | None = None
