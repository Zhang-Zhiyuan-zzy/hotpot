"""Structural protocol for private conformer providers."""

from __future__ import annotations

from typing import Protocol, TYPE_CHECKING

from .contracts import ConformerBatch, ConformerSamplingRequest


if TYPE_CHECKING:
    from ..core import Molecule


__all__ = ("ConformerSampler",)


class ConformerSampler(Protocol):
    """Generate coordinate candidates without owning downstream policy."""

    def sample(
        self,
        molecule: "Molecule",
        request: ConformerSamplingRequest,
    ) -> ConformerBatch: ...
