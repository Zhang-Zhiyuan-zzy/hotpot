"""Value contracts for the private conformer-provider seam."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


__all__ = (
    "ConformerBatch",
    "ConformerSamplingError",
    "ConformerSamplingRequest",
)


class ConformerSamplingError(RuntimeError):
    """Raised when a coordinate provider cannot produce one conformer."""


@dataclass(frozen=True, kw_only=True)
class ConformerSamplingRequest:
    """Backend-neutral request made at the force-field builder seam."""

    conformer_count: int = 1
    seed: int | None = None


@dataclass(frozen=True)
class ConformerBatch:
    """Independent coordinate sets in stable provider ordinal order."""

    coordinates: NDArray[np.float64]
    ordinals: tuple[int, ...]
