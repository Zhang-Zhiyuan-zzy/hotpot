"""Public value contracts for native RDKit conformer generation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray


__all__ = (
    "ConformerBatch",
    "ConformerRecord",
    "CoordinationEmbeddingConstraints",
    "ETKDGOptions",
    "GenerationStatus",
    "RuntimeInfo",
)


class GenerationStatus(IntEnum):
    """Stable outcome classification for one embedding request."""

    SUCCESS = 0
    PARTIAL = 1
    UNSUPPORTED = 2
    FAILED = 3


@dataclass(frozen=True, kw_only=True)
class ETKDGOptions:
    """Options accepted by the pinned ETKDGv3 sampler."""

    conformer_count: int = 1
    seed: int | None = None
    prune_rms_threshold: float | None = None
    use_random_coordinates: bool = False


@dataclass(frozen=True)
class CoordinationEmbeddingConstraints:
    """Optional Angstrom bounds for coordination-aware embedding experiments.

    ``coordination_pairs`` rows are directed, zero-based ``(donor, metal)``
    atom indices and must name donor-to-metal dative bonds already present in
    the molecule. ``donor_pair_indices`` rows are unordered distance pairs.
    """

    coordination_pairs: NDArray[np.int32]
    metal_donor_bounds: NDArray[np.float64]
    donor_pair_indices: NDArray[np.int32] | None = None
    donor_pair_bounds: NDArray[np.float64] | None = None


@dataclass(frozen=True)
class ConformerRecord:
    """Stable ordering metadata for one generated conformer."""

    ordinal: int
    engine_conformer_id: int


@dataclass(frozen=True)
class ConformerBatch:
    """Coordinates and diagnostics returned by a conformer sampler."""

    status: GenerationStatus
    coordinates: NDArray[np.float64]
    records: tuple[ConformerRecord, ...]
    engine: str
    engine_version: str
    diagnostic: str | None


@dataclass(frozen=True)
class RuntimeInfo:
    """Hotpot and compiled native-engine identities."""

    hotpot_version: str
    rdkit_version: str
    native: bool
