"""Pickle-safe adapters for current conformer-generation backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING

import numpy as np

from .contracts import (
    ConformerBatch,
    ConformerSamplingError,
    ConformerSamplingRequest,
)


if TYPE_CHECKING:
    from ..core import Molecule


__all__ = ("OpenBabelConformerAdapter", "RDKitConformerAdapter")


@dataclass(frozen=True)
class OpenBabelConformerAdapter:
    """Adapt the existing single OBBuilder call to the private protocol."""

    _builder: Callable[["Molecule"], None] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def sample(
        self,
        molecule: "Molecule",
        request: ConformerSamplingRequest,
    ) -> ConformerBatch:
        del request
        builder = self._builder
        if builder is None:
            from ..forcefields.backend import _ob_build

            builder = _ob_build
        builder(molecule)
        coordinates = np.asarray(molecule.coordinates, dtype=float).copy()
        return ConformerBatch(coordinates[None, :, :], (0,))


@dataclass(frozen=True)
class RDKitConformerAdapter:
    """Adapt pinned native ETKDG to one deterministic workflow conformer."""

    def sample(
        self,
        molecule: "Molecule",
        request: ConformerSamplingRequest,
    ) -> ConformerBatch:
        from .. import rdWrappers

        try:
            batch = rdWrappers.generate_conformers(
                molecule,
                options=rdWrappers.ETKDGOptions(
                    conformer_count=1,
                    seed=request.seed,
                ),
            )
        except Exception as exc:
            raise ConformerSamplingError(
                f"RDKit conformer generation failed: {exc}"
            ) from exc
        if batch.status not in (
            rdWrappers.GenerationStatus.SUCCESS,
            rdWrappers.GenerationStatus.PARTIAL,
        ):
            detail = f": {batch.diagnostic}" if batch.diagnostic else ""
            raise ConformerSamplingError(
                "RDKit conformer generation returned "
                f"{batch.status.name}{detail}"
            )

        coordinates = np.asarray(batch.coordinates, dtype=float)
        if (
            coordinates.ndim != 3
            or coordinates.shape[1:] != (len(molecule.atoms), 3)
            or coordinates.shape[0] != len(batch.records)
            or coordinates.shape[0] == 0
            or not np.all(np.isfinite(coordinates))
        ):
            raise ConformerSamplingError(
                "RDKit conformer generation returned no valid mapped conformer"
            )

        selected_index, selected_record = min(
            enumerate(batch.records),
            key=lambda item: (item[1].ordinal, item[0]),
        )
        selected = coordinates[selected_index : selected_index + 1].copy()
        return ConformerBatch(selected, (selected_record.ordinal,))
