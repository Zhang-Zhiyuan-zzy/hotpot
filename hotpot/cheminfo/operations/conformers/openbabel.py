"""OpenBabel conformer preparation through the existing build operation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np

from .contracts import ConformerBatch, ConformerSamplingRequest

if TYPE_CHECKING:
    from .sampler import Molecule


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
