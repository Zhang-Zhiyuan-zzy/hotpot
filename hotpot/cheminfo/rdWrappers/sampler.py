"""Public sampler facade for the native RDKit component."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .contracts import (
    ConformerBatch,
    CoordinationEmbeddingConstraints,
    ETKDGOptions,
    RuntimeInfo,
)

if TYPE_CHECKING:
    from hotpot.cheminfo.core import Molecule


class ETKDGSampler:
    """Generate conformers through the pinned native RDKit backend."""

    def __init__(self, options: ETKDGOptions | None = None):
        self.options = options if options is not None else ETKDGOptions()

    def generate(
        self,
        mol: Molecule,
        *,
        constraints: CoordinationEmbeddingConstraints | None = None,
    ) -> ConformerBatch:
        from . import native

        return native.generate(mol, self.options, constraints)


def generate_conformers(
    mol: Molecule,
    *,
    options: ETKDGOptions | None = None,
    constraints: CoordinationEmbeddingConstraints | None = None,
) -> ConformerBatch:
    """Generate conformers without mutating the input molecule."""

    return ETKDGSampler(options).generate(mol, constraints=constraints)


def runtime_info() -> RuntimeInfo:
    """Return the identities compiled into the native component."""

    from . import native

    return native.runtime_info()
