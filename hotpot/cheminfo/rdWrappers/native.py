"""Internal access to the compiled RDKit wrapper."""

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


def _backend():
    try:
        from . import _rd_native
    except ImportError as exc:
        raise RuntimeError("the native RDKit wrapper is not built") from exc
    return _rd_native


def generate(
    mol: Molecule,
    options: ETKDGOptions,
    constraints: CoordinationEmbeddingConstraints | None,
) -> ConformerBatch:
    """Dispatch one generation request to the compiled component."""

    return _backend().generate(mol, options, constraints)


def runtime_info() -> RuntimeInfo:
    """Read compiled dependency identities from the native component."""

    return _backend().runtime_info()
