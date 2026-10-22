from .contracts import (
    ConformerBatch,
    CoordinationEmbeddingConstraints,
    ETKDGOptions,
    RuntimeInfo,
)
from .._native.buffers import NativeMoleculeBuffers

C_API_VERSION: int
RDKIT_COMMIT: str


def generate(
    molecule_buffers: NativeMoleculeBuffers,
    options: ETKDGOptions,
    constraints: CoordinationEmbeddingConstraints | None = ...,
) -> ConformerBatch: ...


def runtime_info() -> RuntimeInfo: ...
