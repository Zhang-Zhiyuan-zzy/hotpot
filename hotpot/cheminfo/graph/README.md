# `hotpot.cheminfo.graph`

This package contains Hotpot's graph utilities and its native Relevant Cycle
perception API. The native cycle implementation accepts topology only: atom
types, bond orders, coordinates, aromaticity, and metal chemistry do not alter
its result.

## Public API

| Name | Purpose |
|---|---|
| `relevant_cycles` | Return all requested Relevant Cycles of a simple undirected graph. |
| `RelevantCycleLimitExceeded` | Report that a complete result would exceed `max_cycles`. |
| `DEFAULT_RELEVANT_CYCLE_LIMIT` | Default result limit, currently 10,000 cycles. |
| `linkmat2adj` | Convert an edge matrix to an adjacency matrix. |
| `adj2laplacian` | Convert an adjacency matrix to a graph Laplacian. |
| `calc_spectrum` | Calculate the existing Hotpot graph spectrum. |
| `GraphSpectrum` | Store and compare graph spectra. |
| `calc_electron_config` | Calculate the existing electron-configuration feature. |
| `atoms_electron_configurations` | Calculate electron-configuration features for atoms. |
| `graph_dfs_path` | Return one depth-first path under the existing traversal contract. |
| `graph_dfs_paths` | Preserve the existing multi-path traversal API. |

The package migration preserves the behavior of the former
`hotpot.cheminfo.graph` module. Relevant Cycle perception is the only new
capability.

## Relevant Cycles

```python
relevant_cycles(
    edges,
    *,
    max_size=None,
    max_cycles=10_000,
) -> tuple[tuple[int, ...], ...]
```

A simple cycle $C$ is relevant exactly when its edge-incidence vector is not in
the $\operatorname{GF}(2)$ span of cycles strictly shorter than $C$. Equivalently,
$C$ occurs in at least one minimum cycle basis. Relevant Cycles are therefore
not the same as one arbitrary cycle basis, SSSR, or the set of all simple
cycles.

The input is an iterable of two-integer undirected edges. Node indices may be
sparse but must be nonnegative. Self-loops and duplicate undirected edges are
rejected. Each returned tuple follows the cycle boundary without repeating its
first node. Rotation and direction are canonicalized, and the complete result
is sorted by cycle length and then lexicographically.

```python
from hotpot.cheminfo.graph import relevant_cycles

triangle = relevant_cycles([(4, 7), (7, 9), (9, 4)])
print(triangle)
# ((4, 7, 9),)

fused_squares = relevant_cycles(
    [(0, 1), (1, 2), (2, 3), (3, 0), (2, 4), (4, 5), (5, 3)]
)
print(fused_squares)
# ((0, 1, 2, 3), (2, 3, 5, 4))
```

`max_size=k` returns the exact subset of globally Relevant Cycles whose length
is at most `k`; it does not redefine relevance on a truncated graph. If the
number of requested cycles exceeds `max_cycles`, the function raises
`RelevantCycleLimitExceeded` instead of returning a partial collection. Use
`max_cycles=None` only when unbounded output is intentional because the number
of Relevant Cycles can be exponential.

## Implementation and scope

The public Python facade normalizes node labels and canonicalizes output. A
C++17/pybind11 core performs biconnected-component decomposition, Vismara cycle
family construction, $\operatorname{GF}(2)$ relevance filtering, and tied
shortest-path expansion. It releases the GIL and has no mutable global state.

The implementation is derived from RingDecomposerLib's Vismara algorithm. RDL
is not a runtime dependency. Its BSD 3-Clause license and provenance notice are
included under `_native/`; the test-only oracle is pinned to commit
`3a7ff93de0d9c4f6a5661508549c6063573f39c7`.

`Molecule.rings` still retains its existing NetworkX cycle-basis semantics.
Selecting Relevant Cycles in Core or geometry is a separate chemistry-policy
decision and is not silently enabled by this package.
