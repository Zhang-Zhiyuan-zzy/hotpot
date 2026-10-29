# `hotpot.cheminfo.rules.mathematics.graph`

This mathematical rules package contains Hotpot's graph utilities and its
Relevant Cycle perception API. The native extension remains in
`hotpot.cheminfo.graph._graph_native`. Chemical graph spectrum descriptors live
in `hotpot.cheminfo.rules.chemistry.descriptors.graph_spectrum`. The native cycle implementation accepts topology only: atom
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
| `calc_spectrum` (chemical descriptor) | Calculate the existing Hotpot graph spectrum. |
| `GraphSpectrum` (chemical descriptor) | Store and compare graph spectra. |
| `calc_electron_config` (chemical descriptor) | Calculate the existing electron-configuration feature. |
| `atoms_electron_configurations` (chemical descriptor) | Calculate electron-configuration features for atoms. |
| `graph_dfs_path` | Return one depth-first path from explicit node/edge inputs. |

The four chemical descriptor names above are imported from
`hotpot.cheminfo.rules.chemistry.descriptors.graph_spectrum`; the remaining
names are exported by `hotpot.cheminfo.rules.mathematics.graph`.

The mathematical package contains numerical public APIs and direct topology algorithms; it
does not expose a Graph data object. Native algorithms construct one private,
task-local topology only when repeated adjacency work benefits from it.

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
from hotpot.cheminfo.rules.mathematics.graph import relevant_cycles

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
included under `hotpot/cheminfo/graph/_native/`; the test-only oracle is pinned to commit
`3a7ff93de0d9c4f6a5661508549c6063573f39c7`.

`Molecule.rings` and `Molecule.ligand_rings` use this Relevant Cycle family.
`Molecule.cycle_basis` exposes one deterministic native Paton basis, while
`all_cycle_basis` exposes every complete simple-cycle basis up to its fixed
complete-result limit. `basis_rings` and `all_basis_rings` are exact Ring
projections of those index-valued results. Existing trained-model extractors
use the same native default-basis algorithm over their ligand-skeleton edge
scope. Aromaticity perception itself uses Relevant Cycles.
