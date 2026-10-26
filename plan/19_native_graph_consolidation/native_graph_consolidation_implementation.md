# Single-Source Molecular Topology, Private Native Graph Consolidation, and NetworkX Removal Plan

Status: owner-approved normative implementation contract; implementation
started on 2026-10-23. Sections below continue to specify target behavior and
must not be read as completion claims until the implementation report records
the corresponding evidence.

Planning branch: `clean/networkx`

Planning worktree: `/tmp/hotpot-clean-networkx`

Planning baseline: `2704f3c`

## 1. Objective

Establish one authoritative Python source for every molecular-topology fact,
remove NetworkX from Hotpot production code, and consolidate Hotpot-owned
graph computation under `hotpot.cheminfo.graph` without replacing NetworkX
with another public, persistent, general-purpose graph framework.

The authoritative domain representation remains the existing object model:
`Molecule` owns ordered atom and bond membership, each `Atom` owns its
atom-local chemistry, and each `Bond` owns its endpoints and bond-local
chemistry. Neither `Atom._neighbours` nor the Atom-specific use of
`Atom._bonds` remains as a second owner of connectivity. `Molecule` maintains
one private, read-only, deterministically rebuildable topology index so that
ordinary Python neighbor access does not scan the whole bond table. That
index is derived state, not another source of chemical truth and not a
computational Graph.

The implementation introduces one minimal private C++ topology type derived
from the existing native-search `GraphData`. That type is an ephemeral
computational representation: a task constructs it from the current source
objects, composes only the typed properties required by that task, consumes it
inside native work, and destroys it when the task completes.

The plan has six simultaneous goals:

1. make Bond endpoints the sole authoritative connectivity facts and remove
   duplicate Atom adjacency state;
2. eliminate every production NetworkX import, object, annotation, exception,
   and dependency declaration;
3. retain one native RI search implementation and preserve chemical/search,
   ring, MCA, and model semantics;
4. centralize reusable graph algorithms and deterministic ordering rules; and
5. avoid mandatory graph construction when a direct edge range, bond range,
   incident-bond range, matrix, or domain snapshot is the cheaper input; and
6. replace the ambiguous legacy cycle-basis surface with one deterministic
   default simple-cycle basis and an explicitly complete collection of all
   simple-cycle bases for bounded feasible inputs.

## 2. Owner decisions and non-negotiable boundaries

The following decisions are normative.

1. A topology fact has exactly one authoritative domain owner. `Molecule`
   owns ordered membership, `Atom` owns atom-local chemical facts, and `Bond`
   owns its two endpoints and bond-local chemical facts. A derived index or
   task snapshot may repeat values only when it is read-only, disposable, and
   reconstructible solely from those owners.
2. The native `Graph` and every Graph-like computational view are private.
   Hotpot does not expose them through Python or any public return value.
3. A computational Graph is not a domain-data owner and is never persistent
   application state. It may own compact topology buffers for the duration of
   one task, but it does not own `Molecule`, `Atom`, `Bond`, chemical identity,
   query identity, or coordinates.
4. `Molecule` never caches a native Graph. A new task snapshots the molecule's
   current topology and required attributes again. The one permitted Python
   topology index is a domain query index, not an algorithm workspace or
   native topology handle.
5. Target-specific structures use composition. Search, automorphism, cycle,
   and other applications compose one base Graph with typed property columns,
   masks, maps, or workspaces. The base type does not accumulate application
   policy.
6. Graph exists only where repeated adjacency, degree, or edge lookup makes it
   a useful computational medium. If graph construction cannot provide a
   measurable or structurally unavoidable computational benefit, the
   application consumes its source directly.
7. All reusable Hotpot graph algorithms and ordering rules live under the
   graph sublibrary, even when an algorithm accepts a direct target rather than
   constructing Graph.
8. `hotpot.cheminfo.core` contains no NetworkX object or residual reference.
   Hotpot exposes no native, NetworkX, or replacement Graph object to users.
9. This stage deliberately establishes explicit deterministic ordering for
   topology normalization, traversal, paths, cycles, components, and search
   inputs. Unordered-container iteration must not influence observable output.
10. RDKit and Open Babel internal graph representations and internal graph
   algorithms are outside this consolidation. Hotpot may extract atom and bond
   facts from their public APIs, but this plan does not alter their internals.
11. There is one implementation, not a selectable backend. No runtime graph
    backend flag, profile, compatibility mode, NetworkX fallback, or retained
    old production implementation is permitted.
12. Core receives no new class, public method, or public property. A new
    private Core attribute or replacement private method is allowed only when
    Section 4.3 names it, fixes its shape or signature, and demonstrates why
    existing state and interfaces cannot meet the requirement. The four
    owner-mandated Molecule cycle-basis properties fixed in Sections 4.3 and
    6.4 are the only public-property exception.

Immutable query IR is not a target-derived Graph and may remain reusable.
Likewise, chemical results such as `Ring`, numerical results such as
`GraphSpectrum`, and neural-network layers whose name contains “Graph” are not
the private topology object governed by this plan.

## 3. Confirmed current-state audit

### 3.1 Production NetworkX ownership

Seven production modules directly import NetworkX at the planning baseline.

| Module | Current use | Required disposition |
|---|---|---|
| `cheminfo/core.py` | persistent `Molecule._graph`, graph-returning properties, components, paths, legacy cycle basis, canonical tree, and Z-matrix traversal | remove every NetworkX object and reference; route work through domain objects or graph algorithms |
| `AImodels/mca/site_detection.py` | labeled NetworkX graph, 1-WL access, and rooted exact isomorphism for automorphic site removal | retain 1-WL semantics; replace rooted isomorphism with the native RI topology and exact anchored mode |
| `forcefields/coordination.py` | connected components after excluding coordination bonds | consume atom/bond facts directly through graph-subpackage component labeling |
| `forcefields/working_copy.py` | `nx.Graph` snapshot annotation and restoration of `Molecule._graph` | delete the graph snapshot field and graph restoration |
| `graph/traversal.py` | public NetworkX-typed DFS helpers | replace or retire the legacy signatures; no public Graph argument remains |
| `ob2chem.py` | unused import | delete immediately |
| `search/search.py` | cached public query `nx.Graph` returned by `construct_graph()` | delete the query graph cache and public graph constructor |

NetworkX is also a base runtime dependency in `pyproject.toml` and
`requirements.txt`, appears in the runtime-distribution identity list, and is
installed by inference tests. Production declarations are removed only after
production imports reach zero. NetworkX remains allowed as a test-only
independent oracle and graph generator.

### 3.2 Existing native topology structures

Hotpot currently contains several independent computational topology forms:

- `search/_native/GraphData`: sorted external keys, canonical edges, CSR
  offsets and neighbors, adjacency edge IDs, degrees, self-loop lookup, and
  binary edge lookup;
- `graph/_native/LocalGraph`: relevant-cycle component edges, original edge
  IDs, and adjacency vectors;
- force-field-native ligand adjacency vectors and component traversal;
- Open-Babel-wrapper task snapshots containing incident-bond adjacency; and
- Python `Atom._neighbours`, the Atom-specific incident relationship stored in
  `Atom._bonds`, `Molecule._graph`, and authoritative Bond endpoint facts.

The Python forms currently repeat connectivity in three mutable places:
`Bond` endpoints, Atom neighbor lists, and Atom incident-bond lists. In
addition to wasting state, `forcefields/working_copy.py` must snapshot and
restore all three forms. Plan 19 removes both Atom copies. The final Python
domain state has one authoritative connectivity representation and one
Molecule-owned, rebuildable query index.

The native-search representation is the correct base for the one private
Graph because it is compact, deterministic, immutable after construction, and
already proven by the RI matcher. The relevant-cycle local graph remains a
task-local composition or view because biconnected components require local
vertex remapping and original-edge provenance.

Open Babel wrapper snapshots and RDKit molecules are not migrated by this
plan. Hotpot force-field algorithms that are independent of Open Babel may use
the graph sublibrary through direct ranges or task-local Graph composition.

### 3.3 Current graph logic inventory

| Logic | Current location | Target input policy |
|---|---|---|
| subgraph monomorphism | `search/_native` | compose one Graph with search predicate/property columns; Graph required |
| MCA rooted automorphism | Python/NetworkX | reuse the task's native target topology and compose exact labels; Graph required |
| Relevant Cycles | `graph/_native` | edge input at Python boundary; task-local Graph and biconnected local compositions internally |
| default and all simple-cycle bases | legacy `core.py`/NetworkX provides one insertion-order-sensitive Paton basis only | native Graph; deterministic Paton default plus complete bounded enumeration of all simple-cycle bases |
| connected components | `core.py`, force fields, native force-field session | direct node count and edge range; union-find or direct traversal; no mandatory Graph |
| atom neighbors | persistent Atom `_neighbours` and `_bonds` | use the Molecule derived topology index and authoritative Bond endpoints; no Graph |
| bond lookup | NetworkX edge attributes | search the smaller indexed incident-bond range; no persistent bond-pair hash |
| shortest paths | `core.py`, MCA feature preparation, Relevant Cycles | direct neighbor provider for one-shot domain work; task-local Graph for repeated native queries |
| simple and longest paths | `core.py` | graph-sublibrary algorithm over a direct provider; no persistent Graph |
| Z-matrix DFS | `core.py` and `graph/traversal.py` | edge array/direct adjacency input; no public Graph |
| adjacency/Laplacian/spectrum | `graph/matrix.py`, `graph/spectrum.py` | dense matrices remain direct numerical inputs; no Graph |
| GNN `edge_index` creation | AI and plugin adapters | generate tensors directly from bonds; no Graph |
| aromaticity and Kekule assignment | `kekulize` | consume rings and incident bonds directly; not moved into Graph |

## 4. Normative semantic contract

### 4.1 Task lifetime and mutation

Each operation snapshots topology and required attributes at task entry. That
snapshot remains stable until the operation returns or its iterator is closed.
Mutating a target molecule, query, or task-observed predicate object while a
native operation is in progress is unsupported. The next task must repack the
current source state.

No Graph, Graph view, target topology capsule, or native target is stored on a
`Molecule`, `Atom`, or `Bond`. Search cursors and other iterators may own their
task snapshot only for the cursor lifetime. MCA prepared queries may remain
compiled because they represent immutable rule programs rather than a target
molecule.

The Python/native boundary is coarse-grained and field-selective. It packs the
current Molecule atom order, resolves Bond endpoints through the derived Atom
rows, copies only task-required chemistry columns into contiguous buffers,
and enters native code once per coarse operation. Multiple native stages
inside that operation reuse the same snapshot. Native code does not perform
per-neighbor calls back into Python and does not retain Python domain objects
after the operation or cursor closes. This one O(V + E) immutable snapshot is
the communication boundary; it is not a second authority.

### 4.2 Domain ownership

The final authority table is normative:

| Fact | Sole authoritative owner |
|---|---|
| ordered atom membership | `Molecule._atoms` |
| ordered active-bond membership | `Molecule._bonds` |
| atomic number, charge, isotope, aromaticity, hydrogen state, and other atom-local chemistry | the corresponding `Atom` |
| bond endpoints, order, kind, direction, and other bond-local chemistry | the corresponding `Bond` |
| whether a bond is temporarily hidden from active topology | exactly one of the existing Molecule active/hidden collections; a Bond may not occur in more than one |
| current coordinates and conformer data | unchanged existing owners; consolidation is explicitly outside Plan 19 |
| neighbor rows and incident-bond rows | no domain owner; they are deterministic projections in the private topology index |
| rings, paths, components, cycle bases, and spectra | algorithm results, never alternative owners of molecular connectivity; cycle-basis outputs obey Section 6.4 |

The Atom-specific `_neighbours` and `_bonds` attributes are removed. This
does not remove `AtomSeq._bonds`, which represents constituent bonds of Bond,
Angle, Torsion, and Ring domain sequences and has different semantics. Public
`Atom.neighbours`, `Atom.neigh_idx`, `Atom.connectivity`, `Atom.hydrogens`,
`Atom.heavy_neighbours`, and `Atom.bonds` remain and derive their results from
the Molecule index plus the authoritative Bond endpoints.

`Molecule` owns one private derived topology index with this fixed conceptual
shape:

```python
_topology_index: dict[int, tuple[int, tuple[Bond, ...]]]
```

The key is `id(atom)`. The value contains that Atom's current row in
`Molecule._atoms` and its incident active Bonds ordered by adjacent atom row.
The index has no setter or public accessor. It contains Bond references, not
copied Bond objects or chemical properties. It must be possible to discard it
and reconstruct an identical index solely from `_atoms`, `_bonds`, and Bond
endpoints. It may never repair or override those owners.

No persistent `_bond_by_edge`, native topology handle, replacement Python
Graph, topology revision counter, or second topology index is added.

### 4.3 Core change budget and required additions

The Core change budget is deliberately closed:

| Change | Decision | Why it is necessary |
|---|---|---|
| new Core classes | forbidden | existing `Molecule`, `Atom`, and `Bond` are the legitimate domain owners; an index class would add ownership and lifecycle ambiguity |
| new public methods | forbidden | the existing Molecule/Atom/Bond behavioral API is sufficient after graph-object APIs are removed |
| `Molecule.cycle_basis -> list[tuple[int, ...]]` | required owner-mandated public property | exposes the deterministic default basis as index values without forcing Ring allocation or exposing Graph |
| `Molecule.all_cycle_basis -> list[list[tuple[int, ...]]]` | required owner-mandated public property | exposes every complete simple-cycle basis as a distinct deterministic index collection; no existing interface provides this information |
| `Molecule.basis_rings -> list[Ring]` | required owner-mandated public property | materializes exactly the default `cycle_basis` through the existing Ring domain type |
| `Molecule.all_basis_rings -> list[list[Ring]]` | required owner-mandated public property | materializes every basis from `all_cycle_basis` without introducing a Graph result type |
| any other new public property | forbidden | prevents expansion beyond the owner-approved cycle-basis replacement surface |
| `Molecule._topology_index` | required new private attribute | without one derived incidence index, deleting both Atom adjacency stores makes each one-hop query scan every molecular Bond; no existing non-Graph field provides Atom row and incident-Bond access |
| `Molecule._update_topology(clear_conformers: bool = True) -> None` | required replacement private method | it replaces both `_update_graph()` and `_update_atom_neighbours_bonds()` so every mutation has one place to rebuild the index and invalidate existing derived chemistry state without retaining a NetworkX-named updater |
| any other new Core attribute or method | forbidden unless the owner first amends and re-approves this plan | prevents convenience helpers, caches, revisions, and compatibility surfaces from accumulating during implementation |

All four cycle-basis interfaces are read-only `@property` values over the
current active full molecular topology. Hidden Bonds are excluded because they
are not members of `_bonds` at evaluation time. The properties accept no
profile, backend, scope, root, ordering, or limit argument. Ligand-skeleton
model preparation filters coordination Bonds internally and calls the same
native basis algorithm; it does not require another public Molecule property.

The old public `Molecule.cycle_basis_rings` and
`Molecule.ligand_cycle_basis_rings` properties and the unused Atom-level
`cycle_basis_rings` view are removed after their active consumers migrate.
They are not retained as aliases. Their private `_cycle_basis_rings` and
`_ligand_cycle_basis_rings` caches are also deleted. None of the four new
properties persists its result on Molecule; every call derives from current
authoritative topology, and the potentially large all-bases family exists
only for the duration of the call and the caller-owned returned value.

The replacement updater is not an additional parallel interface:
`_update_graph()` and `_update_atom_neighbours_bonds()` are deleted when all
repository consumers have been migrated. During incremental commits,
`_update_topology()` may temporarily refresh the old private NetworkX
projection for still-unmigrated algorithms, but that projection is explicitly
derived, cannot receive writes, and is removed before the Core-removal stage
passes. No compatibility alias remains in the final tree.

The updater validates that every active Bond endpoint is present exactly once
in `_atoms`, every active undirected endpoint pair is unique, and active and
hidden Bond membership is exclusive. It then rebuilds the index in
deterministic order and performs the same angle, torsion, ring, Open Babel,
and optional conformer invalidation currently owned by `_update_graph()`.
Bulk mutators call it once at commit, not once per inserted atom or Bond.
Existing mutation methods remain the only mutation surface; Plan 19 adds no
transaction class or public builder.

`Molecule.__init__` initializes the index to an empty dictionary. The existing
`Atom(mol)` construction path already appends the new Atom directly to
`Molecule._atoms`; it therefore adds that one isolated-Atom index record in
O(1) before returning. This is incremental maintenance of derived state, not a
second authority, and prevents direct Atom construction from observing a
missing `Atom.idx`. Existing private bulk builders may append atoms and Bonds
without rebuilding after every item, but they must call `_update_topology()`
exactly once before returning a public Molecule. Existing public bulk mutators,
including `add_bonds()`, must likewise commit once before returning. No lazy
fallback scan, dirty flag, or partially valid public index is permitted.

### 4.4 Chemical and model parity

The change must preserve:

- atom and bond identity;
- full-graph versus ligand-skeleton SMARTS semantics;
- complete search mapping sets and query automorphisms;
- MCA ordered rule priority, site identity, site type, automorphic
  representative policy, and `is_Nuc_site`;
- isotope, charge, aromaticity, implicit-hydrogen, bond-order, and bond-kind
  effects on MCA symmetry;
- Relevant Cycle membership and limit behavior;
- the owner-mandated default/all-basis semantics in Section 6.4;
- current trained-model prediction parity after migrating internal
  ligand-skeleton ring features to the same native default-basis algorithm;
- molecule/component atom and bond attributes;
- existing public Atom neighbor and incident-Bond return types;
- Atom and Bond index meanings under the Molecule's current order;
- Z-matrix results for supported current inputs; and
- graph-spectrum numerical behavior.

Removing graph-returning public APIs is intentional and is not subject to
type compatibility. Scientific outputs and non-graph public interfaces remain
subject to parity except for the explicitly replaced legacy cycle-basis API
and its newly fixed deterministic ordering. Stage 0 records old ring tensors;
the report must explain any representational reordering. Prediction drift is
not authorized: if the new default basis changes a current trained-model
prediction beyond its existing exact/tolerance contract, the basis cutover
stops for owner review.

### 4.5 No hidden fallback

A missing native graph extension is an installation error. It does not invoke
NetworkX or a slower Python graph engine. Python-callable query predicates
retain the Plan 16 leaf-only callback contract; their presence never changes
the topology executor.

## 5. The one private native Graph

### 5.1 Base topology

The single base type moves from `search/_native/graph_data.*` to
`graph/_native/graph.*` under namespace `hotpot::graph`.

The intended internal interface is:

```cpp
namespace hotpot::graph {

using NodeKey = std::int64_t;
using Position = std::uint32_t;
using EdgeIndex = std::uint32_t;

struct Edge {
    Position first;
    Position second;
};

class Graph final {
public:
    Graph(
        std::vector<NodeKey> ordered_keys,
        std::vector<std::pair<NodeKey, NodeKey>> ordered_edges
    );

    std::size_t node_count() const noexcept;
    std::size_t edge_count() const noexcept;
    std::span<const NodeKey> keys() const noexcept;
    std::span<const Edge> edges() const noexcept;
    std::span<const Position> neighbors(Position node) const noexcept;
    std::span<const EdgeIndex> incident_edges(Position node) const noexcept;
    Position degree(Position node) const noexcept;
    std::int32_t edge_between(Position first, Position second) const noexcept;
    std::int32_t self_loop_edge(Position node) const noexcept;
};

}  // namespace hotpot::graph
```

The exact C++17 implementation may return iterator pairs or a small span-like
view because `std::span` is C++20. The semantic interface above is normative.

Graph stores only compact integer topology. It supports isolated nodes and
self-loops because generic search supports them. It rejects unknown endpoints,
duplicate node keys, duplicate undirected edges, and parallel edges. An
algorithm such as Relevant Cycles may impose the stricter no-self-loop simple
graph contract at its own boundary.

Graph has no Python binding and no public `.pyi` declaration.

### 5.2 Composition and property columns

Application properties are indexed parallel storage owned by the task:

```text
SearchTarget
├── Graph topology
├── atom property columns required by compiled queries
├── bond property columns required by compiled queries
└── optional Python object handles required by opaque leaves

AutomorphismProblem
├── Graph topology
├── exact node-label column
├── exact edge-label column
└── fixed query-to-target pair

RelevantCycleComponent
├── Graph local topology or a local Graph view
├── local-to-source node map
└── local-to-source edge map
```

No `dict`, `std::any`, polymorphic attribute base, or universal chemistry
record is stored in Graph. A property not requested by the application is not
packed.

Composition is the default and required architecture. Application classes do
not inherit from Graph merely to gain neighbor access.

### 5.3 Source-level reuse versus runtime sharing

The Graph source is compiled into each native extension that needs it. Plan 19
does not introduce a shared native library, cross-extension Graph handle, or
Python capsule merely to share one runtime instance. Such packaging would add
ownership and ABI complexity without demonstrated benefit.

Within one coarse-grained native task, the application must reuse the Graph it
already constructed. It must not rebuild equivalent adjacency for each query,
candidate, rooted isomorphism, or stage.

## 6. Unified deterministic ordering

### 6.1 Base normalization

Every Graph instance uses one total order:

1. external node keys are unique and ascending;
2. an undirected edge is normalized as `(min_key, max_key)`;
3. normalized edges are unique and lexicographically ascending;
4. internal positions follow ascending external key order;
5. each adjacency range is ordered by neighbor position, then edge index; and
6. no hash-map iteration determines observable traversal or output.

Direct-target algorithms that do not construct Graph must implement the same
logical order whenever their result is ordered.

### 6.2 Algorithm output order

| Output | Ordering contract |
|---|---|
| neighbors and incident bonds | adjacent atom index ascending, then canonical bond endpoint key |
| connected components | nodes ascending inside a component; components by smallest node |
| single shortest path | BFS in ascending-neighbor order, yielding the lexicographically smallest shortest path |
| all-pairs distances | dense matrix indexed by ascending normalized node position |
| DFS paths | roots and neighbors ascending; depth limits applied after the same visit event as the frozen contract |
| Relevant Cycles | canonical rotation and direction; result by cycle length then lexicographic tuple, as today |
| default cycle basis | deterministic Paton traversal and emission order defined in Section 6.4 |
| all cycle bases | default basis first; every remaining basis in canonical basis-key order |
| native search | preserve Plan 16 deterministic RI query plan and ascending target-candidate order |
| MCA site representatives | existing rule order, candidate atom index ascending, first non-automorphic representative retained |

The legacy cycle-basis family requires special care because its atom-index
tuples are a trained-model input. Stage 0 records the exact old basis, tensor,
and prediction corpus before NetworkX is removed. Section 6.4 intentionally
replaces insertion-order sensitivity with one fixed ordering. A different but
mathematically valid basis is not automatically model-compatible; prediction
gates govern cutover.

### 6.3 Neighbor order and complexity after removing Atom adjacency state

`_update_topology()` builds each `_topology_index` incident range by adjacent
atom row and then canonical Bond endpoint key. `Atom.neighbours` obtains that
range in expected O(1), derives the other endpoint of every Bond, and returns
the existing public list type. `Atom.bonds` materializes the same ordered Bond
list. `Atom.neigh_idx` emits the corresponding integer array, while
`Atom.connectivity` reads the incident tuple length without constructing a
neighbor list.

The complexity contract is explicit:

| Operation | Required complexity |
|---|---:|
| locate an Atom's topology-index record | expected O(1) |
| `Atom.idx` | expected O(1) |
| `Atom.connectivity` | O(1) |
| materialize `Atom.bonds`, `Atom.neighbours`, or `Atom.neigh_idx` | Theta(degree) |
| filter hydrogen or heavy neighbors | Theta(degree) |
| one direct Bond lookup | O(min(degree(first), degree(second))) |

Materializing all neighbors cannot be O(1), because the result contains one
item per neighbor. The current implementation also copies `_neighbours` and
is already Theta(degree). Plan 19 guarantees O(1) access to the incident
range, not an impossible O(1) materialization claim.

This removes dependence on bond insertion history and makes direct-domain
access consistent with native Graph. No Python/native round trip occurs for
an ordinary Atom neighbor query.

Code whose chemistry depended on incidental bond insertion order is not a
valid retained contract; nevertheless, Stage 0 captures affected outputs and
the complete suite must expose any behavioral change before cutover.

### 6.4 Default and complete simple-cycle-basis semantics

Cycle-basis results are defined over the current active undirected molecular
graph. Nodes are current Atom indices; edges are active Bond endpoint pairs.
Hidden Bonds are absent. Parallel Bonds or a Bond from an Atom to itself are
outside the molecular ring-basis domain and raise `ValueError`; generic search
Graph support for self-loops does not widen this chemistry API. The molecular
cycle rank is:

```text
mu = edge_count - node_count + connected_component_count
```

A cycle is a simple cyclic Atom-index sequence with at least three distinct
nodes. Its GF(2) incidence vector is indexed by the normalized native Graph
edge order. A simple-cycle basis contains exactly `mu` distinct simple cycles
whose incidence vectors have rank `mu`. “All bases” means all such
simple-cycle bases, not arbitrary disconnected even-degree vectors in the
abstract cycle space.

The default `Molecule.cycle_basis` is selected without enumerating all bases.
It uses one native implementation of Paton's fundamental-cycle algorithm with
the following complete tie-breaking contract:

1. native node positions follow ascending Atom index;
2. the root of each remaining component is its greatest Atom index, matching
   the existing NetworkX default-root direction on an ascending node table;
3. neighbors are inspected in ascending Atom-index order;
4. traversal uses the Paton LIFO stack and predecessor/used-edge events in the
   same order as the frozen algorithm;
5. cycles remain in deterministic Paton discovery order; and
6. each returned tuple omits the repeated closing node.

This produces one unique result for the same indexed topology regardless of
Bond insertion order, Python hash order, or native thread scheduling. It is an
index-relative deterministic basis, not a claim that an unlabeled graph has a
chemically privileged or automorphism-invariant basis. Symmetric graphs do
not possess such a naturally distinguished basis without a tie-break.

`Molecule.all_cycle_basis` enumerates the complete family in native code:

1. enumerate distinct simple cycles using a rotation/reversal-invariant
   internal cycle key;
2. represent each candidate by its GF(2) edge-incidence vector;
3. use rank-pruned backtracking to enumerate every size-`mu`, rank-`mu`
   candidate set exactly once;
4. identify a basis independently of cycle order;
5. emit `cycle_basis` as the first outer entry; and
6. emit every remaining basis by ascending canonical key, with cycles inside
   those remaining bases ordered by `(cycle_length, canonical_cycle_tuple)`.

Consequently:

```python
mol.all_cycle_basis[0] == mol.cycle_basis
```

For an acyclic graph, `cycle_basis == []` and `all_cycle_basis == [[]]`, because
the zero-dimensional cycle space has exactly one empty basis. Ring-valued
properties are exact structural projections:

```text
basis_rings[i] corresponds to cycle_basis[i]
all_basis_rings[b][i] corresponds to all_cycle_basis[b][i]
```

Complete basis enumeration is inherently exponential and its list return type
can be combinatorially large. The implementation has one fixed, non-profiled
safety ceiling of 10,000 complete bases per call. Detection of a 10,001st
basis raises `OverflowError` and discards the partial result; neither
`all_cycle_basis` nor `all_basis_rings` may return an incomplete collection.
The default `cycle_basis` and `basis_rings` do not enumerate all bases and are
not subject to this ceiling.

## 7. Direct-target algorithm policy

The graph sublibrary owns algorithms, not mandatory data conversion. Internal
algorithm seams support either a Graph/view or the cheapest source contract.

Conceptually:

```cpp
result algorithm(const Graph& graph);
result algorithm(std::size_t node_count, EdgeRange edges);
result algorithm(const DomainNeighborProvider& source);
```

These are internal overloads or templates, not public Python Graph APIs. There
must be one semantic algorithm per operation, with adapters only at the input
boundary.

| Operation | Construction decision |
|---|---|
| RI matching | construct one Graph because backtracking repeatedly queries adjacency and edges |
| rooted exact automorphism | reuse the RI task Graph; do not construct a second graph |
| Relevant Cycles | construct task topology from edges, then compose biconnected local topologies/views |
| default cycle basis | consume one native Graph and run deterministic Paton traversal in O(V + E); never enumerate all bases |
| all cycle bases | consume one native Graph and enumerate complete simple-cycle bases with GF(2) rank pruning; exponential by definition and complete-or-error at the fixed ceiling |
| component IDs | prefer union-find over the edge range; no Graph |
| `Atom.neighbours` | consume the Molecule topology index and authoritative Bond endpoints; no Graph |
| one molecule shortest path | traverse the Molecule topology index directly; no Graph |
| repeated/all-pairs shortest paths | construct one task Graph or compact adjacency, then reuse it |
| Z-matrix traversal | consume edge array/direct adjacency; no persistent Graph |
| GNN edge index | emit directly from bonds; no Graph |
| matrix/spectrum | consume the requested numerical matrix; no Graph |

## 8. Public API retention, dependency audit, and removals

### 8.1 Retained domain API

Supported domain access remains explicit and retains its current signatures
and return categories:

- `Molecule.atoms`, `Molecule.bonds`, and `Molecule.link_matrix`;
- `Molecule.bond(first, second)` implemented through the smaller derived
  incident-Bond range;
- `Atom.idx`, `Atom.neighbours`, `Atom.neigh_idx`, `Atom.connectivity`, and
  `Atom.bonds` backed by the private Molecule topology index;
- `Molecule.cycle_basis`, `all_cycle_basis`, `basis_rings`, and
  `all_basis_rings` with the exact Section 6.4 semantics;
- `Substructure.query_atoms` and `Substructure.query_bonds`; and
- graph algorithms that accept documented arrays, edges, node counts, or
  chemistry objects without returning Graph.

`InternalCoordinates.calc_zmat_idx()` and `calc_zmat()` accept the documented
edge/link array representation, not NetworkX. Graph traversal helpers are
either private or receive explicit node/edge inputs. They do not accept an
opaque graph object.

### 8.2 Removal audit and migration gate

No obsolete interface is deleted merely because its replacement is planned.
The baseline audit found the following active repository dependencies; every
listed consumer must be migrated and tested before the declaration is
removed:

| Interface/state to remove | Active baseline dependencies | Required migration before deletion |
|---|---|---|
| `Atom._neighbours` and Atom-specific `_bonds` | Core rebuild/access, force-field working-copy snapshot/rollback, complex-hydrogen rollback tests | use Bond endpoints plus `_topology_index`; rollback restores authoritative atoms/bonds and rebuilds the index instead of restoring derived lists |
| `Molecule._graph`, `graph`, `simple_graph`, and `atom_bond_graph` | Core algorithms, force-field coordination, dataset SMARTS capability detection, geometry/ring/hidden-bond tests | move algorithms to direct/native graph functions; detect supported molecule input without a Graph property; rewrite tests against atoms, bonds, and scientific outputs |
| cached `Molecule.adjacency_matrix` | no active production or test consumer found; `graph_spectral()` constructs its own task-local dense matrix | delete the property and its persistent instance cache; retain exact graph-spectrum behavior through the direct numerical path |
| `Molecule.super_acyclic_graph()` | no active repository consumer found and the method always raises `NotImplementedError` | delete directly as an unimplemented obsolete interface |
| `_update_graph()` and `_update_atom_neighbours_bonds()` | Core mutators, converters, MCA serialization, force-field working copy, CCDC adapter, and tests | migrate atomically to the fixed `_update_topology()` replacement and prove zero remaining calls before deletion |
| `Substructure.construct_graph()`, `query_graph`, and cache/signature state | SMARTS conformance adapter/tests and native-contract tests | construct any independent test oracle from `query_atoms/query_bonds`; native RI production already consumes compiled query facts and must remain independent |
| public MCA `MoleculeGraph` and lazy export | MCA public input/serialization/import tests | migrate repository callers to already-supported `Molecule`, RDKit, Open Babel, SMILES, or generic `to_rdmol` input; prove the prediction and DTO pipeline has no intrinsic dependency on the wrapper |
| public NetworkX-typed traversal helpers | traversal tests and any repository callers found by the final audit | use explicit node/edge arrays or private internal functions; preserve scientific traversal outputs |
| `Molecule.cycle_basis_rings`, `Molecule.ligand_cycle_basis_rings`, Atom `cycle_basis_rings`, `_cycle_basis_rings`, and `_ligand_cycle_basis_rings` | model/PyG ring extraction, force-field working-copy snapshots, relevant-ring integration tests, geometry/cache tests, and complex-hydrogen rollback tests | migrate public consumers to the four Section 6.4 properties; migrate ligand-skeleton model extraction to the same native algorithm over coordination-filtered edges; remove properties, private caches, snapshots, tests, and documentation together with no aliases |

After those migrations, Hotpot removes rather than redirects every public
topology-Graph surface named above. The deletion commit must include a
repository-wide reference audit. A still-active production consumer blocks
that deletion stage; it is not bypassed with a compatibility alias, optional
NetworkX adapter, `to_networkx()` method, or retained legacy backend.

## 9. Algorithm migrations

### 9.1 Native search

Move `GraphData` into the graph sublibrary without changing the Plan 16 RI
algorithm. Search-specific query plans, compatibility domains, typed property
columns, opaque Python handles, cursor state, and predicate IR remain owned by
search.

The cutover must prove:

- identical unbounded mapping sets;
- unchanged `has_match()`, `max_matches`, and `truncated` semantics;
- unchanged callable-leaf behavior;
- unchanged deterministic mapping sequence for identical current inputs; and
- no material search performance or memory regression.

`Substructure.construct_graph()` is not used by native execution and is
deleted independently of matcher correctness.

### 9.2 MCA exact automorphism

Replace NetworkX graph copies and `nx.is_isomorphic()` with exact anchored
native matching composed over the already packed target topology for one site
detection task.

The exact node label is:

```text
(atomic_number, formal_charge, is_aromatic, implicit_hydrogens, isotope)
```

The exact edge label retains current aromatic, bond-order, and bond-kind
semantics. One-WL remains a rejection-only prepartition. For two candidate
sites in the same refined color, native exact matching fixes the first source
node to the representative target node and answers only Yes/No.

Equal node and edge counts, bijection, exact node labels, exact edge labels,
and fixed-root mapping are mandatory. Subgraph monomorphism alone is not an
adequate public description of this operation even if equal graph sizes make
the current case equivalent.

### 9.3 Relevant Cycles and rings

Keep the existing Vismara-derived algorithm and scientific Relevant Cycle
definition. Replace duplicate whole-graph adjacency construction with base
Graph only where the benchmark shows no regression and the local component
mapping remains clear.

The algorithm still owns:

- biconnected decomposition;
- local component renumbering;
- ordered shortest-path predecessor workspaces;
- binary cycle-basis reduction; and
- Relevant Cycle family expansion and limits.

Relevant Cycle Python input remains an edge iterable and output remains
canonical node cycles. Graph is not exposed.

### 9.4 Default and all cycle bases

Implement both operations in the graph native extension over the one private
Graph:

- default basis: deterministic Paton traversal from Section 6.4 in O(V + E)
  time and O(V + E) workspace; and
- all bases: native simple-cycle enumeration plus incremental GF(2) rank
  reduction and rank-pruned backtracking, with complete-or-error semantics.

Relevant Cycles are not a substitute. They are the union of cycles occurring
in minimum cycle bases, whereas `all_cycle_basis` requires every
simple-cycle basis and the default property requires the fixed Paton basis.
The three families remain explicitly distinct.

The native binding returns integer cycle sequences only. Core materializes
Ring objects after the native call and exposes no Graph. Full-topology
properties pass active Bond edges directly. Existing ligand-skeleton ML
features pass the same native function a coordination-filtered edge range;
there is no second basis implementation and no ligand-specific public
Molecule property.

Acceptance covers:

- exact Section 6.4 default ordering over graph-atlas and generated graphs;
- differential default-basis comparison with a frozen independent Paton
  oracle using the same root and neighbor policy;
- exhaustive all-bases comparison with an independent small-graph oracle;
- proof that every reported basis has size/rank `mu`, every candidate is a
  simple cycle, and no basis is duplicated or omitted;
- disconnected, acyclic, fused, bridged, spiro, symmetric, metal-containing,
  full-topology, and ligand-skeleton cases;
- `all_cycle_basis[0] == cycle_basis` and exact Ring/index projection;
- 10,000-result success and 10,001st-result atomic `OverflowError`; and
- current feature-extraction and trained-model prediction gates.

### 9.5 Components, paths, and Z-matrix traversal

Move semantic ownership into the graph sublibrary while retaining direct
inputs:

- component labeling uses edge-range union-find where appropriate;
- one-shot molecular shortest path traverses incident bonds;
- generic repeated shortest paths use compact task adjacency;
- simple-path and longest-path enumeration retain their inherent complexity
  and receive explicit bounds where already supported; and
- Z-matrix traversal consumes a link/edge array.

Replacing NetworkX does not justify changing longest-simple-path complexity or
claiming a native speedup without measurement.

### 9.6 MCA all-pairs shortest paths

The MCA featurizer currently applies a dense Floyd-Warshall-style NumPy loop.
Plan 19 adds a graph-sublibrary sparse all-pairs distance candidate operating
on the atom count and extracted bond edges. RDKit remains the chemical source;
its internals are not modified.

The native candidate is adopted only if it preserves the exact capped distance
matrix and improves the validated workloads. Otherwise the current numerical
implementation remains, moved only if a source-organization change does not
add indirection or complexity. Graph construction is not accepted merely for
API uniformity.

### 9.7 Force-field topology utilities

Hotpot-owned component labeling and edge filtering may call direct graph
algorithms. Open Babel wrapper `MoleculeSnapshot` adjacency, Open Babel
`OBMol`, and RDKit graph operations remain outside scope.

Force-field placement or session code may compose the base Graph only when the
same task performs repeated adjacency/edge queries and the before/after
benchmark proves the construction is worthwhile. No force-field domain object
inherits from or stores Graph.

## 10. Target source layout

```text
hotpot/cheminfo/graph/
├── __init__.py                    # public value/array algorithms only
├── cycles.py                      # Python validation and result conversion
├── components.py                  # direct node/edge component facade
├── paths.py                       # direct path/distance facades
├── matrix.py                      # retained dense conversions
├── spectrum.py                    # retained numerical spectrum API
├── native.py                      # private native loader/adapters
├── _graph_native.pyi             # functions/results only; no Graph binding
└── _native/
    ├── graph.hpp/.cpp             # one private immutable CSR Graph
    ├── components.hpp/.cpp        # direct edge-range algorithms
    ├── paths.hpp/.cpp             # bounded path/distance algorithms as justified
    ├── cycle_basis.hpp/.cpp       # deterministic default and complete all-basis enumeration
    ├── relevant_cycles.hpp/.cpp   # retained Vismara-derived implementation
    └── bindings.cpp               # coarse-grained function bindings only

hotpot/cheminfo/search/_native/
├── matcher.hpp/.cpp               # RI matcher over hotpot::graph::Graph
├── production.hpp/.cpp            # composed properties and predicate engine
└── bindings.cpp                   # no Graph exposure
```

`hotpot/cheminfo/core.py` retains its existing Molecule, Atom, Bond, and
AtomSeq classes. Plan 19 creates no Core topology module, index class, manager,
service, transaction, or facade. The sole derived index is the private
Molecule dictionary fixed in Section 4.2.

The existing private `_relevant_cycles` extension may be renamed to
`_graph_native` once all graph functions are bound. No compatibility import is
required for the old private extension name. `setup.py` compiles the one Graph
source into each extension that needs it; it does not duplicate Graph source
definitions.

## 11. Implementation stages and commit discipline

Every stage receives a dated entry in
`plan/19_native_graph_consolidation/implementation_log.md`. Each completed
stage is committed independently after its focused tests pass.

### Stage 0 — freeze evidence before deletion

1. Record Python, compiler, NetworkX, NumPy, RDKit, Open Babel, and Hotpot
   revision metadata.
2. Record exact ordered outputs for components, neighbors, paths, DFS,
   Relevant Cycles, legacy NetworkX cycle basis, search mappings, and MCA
   sites. Build an independent exhaustive all-simple-cycle-bases oracle for
   small graphs before writing the native enumerator.
3. Record old-model ring tensors and prediction digests.
4. Record current search, MCA site-detection, Relevant Cycle, cycle-basis,
   Molecule topology refresh/copy/rollback, Atom indexing and neighbor access,
   components, paths, and feature-preparation timings.
5. Record every active dependency in Section 8.2, including production,
   adapters, tests, documentation, and plugins; classify each as migration,
   intentional test-oracle retention, or deletion.
6. Add source audits for every public Graph exposure, every production
   NetworkX reference, and every direct access to the Atom-specific
   `_neighbours` or `_bonds` state.

### Stage 1 — establish single-source Python topology

1. Add the sole permitted new Core attribute, `_topology_index`, with the
   exact shape and derived-only contract in Section 4.2.
2. Replace `_update_graph()` and `_update_atom_neighbours_bonds()` with the
   sole `_update_topology(clear_conformers=True)` updater. Until dependent
   algorithms migrate, this updater may regenerate the old private NetworkX
   projection as temporary derived state; no old updater alias remains.
3. Delete Atom-specific `_neighbours` and `_bonds`; preserve the unrelated
   `AtomSeq._bonds` constituent-Bond state.
4. Reimplement existing `Atom.idx`, `bonds`, `neighbours`, `neigh_idx`,
   `connectivity`, hydrogen-neighbor, and heavy-neighbor properties from the
   index and authoritative Bond endpoints without changing public signatures
   or return categories.
5. Reimplement `Molecule.bond()` and duplicate-Bond checks without NetworkX
   edge attributes or a new bond-pair cache.
6. Migrate force-field working-copy rollback, converters, adapters, tests, and
   every other direct private-state consumer. Rollback restores authoritative
   objects and calls the updater; it does not snapshot or restore the derived
   index as fact.
7. Audit every `_create_atom()`, `_add_atom()`, `_add_bond()`, and direct
   `_atoms`/`_bonds` builder path. Public methods return only after one index
   commit; the direct `Atom(mol)` path installs its isolated record in O(1).
8. Prove mutation, copy, hidden-bond, rollback, ordering, and performance
   parity before committing this stage.

### Stage 2 — introduce the private base Graph

1. Move and generalize native-search `GraphData` under graph ownership.
2. Preserve its compact CSR layout and self-loop behavior.
3. Pack topology once per coarse native task from Molecule atoms, Bonds, and
   the current deterministic row order; the native snapshot never writes back
   into `_topology_index`.
4. Add deterministic-construction, isolated-node, self-loop, duplicate-edge,
   sparse-key, incident-edge, and edge-lookup tests.
5. Do not bind Graph to Python.

### Stage 3 — cut native search over to the base Graph

1. Compose search query/target properties around Graph.
2. Remove the search-private GraphData implementation.
3. Run the complete search contract, differential, callable, SMARTS, and
   performance gates.
4. Migrate SMARTS adapters and tests from `construct_graph()` to existing
   query atoms/bonds, then delete query NetworkX construction, cache state,
   and its public Graph API.

### Stage 4 — native MCA automorphism

1. Add exact labels and fixed-root exact matching to the existing RI task.
2. Reuse one task target topology across ordered MCA rules and symmetry checks.
3. Remove NetworkX from site detection.
4. Prove exact ordered site and `is_Nuc_site` parity, including disconnected
   symmetric components and isotopic asymmetry.

### Stage 5 — cycle algorithms and ordering

1. Put Relevant Cycle native sources behind the graph-native extension.
2. Implement the deterministic default Paton basis and complete bounded
   all-simple-cycle-bases enumerator from Sections 6.4 and 9.4.
3. Add exactly the four approved Molecule properties with their fixed return
   shapes; add no other Core public interface or basis-result cache.
4. Migrate model/PyG ligand-skeleton consumers to the same native algorithm
   over filtered edges and migrate repository users to the new properties.
5. Delete `Molecule.cycle_basis_rings`,
   `Molecule.ligand_cycle_basis_rings`, Atom `cycle_basis_rings`, both old
   private basis caches, their working-copy snapshot fields, and every test or
   document that names the old interfaces. No alias or deprecation shim
   remains.
6. Apply the unified ordering and complete-or-error contracts.
7. Verify Relevant Cycle separation, exhaustive small-graph completeness,
   Ring/index projection, model tensors, and prediction parity before removing
   `nx.cycle_basis()`.

### Stage 6 — remove Core graph state

1. Route components, paths, ring topology filtering, canonical-tree work,
   coordination component labeling, and Z-matrix traversal through the
   topology index or direct graph-sublibrary algorithms.
2. Migrate dataset SMARTS capability detection and all remaining repository
   consumers away from Graph-returning Molecule properties.
3. Delete `Molecule._graph` initialization, transitional rebuild, snapshot,
   copy, and restoration.
4. Delete `Molecule.graph`, `simple_graph`, and `atom_bond_graph` only after
   the Section 8.2 active-dependency audit reaches zero.
5. Delete the unused cached `Molecule.adjacency_matrix`, the unimplemented
   `super_acyclic_graph()`, and private node/edge Graph-construction helpers;
   retain `graph_spectral()` and its exact task-local numerical behavior.
6. Delete all NetworkX annotations, exceptions, docstrings, commented code,
   and imports from Core.
7. Prove that `_topology_index` is the only persistent derived topology index
   and remains reconstructible from authoritative domain state.

### Stage 7 — remove remaining public Graph-like surfaces

1. Migrate active repository MCA callers, tests, and documentation to existing
   supported molecule inputs; prove the prediction/DTO pipeline is independent
   before deleting the public `MoleculeGraph` wrapper/export.
2. Remove or redesign public graph traversal helpers around explicit arrays.
3. Update examples and remaining internal consumers to use atom, Bond, query,
   or explicit edge collections directly.
4. Confirm Hotpot exports no Graph type or graph-returning method.

### Stage 8 — direct-target consolidation and optional performance work

1. Consolidate component labeling, one-shot shortest path, and Z-matrix
   traversal without constructing Graph where the direct target is cheaper.
2. Benchmark the sparse MCA all-pairs distance candidate and adopt only on a
   measured win with exact output.
3. Audit force-field-native duplicate adjacency construction, migrating only
   Hotpot-owned repeated work that benefits from task composition.
4. Reject any optional optimization that requires another Core attribute,
   method, class, persistent index, or topology cache without owner-approved
   revision of this plan.

### Stage 9 — dependency and packaging cutover

1. Remove NetworkX from `project.dependencies` and base requirements.
2. Remove it from the production runtime-distribution identity.
3. Retain it only in test requirements for independent oracles and fixtures.
4. Build wheels for all supported Python versions and verify that wheel
   metadata does not require NetworkX.
5. Verify native extension import and absence-of-fallback failures.

### Stage 10 — final closure

1. Run all focused and full suites.
2. Run prediction, ordering, packaging, and performance gates.
3. Run ownership, zero-reference, Core-change-budget, and public-surface
   audits.
4. Complete the implementation log and implementation report.
5. Commit only after all mandatory gates pass.

## 12. Validation matrix

### 12.1 Static closure

The following commands, or equivalent repository-safe audits, must prove:

```text
production imports of networkx: 0
production references to nx.: 0
Core references to networkx/NetworkX/nx.: 0
public Graph classes or Graph-returning APIs: 0
runtime dependency declarations for networkx: 0
Molecule computational/native Graph caches: 0
Atom-specific _neighbours definitions/accesses: 0
Atom-specific _bonds definitions/accesses: 0
Molecule derived topology indexes other than _topology_index: 0
Core classes added by Plan 19: 0
Core public methods added by Plan 19: 0
Core public properties added outside cycle_basis/all_cycle_basis/basis_rings/all_basis_rings: 0
definitions of each approved cycle-basis property: exactly 1
Core private attributes added by Plan 19 other than _topology_index: 0
Core private methods added by Plan 19 other than replacement _update_topology: 0
calls or definitions of _update_graph/_update_atom_neighbours_bonds: 0
production calls to nx.cycle_basis: 0
production definitions/references of cycle_basis_rings/ligand_cycle_basis_rings: 0
production definitions/references of _cycle_basis_rings/_ligand_cycle_basis_rings: 0
active repository references to each removed API: 0
```

Historical plan/report prose and test-only NetworkX oracles are not production
violations. The Atom-specific `_bonds` audit is scope-aware and must not flag
the semantically distinct `AtomSeq._bonds` constituent-Bond field.

### 12.2 Unit and differential tests

Mandatory suites cover:

- authoritative ownership and deterministic `_topology_index`
  reconstruction from atoms, active Bonds, and Bond endpoints;
- rejection of missing or foreign endpoints and duplicate active edges under
  the current Molecule simple-topology contract;
- Atom index, incident-Bond, neighbor, connectivity, hydrogen-neighbor, and
  heavy-neighbor behavior and return-type parity;
- add/replace/remove Atom, add/remove/hide/restore Bond, bulk mutation,
  molecule copy, component merge, and force-field rollback, including the
  invariant that a Bond is present in exactly one active/hidden collection;
- direct `Atom(mol)` construction, private parser/converter bulk construction,
  and public `add_bonds()` commit behavior, with no observable partial index;
- Graph construction and ordering;
- components, neighbors, paths, DFS, Z-matrix, and isolated atoms;
- Relevant Cycles, graph atlas, RDL differential evidence, and limits;
- deterministic default Paton basis under shuffled Bond insertion, repeated
  calls, disconnected components, isolates, and symmetric indexed graphs;
- exhaustive all-simple-cycle-bases equality against an independent oracle on
  bounded graph-atlas/generated graphs, including GF(2) rank, completeness,
  duplicate rejection, default-first ordering, empty-basis semantics, and the
  atomic 10,000-basis ceiling;
- exact `basis_rings`/`all_basis_rings` projection from their index-valued
  counterparts and proof that none of the four values is cached on Molecule;
- native search kernel, contracts, mappings, predicates, mutation, recursion,
  SMARTS conformance, and callable behavior;
- MCA ordered rules, exact automorphism, serialization, coordination, isotope,
  and `is_Nuc_site`;
- force-field coordination environments and working-copy rollback;
- Core mutation, hidden-bond restoration, ring-cache invalidation, molecule
  copy, and component extraction;
- native task packing from authoritative facts, with proof that native results
  never mutate or become the source of `_topology_index`;
- inference model feature tensors and old-model predictions; and
- README/public-interface examples after intentional Graph API removal.

NetworkX may act as a test oracle where its semantics define the frozen
baseline. Production code must not import it during these tests.

### 12.3 Prediction parity

For fixed inputs and model artifacts, the report records:

- exact ordered MCA sites;
- exact site types and `is_Nuc_site` ownership;
- exact MCA float bit patterns where the graph change does not intentionally
  alter model input;
- exact ring tensors consumed by existing CBond/PyG extractors;
- exact or explicitly approved model prediction digests; and
- zero unexplained structural or ordering differences.

The report separately identifies old insertion-order-dependent tensors and
the new deterministic tensor. A basis-set difference on a frozen accepted
model workload is a failed gate. A tuple orientation/order difference caused
only by the owner-approved deterministic contract must be explicit and still
pass the existing prediction contract; it is never hidden as a benign native
refactor.

### 12.4 Performance evidence

Performance comparisons use fixed inputs, warmups, repeated trials, medians,
machine identity, CPU affinity, and build metadata. At minimum they measure:

- native Graph construction;
- `_topology_index` construction and peak retained memory;
- molecule creation/copy/rollback and single/bulk topology mutation before and
  after removal of NetworkX and Atom adjacency state;
- `Atom.idx`, connectivity, one-hop neighbor materialization, repeated
  neighbor access, and direct Bond lookup across molecule sizes and degrees;
- components and shortest paths;
- native default basis versus legacy NetworkX basis, Relevant Cycles, and
  all-bases enumeration scaling up to the fixed success/error boundary;
- generic search workloads from Plan 16;
- MCA site detection on simple, complex, and symmetric molecules;
- MCA feature preparation/all-pairs distances; and
- complete MCA CPU/GPU workflows where graph work is material.

Acceptance rules:

1. Atom-row and incident-range location must exhibit expected O(1) scaling;
   materializing neighbors may scale only with returned degree, not total
   Molecule atom or Bond count;
2. the derived index must use less retained topology state than the removed
   NetworkX plus per-Atom neighbor and incident-Bond copies, with measured
   object/byte evidence rather than an estimate;
3. replacing Core topology state must not introduce a statistically material
   whole-workflow or common-mutation regression; any regression outside the
   recorded baseline confidence/noise interval requires owner review;
4. extracting the already-existing search Graph into graph ownership must not
   add a second topology construction or material memory regression;
5. adopting Graph in a new application requires a measured wall-time,
   allocation, or repeated-construction improvement on its target workload;
6. a neutral source-ownership refactor is acceptable only when it does not add
   runtime indirection and is necessary to keep one implementation;
7. a slower Graph-mediated path is rejected in favor of the direct-target
   algorithm; and
8. complete workflow regressions outside controlled measurement noise require
   owner review before cutover.

The default native basis must additionally avoid a material regression against
the frozen NetworkX path while eliminating Python Graph construction. No
polynomial-performance promise applies to all-bases enumeration; its required
performance property is deterministic scaling, bounded retained memory per
completed result, and atomic failure at the fixed ceiling.

## 13. Compatibility and migration policy

This stage deliberately removes graph-object APIs. Repository consumers are
migrated before the relevant deletion commit on the branch. The Section 8.2
audit is a deletion gate, not documentation after the fact. No compatibility
layer remains in the final tree.

Existing non-Graph public domain access remains compatible. In particular,
Plan 19 does not rename `Atom.neighbours`, change `Atom.bonds` into a native
view, introduce a new topology service, or require a user to obtain Graph in
order to inspect a Molecule. The public list/array return categories named in
Section 6.3 remain unchanged.

Cycle-basis access is an intentional breaking replacement. The only final
Molecule surface is `cycle_basis`, `all_cycle_basis`, `basis_rings`, and
`all_basis_rings`. The former Molecule/Atom basis-ring properties and both old
private basis caches are deleted in the same migration stage. Documentation,
type information, tests, examples, model extraction, and plugins must use the
new names or the one internal native function; no alias or deprecation path is
permitted.

Public documentation must state that molecular topology is accessed through
atoms, bonds, incident relationships, explicit link arrays, search methods,
ring results, and dedicated graph algorithms—not by retrieving a mutable graph
container.

Users requiring NetworkX construct it in their own application from public
atom/bond data and own that dependency. Hotpot does not provide a converter or
promise compatibility with NetworkX object behavior.

## 14. Explicit non-goals

Plan 19 does not:

- alter RDKit graph storage, sanitization, conformer generation, or feature
  semantics;
- alter Open Babel `OBMol`, perception, force-field internals, or plugin state;
- consolidate ownership of Atom `x/y/z`, current coordinates, conformer
  coordinates, energies, forces, or other geometry facts;
- convert Atom/Bond chemical attributes into Molecule-owned structure-of-
  arrays storage or introduce a general zero-copy molecule ABI;
- redesign the existing active/hidden-Bond workflow beyond enforcing
  exclusive membership and rebuilding the derived index;
- create a public graph framework;
- cache native topology on Molecule;
- add a topology revision counter, persistent bond-pair hash, second incidence
  index, transaction class, public builder, or convenience topology method;
- make graph algorithms selectable by backend;
- replace Relevant Cycles with cycle basis or vice versa;
- describe the deterministic Paton default as a minimum cycle basis, a
  chemically preferred ring set, or an atom-renumbering-invariant basis;
- promise polynomial runtime or unbounded materialization for
  `all_cycle_basis`/`all_basis_rings`;
- change RI-style search to another matcher;
- remove Python predicate leaves;
- promise polynomial longest-simple-path behavior;
- change the scientific definition or public numerical behavior of graph
  spectrum and spectrum similarity;
- nativeize dense spectrum eigensolvers merely because they consume a graph
  matrix; or
- retain obsolete Graph APIs for compatibility.

## 15. Risks and controls

| Risk | Control |
|---|---|
| hidden dependence on NetworkX object identity or mutation | delete the public surface, migrate repository consumers, and run a zero-reference audit |
| Bond endpoints, Atom state, and the Molecule index diverge | Bond endpoints remain authoritative; an isolated `Atom(mol)` installs only its own empty record and every multi-object topology mutation converges on one updater; rebuild-oracle tests compare every index entry after every mutation family |
| direct private `_atoms`/`_bonds` mutation bypasses index rebuild | migrate active repository writes, audit remaining writes, and confine authoritative mutation to existing Molecule methods |
| working-copy rollback restores derived state as fact | snapshot only authoritative atoms/Bonds and rebuild the index after rollback; differential failure-injection tests cover every commit step |
| Atom-specific `_bonds` removal accidentally removes `AtomSeq._bonds` | scope-aware source audit and explicit Bond/Angle/Torsion/Ring regression tests |
| deterministic default basis alters trained-model input | freeze old basis/tensors, use the same native default algorithm for ligand filtering, report deterministic representation changes, and block prediction drift |
| “unique default” is mistaken for a chemically unique or relabeling-invariant basis | document that uniqueness is relative to the indexed topology and fixed Paton tie-breaks; symmetric graphs require an index tie-break |
| all-bases enumeration exhausts time or memory | native GF(2) pruning, fixed 10,000-basis ceiling, atomic `OverflowError`, no Molecule cache, and explicit exponential-complexity documentation |
| an all-bases path returns a partial family | construct caller-visible results only after native completion; ceiling or allocation failure raises and discards partial work |
| unified sorting changes chemistry behavior | explicit total-order contract plus before/after domain-output fixtures |
| target mutation produces stale native topology | native topology is a task-local snapshot only; mutation during an active task remains unsupported and the next task repacks authoritative facts |
| Graph becomes a universal chemistry object | topology-only base plus composed typed columns |
| pybind round trips erase native benefit | coarse-grained functions and task-local reuse; no per-neighbor Python/native calls |
| direct one-shot work becomes slower after forced construction | dual internal input seams and benchmarked go/no-go adoption |
| separate extensions duplicate runtime state | share source/type definition, not cross-extension objects or mutable globals |
| removing Atom adjacency state slows common properties | use the one Molecule incidence index, compute connectivity without list materialization, and benchmark representative high-frequency consumers |
| intentional API removal breaks an active repository consumer | Section 8.2 consumer migration and zero-reference evidence are mandatory before each deletion |
| geometry ownership is mistaken as solved by this plan | geometry consolidation is an explicit non-goal and must receive its own owner-approved plan |

## 16. Required implementation records

Implementation must create and maintain:

1. `implementation_log.md`, updated after every small committed stage with
   files changed, commands, results, decisions, and remaining risks;
2. `implementation_report.md`, summarizing final architecture, deleted public
   APIs, semantic evidence, performance evidence, packaging results, and
   residual limitations; and
3. machine-readable benchmark artifacts where existing plan conventions
   provide a runner and schema.

The implementation report must distinguish improvements caused by removing
NetworkX, improvements caused by native algorithms, and neutral source
consolidation. It must not attribute all workflow improvement to the presence
of the Graph class. It must also include the final authority table, every Core
class/method/attribute addition and deletion, proof that the Core change budget
was respected, the Section 8.2 consumer disposition, and retained-memory
evidence for the old and new Python topology representations.

For cycle bases it also records the default-selection contract, exhaustive
small-graph oracle results, all-basis counts, ceiling behavior, removed legacy
symbols, old/new model tensors, and final prediction evidence.

## 17. Definition of done

Plan 19 is complete only when all of the following are true:

1. Molecule membership plus Bond endpoints are the sole authoritative Python
   connectivity representation;
2. Atom instances contain neither `_neighbours` nor an Atom-specific `_bonds`
   field, while the distinct `AtomSeq._bonds` semantics remain intact;
3. `_topology_index` is the sole persistent derived Python topology index, is
   deterministic and reconstructible, and meets the complexity contract;
4. Core adds no class or public method, adds exactly the four approved
   cycle-basis properties and no other public property, and adds no private
   state or method beyond `_topology_index` and the replacement
   `_update_topology()` expressly authorized by Section 4.3;
5. one private C++ CSR Graph definition exists under graph ownership;
6. native search composes that Graph and retains Plan 16 semantics;
7. MCA automorphic site removal is native and exactly preserves ordered sites
   and `is_Nuc_site`;
8. Relevant Cycles, deterministic default basis, and complete bounded
   all-simple-cycle-bases enumeration are owned by the graph sublibrary and
   retain their three distinct semantics;
9. Core contains no NetworkX, computational Graph cache, duplicated Atom
   adjacency state, old updater, or stale Graph documentation;
10. every active consumer identified in Section 8.2 is migrated before its
    obsolete interface is deleted;
11. `cycle_basis`, `all_cycle_basis`, `basis_rings`, and `all_basis_rings`
    satisfy Section 6.4, while every old Molecule/Atom cycle-basis interface
    and private basis cache is absent;
12. Hotpot exposes no Graph or Graph-like topology holder;
13. one-shot algorithms avoid Graph construction where direct input is cheaper;
14. production code and runtime metadata contain no NetworkX dependency;
15. all mandatory scientific, ordering, prediction, packaging, and performance
   gates pass;
16. no old production backend, compatibility shim, or hidden fallback remains;
    and
17. the implementation log and report contain reproducible final evidence.
