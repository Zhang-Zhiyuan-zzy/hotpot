# Plan 19 implementation log

## Recording policy

This log is updated during implementation, not reconstructed after the work.
Every stage records its intended result, files changed, verification commands,
results, unexpected findings, decisions, residual risks, and Git checkpoint.

The owner authorized implementation on 2026-10-23.  The final tree may contain
one deterministic RI search implementation, one private native topology type,
and no production NetworkX fallback or compatibility alias for a removed graph
or cycle-basis interface.

## Baseline

| Item | Result |
| --- | --- |
| Date | 2026-10-23 |
| Branch | `clean/networkx` |
| Worktree | `/tmp/hotpot-clean-networkx` |
| Starting revision | `2704f3c9e0ca449f0d9d1445d6320714d5dbbc0c` |
| Python | 3.11.16, GCC 14.3.0 runtime build |
| Platform | Linux 5.15.0-139-generic x86_64, glibc 2.31 |
| Native compiler | GCC 9.4.0, C++17 |
| NetworkX | 3.6.1 |
| NumPy | 2.4.6 |
| RDKit | 2026.09.1 |
| Open Babel | 3.2.1 |

## Stage record

### Stage 0 - freeze evidence before deletion

Status: complete

Planned:

- freeze graph, topology, search, MCA-site, cycle-basis, model-ring-tensor,
  dependency, timing, and packaging evidence before deleting any behavior;
- add an implementation-independent exhaustive small-graph cycle-basis oracle;
  and
- establish a source-built test environment tied to the exact starting
  revision.

Performed:

- recorded the environment and exact starting revision above;
- audited all seven production NetworkX imports, runtime dependency entries,
  graph-returning public surfaces, old updater calls, duplicated Atom adjacency
  state, old cycle-basis consumers, force-field rollback state, model/PyG ring
  feature consumers, and active tests/documents;
- added `capture_legacy_semantics.py`, which emits ordered legacy topology,
  DFS, Relevant Cycle, Paton cycle-basis, ring-tensor, RI mapping, and MCA-site
  evidence as JSON;
- added an independent pure-Python oracle for undirected simple-cycle
  enumeration, GF(2) rank, deterministic Paton traversal, and exhaustive
  simple-cycle-basis enumeration; and
- self-validated the oracle on acyclic, triangular, diagonal-square, K4, and
  disconnected graphs.  An additional one-off comparison covered all 1,253
  NetworkX graph-atlas entries with no Paton mismatch;
- rebuilt all four native extensions from the exact starting revision with
  CPython 3.11, GCC 9.4.0, C++17, pybind11 3.x, and setuptools 77-81;
- wrote the exact ordered baseline to `legacy_semantics.json`, including the
  five-cycle cubane Paton basis, six-cycle Relevant Cycle family, ring tensor,
  topology/traversal results, RI mapping order, MCA site order, isotope case,
  and the three accepted MCA prediction hashes already enforced by the test
  suite;
- wrote `baseline_dependency_inventory.md`, classifying every active
  production/import/API/model/test/document consumer as migration, deletion,
  or test-only oracle retention; and
- recorded fixed-input microbenchmarks and targeted retained duplicate-state
  size in `legacy_topology_benchmark.json`.

Verification completed so far:

```text
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_cheminfo/graph/test_cycle_basis_oracle.py
```

Result: 8 passed.

Additional focused baseline results:

```text
graph/topology/ring group: 62 passed in 1.22 s
native search/API group: 80 passed, 49 subtests passed in 4.70 s
MCA site/coordination/serialization group: 59 passed in 1.32 s
MCA accepted prediction-digest gate: 1 passed in 6.30 s
```

The frozen 512-atom chain/cubane/search/MCA microbenchmark medians were:

```text
Atom.idx                         9.359429 us
Atom.connectivity               3.315981 us
Atom.neighbours                 3.240479 us
Molecule.bond                   0.454011 us
topology refresh            15720.257000 us
Molecule.copy               61901.120000 us
legacy cubane Paton basis      60.786000 us
cubane Relevant Cycles         75.383000 us
12-carbon pair search         259.833000 us
MCA site detection           1557.816000 us
```

The targeted deep-size measurement for the legacy NetworkX projection plus
per-Atom neighbor and incident-Bond lists on the 512-atom chain was 653,181
bytes.  This is a controlled before/after metric for these specific derived
objects, not whole-process RSS.

Unexpected findings:

- copied native extensions from the main worktree did not match this source
  checkout (`NativeTarget` rejected a missing `degree` field), so they are
  invalid as baseline evidence;
- the first isolated `uv run` selected CPython 3.14 from project metadata; it
  was stopped and its exact `.venv` was removed; and
- authoritative baseline tests therefore require a fresh CPython 3.11 native
  build from this worktree.  That rebuild completed successfully; and
- the broad legacy `tests/test_cheminfo/test_core.py` module contains an
  unbounded `test_atompair` path that invokes 3D construction for a very large
  molecule and is unsuitable as a focused graph baseline.  The graph-relevant
  behavior is instead frozen by the targeted suites and semantic snapshot;
  final closure still runs the maintained repository suite with its normal
  exclusions/constraints.

Adjustment:

- do not record any mixed-source/native result;
- run semantic snapshots and focused suites only after the exact checkout has
  rebuilt its native extensions; and
- keep NetworkX only in test-oracle and historical evidence after production
  cutover.

Git checkpoint: `089fed8` (`plan(graph): freeze networkx removal baselines`).

### Stage 1 - establish single-source Python topology

Status: complete.

Performed:

- added the sole authorized `_topology_index` dictionary and replaced both old
  refresh methods with `_update_topology(clear_conformers=True)`;
- made `_atoms`, `_bonds`, and Bond endpoints the authority while validating
  duplicate Atom membership, missing endpoints, self-bonds, parallel active
  edges, duplicate hidden membership, and active/hidden identity overlap;
- removed Atom-specific `_neighbours` and `_bonds` state while retaining the
  unrelated `AtomSeq._bonds` constituent state;
- implemented O(1) Atom-row and incidence-record location, O(1)
  `connectivity`, degree-linear neighbor/Bond materialization, and
  smaller-incidence-range `Molecule.bond()` lookup;
- made direct `Atom(mol)` and private atom append install an isolated index
  entry immediately, and made public `add_bonds()` commit once;
- migrated every production old-updater call in Core, RDKit/Open Babel/CCDC
  converters, MCA DTO conversion, and force-field commit/rollback;
- changed rollback to restore authoritative atoms/Bonds/Atom attributes, then
  reconstruct topology rather than restoring graph/list/cache identities;
- invalidated `_row2idx` together with `_obmol`; and
- rebuilt the identity-keyed derived index at force-field spawn task entry,
  because deserialization creates new Atom identities.  This avoids a new
  Core pickle method or lazy/dirty fallback.

Core change-budget result:

```text
new Core classes: 0
new Core public methods/properties: 0
new Core private attributes: _topology_index only
new/replacement Core private methods: _update_topology only
removed Atom fields: _neighbours and Atom-specific _bonds
removed methods: _update_graph and _update_atom_neighbours_bonds
```

Verification:

```text
topology/rollback/ring/converter/MCA focused group:
  169 passed, 3 xfailed, 1 existing warning in 1.89 s
real force-field multiprocessing integration:
  11 passed in 23.68 s
direct initial topology group:
  43 passed in 1.17 s
```

The new tests cover direct Atom construction, index reconstruction, shuffled
Bond insertion, no Atom adjacency fields, O(1)-path Atom indexing, one-commit
bulk bond addition, direct Bond lookup, invalid authority states, removal
reindexing, hidden restoration, and failed working-copy rollback.

Performance evidence on the same 512-Atom chain:

| Operation | Legacy median (us) | Index median (us) |
| --- | ---: | ---: |
| `Atom.idx` | 9.359429 | 3.177599 |
| `Atom.connectivity` | 3.315981 | 3.191546 |
| `Atom.neighbours` | 3.240479 | 8.222773 |
| `Molecule.bond` | 0.454011 | 5.007222 |
| topology refresh | 15,720.257 | 12,783.047 |
| Molecule copy | 61,901.120 | 52,559.716 |

The targeted retained derived-topology size fell from 653,181 bytes to
135,176 bytes (79.3%).  Degree-linear Python materialization and direct Bond
lookup have small absolute microbenchmark costs relative to the former
NetworkX hash lookup/cached neighbor list; they implement the owner-approved
single-source and complexity contract without a forbidden second pair or
neighbor index.  Whole force-field and MCA focused gates show no material
workflow regression.

Git checkpoint: `0a7fcdc` (`refactor(core): establish single-source topology
index`).

### Stage 2 - introduce the private base Graph

Status: complete.

Performed:

- moved the proven search topology design into
  `hotpot/cheminfo/graph/_native/graph.hpp/.cpp` as the sole private
  `hotpot::graph::Graph` definition;
- retained sorted sparse external keys, canonical unique edges, CSR offsets,
  ordered neighbors, incident-edge IDs, degrees, self-loop lookup, and binary
  edge lookup;
- added iterator-range access for a node's neighbors and incident edges; and
- exposed no Python Graph binding, handle, capsule, class, or type stub.

Verification:

- the standalone C++17 source contract compiled with `-Wall -Wextra -Werror`
  and covered sparse signed keys, isolates, self-loops, degree semantics,
  neighbor/incident ranges, edge lookup, unsorted/duplicate keys, missing
  endpoints, noncanonical edges, duplicate edges, and edge ordering;
- the exact CPython 3.11 extension build succeeded; and
- source audits found one native base Graph definition and no public binding.

Git checkpoint: `51cc4b6` (`feat(graph): add private native topology graph`).

### Stage 3 - cut native search over to the base Graph

Status: complete.

Performed:

- changed the RI matcher, compatibility boundary, production target, and
  production problem to compose `hotpot::graph::Graph`;
- compiled `graph.cpp` into the private search extension and deleted the
  search-owned `GraphData` source and definition;
- retained the Plan 16 query plan, compatibility domains, lazy cursor,
  callable-leaf behavior, result facade, and deterministic candidate order;
- deleted `Substructure.construct_graph()`, `query_graph`, cache signatures,
  invalidation state, and the production NetworkX import with no alias; and
- changed SMARTS adapters and test oracles to construct any independent
  NetworkX test graph from `query_atoms` and `query_bonds` only.

Verification:

```text
517 passed, 49 subtests passed in 7.38 s
```

This set includes the complete SMARTS conformance and Plan 16 native contract,
kernel, production, prepared-query, callable, mapping, cutover, predicate
parity, and C++ Graph source tests.  Static audits report no exact `GraphData`,
old source filename, production `construct_graph`/`query_graph`, or production
NetworkX reference under search.  The frozen performance/semantic gate retained
deterministic results with SHA-256
`a2c1f6c232b1b69f87599eb9bfa843a62846f726dbbe7fc8077fdd2f414c43f4`.

Git checkpoint: `cb17165` (`refactor(search): compose shared native graph`).

### Stage 4 - native MCA automorphism

Status: complete.

Performed:

- added a private fixed-root exact automorphism problem that reuses the same
  `NativeTarget` and private Graph as all ordered MCA rules for one molecule;
- retained RI planning, domains, cursor state, and candidate order while
  requiring equal degree, exact node labels, and exact edge labels;
- encoded the full MCA atom label
  `(atomic_number, formal_charge, is_aromatic, implicit_hydrogens, isotope)`
  and the pre-existing aromatic/bond-order/bond-kind edge semantics without
  lossy hashing;
- replaced NetworkX copies and `nx.is_isomorphic()` with the anchored native
  Yes/No check;
- retained one-WL as a rejection-only Python prepartition over direct packed
  adjacency; and
- preserved ordered rules, representative selection, site types, and
  `is_Nuc_site` ownership.

Verification:

```text
native search and complete MCA focused group:
  377 passed in 23.70 s
```

The added contracts cover shared target identity, exact node and edge labels,
disconnected symmetric components, isotope/asymmetric rejection, and absence
of a production NetworkX reference in MCA site detection.

Git checkpoint: `6eac3fa` (`refactor(mca): replace networkx automorphism`).

### Stage 5 - cycle algorithms and ordering

Status: complete.

Performed:

- unified Relevant Cycles, the private base Graph, deterministic Paton, and
  complete simple-cycle-basis enumeration behind `_graph_native`, with no
  Python Graph binding;
- renamed Relevant Cycle's local edge type to avoid a second conflicting
  `hotpot::graph::Edge` definition and migrated its native force-field
  consumer;
- implemented maximum-index component roots, ascending neighbors, the Paton
  LIFO/predecessor/used events, and exact discovery-order output;
- implemented rotation/reversal-invariant simple-cycle enumeration, dynamic
  GF(2) masks, rank-pruned basis backtracking, default-first ordering, and the
  atomic fixed 10,000-basis ceiling;
- added exactly `cycle_basis`, `all_cycle_basis`, `basis_rings`, and
  `all_basis_rings` to Molecule, with no result cache or Graph exposure;
- moved full and ligand Relevant Cycle inputs to direct active Bond ranges;
- moved both NumPy and optional PyG trained-model consumers to the same native
  default-basis implementation over coordination-filtered edges; and
- deleted both old Molecule basis-ring properties, the Atom view, both private
  caches, their docs/tests, and the old private extension/type-stub name with
  no alias.

Ordering evidence:

The owner-approved topology-relative ordering intentionally replaces bond
insertion history.  Cubane therefore changed from the frozen legacy tensor

```text
4 7 6 5 | 2 1 7 6 | 3 0 1 7 6 5 | 4 0 1 7 | 3 2 6 5
```

to the stable native tensor

```text
4 5 6 7 | 2 3 5 6 | 1 0 3 5 6 7 | 4 0 3 5 | 1 2 6 7
```

Both contain five cycles with sizes `(4, 4, 6, 4, 4)`.  This is recorded as
the explicit unified-ordering change rather than hidden as implementation
parity.  The maintained CBond runtime/prediction suite passed with this tensor,
and all three accepted MCA prediction hashes remain exact because MCA model
inputs are independent of this ring-basis tensor.

Verification:

```text
native basis/oracle, Relevant Cycle, Core ring/cache, force-field link,
and MCA prediction-digest group:
  190 passed, 4 skipped, 1 existing warning in 25.56 s

CBond runtime and prediction group with writable isolated model cache:
  42 passed in 6.15 s

private cycle-basis C++ source contract:
  passed with -Wall -Wextra -Werror
```

The exhaustive differential set covers acyclic, triangle, diagonal-square,
K4, disconnected, sparse-key, shuffled-edge, and bounded graph-atlas cases.
K5 triggers the real fixed 10,000-basis `OverflowError`; the source contract
also proves exact-limit success and next-result failure on a smaller injected
limit.  PyG is not installed in the validation environment, so its migrated
consumer is statically identical at the cycle source boundary; the NumPy
tensor and installed CBond runtime are covered dynamically.

On the same local cubane checkout, uncached `Molecule.cycle_basis` measured
46.36 us median versus the frozen legacy Ring-materializing operation's
60.79 us.  These values have different return layers and are recorded only as
a regression guard, not as a claimed like-for-like speedup.

Git checkpoint: `2848fb4` (`feat(graph): implement deterministic cycle bases`).

### Stage 6 - remove Core graph state

Status: complete; checkpoint pending.

Performed:

- added deterministic direct-edge component, shortest-path, simple-path,
  longest-path, and one-branch DFS algorithms under the graph sublibrary;
- routed Molecule component extraction, path access, canonical-tree work,
  coordination component labeling, and Z-matrix traversal through explicit
  Atom/Bond facts or task-local edge ranges;
- made one-shot Molecule shortest paths traverse the ordered incident-Bond
  range directly, and made longest-path enumeration streaming so it retains
  only the current path and best result;
- rebuilt the identity-keyed topology index only at the force-field CLI
  ProcessPool deserialization boundary, leaving serial execution unchanged;
- migrated dataset SMARTS capability detection and every active test oracle
  away from Graph-returning Molecule properties;
- deleted the transitional `_graph` initialization/rebuild and the public
  `graph`, `simple_graph`, and `atom_bond_graph` properties;
- deleted the unused cached `adjacency_matrix`, unimplemented
  `super_acyclic_graph()`, and private NetworkX node/edge packing helpers;
- removed the final Core, coordination, and Open Babel NetworkX imports; and
- migrated hidden-bond, ring, geometry, kekulization, SMARTS, and coordination
  tests to authoritative Atom/Bond/link-matrix facts or independent test-only
  NetworkX oracles.

Core change-budget result:

```text
new Core classes/methods/properties/attributes in Stage 6: 0
remaining Core NetworkX references: 0
remaining Molecule Graph properties/caches: 0
sole persistent derived topology index: _topology_index
```

Verification:

```text
direct algorithms and migrated scientific consumers:
  131 passed, 3 xfailed in 1.34 s
topology, mutation, hidden-bond, ligand-ring, coordination, and rollback group:
  96 passed, 2 existing warnings in 1.04 s
shortest-path and InternalCoordinates integration:
  2 passed, 1 existing warning
force-field CLI and worker-boundary regression group:
  53 passed in 4.21 s
broad cheminformatics suite excluding the two documented legacy smoke modules:
  1,783 passed, 6 skipped, 3 xfailed, 49 subtests passed in 245.33 s
```

Direct-topology benchmark evidence on fixed inputs is recorded in
`direct_topology_benchmark.json`:

| Operation | Final direct (us) | Reused legacy NetworkX (us) | Task-local NetworkX (us) |
| --- | ---: | ---: | ---: |
| 512-node, 576-edge components | 290.83 | 159.17 | 713.46 |
| 512-chain shortest path 7 -> 503 | 322.20 | 308.80 | n/a |

Component labeling is 1.83x slower than querying the removed persistent
NetworkX cache, but 2.45x faster than constructing a task-local NetworkX Graph.
The absolute 0.13 ms query delta is paired with the Stage 1 topology-refresh,
copy, and 79.3% retained-state improvements. The direct shortest path is 4.3%
slower than a reused NetworkX Graph and constructs no replacement cache.

The first longest-path implementation materialized all paths. On the fixed K10
pair it peaked at 14,933,160 bytes. Streaming retains identical deterministic
output and similar wall time while peaking at 9,312 bytes, a 1,604x reduction.

The direct algorithms intentionally make three previously backend-accidental
edge cases explicit and test them: disconnected longest-path selection returns
the deterministic longest component path instead of aborting on the first
unreachable pair; both public shortest-path-index spellings return `[]` for a
disconnected pair instead of exposing one NetworkX-specific exception; and
Z-matrix construction includes a declared single isolate but rejects a
multi-component topology instead of silently omitting nodes absent from the
edge array.

The broad legacy `test_core.py::test_atompair` remains intentionally outside
the focused graph gate for the reason frozen at Stage 0: it performs 3D
construction on a very large molecule and is not a bounded graph regression
test. The maintained targeted tests cover the affected topology semantics.

Git checkpoint: `fd86063` (`refactor(core): remove persistent graph state`).

### Stage 7 - remove remaining public Graph-like surfaces

Status: complete; checkpoint pending.

Performed:

- changed the retained depth-first traversal helpers to explicit
  `node_count`/edge inputs and removed their public NetworkX-typed boundary;
- deleted the public MCA `MoleculeGraph` dataclass, module, lazy export, and
  import path with no alias;
- migrated MCA API and serialization coverage to the already-supported
  generic `to_rdmol()` adapter protocol;
- migrated the final repository example from the removed ligand-basis
  property to a task-local copy with hidden coordination Bonds and
  `basis_rings`;
- updated current API tables, README examples, MCA installation guidance,
  graph-package documentation, and development guidance to the native RI and
  non-Graph topology contracts; and
- confirmed that GraphSpectrum and neural-network/ONNX `.graph` values remain
  numerical/model objects, not exposed molecular topology holders.

Verification:

```text
MCA public input, serialization, and import layout:
  41 passed
explicit-edge traversal contract:
  7 passed
Python compilation and repository removed-symbol audit:
  passed
```

The only `MoleculeGraph` reference outside historical Plan prose is the
negative import-layout assertion proving that it is absent. Test-only
NetworkX use remains limited to independent scientific oracles, generators,
and layouts.

Git checkpoint: pending Stage 7 commit.

### Stage 8 - direct-target consolidation and optional performance work

Status: not started.

### Stage 9 - dependency and packaging cutover

Status: not started.

### Stage 10 - final closure

Status: not started.
