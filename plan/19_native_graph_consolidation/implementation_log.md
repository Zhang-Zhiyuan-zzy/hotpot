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

Git checkpoint: pending Stage 5 commit.

### Stage 6 - remove Core graph state

Status: not started.

### Stage 7 - remove remaining public Graph-like surfaces

Status: not started.

### Stage 8 - direct-target consolidation and optional performance work

Status: not started.

### Stage 9 - dependency and packaging cutover

Status: not started.

### Stage 10 - final closure

Status: not started.
