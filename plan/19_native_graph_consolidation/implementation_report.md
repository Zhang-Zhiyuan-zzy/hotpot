# Plan 19 implementation report

Status: implementation complete on `clean/networkx`.

Baseline: `2704f3c`.

This report closes `native_graph_consolidation_implementation.md` using only
evidence recorded on this branch. It does not claim a local Python 3.9--3.14
matrix run, a live GPU run, or dynamic PyG coverage.

## 1. Result and delivery sequence

Plan 19 replaced the persistent NetworkX molecular graph and duplicated Atom
adjacency state with one authoritative Python topology and one rebuildable
query index. It introduced one private, ephemeral C++ CSR-style Graph under
`hotpot.cheminfo.graph`, made native search and MCA exact automorphism compose
that Graph, moved cycle-basis work into the native graph sublibrary, and kept
one-shot topology algorithms on direct inputs where Graph construction did
not pay.

The final production tree has no NetworkX import, runtime dependency, public
Graph object, compatibility backend, or old cycle-basis alias. Search mapping
semantics, MCA ordered sites and `is_Nuc_site` ownership, accepted MCA
prediction hashes, Relevant Cycle semantics, and maintained model tests pass.
The only intentional ring-tensor change is the approved deterministic ordering
of the default cycle basis.

| Stage | Commit | Result |
| --- | --- | --- |
| baseline | `089fed8` | froze dependency, semantic, ordering, prediction, and performance evidence |
| topology authority | `0a7fcdc` | established the single derived topology index and removed Atom adjacency ownership |
| private Graph | `51cc4b6` | added the graph-owned C++ primitive without a Python Graph binding |
| search | `cb17165` | changed RI search to compose the shared Graph and removed its NetworkX query surface |
| MCA | `6eac3fa` | replaced NetworkX rooted isomorphism with native fixed-root exact matching |
| cycles | `2848fb4` | added deterministic default/all cycle bases and unified native cycle ownership |
| Core | `fd86063` | removed persistent Graph state and migrated direct topology algorithms |
| public API | `c355088` | deleted remaining Graph-like public surfaces and migrated consumers |
| performance | `310921a` | added the measured MCA sparse all-pairs cutover |
| packaging | `57a0418` | removed the production NetworkX dependency and validated a clean wheel |
| release CI | `c788ac4` | migrated wheel smoke coverage to the final native graph extension |

## 2. Final architecture and authority

| Fact or result | Final owner | Lifetime and access |
| --- | --- | --- |
| ordered atom membership | `Molecule._atoms` | persistent domain fact |
| ordered active-bond membership | `Molecule._bonds` | persistent domain fact |
| atom-local chemistry | the corresponding `Atom` | persistent domain fact |
| bond endpoints and bond-local chemistry | the corresponding `Bond` | persistent fact and sole connectivity authority |
| active versus hidden bond membership | existing Molecule active/hidden collections | persistent fact; membership is exclusive |
| atom row and ordered incident Bonds | `Molecule._topology_index` | private read-only derived index, deterministically rebuildable from the facts above |
| coordinates and conformers | unchanged existing owners | outside Plan 19 |
| native adjacency, degree, and edge lookup | private `hotpot::graph::Graph` | immutable task-local snapshot, destroyed after the operation/cursor |
| search predicates and MCA exact labels | task-specific columns composed with Graph | task-local, never chemical authority |
| rings, paths, components, bases, spectra | returned algorithm results | caller-owned results, never connectivity authority |

`Molecule._topology_index` has the contracted conceptual shape
`dict[id(atom), (atom_row, ordered_incident_bonds)]`. It contains references to
authoritative Bonds rather than copied chemistry and is the sole persistent
derived topology index. `Atom.neighbours`, `Atom.bonds`, `Atom.idx`, and
connectivity-related access derive from this index and Bond endpoints. The
unrelated `AtomSeq._bonds` constituent-bond representation remains intact.

The single native base is `hotpot::graph::Graph` in
`hotpot/cheminfo/graph/_native/graph.hpp/.cpp`. It stores normalized topology
and CSR-style adjacency only. It is neither exposed to Python nor cached on a
Molecule. Search, anchored MCA automorphism, Relevant Cycles, and cycle-basis
algorithms compose it for a coarse native task. Components, ordinary Molecule
shortest paths, Z-matrix traversal, and similar one-shot operations consume
explicit node/edge or incident-Bond inputs directly.

The Python/C++ boundary is coarse: current topology and only required property
columns are packed once; native work makes no per-neighbor Python callback;
and the snapshot cannot mutate or supersede the Python domain facts.

## 3. Core change-budget proof

| Category | Added | Removed | Result |
| --- | --- | --- | --- |
| classes | none | none relevant | pass: no new Core class |
| public methods | none | obsolete Graph surfaces below | pass: no new public method |
| public properties | exactly `cycle_basis`, `all_cycle_basis`, `basis_rings`, `all_basis_rings` | old Graph/basis properties | pass: exactly four approved exceptions |
| private Molecule methods | `_update_topology(clear_conformers: bool = True)` | `_update_graph`, `_update_atom_neighbours_bonds`, `_edge_with_attrs`, `_node_with_attrs`, `_ring_graph`, `_uncached_cycle_basis_rings` | pass: one authorized replacement updater, no parallel updater |
| private Molecule attributes | `_topology_index` | `_graph`, `_cycle_basis_rings`, `_ligand_cycle_basis_rings` | pass: one authorized index; no native handle, revision, pair map, or second index |
| private Atom attributes | none | Atom-specific `_neighbours`, `_bonds` | pass: duplicate adjacency ownership removed |
| Bond/InternalCoordinates additions | none | none | pass |

Static closure found one native `class Graph final`, one definition of each of
the four cycle properties, no `self._graph` or `self._neighbours` residue in
Core, and no added Core class. Runtime inspection confirms the removed
Molecule surfaces are absent and the four replacements are present.

## 4. Deleted API and consumer disposition

No compatibility alias or backend switch remains.

| Deleted API/state | Consumer disposition |
| --- | --- |
| `Molecule.graph`, `simple_graph`, `atom_bond_graph` | algorithms/tests use Atom/Bond facts, `_topology_index`, explicit edges, or independent test-only NetworkX oracles |
| `Molecule.adjacency_matrix` | no active consumer; spectrum retains direct numerical matrix behavior |
| `Molecule.super_acyclic_graph` | deleted as unimplemented and unused |
| `Substructure.construct_graph()`, `query_graph`, cache/signature | production RI consumes compiled facts; tests build independent graphs from `query_atoms`/`query_bonds` where needed |
| MCA `MoleculeGraph`, module, lazy export | callers/tests use Molecule, SMILES, RDKit, Open Babel, or generic `to_rdmol()` input |
| NetworkX-typed traversal API | traversal accepts explicit `node_count` and edges |
| Molecule `cycle_basis_rings`, `ligand_cycle_basis_rings`; Atom `cycle_basis_rings` | replaced by four explicit basis properties; ligand model extraction calls the private default-basis algorithm on coordination-filtered edges |
| `_cycle_basis_rings`, `_ligand_cycle_basis_rings`, `_uncached_cycle_basis_rings` | deleted; basis results are recomputed from current topology and caller-owned |
| persistent `_graph` and working-copy Graph snapshots | rollback restores authoritative Atom/Bond collections and rebuilds `_topology_index` |
| Atom `_neighbours` and Atom-specific `_bonds` | public relationship queries use the Molecule index plus Bond endpoints |
| seven production NetworkX import sites | removed; NetworkX remains test-only |

The ligand-ring example now makes a task-local Molecule copy, hides
coordination Bonds, and reads `basis_rings`. Dataset SMARTS checks,
force-field components, hidden-bond/ring tests, and geometry/kekulization tests
were migrated before deletion of their Graph access.

## 5. Semantic and ordering evidence

### 5.1 Search

There remains one matcher: RI-style C++. It composes the graph-owned base
rather than search-owned `GraphData`; no second target construction or Python
fallback was added. Query planning, domains, lazy cursor behavior, truncation,
mappings, and deterministic candidate order remain unchanged. Python callable
predicates still execute only at their leaf boundary.

The native/search/SMARTS gate passed 517 tests plus 49 subtests. Its frozen
semantic/performance digest remained
`a2c1f6c232b1b69f87599eb9bfa843a62846f726dbbe7fc8077fdd2f414c43f4`.

### 5.2 MCA site detection and prediction

NetworkX copies and rooted `is_isomorphic` calls were replaced with a
fixed-root exact native problem reusing the packed target. Exact node labels
are `(atomic_number, formal_charge, is_aromatic, implicit_hydrogens, isotope)`;
edge comparison retains aromatic, bond-order, and bond-kind semantics. Python
1-WL is rejection-only, not semantic authority.

Ordered rules, representative selection, site types, isotope/coordination
behavior, and `is_Nuc_site` ownership are unchanged. The complete MCA suite
passed 194 tests, and the accepted prediction SHA-256 values remain exact:

| Workload | SHA-256 |
| --- | --- |
| batch 5 | `1b39a16d8c84f718184d89ed623ba56ec69aa5ffeacd6eb84138ed5949538a7f` |
| batch 8 | `ddeb4f8635c47a7f40aa8fff8576a5dbf94c8c2bb525381752d952d0a44c4f11` |
| batch 64 | `6e60a9f38359c05702f503f76bbbd710476469eac0d3837377a7044b5d73ec92` |

These are CPU/model parity results. No live GPU result is claimed.

### 5.3 Relevant Cycles and cycle bases

Relevant Cycles, the default basis, and all simple-cycle bases remain three
distinct concepts. None is redefined as SSSR, a minimum basis, or a chemically
unique ring set. The final public surface is:

```text
Molecule.cycle_basis       -> list[tuple[int, ...]]
Molecule.all_cycle_basis   -> list[list[tuple[int, ...]]]
Molecule.basis_rings       -> list[Ring]
Molecule.all_basis_rings   -> list[list[Ring]]
```

The default is deterministic relative to indexed topology: maximum-index root
per component, ascending neighbors, fixed Paton LIFO/predecessor/used events,
and discovery-order output. It is not atom-renumbering invariant. All-bases
enumeration canonicalizes under rotation/reversal, uses GF(2) rank-pruned
backtracking, emits the default first and the rest canonically, and has an
atomic 10,000-complete-basis ceiling. Acyclic semantics are
`cycle_basis == []` and `all_cycle_basis == [[]]`.

Independent oracle checks cover acyclic, triangle, square-with-diagonal, K4,
disconnected, sparse-key, shuffled-edge, and bounded graph-atlas/generated
graphs. Recorded counts include one empty basis for an acyclic graph, three
bases for a square with diagonal, and 28 for K4. K5 exceeds the real 10,000
ceiling and raises `OverflowError` with no partial result; native tests also
prove exact-limit success and next-result failure using a smaller injected
limit. Ring properties are exact projections of index properties, and none is
cached on Molecule.

Approved deterministic ordering changes cubane from

```text
4 7 6 5 | 2 1 7 6 | 3 0 1 7 6 5 | 4 0 1 7 | 3 2 6 5
```

to

```text
4 5 6 7 | 2 3 5 6 | 1 0 3 5 6 7 | 4 0 3 5 | 1 2 6 7
```

Both contain five cycles sized `(4, 4, 6, 4, 4)`. This is an intentional
orientation/order change, not hidden basis-set drift. NumPy and installed
CBond consumers use the new tensor; the CBond runtime/prediction group passed
42 tests. PyG was unavailable and its shared cycle-source boundary was
inspected statically only.

### 5.4 Direct topology semantics

Components, paths, and traversal now have deterministic ordering without a
Graph cache. Three backend-accidental edge cases were made explicit and
tested: disconnected longest-path selection returns the deterministic longest
component path; both shortest-path-index spellings return `[]` for a
disconnected pair; and Z-matrix traversal includes one declared isolate but
rejects multi-component topology instead of silently omitting nodes.

## 6. Performance evidence and attribution

Figures are recorded 21-sample medians on Linux 5.15/CPython 3.11.16. The
JSON files preserve revision and dirty-state metadata. Categories are kept
separate because the private Graph did not cause every improvement.

### 6.1 Removing NetworkX and duplicate Python state

| 512-atom chain | Legacy | Final | Change |
| --- | ---: | ---: | ---: |
| retained derived topology | 653,181 B | 135,176 B | -79.3% |
| `Atom.idx` | 9.359 us | 3.178 us | 2.95x faster |
| connectivity | 3.316 us | 3.192 us | 3.8% faster |
| topology refresh | 15,720 us | 12,783 us | 18.7% faster |
| Molecule copy | 61,901 us | 52,560 us | 15.1% faster |
| neighbors materialization | 3.240 us | 8.223 us | 2.54x slower |
| direct Bond lookup | 0.454 us | 5.007 us | 11.0x slower |

The last two are the explicit cost of removing redundant Atom lists and not
adding a persistent bond-pair map. They remain small in absolute terms and
preserve the authority/state budget. Atom-row and incident-tuple location are
O(1); materialization scales with returned degree, not molecule size.

### 6.2 Direct algorithms rather than a persistent Graph

| Workload | Final direct | Reused legacy NetworkX | Task-local NetworkX |
| --- | ---: | ---: | ---: |
| components, 512 nodes/576 edges | 290.83 us | 159.17 us | 713.46 us |
| chain shortest path, 7 to 503 | 322.20 us | 308.80 us | n/a |

Components are 1.83x slower than querying the removed persistent cache but
2.45x faster than rebuilding NetworkX. Shortest path is 4.3% slower than a
reused NetworkX Graph. These are accepted consolidation trade-offs, not native
speed claims. Streaming longest-simple-path retained output and similar K10
time (79.54 versus 78.31 ms) while reducing peak traced memory from 14,933,160
to 9,312 bytes, approximately 1,604x.

### 6.3 Native algorithms versus neutral consolidation

- Search's Graph extraction is neutral ownership consolidation: no second
  definition or target construction, and no generic search speedup claim.
- MCA exact matching replaces NetworkX and reuses the packed target; scientific
  and ordering parity, not a whole-workflow Graph speedup, is the acceptance.
- Uncached native `cycle_basis` measured 46.36 us on cubane versus 60.79 us
  for the legacy Ring-materializing call. Different return layers make this a
  regression guard, not a like-for-like speedup claim.
- Force-field ring workspaces already reuse topology/geometry revisions within
  coarse stages. Further consolidation was rejected because remaining scopes
  differ and a cache would add API/invalidation state without measured gain.

### 6.4 MCA all-pairs optimization

This gain comes from a measured sparse-algorithm cutover, not Graph. At 256+
atoms, feature construction lazily uses SciPy sparse shortest paths; smaller
inputs retain dense NumPy. There is one fixed threshold and no profile.

| Atoms | Dense | Final | Speedup | Exact parity |
| ---: | ---: | ---: | ---: | --- |
| 64 | 309.97 us | 305.69 us | 1.01x | yes |
| 128 | 1,263.42 us | 1,250.87 us | 1.01x | yes |
| 255 | 8,259.74 us | 8,518.59 us | 0.97x | yes |
| 256 | 7,616.41 us | 3,399.33 us | 2.24x | yes |
| 512 | 55,432.33 us | 14,038.67 us | 3.95x | yes |

Directed weighted behavior, `int32`, disconnected/clipped value 510, shapes,
and downstream values are exact at every measured size.

## 7. Packaging and dependency evidence

NetworkX was removed from `project.dependencies`, base requirements, and the
production runtime-distribution identity. It remains only in
`tests/requirements-inference.txt` as an independent oracle/generator/layout
dependency. The pre-existing `cryptography` project dependency was synchronized
into base requirements to retain the identical-list contract.

Packaging metadata/runtime identity passed 16 tests. A clean CPython 3.11
wheel built with uv/PEP 517; METADATA has no `Requires-Dist: networkx`; and
extracted `_graph_native`/`_search_native` import successfully. The wheel has
neither obsolete `_relevant_cycles` nor `graph_adapter`.

An initial incremental wheel exposed ignored stale `_relevant_cycles` output
in an old `build/` directory. A clean rebuild passed. This was generated
residue, not tracked source or a retained backend.

The cibuildwheel release smoke command was also migrated from the deleted
`_relevant_cycles` import to `_graph_native` and now asserts the final
cycle-basis symbol before exercising search and Relevant Cycles.

Final static audits found zero production NetworkX references, zero deleted
API references, zero runtime NetworkX declarations, one private native Graph
definition, no Python Graph binding, and no obsolete backend/fallback.

## 8. Test record

| Area | Result |
| --- | --- |
| native Graph/cycle source | C++17 build passed; cycle contract passed `-Wall -Wextra -Werror` |
| native search/SMARTS/contracts | 517 passed, 49 subtests |
| native search plus focused MCA cutover | 377 passed |
| bases/oracle/Relevant Cycles/Core rings/force-field/MCA digests | 190 passed, 4 skipped |
| direct algorithms/scientific consumers | 131 passed, 3 xfailed |
| topology/mutation/hidden Bonds/coordination/rollback | 96 passed |
| force-field CLI/worker boundary | 53 passed |
| MCA public input/serialization/import layout | 41 passed |
| explicit-edge traversal | 7 passed |
| complete MCA after final optimization | 194 passed in 18.69 s |
| final graph directory | 118 passed, 4 skipped in 6.03 s |
| final SMARTS conformance | 282 passed in 2.06 s |
| final CBond | 42 passed in 1.49 s |
| packaging/runtime identity | 16 passed in 2.73 s |
| independent final semantic/packaging audit | 126 passed in 6.27 s |
| native-extension absence behavior | 2/2 raised explicit `ImportError`; no Python fallback |
| broad cheminformatics suite excluding documented smoke modules | 1,783 passed, 6 skipped, 3 xfailed, 49 subtests in 245.33 s |

The broad exclusion is explicit: `test_core.py::test_atompair` performs large
3D construction and is not a bounded graph regression; affected behavior is
covered by focused tests. `test_elements.py` was not collected because Numba
was unavailable. Existing skips/xfails are not presented as new passes.

Machine-readable evidence is in `legacy_topology_benchmark.json`,
`topology_index_benchmark.json`, `direct_topology_benchmark.json`, and
`mca_shortest_paths_benchmark.json`.

## 9. Residual limitations and non-claims

1. Local validation used CPython 3.11. The repository's Python 3.9--3.14 CI
   matrix was not run locally and must be judged from pushed-branch CI.
2. No live GPU workflow or throughput benchmark was run for this closure.
3. PyG was unavailable; its migration has static boundary evidence only.
4. All-bases enumeration is exponential, bounded at 10,000 complete bases,
   and fails atomically beyond it.
5. Default-basis uniqueness is index-relative, not chemical uniqueness,
   minimum-basis semantics, or atom-renumbering invariance.
6. Neighbor materialization and Bond lookup regress in microbenchmarks;
   restoring another persistent cache would violate the approved authority and
   Core-state budget.
7. RDKit/Open Babel internals, conformer generation, and geometry ownership
   were outside Plan 19 and remain unchanged.
8. Direct private mutation of `_atoms`/`_bonds` outside supported Molecule
   mutation paths remains unsupported; supported mutations rebuild the index.

## 10. Completion assessment

Plan 19 meets its architectural and semantic definition of done: Bond
endpoints are the sole Python connectivity facts; `_topology_index` is the
sole persistent derived index; the only computational Graph is private,
native, and ephemeral; active consumers were migrated before API deletion;
search, MCA, cycle, model, packaging, and ordering gates passed; and production
code/runtime metadata are free of NetworkX.

The measured outcome is intentionally mixed rather than a blanket speedup:
topology retention, refresh/copy, and large MCA all-pairs work improve; some
direct one-hop queries regress in small absolute time; search consolidation is
structurally neutral; and native cycle/MCA changes are accepted primarily on
deterministic/scientific parity. Subject to the explicit unrun cross-version,
live-GPU, and PyG validations above, Plan 19 is ready for branch review and CI.
