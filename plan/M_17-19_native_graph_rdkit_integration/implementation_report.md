# M_17-19 implementation report

Status: complete on `merge/m17-19-native-graph-rdwrapper`.

## 1. Outcome

Plan 17 and Plan 19 are now integrated by the two-parent merge commit
`b5e04d6`. The final implementation uses Plan 19's graph-facing interfaces,
single topology authority, native graph/search ownership, and deletion policy.
It retains Plan 17's Level-1/Level-2 stereochemistry, common molecule
transport, locked native RDKit, rdWrapper Python/C/C++ components, and exact
force-field transaction semantics.

The result has no compatibility Graph API, NetworkX production dependency,
old topology updater, GraphMatcher-shaped search adapter, archived SymPy
search island, MCA `MoleculeGraph`, RDKit legacy hydrogen namespace, or second
molecule buffer representation.

| Item | Identity |
|---|---|
| integration branch | `merge/m17-19-native-graph-rdwrapper` |
| worktree | `/tmp/hotpot-m17-19` |
| initial parent | `6639754` |
| plan checkpoint | `f4b9a94` |
| graph parent | `b912c8a` |
| merge base | `2704f3c` |
| semantic merge | `b5e04d6` |
| merge parents | `f4b9a94`, `b912c8a` |

Both requested histories are ancestors of the result. The original main
worktree and its pre-existing uncommitted Plan 17 files were not modified.

## 2. Final authority and unique implementations

| Concern | Final sole owner |
|---|---|
| atom membership/order | `Molecule._atoms` |
| active Bond membership/order | `Molecule._bonds` |
| connectivity facts | Bond endpoints and active/hidden membership |
| derived atom row/incidence index | `Molecule._topology_index` |
| topology refresh | `Molecule._update_topology()` |
| native adjacency snapshot | task-local `hotpot::graph::Graph` |
| generic subgraph matching | RI-style `_search_native` |
| Python packed target | private `_NativeTarget` |
| stereo chemical state | explicit `Molecule._stereo_payload` |
| Python/C++ molecule transport | `hotpot.cheminfo._native` common buffers |
| RDKit conformer component | locked `hotpot.cheminfo.rdWrappers` |
| complex default route | existing O0 OpenBabel/Hotpot workflow |

Removed Atom objects retain the established empty `bonds`/`neighbours`
inspection behavior by detecting non-membership in the sole topology index.
No Atom-local adjacency cache was reintroduced.

The exact final public/private class and function API trees are frozen in
`native_graph_rdkit_integration_implementation.md`. The exhaustive 97-row
adjustment/deletion inventory is
`api_implementation_migration_matrix.md`; all rows are `Implemented` and
`Verified`, every `Compatibility retained` value is `No`, and each row records
its final owner and validation gate.

## 3. Conflict resolution

The merge produced six text-conflicted files. Each was resolved semantically:

| File | Resolution |
|---|---|
| `.github/workflows/publish_pypi.yml` | one smoke path for graph/search/geometry/OpenBabel/rdWrapper native modules |
| `hotpot/cheminfo/core.py` | Plan 19 topology and APIs plus Plan 17 stereo lifecycle and hidden-Bond semantics |
| `hotpot/cheminfo/forcefields/working_copy.py` | authoritative topology snapshot/refresh plus exact stereo commit/rollback |
| `plan/README.md` | retain Plans 17, 18, 19, and M_17-19 |
| `pyproject.toml` | exact RDKit/OpenBabel requirements, no runtime NetworkX/SymPy legacy extra |
| `requirements.txt` | same runtime policy as project metadata |

No unresolved conflict or whole-file parent selection remains.

## 4. Main implementation changes

### 4.1 Core graph and mutation

- Deleted persistent `_graph`, Graph-returning properties, duplicate Atom
  adjacency state, old cycle caches, `_update_graph()`, and
  `_update_atom_neighbours_bonds()`.
- Folded single mutation into the unique bulk paths and deleted
  `_rm_atom(s)`/`_rm_bond(s)`.
- Preserved stereo remap/prune semantics for atom, active Bond, hidden Bond,
  AXIS, group, copy, component, and composition operations.
- Consolidated components and paths under direct integer topology algorithms;
  `Molecule.shortest_path_index()` now uses the shared shortest-path function.
- Replaced Graph/duck-typed Z-matrix and traversal inputs with explicit
  `node_count`, `edges`, or `link_matrix` signatures.

### 4.2 Search and MCA

- Retained one C++ RI-style matcher; no VF2 or Python fallback was added.
- Made native mapping sources emit query-to-molecule mappings directly.
- Consolidated generic search, MCA automorphism, and stereo Gate 6 on one
  `_NativeTarget` packer.
- Deleted GraphMatcher adapters, old SMARTS/SymPy modules, and their optional
  dependency extra.
- Preserved MCA ordered sites, native fixed-root automorphism, sparse distance
  cutover, accepted prediction digests, and `is_Nuc_site` ownership.

### 4.3 Stereo, transport, and rdWrapper

- Every Molecule now owns an explicit `StereoPayload`; the missing-field
  fallback was deleted.
- Retained opaque source namespaces/codes because they are chemical transport
  facts, not compatibility shims.
- Moved implicit-H completion into the common field-selective molecule packer.
- Deleted `_HOTPOT_LEGACY_NAMESPACE`,
  `_add_legacy_hydrogen_transport()`, and native `hotpot-legacy/1` parsing.
- Retained one exact C ABI v1, one C++ facade/target, and one Python rdWrapper
  contract pinned to RDKit 2026.09.1 (Python distribution 2026.9.1).

### 4.4 Force fields

- Working copies snapshot authoritative molecule facts and exact stereo only;
  Graph/neighbour/ring-cache snapshots were deleted.
- Rollback restores facts, refreshes topology once, and restores the exact
  payload. Commit exposes the working payload before downstream operations so
  exception rollback is atomic.
- Spawned force-field boundaries rebuild the identity-keyed topology index.
- The overall force-field workflow and O0 complex default were not changed.

## 5. Deleted files and source relocation

Deleted production modules:

| Area | Deleted files |
|---|---|
| MCA | `AImodels/mca/graph_adapter.py` |
| archived search | `_logic.py`, `_logic_tuple.py`, `_smarts.py`, `fields.py`, `logic.py`, `logic2.py`, `match_func.py` |
| archived parser | `smarts_parser/__init__.py`, `lexer.py`, `parser.py`, `tonx.py` |

Relocations/consolidations:

| Old | Final |
|---|---|
| `graph/_relevant_cycles.pyi` | `graph/_graph_native.pyi` |
| `search/_native/graph_data.cpp/.hpp` | `graph/_native/graph.cpp/.hpp` |
| `test_search_native_spec_target.py` | `test_search_native_target.py` |

The semantic merge commit changes 162 files with 9,555 insertions and 4,232
deletions relative to the initial parent. The normative migration matrix,
rather than this summary, is the complete interface/implementation table.

## 6. Test and validation results

| Gate | Result |
|---|---:|
| focused topology/search/graph/stereo | 358 passed, 4 skipped |
| full stereo/lifecycle | 600 passed |
| search/SMARTS | 521 passed, 49 subtests |
| complete MCA | 194 passed |
| real native rdWrapper/ETKDG/stereo | 39 passed |
| force-field integration | 252 passed; corrected fixture rerun passed |
| CBond/cycle consumers | 53 passed |
| broad cheminformatics | 2459 passed, 7 skipped, 3 xfailed, 49 subtests |
| packaging metadata | 10 passed |
| standalone rdWrapper C/C++ SDK | 7 passed |
| cohort/route/runner/working-copy contracts | 32 passed |
| final focused regression | 79 passed |

The broad run used CPython 3.11, `--import-mode=importlib`, a writable temporary
model cache, and the rebuilt native rdWrapper. Its four warnings are existing
tests that deliberately retain rejected geometry for inspection.

MCA validation includes exact historical prediction digests and the Boolean
`is_Nuc_site` contract. It does not claim a live GPU run.

## 7. 182-case O0 result and efficiency

The cohort was rebuilt from the tracked 187-entry extractant SMILES file and
current CBond output. It produced 182 connected coordination cases with
canonical SHA-256
`8f68e8b74d51eac07f28846cb901e92b94e5697aa0495733dbc44e03ae198102`.

| O0 result | Value |
|---|---:|
| planned/observed | 182 / 182 |
| missing/duplicate/unexpected | 0 / 0 / 0 |
| automated accepted/rejected | 182 / 0 |
| A01-A10 | each 182 / 182 |
| per-case total p50 | 7.448 s |
| per-case total p95 | 18.499 s |
| per-case total maximum | 41.559 s |
| summed case time | 1,598.237 s |
| summed user/system CPU | 842.670 / 0.536 s |
| maximum observed process high-water RSS | 235,572 KiB |

Stage timing:

| Stage | p50 (s) | p95 (s) | sum (s) |
|---|---:|---:|---:|
| ligand build | 2.683 | 5.486 | 527.800 |
| optimization | 2.770 | 11.324 | 758.436 |
| coordination restoration | 0.284 | 0.662 | 57.999 |
| acceptance | 0.031 | 0.058 | 5.883 |
| semantic reconstruction | 0.005 | 0.008 | 0.882 |
| selected-structure write | 0.013 | 0.027 | 2.527 |

A01-A10 respectively verify mapped proposal, setup, finite selected
coordinates, finite optimal energy, finite selected gradients, recovery from
unavailable-gradient events, non-exploded selected frame, standard geometry,
graph/stereo semantics, and explicit stage failure/recovery records.

This was the full 182-case O0 route, not a reference-coordinate comparison.
The runner marks the invocation non-formal because its formal mode requires
the complete O0/F1/F2/C1/C2 plus repeat matrix (1,092 results). That alternative
matrix was not rerun because O0 remains the decided complex default. No new
human visual review was performed.

## 8. Clean artifact evidence

A detached clean worktree at `b5e04d6` produced:

- `hotpot_zzy-0.5.4.0.tar.gz` (1.4 MiB); and
- `hotpot_zzy-0.5.4.0-cp311-cp311-linux_x86_64.whl` (5.2 MiB), built from
  that sdist.

The wheel contains and imports `_graph_native`, `_search_native`,
`_geometry_native`, `_ob_native`, and `_rd_native`. It contains none of the
old Relevant-Cycles binary, MCA graph adapter, or archived search modules.
METADATA has no runtime NetworkX requirement and retains exact RDKit 2026.9.1
plus OpenBabel `>=3.2.1,<3.3`. An installation outside the checkout passed
native imports, substructure search, Relevant Cycles, and rdWrapper runtime
identity (`2026.09.1`).

The separately installed `Hotpot::rdwrappers` SDK passed C11/header and C++17
consumer tests under both libstdc++ ABI macro values.

## 9. Deviations and non-claims

- Live GPU execution was unavailable; no GPU throughput/parity result is
  claimed by this merge.
- Local execution covered CPython 3.11, not the complete 3.9-3.14 release
  matrix.
- Dynamic PyG coverage was unavailable where its optional dependency was not
  installed; shared cycle-source behavior was covered by static and other
  model consumers.
- The full five-route Plan 17 formal matrix and F1 repeat were not rerun.
- The 182 O0 structures received automated A01-A10 evaluation but no new human
  visual review.

These limitations do not require a compatibility implementation or parallel
API. They remain environment/review gates for release qualification if the
owner requests those broader claims.
