# M_17-19 implementation log

## Baseline

| Item | Value |
|---|---|
| Integration branch | `merge/m17-19-native-graph-rdwrapper` |
| Integration worktree | `/tmp/hotpot-m17-19` |
| Current parent | `6639754` |
| Graph parent | `clean/networkx@b912c8a` |
| Merge base | `2704f3c` |
| Started | 2026-10-26 |

## Stage 0 — Freeze plan, current assets, and disposition matrix

Status: complete

Intent:

- preserve the current main worktree exactly as found;
- create a separate branch/worktree for all integration work;
- carry the uncommitted Plan 17 assessment, benchmarks, and proposed organic
  native conformer extension into the integration branch;
- freeze merge precedence, the final API tree, conflict semantics, deletion
  policy, validation gates, and the complete API/implementation disposition
  matrix before running `git merge`.

Performed:

- created branch `merge/m17-19-native-graph-rdwrapper` at `6639754`;
- created worktree `/tmp/hotpot-m17-19`;
- copied the three modified and five untracked Plan 17 files without changing
  the source worktree;
- recorded both parents and merge-base;
- created this integration plan and the normative migration matrix;
- expanded the deletion audit beyond Plan 19 to include the constant-empty
  traversal compatibility API, the disconnected SymPy search implementation,
  GraphMatcher-shaped native-search adapters, missing-stereo-field fallback,
  RDKit `hotpot-legacy/1` transport, and proposed old-prefix ABI handling;
- explicitly retained RI candidate-domain `compatibility.hpp`, opaque stereo
  source facts, the current exact-version C API/SDK target, and the private
  conformer sampler because they are active business mechanisms rather than
  compatibility layers; and
- corrected the proposed Plan 17 organic-native plan so a future ABI extension
  accepts only its one current full structure/version.

Conflict/deviation:

- none; merge has not started.

Validation:

- `git diff --check`: passed;
- Python syntax compilation for all three copied benchmark drivers: passed;
- relative Markdown link resolution for Plan 17, M_17-19, and the Plan index:
  zero missing targets;
- stale legacy-prefix wording scan: only the intentional deletion row in the
  migration matrix remains.

Git checkpoint: `f4b9a94 plan(M_17-19): freeze graph RDKit integration`.

## Stage 1 — Merge Plan 19 and resolve the six textual conflicts

Status: complete

Performed:

- started `git merge --no-ff --no-commit clean/networkx` from the frozen plan
  checkpoint;
- resolved the six conflicted files without selecting either parent wholesale:
  `publish_pypi.yml`, `core.py`, `working_copy.py`, `plan/README.md`,
  `pyproject.toml`, and `requirements.txt`;
- retained Plan 19's native graph/search/MCA topology and Plan 17's exact
  RDKit/OpenBabel pins, rdWrapper build, stereo state, and force-field
  transaction semantics; and
- combined native build sources and CI smoke imports for `_graph_native`,
  `_search_native`, `_geometry_native`, `_ob_native`, and `_rd_native`.

Conflict/deviation:

- none remains unresolved; `git diff --name-only --diff-filter=U` is empty.

## Stage 2 — Establish one topology/mutation/stereo implementation

Status: complete

Performed:

- made `_atoms`, `_bonds`, Bond endpoints, and `_topology_index` the only
  persistent topology owners;
- initialized every `Molecule` with an explicit empty `StereoPayload` and
  deleted missing-field fallback behavior;
- integrated copy, component extraction, component composition, hidden Bonds,
  and stereo remapping on the Plan 19 topology path;
- folded single Bond removal into the bulk implementation and deleted
  `_rm_bond()`/`_rm_bonds()`;
- consolidated atom removal into `_remove_atoms_from_topology()` and deleted
  `_rm_atom()`/`_rm_atoms()`;
- preserved empty adjacency inspection for externally retained, removed Atom
  objects without restoring Atom-local caches;
- deleted the unused OpenBabel `add_hydrogens()` compatibility helper; and
- made `Molecule.shortest_path_index()` call the sole direct shortest-path
  implementation in `graph.paths`.

Unexpected finding and resolution:

- the first lifecycle run exposed that a retained removed Atom no longer had
  an index entry. `Atom.bonds` now treats non-membership as empty incidence,
  and all derived neighbor/connectivity queries use that one result; no
  detached-state cache was added.

Verification:

- initial focused run: 291 passed, 2 failed, 4 skipped;
- after the lifecycle correction: 358 passed, 4 skipped;
- full stereo/lifecycle selection: 600 passed.

## Stage 3 — Consolidate native RI search and delete archived search code

Status: complete

Performed:

- retained one RI-style C++ matcher and one private `_NativeTarget` packed
  target owner for generic search, MCA automorphism, and stereo Gate 6;
- changed mapping sources to emit query-to-molecule mappings directly;
- renamed the result dependency from GraphMatcher-shaped `graph_matcher` to
  `mapping_source` and deleted `_node_match`, `_edge_match`,
  `_get_nodes_set`, `_NativeSpecTarget`, and the optional empty Hit mapping;
- deleted the disconnected SymPy/old-SMARTS module island and the
  `legacy-search` extra; and
- retained NetworkX only in independent test oracles.

Unexpected finding and resolution:

- a mechanical rename accidentally changed the independent NetworkX oracle to
  call a non-existent `iter_mappings()`. The oracle was restored to
  `subgraph_monomorphisms_iter()` while production remained GraphMatcher-free.

Verification:

- search/SMARTS selection: 521 passed plus 49 subtests;
- production GraphMatcher and NetworkX scans: zero references/imports.

## Stage 4 — Consolidate generic molecule transport and rdWrapper

Status: complete

Performed:

- made the common contiguous molecule packer the only owner of current
  implicit-hydrogen completion;
- deleted `_HOTPOT_LEGACY_NAMESPACE`,
  `_add_legacy_hydrogen_transport()`, and native `hotpot-legacy/1` parsing;
- retained the exact current C ABI v1, C++ facade, Python contracts, ETKDG
  sampler, and locked RDKit 2026.09.1 build; and
- retained absent namespace as the canonical Hotpot-origin current transport
  record instead of introducing another compatibility namespace.

Verification:

- locked RDKit/rdWrapper build: passed, 162 native build actions;
- real-extension rdWrapper/ETKDG/coordination/stereo selection: 39 passed;
- static legacy transport scan: zero active references.

## Stage 5 — Integrate force-field process and working-copy boundaries

Status: complete

Performed:

- removed Graph/neighbour/ring-cache snapshots from working copies;
- restored authoritative facts, rebuilt topology once, then restored the exact
  stereo payload on rollback and commit paths;
- rebuilt identity-keyed topology at spawned worker/process deserialization
  boundaries; and
- retained the private conformer-sampler seam without changing the public
  complex optimization workflow or its O0 default.

Unexpected finding and resolution:

- a Plan 17 worker unit fixture passed `object()` and therefore could not
  exercise Plan 19's mandatory child topology rebuild. The fixture now passes
  the real Molecule it already constructs; production behavior was unchanged.

Verification:

- force-field integration selection: 252 passed with one corrected fixture;
- corrected worker/working-copy/component selection: 5 passed.

## Stage 6 — Scientific and broad local validation before artifact closure

Status: complete

Results:

| Selection | Result |
|---|---:|
| topology/search/graph/stereo focused | 358 passed, 4 skipped |
| full stereo/lifecycle | 600 passed |
| search and SMARTS | 521 passed, 49 subtests |
| MCA complete CPU and simulated GPU contracts | 194 passed |
| rdWrapper with rebuilt native extension | 39 passed |
| force-field integration | 252 passed before fixture correction; correction passed |
| CBond/cycle consumers with writable model cache | 53 passed |
| broad cheminformatics (`--import-mode=importlib`) | 2459 passed, 7 skipped, 3 xfailed, 49 subtests |

The MCA run includes the accepted prediction digests and `is_Nuc_site`
contracts. No live GPU result is claimed. The first CBond attempt failed only
because its default model cache was read-only; rerunning with
`HOTPOT_MODEL_HOME=/tmp/hotpot-m17-19-models` passed all selected tests.

Remaining closure work:

- clean wheel/sdist audit;
- 182-case O0 integration validation from the tracked extractant source;
- final zero-reference gates, migration-matrix closure, and report.
