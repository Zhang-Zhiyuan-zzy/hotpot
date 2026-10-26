# M_17-19 API and implementation migration matrix

Status: implementation and local validation complete.

This is the sole normative inventory of interfaces and implementations changed
or deleted by M_17-19. `Compatibility retained` is intentionally `No` for every
row. Historical Plan 17/19 documents may retain names as evidence; production,
active tests, packaging, and current API documentation may not retain deleted
surfaces.

`Implemented` means the row's final disposition was completed: retained items
have one owner, replacement callers use only the final interface, and deleted
items have no active implementation or compatibility alias. `Verified` means
the evidence named in the final column passed. Live GPU execution and optional
dynamic PyG coverage were unavailable locally and are not claimed; MCA CPU
digests/`is_Nuc_site` and the shared cycle-source boundary were verified
instead.

| Layer | Existing symbol/artifact | Origin | Final disposition | Final unique owner/API | Required preservation | Compatibility retained | Implementation status | Validation status | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| Core authority | `Molecule._atoms`, `_bonds`, Bond endpoints | Both | Retain as sole topology facts | `Molecule`/`Bond` | identity, order, attributes | No | Implemented | Verified | topology tests |
| Core derived state | `Molecule._topology_index` | P19 | Retain as sole persistent topology index | `_update_topology()` | deterministic row/incidence order | No | Implemented | Verified | topology-index tests/static scan |
| Core refresh | `_update_topology(clear_conformers=True)` | P19 | Retain as sole refresh implementation | `Molecule._update_topology` | all current invalidation behavior | No | Implemented | Verified | mutation tests/static scan |
| Core refresh | `_update_graph()` | P17/legacy | Delete | `_update_topology()` | callers migrated, stereo handled separately | No | Implemented | Verified | zero-reference scan |
| Core refresh | `_update_atom_neighbours_bonds()` | Legacy | Delete | `_update_topology()` | public neighbor results unchanged | No | Implemented | Verified | zero-reference scan |
| Core graph helpers | `_edge_with_attrs`, `_node_with_attrs`, `_ring_graph`, `_uncached_cycle_basis_rings` | Legacy | Delete | explicit facts/native algorithms | scientific outputs | No | Implemented | Verified | zero-reference scan |
| Core graph state | `Molecule._graph` | Legacy | Delete | no persistent computational Graph | algorithms preserve outputs | No | Implemented | Verified | source/runtime scan |
| Core adjacency | `Atom._neighbours`, Atom-specific stored `_bonds` | Legacy | Delete | `_topology_index` queries | return types/order | No | Implemented | Verified | state-budget tests |
| Core adjacency lifecycle | removed `Atom.bonds`/`neighbours` empty-list behavior | P17 | Preserve without duplicate state | membership-aware queries over `_topology_index` | detached references remain inspectable | No | Implemented | Verified | structural lifecycle tests |
| OpenBabel adapter cache | `_obmol` / `_row2idx` | Active | Retain and invalidate together | one OpenBabel conversion cache | row-to-OB index mapping | No | Implemented | Verified | converter/cache tests |
| Core cache | cached `adjacency_matrix` instance state/property | Legacy | Delete | task-local numerical construction | graph spectrum parity | No | Implemented | Verified | spectrum tests |
| Core public graph | `Molecule.graph` | Legacy | Delete | domain APIs/direct algorithms | no Graph return | No | Implemented | Verified | API/static tests |
| Core public graph | `Molecule.simple_graph` | Legacy | Delete | domain APIs/direct algorithms | scientific callers migrated | No | Implemented | Verified | API/static tests |
| Core public graph | `Molecule.atom_bond_graph` | Legacy | Delete | native search snapshots | search semantics | No | Implemented | Verified | search tests |
| Core public graph | `Molecule.super_acyclic_graph()` | Legacy | Delete | none | none; unimplemented | No | Implemented | Verified | source scan |
| Core cycles | `cycle_basis_rings` on Molecule/Atom | Legacy | Delete | `basis_rings` | deterministic default basis | No | Implemented | Verified | cycle/API tests |
| Core cycles | `ligand_cycle_basis_rings` | Legacy | Delete | private filtered native basis in consumer | trained-model tensors | No | Implemented | Verified | model/CBond tests |
| Core cycles | `_cycle_basis_rings`, `_ligand_cycle_basis_rings` | Legacy | Delete caches | no persistent cycle cache | current topology reflected | No | Implemented | Verified | state/static tests |
| Core cycles | `cycle_basis` | P19 | Retain public property | `Molecule` + native cycle basis | specified deterministic ordering | No | Implemented | Verified | oracle tests |
| Core cycles | `all_cycle_basis` | P19 | Retain public property | `Molecule` + native cycle basis | complete bounded family | No | Implemented | Verified | oracle/limit tests |
| Core cycles | `basis_rings` | P19 | Retain public property | `Molecule` | exact index-to-Ring projection | No | Implemented | Verified | projection tests |
| Core cycles | `all_basis_rings` | P19 | Retain public property | `Molecule` | exact nested projection | No | Implemented | Verified | projection tests |
| Bond mutation | `_rm_bond()` | Both | Delete after folding semantics | one `remove_bonds()` implementation | BOND/AXIS/fact pruning | No | Implemented | Verified | stereo lifecycle tests |
| Bond mutation | `_rm_bonds()` | Both | Delete after folding semantics | one `remove_bonds()` implementation | one batch cleanup/refresh | No | Implemented | Verified | bulk mutation tests |
| Bond mutation | `remove_bond()` | Both | Retain; delegate to bulk path | public Molecule API | signature and semantics | No | Implemented | Verified | mutation tests |
| Bond mutation | `remove_bonds()` | Both | Retain as unique implementation | public Molecule API | single refresh and stereo prune | No | Implemented | Verified | mutation tests |
| Atom mutation | `_rm_atom()`/`_rm_atoms()` | P17/legacy | Consolidate without compatibility duplicate | `_remove_atoms_from_topology()` | stereo index remap | No | Implemented | Verified | stereo lifecycle tests |
| OpenBabel helper | unused `obconvert.add_hydrogens()` | Legacy | Delete with its `_rm_atoms()` dependency | Molecule/force-field hydrogen workflows | no active caller | No | Implemented | Verified | zero-reference scan |
| Hidden Bonds | hide/recover APIs | Both | Retain | existing Molecule APIs | payload identity on temporary hide | No | Implemented | Verified | hidden-Bond tests |
| Hidden Bonds | permanent hidden-Bond clear | P17 | Retain | clear methods + stereo helper | affected stereo pruning | No | Implemented | Verified | lifecycle tests |
| Clone | `Molecule.copy()`/`__copy__` | Both | Integrate | Molecule clone path | raw attributes and exact payload remap | No | Implemented | Verified | clone tests |
| Components | `Molecule.components` | Both | Integrate | P19 traversal + P17 construction | order, hidden Bonds, stereo scopes | No | Implemented | Verified | component tests |
| Composition | `Molecule.add_component()` | Both | Integrate | one Molecule method | group lineage and payload concatenation | No | Implemented | Verified | composition tests |
| Stereo state | `Molecule._stereo_payload` | P17 | Retain as chemical state | `hotpot.cheminfo.stereo` | Level-1 transport | No | Implemented | Verified | stereo tests |
| Stereo API | Level-1 public enums/dataclasses/functions | P17 | Retain unchanged | `hotpot.cheminfo.stereo` | full transport semantics | No | Implemented | Verified | public API/golden tests |
| Stereo API | Level-2 reports/functions | P17 | Retain unchanged | `hotpot.cheminfo.stereo` | gates 1-7 | No | Implemented | Verified | Level-2 tests |
| Stereo initialization | `_EMPTY_STEREO_PAYLOAD` and missing-field fallback in `get_stereo_payload()` | P17 compatibility | Delete | constructor-owned empty payload + direct getter | serialization/pickle semantics | No | Implemented | Verified | lifecycle/serialization tests |
| Stereo source facts | opaque namespaces and unknown source codes | P17 business data | Retain | Level-1 payload | lossless transport | No | Implemented | Verified | transport tests |
| Stereo internal | `_remove_stereo_bond_rows()` | P17 | Retain one private semantic operation or fold once | unique bulk mutation helper | ordered BOND/AXIS/fact behavior | No | Implemented | Verified | lifecycle tests |
| Working copy | Graph/neighbor/ring-cache snapshot fields | Legacy/P19 | Delete | restore authoritative facts then refresh | rollback correctness | No | Implemented | Verified | working-copy tests |
| Working copy | `StereoPayload` snapshot/commit/restore | P17 | Retain | force-field working-copy transaction | atomic exact restore | No | Implemented | Verified | rollback/failure tests |
| Generic transport | common contiguous molecule buffers | P17 | Retain as sole transport | `hotpot.cheminfo._native` | field-selective views | No | Implemented | Verified | native buffer tests |
| RDKit transport | any second RDKit-specific molecule buffers | Forbidden | Do not add/delete if found | common transport | no translation duplication | No | Implemented | Verified | source/ABI scan |
| rdWrapper Python | public contracts/sampler/functions | P17 | Retain unchanged | `hotpot.cheminfo.rdWrappers` | ETKDG/status/runtime behavior | No | Implemented | Verified | component tests |
| rdWrapper C | versioned C ABI v1 | P17 | Retain unchanged | rdWrapper native library | ABI ownership/lifetimes | No | Implemented | Verified | C ABI tests |
| rdWrapper C++ | facade and `Hotpot::rdwrappers` | P17 | Retain unchanged | installed SDK component | C++17 consumer contract | No | Implemented | Verified | C++ consumer tests |
| RDKit pin | native 2026.09.1 / Python 2026.9.1 | P17 | Retain exact pin | locked source closure | runtime identity | No | Implemented | Verified | packaging tests |
| RDKit hydrogen adapter | `_HOTPOT_LEGACY_NAMESPACE`, `_add_legacy_hydrogen_transport()` | P17 legacy bridge | Delete | canonical generic molecule packer | implicit-H input semantics | No | Implemented | Verified | rdWrapper transport/ETKDG tests |
| RDKit native parser | `hotpot-legacy/1` branches | P17 legacy bridge | Delete | ordinary current transport records | implicit-H application | No | Implemented | Verified | native/C++ tests |
| RDKit C ABI | legacy-prefix/additive-size acceptance proposed in Plan 17 extension | Proposed compatibility | Reject/delete from plan and tests | one exact current struct size/version | one generate pipeline | No | Implemented | Verified | ABI validation tests/static scan |
| RDKit CMake | `Hotpot::rdwrappers` alias and exact-version package metadata | P17 current API | Retain | one current SDK target | installed consumer contract | No | Implemented | Verified | CMake consumer tests |
| Search API | Query/Substructure/Searcher/Hits/Hit | P16/P19 | Retain | RI-style native search | mapping/order/callable semantics | No | Implemented | Verified | search/SMARTS tests |
| Search mapping protocol | `subgraph_monomorphisms_iter()` and mol-to-query inversion | P16 GraphMatcher adapter | Delete | `iter_mappings()` emits query-to-molecule directly | exact order/mapping sets | No | Implemented | Verified | mapping tests |
| Search result constructor | `Hits(..., graph_matcher, ...)` and `.graph_matcher` | P16 GraphMatcher adapter | Replace | `mapping_source` | search result semantics | No | Implemented | Verified | API/search tests |
| Search matcher helpers | `Searcher._node_match/_edge_match` | Legacy NetworkX adapter | Delete | compiled typed predicates | predicate semantics | No | Implemented | Verified | conformance/static tests |
| Search result helper | `Hits._get_nodes_set()` | Legacy private compatibility | Delete | normal iteration/result state | node-set results | No | Implemented | Verified | result tests |
| Search Hit | optional `mappings=()` default | Legacy invalid default | Delete default | mappings required | mapped atoms/Bonds | No | Implemented | Verified | Hit tests |
| Search graph API | `Substructure.construct_graph`, `query_graph`, cache/signature | Legacy | Delete | `query_atoms`/`query_bonds` + native IR | independent oracle migration | No | Implemented | Verified | API/static tests |
| Graph traversal | `graph_dfs_paths()` constant-empty compatibility function | P19/legacy | Delete | none | no production behavior exists | No | Implemented | Verified | export/static tests |
| Graph traversal | `graph_dfs_path()` | Active/P19 | Retain as sole explicit-edge DFS API | `hotpot.cheminfo.graph` | InternalCoordinates outputs | No | Implemented | Verified | traversal/Z-matrix tests |
| Graph traversal signature | `graph_dfs_path(graph, ...)` | Legacy | Replace, no duck-typed compatibility | `graph_dfs_path(node_count, edges, start_node=None, scope_nodes=None, min_deep=None, max_deep=None)` | deterministic one-branch event | No | Implemented | Verified | traversal/Z-matrix tests |
| Internal coordinates | `calc_zmat_idx(graph_or_link)` | Legacy | Replace, no Graph input branch | `InternalCoordinates.calc_zmat_idx(link_matrix, node_count=None)` | connected/isolate semantics | No | Implemented | Verified | direct-algorithm tests |
| Internal coordinates | `calc_zmat(graph_or_link, coords)` | Legacy | Replace, no Graph input branch | `InternalCoordinates.calc_zmat(link_matrix, coords)` | Z-matrix values | No | Implemented | Verified | Core/direct-algorithm tests |
| Direct graph algorithms | duplicated components/path/traversal implementations | P19 | Consolidate | `graph.components.connected_components`; `graph.paths.{shortest_path,all_simple_paths,longest_simple_path,depth_first_path}` | deterministic explicit-edge behavior | No | Implemented | Verified | direct-algorithm tests |
| Molecule shortest path | private in-method BFS | P19 merge residue | Delete duplicate implementation | `graph.paths.shortest_path()` called by `Molecule.shortest_path_index()` | disconnected and ordering semantics | No | Implemented | Verified | direct/Core tests |
| Archived search | `_logic.py`, `logic.py`, `logic2.py`, `_logic_tuple.py`, `fields.py`, `match_func.py`, `_smarts.py`, `smarts_parser/` | Legacy | Delete module island | Plan 16 query IR + current SMARTS parser | current public search semantics | No | Implemented | Verified | import/conformance/static tests |
| Archived search dependency | `legacy-search` optional extra / SymPy | Legacy | Delete | none | active extras remain unchanged | No | Implemented | Verified | metadata tests |
| Native search graph | `search/_native/graph_data.*` | P16 | Delete old location/implementation | shared `graph/_native/graph.*` | RI target semantics | No | Implemented | Verified | build/search tests |
| RI candidate logic | `search/_native/compatibility.hpp` | P16 | Retain | native RI matcher | candidate-domain semantics | No | Implemented | Verified | native search tests |
| Native search | Python fallback executor | Forbidden | Absent | `_search_native` only | explicit ImportError | No | Implemented | Verified | absence tests |
| Native graph | `graph/_native/graph.*` | P19 | Retain sole base Graph | private C++ graph library | deterministic CSR behavior | No | Implemented | Verified | native C++ tests |
| Native target ownership | `_NativeSpecTarget` plus direct `NativeTarget` construction | P17/P19 divergence | Consolidate then delete duplicate wrapper/path | one private packed-target owner | search/MCA/stereo reuse | No | Implemented | Verified | construction-count/semantic tests |
| Native cycles | `graph/_native/cycle_basis.*` | P19 | Retain sole cycle implementation | private C++ graph library | basis contracts/limits | No | Implemented | Verified | oracle/C++ tests |
| Relevant cycles | `_relevant_cycles` extension/stub/build target | Legacy | Delete | `_graph_native` | Relevant Cycle semantics | No | Implemented | Verified | wheel/import tests |
| MCA adapter | `MoleculeGraph` | Legacy/P17 baseline | Delete public type/export | accepted molecule inputs | prediction/API behavior | No | Implemented | Verified | import/API tests |
| MCA adapter | `AImodels/mca/graph_adapter.py` | Legacy | Delete file | serialization/conversion paths | supported inputs | No | Implemented | Verified | wheel/static tests |
| MCA automorphism | NetworkX rooted isomorphism | Legacy | Delete | native fixed-root exact matcher | labels/order/representatives | No | Implemented | Verified | MCA site tests |
| MCA distances | dense-only all-pairs path | Legacy | Replace above fixed threshold | P19 dense/sparse cutover | exact dtype/clip/values | No | Implemented | Verified | distance/prediction tests |
| MCA semantics | ordered sites and `is_Nuc_site` | Both | Preserve exactly | MCA pipeline/results | business meaning | No | Implemented | Verified | accepted hashes/site tests |
| NetworkX production | imports and runtime dependency | Legacy | Delete | native/direct algorithms | scientific outputs | No | Implemented | Verified | zero-import/metadata scan |
| NetworkX tests | independent oracle dependency | P19 | Retain test-only | test requirements | oracle independence | No | Implemented | Verified | packaging metadata test |
| Build | `setup.py` native source lists | Both | Integrate into one list | setuptools build | graph/search/rdWrapper coexistence | No | Implemented | Verified | clean wheel |
| Build | `MANIFEST.in` | Both | Integrate | one source manifest | all required sources, no obsolete stub | No | Implemented | Verified | sdist/wheel audit |
| Build | `publish_pypi.yml` smoke | Both | Integrate | one release workflow | all current native imports | No | Implemented | Verified | workflow/static audit |
| Dependencies | `pyproject.toml`/`requirements.txt` | Both | Integrate | identical runtime lists | P17 pins, P19 no NetworkX | No | Implemented | Verified | metadata tests |
| Force-field components | NetworkX component traversal | P19 | Replace | `graph.components.connected_components()` | component labels/order | No | Implemented | Verified | coordination component tests |
| Process boundaries | stale identity-keyed topology after deserialization | P19 | Rebuild at each process boundary | `_update_topology(clear_conformers=False)` in CLI/workers | child-process topology correctness | No | Implemented | Verified | worker/CLI tests |
| Conversion/serialization | `_update_graph()` calls in RDKit, MCA, CCDC paths | Both | Migrate, no alias | `_update_topology()` | conversion/round-trip semantics | No | Implemented | Verified | conversion/MCA tests |
| Cycle consumers | legacy ligand-cycle APIs in PyG, MCA extraction, examples | P19 | Migrate | current native basis/direct consumer algorithms | tensor/ring meaning | No | Implemented | Verified | model/cycle tests |
| Dataset search | `atom_bond_graph` duck typing | Legacy | Delete | current search capability/API | dataset search behavior | No | Implemented | Verified | dataset/search tests |
| OpenBabel conversion | NetworkX construction/import | Legacy | Delete | authoritative Atom/Bond facts | conversion semantics | No | Implemented | Verified | conversion/static tests |
| Plan index | Plan 17/18/19/M_17-19 rows | Both | Retain all historical plans plus merge plan | `plan/README.md` | unique numbering/context | No | Implemented | Verified | link scan |
| Tests/helpers | current-only `_update_graph` calls | P17 | Migrate; no alias | `_update_topology()` or public seam | test intent unchanged | No | Implemented | Verified | zero-reference scan/tests |
| Tests/helpers | working-copy assertions on `_graph` identity | P17 | Rewrite | normalized `_topology_index` + exact payload identity | rollback intent | No | Implemented | Verified | working-copy tests |
| Documentation | current API docs using deleted symbols | Both | Rewrite | final API tree | history remains clearly historical | No | Implemented | Verified | documentation scan |
| Future Plan 17 extension | additive/old-call compatibility wording | P17 draft | Rewrite | one replacement API and exact ABI | no future parallel interface promise | No | Implemented | Verified | plan review/static scan |
| Release source audit | old `search/_native/graph_data.*` workflow expectations | P16 CI residue | Replace and mark obsolete | `graph/_native/graph.*` + cycle sources | sdist completeness | No | Implemented | Verified | workflow/wheel audit |
