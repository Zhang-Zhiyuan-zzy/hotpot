# M_17-19 API and implementation migration matrix

Status: frozen baseline; final status columns are updated during implementation.

This is the sole normative inventory of interfaces and implementations changed
or deleted by M_17-19. `Compatibility retained` is intentionally `No` for every
row. Historical Plan 17/19 documents may retain names as evidence; production,
active tests, packaging, and current API documentation may not retain deleted
surfaces.

| Layer | Existing symbol/artifact | Origin | Final disposition | Final unique owner/API | Required preservation | Compatibility retained | Initial status | Verification |
|---|---|---|---|---|---|---|---|---|
| Core authority | `Molecule._atoms`, `_bonds`, Bond endpoints | Both | Retain as sole topology facts | `Molecule`/`Bond` | identity, order, attributes | No | Pending merge | topology tests |
| Core derived state | `Molecule._topology_index` | P19 | Retain as sole persistent topology index | `_update_topology()` | deterministic row/incidence order | No | Pending merge | topology-index tests/static scan |
| Core refresh | `_update_topology(clear_conformers=True)` | P19 | Retain as sole refresh implementation | `Molecule._update_topology` | all current invalidation behavior | No | Pending merge | mutation tests/static scan |
| Core refresh | `_update_graph()` | P17/legacy | Delete | `_update_topology()` | callers migrated, stereo handled separately | No | Pending deletion | zero-reference scan |
| Core refresh | `_update_atom_neighbours_bonds()` | Legacy | Delete | `_update_topology()` | public neighbor results unchanged | No | Pending deletion | zero-reference scan |
| Core graph helpers | `_edge_with_attrs`, `_node_with_attrs`, `_ring_graph`, `_uncached_cycle_basis_rings` | Legacy | Delete | explicit facts/native algorithms | scientific outputs | No | Pending deletion | zero-reference scan |
| Core graph state | `Molecule._graph` | Legacy | Delete | no persistent computational Graph | algorithms preserve outputs | No | Pending deletion | source/runtime scan |
| Core adjacency | `Atom._neighbours`, Atom-specific stored `_bonds` | Legacy | Delete | `_topology_index` queries | return types/order | No | Pending deletion | state-budget tests |
| Core cache | `_row2idx` | Legacy | Delete | `_topology_index` | atom index meanings | No | Pending deletion | source scan |
| Core cache | cached `adjacency_matrix` instance state/property | Legacy | Delete | task-local numerical construction | graph spectrum parity | No | Pending deletion | spectrum tests |
| Core public graph | `Molecule.graph` | Legacy | Delete | domain APIs/direct algorithms | no Graph return | No | Pending deletion | API/static tests |
| Core public graph | `Molecule.simple_graph` | Legacy | Delete | domain APIs/direct algorithms | scientific callers migrated | No | Pending deletion | API/static tests |
| Core public graph | `Molecule.atom_bond_graph` | Legacy | Delete | native search snapshots | search semantics | No | Pending deletion | search tests |
| Core public graph | `Molecule.super_acyclic_graph()` | Legacy | Delete | none | none; unimplemented | No | Pending deletion | source scan |
| Core cycles | `cycle_basis_rings` on Molecule/Atom | Legacy | Delete | `basis_rings` | deterministic default basis | No | Pending deletion | cycle/API tests |
| Core cycles | `ligand_cycle_basis_rings` | Legacy | Delete | private filtered native basis in consumer | trained-model tensors | No | Pending deletion | model/CBond tests |
| Core cycles | `_cycle_basis_rings`, `_ligand_cycle_basis_rings` | Legacy | Delete caches | no persistent cycle cache | current topology reflected | No | Pending deletion | state/static tests |
| Core cycles | `cycle_basis` | P19 | Retain public property | `Molecule` + native cycle basis | specified deterministic ordering | No | Pending merge | oracle tests |
| Core cycles | `all_cycle_basis` | P19 | Retain public property | `Molecule` + native cycle basis | complete bounded family | No | Pending merge | oracle/limit tests |
| Core cycles | `basis_rings` | P19 | Retain public property | `Molecule` | exact index-to-Ring projection | No | Pending merge | projection tests |
| Core cycles | `all_basis_rings` | P19 | Retain public property | `Molecule` | exact nested projection | No | Pending merge | projection tests |
| Bond mutation | `_rm_bond()` | Both | Delete after folding semantics | one `remove_bonds()` implementation | BOND/AXIS/fact pruning | No | Pending integration | stereo lifecycle tests |
| Bond mutation | `_rm_bonds()` | Both | Delete after folding semantics | one `remove_bonds()` implementation | one batch cleanup/refresh | No | Pending integration | bulk mutation tests |
| Bond mutation | `remove_bond()` | Both | Retain; delegate to bulk path | public Molecule API | signature and semantics | No | Pending integration | mutation tests |
| Bond mutation | `remove_bonds()` | Both | Retain as unique implementation | public Molecule API | single refresh and stereo prune | No | Pending integration | mutation tests |
| Atom mutation | `_rm_atom()`/`_rm_atoms()` | P17/legacy | Consolidate without compatibility duplicate | current internal atom-removal path | stereo index remap | No | Pending integration | stereo lifecycle tests |
| Hidden Bonds | hide/recover APIs | Both | Retain | existing Molecule APIs | payload identity on temporary hide | No | Pending integration | hidden-Bond tests |
| Hidden Bonds | permanent hidden-Bond clear | P17 | Retain | clear methods + stereo helper | affected stereo pruning | No | Pending integration | lifecycle tests |
| Clone | `Molecule.copy()`/`__copy__` | Both | Integrate | Molecule clone path | raw attributes and exact payload remap | No | Pending conflict | clone tests |
| Components | `Molecule.components` | Both | Integrate | P19 traversal + P17 construction | order, hidden Bonds, stereo scopes | No | Pending conflict | component tests |
| Composition | `Molecule.add_component()` | Both | Integrate | one Molecule method | group lineage and payload concatenation | No | Pending conflict | composition tests |
| Stereo state | `Molecule._stereo_payload` | P17 | Retain as chemical state | `hotpot.cheminfo.stereo` | Level-1 transport | No | Pending integration | stereo tests |
| Stereo API | Level-1 public enums/dataclasses/functions | P17 | Retain unchanged | `hotpot.cheminfo.stereo` | full transport semantics | No | Pending integration | public API/golden tests |
| Stereo API | Level-2 reports/functions | P17 | Retain unchanged | `hotpot.cheminfo.stereo` | gates 1-7 | No | Pending integration | Level-2 tests |
| Stereo initialization | missing-field fallback in `get_stereo_payload()` | P17 compatibility | Delete | constructor-owned empty payload + direct getter | serialization/pickle semantics | No | Pending replacement | lifecycle/serialization tests |
| Stereo source facts | opaque namespaces and unknown source codes | P17 business data | Retain | Level-1 payload | lossless transport | No | Pending integration | transport tests |
| Stereo internal | `_remove_stereo_bond_rows()` | P17 | Retain one private semantic operation or fold once | unique bulk mutation helper | ordered BOND/AXIS/fact behavior | No | Pending integration | lifecycle tests |
| Working copy | Graph/neighbor/ring-cache snapshot fields | Legacy/P19 | Delete | restore authoritative facts then refresh | rollback correctness | No | Pending deletion | working-copy tests |
| Working copy | `StereoPayload` snapshot/commit/restore | P17 | Retain | force-field working-copy transaction | atomic exact restore | No | Pending conflict | rollback/failure tests |
| Generic transport | common contiguous molecule buffers | P17 | Retain as sole transport | `hotpot.cheminfo._native` | field-selective views | No | Pending integration | native buffer tests |
| RDKit transport | any second RDKit-specific molecule buffers | Forbidden | Do not add/delete if found | common transport | no translation duplication | No | Pending scan | source/ABI scan |
| rdWrapper Python | public contracts/sampler/functions | P17 | Retain unchanged | `hotpot.cheminfo.rdWrappers` | ETKDG/status/runtime behavior | No | Pending integration | component tests |
| rdWrapper C | versioned C ABI v1 | P17 | Retain unchanged | rdWrapper native library | ABI ownership/lifetimes | No | Pending integration | C ABI tests |
| rdWrapper C++ | facade and `Hotpot::rdwrappers` | P17 | Retain unchanged | installed SDK component | C++17 consumer contract | No | Pending integration | C++ consumer tests |
| RDKit pin | native 2026.09.1 / Python 2026.9.1 | P17 | Retain exact pin | locked source closure | runtime identity | No | Pending dependency merge | packaging tests |
| RDKit hydrogen adapter | `_HOTPOT_LEGACY_NAMESPACE`, `_add_legacy_hydrogen_transport()` | P17 legacy bridge | Delete | canonical generic molecule packer | implicit-H input semantics | No | Pending replacement | rdWrapper transport/ETKDG tests |
| RDKit native parser | `hotpot-legacy/1` branches | P17 legacy bridge | Delete | ordinary current transport records | implicit-H application | No | Pending replacement | native/C++ tests |
| RDKit C ABI | legacy-prefix/additive-size acceptance proposed in Plan 17 extension | Proposed compatibility | Reject/delete from plan and tests | one exact current struct size/version | one generate pipeline | No | Pending plan correction | ABI validation tests/static scan |
| RDKit CMake | `Hotpot::rdwrappers` alias and exact-version package metadata | P17 current API | Retain | one current SDK target | installed consumer contract | No | Pending integration | CMake consumer tests |
| Search API | Query/Substructure/Searcher/Hits/Hit | P16/P19 | Retain | RI-style native search | mapping/order/callable semantics | No | Pending merge | search/SMARTS tests |
| Search mapping protocol | `subgraph_monomorphisms_iter()` and mol-to-query inversion | P16 GraphMatcher adapter | Delete | `iter_mappings()` emits query-to-molecule directly | exact order/mapping sets | No | Pending replacement | mapping tests |
| Search result constructor | `Hits(..., graph_matcher, ...)` and `.graph_matcher` | P16 GraphMatcher adapter | Replace | `mapping_source` | search result semantics | No | Pending replacement | API/search tests |
| Search matcher helpers | `Searcher._node_match/_edge_match` | Legacy NetworkX adapter | Delete | compiled typed predicates | predicate semantics | No | Pending deletion | conformance/static tests |
| Search result helper | `Hits._get_nodes_set()` | Legacy private compatibility | Delete | normal iteration/result state | node-set results | No | Pending deletion | result tests |
| Search Hit | optional `mappings=()` default | Legacy invalid default | Delete default | mappings required | mapped atoms/Bonds | No | Pending replacement | Hit tests |
| Search graph API | `Substructure.construct_graph`, `query_graph`, cache/signature | Legacy | Delete | `query_atoms`/`query_bonds` + native IR | independent oracle migration | No | Pending deletion | API/static tests |
| Graph traversal | `graph_dfs_paths()` constant-empty compatibility function | P19/legacy | Delete | none | no production behavior exists | No | Pending deletion | export/static tests |
| Graph traversal | `graph_dfs_path()` | Active/P19 | Retain as sole explicit-edge DFS API | `hotpot.cheminfo.graph` | InternalCoordinates outputs | No | Pending integration | traversal/Z-matrix tests |
| Archived search | `_logic.py`, `logic.py`, `logic2.py`, `_logic_tuple.py`, `fields.py`, `match_func.py`, `_smarts.py`, `smarts_parser/` | Legacy | Delete module island | Plan 16 query IR + current SMARTS parser | current public search semantics | No | Pending deletion | import/conformance/static tests |
| Archived search dependency | `legacy-search` optional extra / SymPy | Legacy | Delete | none | active extras remain unchanged | No | Pending deletion | metadata tests |
| Native search graph | `search/_native/graph_data.*` | P16 | Delete old location/implementation | shared `graph/_native/graph.*` | RI target semantics | No | Pending rename | build/search tests |
| RI candidate logic | `search/_native/compatibility.hpp` | P16 | Retain | native RI matcher | candidate-domain semantics | No | Pending merge | native search tests |
| Native search | Python fallback executor | Forbidden | Absent | `_search_native` only | explicit ImportError | No | Pending scan | absence tests |
| Native graph | `graph/_native/graph.*` | P19 | Retain sole base Graph | private C++ graph library | deterministic CSR behavior | No | Pending merge | native C++ tests |
| Native target ownership | `_NativeSpecTarget` plus direct `NativeTarget` construction | P17/P19 divergence | Consolidate then delete duplicate wrapper/path | one private packed-target owner | search/MCA/stereo reuse | No | Pending integration | construction-count/semantic tests |
| Native cycles | `graph/_native/cycle_basis.*` | P19 | Retain sole cycle implementation | private C++ graph library | basis contracts/limits | No | Pending merge | oracle/C++ tests |
| Relevant cycles | `_relevant_cycles` extension/stub/build target | Legacy | Delete | `_graph_native` | Relevant Cycle semantics | No | Pending deletion/rename | wheel/import tests |
| MCA adapter | `MoleculeGraph` | Legacy/P17 baseline | Delete public type/export | accepted molecule inputs | prediction/API behavior | No | Pending deletion | import/API tests |
| MCA adapter | `AImodels/mca/graph_adapter.py` | Legacy | Delete file | serialization/conversion paths | supported inputs | No | Pending deletion | wheel/static tests |
| MCA automorphism | NetworkX rooted isomorphism | Legacy | Delete | native fixed-root exact matcher | labels/order/representatives | No | Pending deletion | MCA site tests |
| MCA distances | dense-only all-pairs path | Legacy | Replace above fixed threshold | P19 dense/sparse cutover | exact dtype/clip/values | No | Pending merge | distance/prediction tests |
| MCA semantics | ordered sites and `is_Nuc_site` | Both | Preserve exactly | MCA pipeline/results | business meaning | No | Pending integration | accepted hashes/site tests |
| NetworkX production | imports and runtime dependency | Legacy | Delete | native/direct algorithms | scientific outputs | No | Pending deletion | zero-import/metadata scan |
| NetworkX tests | independent oracle dependency | P19 | Retain test-only | test requirements | oracle independence | No | Pending dependency merge | packaging metadata test |
| Build | `setup.py` native source lists | Both | Integrate into one list | setuptools build | graph/search/rdWrapper coexistence | No | Pending auto-merge review | clean wheel |
| Build | `MANIFEST.in` | Both | Integrate | one source manifest | all required sources, no obsolete stub | No | Pending auto-merge review | sdist/wheel audit |
| Build | `publish_pypi.yml` smoke | Both | Integrate | one release workflow | all current native imports | No | Pending conflict | workflow/static audit |
| Dependencies | `pyproject.toml`/`requirements.txt` | Both | Integrate | identical runtime lists | P17 pins, P19 no NetworkX | No | Pending conflict | metadata tests |
| Plan index | Plan 17/18/19/M_17-19 rows | Both | Retain all historical plans plus merge plan | `plan/README.md` | unique numbering/context | No | Pending conflict | link scan |
| Tests/helpers | current-only `_update_graph` calls | P17 | Migrate; no alias | `_update_topology()` or public seam | test intent unchanged | No | Pending migration | zero-reference scan/tests |
| Tests/helpers | working-copy assertions on `_graph` identity | P17 | Rewrite | normalized `_topology_index` + exact payload identity | rollback intent | No | Pending migration | working-copy tests |
| Documentation | current API docs using deleted symbols | Both | Rewrite | final API tree | history remains clearly historical | No | Pending audit | documentation scan |
