# M_17-19 native graph, stereochemistry, and RDKit integration plan

Status: approved for implementation on 2026-10-26.

Branch: `merge/m17-19-native-graph-rdwrapper`

Worktree: `/tmp/hotpot-m17-19`

## 1. Objective and merge lineage

Integrate `clean/networkx@b912c8a` into the current
`perf/mca-resource-efficient-workflow` lineage without selecting one parent's
whole-file version where both parents own distinct required semantics.

The parents have diverged substantially:

- current parent `6639754`: 118 commits after merge-base `2704f3c`, including
  native search, MCA resource work, native RDKit, full stereochemistry, and
  force-field validation;
- Graph parent `b912c8a`: 12 commits after the same merge base, including the
  private native Graph, topology-index cutover, NetworkX removal, native MCA
  automorphism, native cycle bases, API deletion, and packaging closure.

The result must contain both feature sets, not merely compile after resolving
text markers.

## 2. Fixed precedence and design decisions

1. Plan 19 is normative for topology authority, Graph APIs, cycle-basis APIs,
   native Graph ownership, NetworkX removal, MCA graph-adapter removal, and
   deletion of obsolete implementations.
2. Plan 17 is normative for `StereoPayload`, Level-1 transport, Level-2 stereo
   semantics, RDKit conversion, `rdWrappers`, generic molecule buffers,
   working-copy transactions, and force-field behavior.
3. Molecule atom membership and Bond endpoints are the sole Python topology
   facts. `Molecule._topology_index` is the sole persistent derived topology
   index.
4. The native Graph is private, ephemeral, and task-owned. Search retains one
   RI-style implementation; MCA exact automorphism and cycle algorithms reuse
   the shared native graph implementation.
5. `StereoPayload` is chemical state, not a topology cache. Rebuilding
   `_topology_index` must never silently create, discard, repair, or reinterpret
   stereo facts.
6. Permanent atom/Bond removal explicitly remaps or removes affected stereo
   records and transport facts. Hide/recover remains non-destructive.
7. No compatibility alias, deprecated Graph view, NetworkX fallback, old
   native extension, alternate updater, `MoleculeGraph`, or legacy cycle API is
   retained.
8. Public force-field control flow and defaults remain unchanged. This merge
   integrates components; it does not authorize the proposed native UFF/MMFF
   cutover.
9. The final tree has one implementation for single- and bulk-Bond deletion.
   The historical `_rm_bond()`/`_rm_bonds()` pair is removed after its stereo
   behavior is folded into that single implementation.

## 3. Final authority model

```text
Molecule._atoms + Molecule._bonds + Bond endpoints
    -> authoritative topology
    -> Molecule._topology_index
       (one rebuildable row/incidence index)
    -> direct Python algorithms or one private ephemeral native Graph

Molecule._stereo_payload
    -> authoritative transported/derived stereo state
    -> explicitly remapped by structural mutation
    -> never inferred or repaired by _update_topology()

generic molecule buffers
    -> one Hotpot Python/C/C++ transport implementation
    -> obWrappers and rdWrappers consume only requested fields
```

No persistent Python computational Graph, native Graph handle, Bond-pair map,
topology revision counter, duplicate Atom adjacency list, or RDKit-specific
second molecule buffer is permitted.

## 4. Frozen final API tree

### 4.1 Core topology and mutation

```text
hotpot.Molecule
|- add_bond(atom1, atom2, bond_order=1.0, **kwargs) -> Bond
|- add_bonds(idx_order, list_kw=None) -> list[Bond]
|- remove_atom(atom: int | Atom) -> None
|- remove_atoms(atoms: Iterable[Atom]) -> None
|- remove_bond(bond: Bond) -> None
|- remove_bonds(bonds: Iterable[Bond]) -> None
|- add_component(component: Molecule) -> None
|- components -> list[Molecule]
|- hide_bonds(*bonds: Bond, clear_conformers=False) -> None
|- restore_bonds(*bonds: Bond, clear_conformers=False) -> None
|- cycle_basis -> list[tuple[int, ...]]
|- all_cycle_basis -> list[list[tuple[int, ...]]]
|- basis_rings -> list[Ring]
`- all_basis_rings -> list[list[Ring]]

private unique refresh:
`- _update_topology(clear_conformers: bool = True) -> None
```

The four cycle properties and `_update_topology()` follow Plan 19 exactly.
Existing mutation signatures remain unchanged. `remove_bond()` delegates to
the one bulk deletion path; that path captures endpoint rows, removes Bonds,
updates stereo once, and refreshes topology once.

### 4.2 Search and graph algorithms

```text
hotpot.cheminfo.search
|- Query
|- QueryAtom
|- QueryBond
|- Substructure
|- Searcher
|- Hits
|- Hit
|- SmartsSyntaxError
|- UnsupportedSmartsError
`- SmartsSemantics

hotpot.cheminfo.graph
|- linkmat2adj(...)
|- adj2laplacian(...)
|- calc_spectrum(...)
|- calc_electron_config(...)
|- atoms_electron_configurations(...)
|- graph_dfs_path(nodes, edges, ...)
|- relevant_cycles(...)
|- GraphSpectrum
|- RelevantCycleLimitExceeded
`- DEFAULT_RELEVANT_CYCLE_LIMIT
```

Search mapping, `has_match()`, truncation, ordering, and leaf-callable behavior
remain the Plan 16 RI contract. No public Graph class or NetworkX-typed input is
added.

The private mapping protocol has one direction and one spelling:

```text
_NativeMappingSource.iter_mappings()
    -> query-index to molecule-index mappings
_MappingIterator
    -> bounds and freezes that same direction without inversion
Hits(sub, mol, mapping_source, get_hit=True, max_matches=None)
Hit(mol, sub, atom_indices, mappings)
```

The GraphMatcher-shaped `subgraph_monomorphisms_iter()`, `graph_matcher`
parameter/property, `Searcher._node_match/_edge_match`,
`Hits._get_nodes_set()`, and optional empty `Hit.mappings` default are deleted.
Their business results remain available through `Searcher`, `Hits`, and `Hit`.

### 4.3 MCA

```text
hotpot.cheminfo.AImodels.mca
|- AtomPrediction
|- MoleculePrediction
|- SitePrediction
|- MCACpuPredictor
|- MCAGpuPredictor
|- MCAGpuOutOfMemoryError
|- predict_mca(...)
|- predict_mca_cpu(...)
`- predict_mca_gpu(...)
```

`MoleculeGraph` and `graph_adapter` are absent. Supported molecule, RDKit, Open
Babel, SMILES, and `to_rdmol()` inputs remain. Site order, exact automorphism,
site type, and `is_Nuc_site` are invariant.

### 4.4 Stereochemistry

The complete Plan 17 public module remains unique:

```text
hotpot.cheminfo.stereo
|- StereoAnchorKind, StereoCarrierKind, StereoGroupKind, StereoType
|- StereoIssueCode
|- StereoGroupRef, StereoCarrier, StereoAnchor, StereoRecord
|- HydrogenTransportState, AtomTransportRecord, BondTransportRecord
|- StereoPayload
|- StereoValidationIssue, StereoValidationReport
|- StereoDerivedDescriptor, StereoAssignmentReport, StereoAlignmentReport
|- get_stereo_payload(mol: Molecule) -> StereoPayload
|- set_stereo_payload(mol: Molecule, payload: StereoPayload) -> None
|- validate_stereochemistry(mol, *, coordinates=None)
|- assign_stereochemistry(mol, *, coordinates=None)
`- align_stereochemistry(reference, target, atom_mapping)
```

Clone, component extraction/composition, atom removal, Bond deletion,
hide/recover, converters, and force-field commit/rollback must preserve the
Plan 17 contracts and ordering.

`Molecule.__init__` explicitly owns an empty `StereoPayload`.
`get_stereo_payload()` reads that required field directly; it does not accept
objects that predate the field through a missing-attribute fallback. Opaque
source namespaces and unknown transported source codes remain lossless
business data and are not compatibility shims.

### 4.5 Native RDKit component

The Plan 17 Python API remains unchanged:

```text
hotpot.cheminfo.rdWrappers
|- GenerationStatus
|- ETKDGOptions
|- CoordinationEmbeddingConstraints
|- ConformerRecord
|- ConformerBatch
|- RuntimeInfo
|- ETKDGSampler
|  `- generate(mol, *, constraints=None) -> ConformerBatch
|- generate_conformers(mol, *, options=None, constraints=None)
`- runtime_info() -> RuntimeInfo
```

The existing versioned C ABI, C++17 facade, `Hotpot::rdwrappers` SDK target,
and generic `MoleculeBufferView` remain. No Python or C++ RDKit object crosses
the public boundary. The private implementation remains pinned to RDKit
2026.09.1; the Python dependency remains `rdkit==2026.9.1`.

## 5. Conflict-resolution contract

### 5.1 Core

- `copy()`: call `_update_topology()` and then retain the exact Plan 17 stereo
  payload remap.
- `add_component()`: retain Plan 17 group/payload composition and use the new
  topology refresh.
- `components`: use Plan 19 connected-components traversal while preserving
  Plan 17 raw Atom arrays, active/hidden Bonds, stereo remapping, and component
  order.
- atom removal: preserve Plan 17 index remapping, then refresh topology.
- Bond deletion: use one implementation; prune BOND anchors, constituent AXIS
  anchors, and Bond transport facts before one refresh.
- hide/recover: retain payload identity; permanent hidden-Bond clearing prunes
  affected stereo facts.

### 5.2 Force-field working copies

Snapshots retain authoritative atom/Bond state, stereo payload, conformers,
and AtomPair state. They do not retain NetworkX Graphs, duplicate Atom
adjacency, obsolete ring caches, or row caches. Restore rebuilds topology once
and restores stereo exactly. Commit refreshes topology once and installs the
prepared stereo payload atomically.

### 5.3 Search, MCA, cycles, and consumers

- retain Plan 16 RI search and compose it with the shared Plan 19 native Graph;
- provide one private packed-target owner for generic search, MCA exact
  automorphism, and stereo Gate 6; remove the parallel `_NativeSpecTarget`
  wrapper/direct-`NativeTarget` construction paths;
- retain Plan 19 native exact MCA automorphism and sparse all-pairs threshold;
- retain the four Plan 19 cycle-basis properties and remove all older ring-basis
  surfaces and caches;
- migrate Plan 17 stereo tests, RDKit converters, benchmark helpers, and formal
  validation helpers from `_update_graph()` to `_update_topology()`;
- preserve MCA exact accepted hashes and `is_Nuc_site` ownership.

Stereo Gate 6 must retain its represented-graph semantics, including active
Bonds, hidden Bonds, and virtual implicit-hydrogen nodes. It may use the shared
native Graph executor but must not substitute Plan 19's active-topology-only
Molecule snapshot for that represented graph.

### 5.4 Build and dependencies

The final wheel builds and packages exactly these native Python modules where
applicable:

```text
hotpot.cheminfo.graph._graph_native
hotpot.cheminfo.search._search_native
hotpot.cheminfo.forcefields._native._geometry_native
hotpot.cheminfo.obWrappers._ob_native
hotpot.cheminfo.rdWrappers._rd_native
```

`setup.py`, `MANIFEST.in`, workflow smoke tests, and packaging tests must
contain both Graph/search sources and the locked RDKit build. Runtime
dependencies exclude NetworkX; test requirements may retain it only as an
independent oracle. Current OpenBabel, Python, RDKit, and cryptography policies
remain.

The constant-empty `graph_dfs_paths()` compatibility function is deleted;
`graph_dfs_path()` remains the one active explicit-edge traversal API because
`InternalCoordinates` consumes it. The disconnected SymPy-era search island
(`_logic.py`, `logic.py`, `logic2.py`, `_logic_tuple.py`, `fields.py`,
`match_func.py`, `_smarts.py`, and `smarts_parser/`) and the `legacy-search`
optional dependency are deleted. The native search
`_native/compatibility.hpp` is retained: it implements RI candidate-domain
compatibility and is not an API-compatibility shim.

The RDKit path also removes `_HOTPOT_LEGACY_NAMESPACE`,
`_add_legacy_hydrogen_transport()`, and native `hotpot-legacy/1` parsing.
Canonical generic packing supplies current hydrogen transport for every atom
that needs it. The versioned C API accepts exactly the installed structure
size/version; it does not accept old prefixes or retain translators. CMake's
`Hotpot::rdwrappers` target alias and `COMPATIBILITY ExactVersion` remain
because they identify the one current SDK and exact package version rather
than preserving an obsolete interface.

## 6. Complete disposition source

[api_implementation_migration_matrix.md](api_implementation_migration_matrix.md)
is the sole normative adjustment/deletion table. Every row must finish as
`Implemented` and `Verified`. Compatibility retained must remain `No`. A row
that cannot be deleted or uniquely owned blocks completion.

## 7. Small-step implementation sequence

1. Freeze this plan, the migration matrix, baseline identities, and the copied
   Plan 17 documentation/benchmark assets; commit before merge.
2. Merge `clean/networkx@b912c8a`; record every textual conflict without
   resolving by whole-file selection.
3. Resolve Core topology and stereo mutation, then run lifecycle tests.
4. Resolve clone/components/add-component and force-field working-copy
   commit/rollback, then run transaction and coordination tests.
5. Migrate all remaining Plan 17 `_update_graph` and deleted-API callers.
6. Combine search/MCA/cycle and RDKit/stereo sources; run focused scientific
   parity tests.
7. Combine build manifests, extension sources, dependencies, CI smoke tests,
   and packaging metadata.
8. Delete all compatibility surfaces and old implementations; run static
   zero-reference gates.
9. Run focused, broad, wheel, MCA CPU/GPU where available, and Plan 17
   force-field validations.
10. Close the matrix, implementation log, and final report; commit each
    independently meaningful stage.

## 8. Static uniqueness and deletion gates

The final tracked tree must prove:

```text
production NetworkX imports/dependency declarations: 0
_update_graph definitions/calls: 0
_update_atom_neighbours_bonds definitions/calls: 0
Molecule._graph and public Graph-like properties: 0
MoleculeGraph/graph_adapter references: 0
cycle_basis_rings/ligand_cycle_basis_rings references: 0
Atom._neighbours and Atom-specific stored _bonds: 0
old _relevant_cycles extension/source declarations: 0
search/_native/graph_data source declarations: 0
Python Graph/search/stereo fallback implementations: 0
compatibility aliases for removed APIs: 0
constant-empty graph_dfs_paths compatibility implementation: 0
archived SymPy search implementations and legacy-search extra: 0
hotpot-legacy/1 hydrogen transport adapters/parsers: 0
old-prefix C ABI translators: 0
native RI implementations: 1
native base Graph implementations: 1
topology refresh implementations: 1
generic contiguous molecule transports: 1
```

Documentation describing historical evidence may name deleted symbols; the
code, active API documentation, packaging declarations, and tests may not use
them.

## 9. Validation matrix

| Gate | Required result |
|---|---|
| Core topology/mutation | all Plan 19 topology and mutation tests pass |
| Stereo lifecycle | all Level-1/Level-2 clone, component, atom/Bond/axis, hidden-Bond, coordinate, projection, and alignment tests pass |
| RDKit conversion | current exact stereo transport and round-trip tests pass |
| rdWrapper | Python, C ABI, C++ API, pinned-runtime, ETKDG, and packaging tests pass |
| Search/SMARTS | exact mapping sets/order, `has_match`, truncation, callable-leaf, and conformance tests pass |
| Cycles/models | default/all-basis, Relevant Cycles, CBond/PyG tensor parity tests pass where dependencies exist |
| MCA | exact ordered sites, `is_Nuc_site`, prediction digests, CPU suite, and live GPU suite where available pass |
| Force field | working-copy rollback, coordination, workers, component routes, and formal 182-case O0 acceptance pass |
| Packaging | clean wheel contains current native modules, no obsolete modules, no runtime NetworkX, and correct RDKit/OpenBabel metadata |
| Broad suite | no new failure outside documented unavailable optional dependencies |

The 182-case O0 validation assesses the integrated topology/stereo/working-copy
path; it does not reopen the completed choice of O0 as the complex default.

## 10. Risks and stopping conditions

Stop for owner review if integration requires changing public force-field
signatures/defaults, changing the pinned RDKit version, retaining a deleted
Graph compatibility API, adding a second topology or molecule-buffer owner,
changing search mapping semantics, changing MCA site ownership or accepted
predictions, or weakening Plan 17 stereo semantics.

Cross-version wheel, live GPU, PyG, or full 182-case execution may be reported
as environment-blocked only after all locally available static and focused
gates run; absence of an environment does not authorize a fallback
implementation.

## 11. Definition of done

The merge is complete only when both parents are in the branch ancestry, the
migration matrix is fully closed, all conflicts are semantically resolved,
all obsolete interfaces and implementations have zero active references, the
unique final API tree matches this plan, required tests pass, and the final
report records commands, results, deviations, limitations, and commit IDs.
