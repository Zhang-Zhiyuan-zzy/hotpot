# Fact-Carrier and Rule Architecture Refactor

Status: **implementation authorized on 2026-10-28; Stage0–8E owning-stage gates
complete within their recorded scopes. Stage8C includes canonical search and
actual MCA workflows; Stage8D includes inference/conformer/artifact migration and
actual CPU/pipeline/one-GPU evidence. Stage8E now has source/installed CP311
assembly, local H/scaffold corrections and explicit exports/resources accepted;
P20-D024 is closed. [Stage12B H representation](stages/stage12/hydrogen-representation/validation/checkpoint.json) closes P20-D026 within affected CP311 source/installed and rebuilt-RDKit-native scope.
Stage9A semantic preparation, Stage9B sole-owner carrier cutover and Stage9C
collection completion are accepted at their recorded CP311 source/installed
boundaries. [Stage10A physical/native migration](stages/stage10/native-migration/validation/checkpoint.json)
is accepted within its clean CP311 source/sdist/wheel/probe-SDK boundary.
Stage11A immutable reference assets and [Stage12A explicit/lazy imports](stages/stage12/lazy-imports/validation/checkpoint.json)
are accepted at their affected CP311 source/installed boundaries. Other Stage10 release/version obligations,
remaining Stage11–12, optional scientific activation and global acceptance remain open**.
Historical C62 Stage6 delivery and C61 validated Stage6 runtime retain their own
source scopes. Current bounded results and remaining semantic/installation limits
are recorded in [the CP311 core-workflow report](cp311_core_workflow_validation_20261103.md). The architecture, lazy invalidation, scientific-reporting, and
performance requirements are incorporated. The owner's implementation request
authorizes the concrete documented route; Stage1 records exact API differences
and consumer evidence. Unspecified public signatures, return semantics and
schema changes still require a concrete decision before their dependent step.

Planning branch: `refactor/fact-rule-architecture`

Implementation branch: `implement/plan20-fact-rule-architecture`

Planning worktree: `/home/zhangzhiyuan/hotpot` (no additional worktree)

Original planning baseline: `0a4b0ee`; reviewed documentation commit: `d9eb80e`.
The executable implementation baseline is frozen in Stage 1 after the selected
Plan17 prerequisite route, not assumed to be the original planning commit.

Initial audit: 2026-10-27. Owner-directed revisions: 2026-10-28, 2026-10-30 and 2026-11-03.

The [current implementation constitution](goal.md), approved and activated by
the owner on 2026-11-03, governs execution, ordinary results, findings and control
decisions in this plan. The earlier goal is archived as `goal_v1.md`. Settled
public contracts, scientific constraints and required final validation/performance
work remain binding; the current goal governs priority, evidence reuse and usable
cost when older planning text conflicts.

Normative companion records:

- [Plan17/Plan20 execution alignment](plan17_plan20_execution_alignment.md)
- [Data-object scope and consolidation audit](data_object_scope_audit.md)
- [Consistent public-interface protocol for chemical types and fact storage](chemical_type_interface_contract.md)
- [Plan21 MetalCluster framework draft](../21_metal_cluster/metal_cluster_implementation_plan.md)

## 1. Objective

Reorganize Hotpot's chemistry kernel around one explicit architectural rule:
domain objects own durable facts, while chemical, physical, and mathematical
rules only calculate from those facts.

All public chemical types follow one interface protocol: each constructor,
property, method, collection access, result, and conversion declares its fact
owner or dependencies, permissions, units, lifecycle, and execution boundary.
This covers Molecule, Atom, Bond, Crystal, collections, views, scientific
results, public transport contracts and the registered future MetalCluster
type. Shared semantics do not require a shared base class, storage layout,
or mutability.

These types remain the ordinary user interface. Users continue to write
`mol.optimize(...)`, `mol.assign_bond_order()`, or `bond.length` without a rule
registry, rule handle, operator object, backend profile, or dependency-injection
setup. Object methods become thin entry points into named rules or operations.
The Atom fact table is a concrete storage implementation of this common protocol.

The refactor has eight simultaneous goals:

1. make each durable chemical fact have exactly one authoritative owner;
2. apply one public-interface protocol across chemical types while separating
   fact storage and interfaces from chemistry, physics, mathematics, toolkit
   integration, and workflow execution;
3. make rule evaluation read-only and return immutable findings, proposals,
   or result objects, including ownership isolation of nested numeric buffers;
4. permit mutation only through a small private operation/commit boundary that
   validates and applies one complete change atomically;
5. remove redundant derived values, toolkit molecules, and algorithm
   workspaces from carriers; explicitly selected P2/P3 values have finite
   dependency-status metadata and are marked for demand-driven recalculation;
6. preserve the useful top-level Hotpot API and scientific behavior while
   deleting obsolete internal paths and duplicate implementations;
7. classify the full chemistry package by responsibility rather than merely
   splitting the current `core.py`; and
8. retain the performance benefits of the native graph, native geometry,
   `rdWrapper`, `obWrapper`, and field-selective molecule buffers without
   allowing any of them to become a second source of truth.

Physical migration preserves the selected aromaticity, force-field, stereo,
graph-search and MCA algorithms. Evidence-bound revisions within approved
scientific scope follow the constitution and owning-stage records; a new
scientific scope still requires its separate decision and evidence. The main
agent may revise the implementation route, diagnostics and warning/break policy
within that authority, without reducing declared work, denominators or acceptance
gates. Scientific outcomes are reported, including changes in eligible cohorts
and geometry/energy monitors; historical outcomes are not a fixed truth or a
fixed success-count gate.

## 2. Scope and non-goals

### 2.1 In scope

- `hotpot/cheminfo` fact carriers, independent values, collections, views,
  scientific results, transport contracts, rules, operations, adapters,
  reference data, native bindings, public exports, and documentation;
- every production, plugin, pipeline, dataset-provider, CLI, and test import
  affected by the chemistry-kernel move;
- conditionally, `hotpot/main` and `hotpot/works`, whose overlapping
  command/workflow roles currently obscure the outer architecture (Decision
  12); this cleanup is not interleaved with carrier extraction;
- setup, wheel, sdist, CMake, C/C++ include, package-data, and typing-stub paths;
- conditionally, relocation of misowned chemistry assets (Decision 13), as a
  separate coherent data-governance commit;
- removal of dead or superseded chemistry implementations discovered by the
  zero-reference/build audit;
- behavior, packaging, scientific-boundary, and performance validation;
- completion of the functional requirements established by
  [Plan12](../12_xtb_workflow/README.md) through the Plan20 fact/rule, operation
  and adapter owners, including their required tests and scientific execution;
  no parallel Plan12 implementation or bypass of this plan is introduced; and
- design registration of MetalCluster and its relationship to Molecule,
  Atom and Bond; runtime implementation belongs to Plan21.

### 2.2 Out of scope

- changing the contents or governance model of `hotpot/dataset` and
  `hotpot/ds`, except for moving chemistry-package assets that are currently in
  the wrong ownership layer;
- redesigning the generic `hotpot.pipeline` engine;
- restructuring independent plugin internals outside the integrated Plan12
  requirements that do not violate the new carrier/rule boundary;
- MetalCluster templates, recognition, runtime objects and superatom
  execution; these are deferred to the Plan21 framework draft;
- changing trained model weights or tuning scientific monitor thresholds to
  conceal differences;
- adding compatibility profiles or selectable old/new implementations; and
- changing storage merely for style. The owner's requested unified Atom fact
  store is now in scope: a benchmarked private typed-column table, schema-driven
  Atom projections, and efficient scalar/batch writes. Bond storage does not
  receive an unrelated layout rewrite. Public access changes remain reviewable.

## 3. Architectural vocabulary and fixed boundaries

### 3.1 Vocabulary

The following terms are normative for this plan.

| Term | Meaning | Lifetime and ownership |
|---|---|---|
| fact | a value deliberately committed as part of a molecular structure, representation, selected state, or durable final result | owned by one canonical fact-store field or typed result record; object properties are projections |
| carrier | a domain object that stores facts and exposes the primary user API | durable; authoritative |
| view | a structurally immutable projection such as a ring, path, angle, torsion, or atom pair | caller-owned result; never authoritative |
| registered aggregate | the future MetalCluster: an internal graph/point-cloud unit used externally as a multi-attachment superatom | type/relationships recorded here; registration, instances and execution deferred to Plan21 |
| materialized partition | a derived connected-component partition returned as independent Molecule carriers | caller-owned Molecules with explicitly subset facts; not a view and not cached on the source |
| snapshot | an immutable, field-selective copy consumed by one operation or backend | task-local and disposable |
| rule | a calculation from facts/snapshots to an immutable finding, proposal, or result; built-in rules are deterministic for fixed inputs | read-only; no carrier writes |
| operation | a small controller that invokes rules/backends and, for mutating work, validates and atomically commits a complete patch | task-local; the only non-carrier write path |
| adapter | I/O, external toolkit, carrier packing, ABI binding/result translation, network service, or presentation translation | may copy facts temporarily; never owns Hotpot facts |
| acceleration state | a private, fully reconstructible index or immutable reference-data cache that cannot alter observable semantics | allowed only under Section 5.6 |
| scientific result | an explicit final output such as an optimized conformer, energy, partial charges, or MCA prediction | returned by default; stored only by an explicit API with identity and provenance |

There will be no public generic `RuleHandle`, `Operator`, `RuleRegistry`,
`apply_rule(name, ...)`, or architecture profile. Such abstractions would make
normal chemistry harder to discover and would create a second, stringly typed
interface beside the object API.

Native code is classified by semantics, not by implementation language. A
pure C++ graph/geometry/force-field kernel belongs beside its rule; a native
session controller belongs beside its operation; only packing, ABI binding,
external-toolkit invocation, and result translation belong to adapters.

User-supplied callable search predicates are the one explicit purity
exception: only the callable leaf executes in Python and its side effects or
non-determinism belong to the caller. The remaining query compilation,
topology packing, RI matching, and enumeration do not fall back to Python and
remain deterministic.

### 3.2 Relationship to Plans 17, 19, and M_17-19

The following previously approved contracts remain normative:

- Plan 19's sole authoritative topology, one private `_topology_index`, one
  shared native CSR Graph representation, deterministic graph/search/cycle
  ordering, and inspectable detached Atom references;
- Plan 17/M_17-19 Level-1 stereo transport semantics, Level-2 Gate 2-7
  semantics, fixed public stereo signatures, pinned RDKit version, and
  standalone `rdWrapper`/`obWrapper` contracts; and
- the owner-approved O0 default complex workflow and existing MCA resource
  policy.

Plan20 explicitly records the following amendments to Plan19 rather than
treating them as ordinary internal moves. Private fact-table and dependency
status design follows the latest owner direction; affected public surfaces
remain subject to Section 16 and the companion API review register:

1. replace persistent active/hidden Bond collections and public hide/recover
   workflow state with one complete authoritative Bond collection plus
   operation-local topology masks; and
2. reopen Plan19's Core state/change budget to split Core, install the sole
   Atom fact table, and add the facts and finite dependency-status metadata in
   Section 3.3. No second topology index, native Graph, or general cache is
   authorized.

Root adopted Decision6's eight-method retirement and accepted the exact
[Stage7B HT1–HT5 contracts](stages/stage07/topology-selection/contracts/hidden_topology_design.md).
The separate implementation release and connected cutover evidence govern delivery;
there is no replacement public selector. The new private storage/dirty-state
design is not excluded by the old suggestion to preserve the Atom array layout.

Stage0 selects **Plan20 first** because Plan17's minimum native MCA semantic
closure is not implemented. Freeze a fresh Plan20 baseline using the existing
single Python-RDKit algorithm and move that algorithm intact into an adapter.
The native MMFF/UFF closure and MCA cutover are deferred until after `F20`, with
their own fresh `B17/F17` evidence. K>1 clustering and ordinary-organic default
replacement remain independent. The companion alignment record freezes this
order; today's ETKDG-only native route does not replace the selected algorithm.

The Plan 17 public stereo facade remains available at
`hotpot.cheminfo.stereo` with these signatures:

```python
def get_stereo_payload(mol: Molecule) -> StereoPayload: ...
def set_stereo_payload(mol: Molecule, payload: StereoPayload) -> None: ...
def validate_stereochemistry(mol: Molecule, *, coordinates=None) -> StereoValidationReport: ...
def assign_stereochemistry(mol: Molecule, *, coordinates=None) -> StereoAssignmentReport: ...
def align_stereochemistry(reference, target, atom_mapping) -> StereoAlignmentReport: ...
```

The facade is a real maintained public API, not a compatibility alias. It
constructs the one required snapshot and calls the new rule/operation
implementation. The old monolithic bodies are deleted.

The same facade continues to export the complete Plan 17 public schema with
unchanged class identity: `AtomTransportRecord`, `BondTransportRecord`,
`HydrogenTransportState`, `StereoAlignmentReport`, `StereoAnchor`,
`StereoAnchorKind`, `StereoAssignmentReport`, `StereoCarrier`,
`StereoCarrierKind`, `StereoDerivedDescriptor`, `StereoGroupKind`,
`StereoGroupRef`, `StereoIssueCode`, `StereoPayload`, `StereoRecord`,
`StereoType`, `StereoValidationIssue`, and `StereoValidationReport`. These are
imports of the one structure/rule contract definitions, not duplicate facade
classes.

### 3.3 Normative final carrier state budget

Subject to owner approval, the final Molecule-owned state is limited to:

| State | Purpose |
|---|---|
| ordered Atom and complete Bond membership | P0 structure truth |
| private `_AtomFactTable` | sole numeric Atom fact owner, including current Cartesian coordinates; Atom/Molecule properties project it |
| monotonic Atom/Bond UID high-water marks | P0 identity-allocation infrastructure; never decremented or reused within a Molecule lifetime |
| Plan 19 `_topology_index` | sole reconstructible structural acceleration index |
| molecular net charge/spin declarations | explicit P1 facts where present |
| current Cartesian coordinate frame | logical P1 geometry projected from the table's one coordinate block, never a duplicate persistent frame |
| optional `Crystal` value | P1 six-parameter unit-cell fact; no Molecule back-reference |
| source coordinate-frame collection | explicitly imported/user-supplied P1 frames with source provenance; distinct from generated results |
| stereo records plus source-representation ledger | canonical stereo facts and lossless source provenance; never a duplicate current Atom/Bond table |
| `ConstraintSet` | coordinate constraints keyed by Atom UID, bond constraints by Bond UID, and angle/torsion constraints by ordered Atom-UID tuples |
| user annotation mapping | user-owned metadata only |
| typed result collection | explicitly attached P3 results, including generated/optimized conformer ensembles, with declared dependency identities |
| finite fact-group validity/source metadata | CURRENT/NEEDS_RECALC/ABSENT/UNRESOLVED/CONFLICT as specified in the companion; no value duplication or revision cache |

Each Atom additionally owns an immutable internal `_uid` that is unique within
its Molecule lifetime. It is distinct from mutable/source `Atom.id`, derived
row `Atom.idx`, Python `id(atom)`, and transient stereo/native row numbers.
Each Bond has the corresponding internal identity required by constraints and
views. These identities are private facts, not user-visible chemical labels.
Molecule owns monotonic `_next_atom_uid` and `_next_bond_uid` high-water marks;
they are P0 identity infrastructure rather than calculated caches. Builders,
copy/component remapping, and serializers install or advance them atomically,
and removal never makes a UID reusable.

`_model` and `_environ` are removed from Molecule. Model/backend/runtime
contexts are operation inputs, not molecular facts. `_hided_*`, `_obmol`,
`_row2idx`, and all persistent view caches are outside this state budget.
MetalCluster is a Plan21 type reservation, not additional Plan20 runtime state.
Its future registration/association metadata requires an explicit ownership
budget; member atoms, bonds and coordinates keep their existing authority.
Internal subgraphs and external contracted graphs do not authorize a second
persistent topology index.

The private declarations serve all chemical types: `_fields.py` records
canonical fields, interface mappings, units/codecs, roles and permissions;
`_dependencies.py` records cross-type dependency groups, invalidation scope and
read/recompute policy. They share a declaration structure without a universal
record of optional fields. `_state.py` references those declarations rather
than copying their catalog. Each role keeps its own ownership and lifecycle;
independent values need no artificial Molecule identity or dirty state.

Atom's `_atom_table.py` implements typed columns/contiguous numeric blocks,
with its identity handle and `_topology_index` supplying row access without a
second index. Other types use storage appropriate to their facts. Decision 11
and the companion's Atom chapter govern this layout choice. Schema extension
updates approved projections and dependency handling together; no private
field automatically becomes a public attribute.

## 4. Confirmed current-state audit

### 4.1 Parts already aligned with the target

The work after version 0.5.3.0 provides important precedents and should be
preserved rather than rewritten mechanically.

| Current area | What is already correct | Required treatment |
|---|---|---|
| `graph/` and Plan 19 topology | Atom/Bond membership is authoritative; native Graph is private, deterministic, task-local, and field-selective | move/classify without changing graph semantics or ordering |
| `geometry/` | factual geometry objects, mathematical relations, and chemical adapters are already separated | retain internal separation while moving under mathematics rules |
| `kekulize/` | protocols, aromaticity tests, assignment, and workflow are separated from most carrier storage | merge into the chemistry-rule tree; keep one implementation |
| `calculator/` | immutable facts/evidence/contracts and explicit state composition already avoid hidden mutation | split pure chemistry rules from orchestration without flattening the evidence model |
| `forcefields/` | contracts, validation, working-copy isolation, trajectory, backend, and workflow responsibilities are substantially separated | split physics policy from execution/adapters; preserve atomic commit and report scientific monitors |
| `conformers/` | backend-neutral request/result contracts and a provider seam exist; frozen dataclasses do not yet guarantee immutable ndarray contents | move under operations; isolate buffers and verify actual result immutability |
| `_native/buffers.py` | `NativeMoleculeBuffers` selects only requested fields and is a generic transient Python/C++ transport | retain as the sole generic native transport; do not create another contiguous-buffer system |
| `rdWrappers/` and `obWrappers/` | both have independent Python/native contracts and coarse-grained calls | preserve standalone component behavior and current component package names; adapters consume them without copying them |
| `stereo.py` transport contracts | frozen Level-1 payload, carrier, anchor, group, and transport records preserve information explicitly | move these contracts intact into the carrier layer |
| pipeline/dataset governance | already outside the molecular object model | remain outer consumers of the chemistry kernel |

### 4.2 Partially aligned areas

Several `Molecule` methods already demonstrate the intended thin-facade form:

- `optimize(...)` delegates to force-field orchestration;
- `determine_rings_aromatic()` delegates to `kekulize`;
- `search_substructure(...)` delegates to the search package; and
- bond/ring geometry queries delegate to `geometry`.

Their public placement is correct, but dependencies, result ownership, and
some surrounding cache behavior still need correction.

### 4.3 Confirmed violations and risks

| Area | Confirmed current state | Architectural consequence |
|---|---|---|
| `core.py` | approximately 6,000 lines containing carriers, topology mutation, graph traversal, rings, valence, charge, hydrogens, aromaticity, geometry, force fields, I/O, RDKit/Open Babel conversion, and thermodynamics | the carrier cannot be imported or understood independently of implementations and toolkits |
| Atom storage | `Atom.attrs` mixes source facts with electron configuration, valence, implicit H, aromaticity, coordinates, constraints, and computed charges; magic numeric offsets select fields | fact dependencies and invalidation are implicit |
| Atom mutation | the `atomic_number` setter immediately recalculates electron configuration, valence, and implicit H | one fact write silently executes chemistry and can commit a partial state |
| Bond mutation | `bond_order` updates `bond_kind` and recalculates neighboring implicit H | one setter changes multiple fact tiers without a transaction |
| bond semantics | `bond_kind`, numeric `bond_order`, Atom aromatic flags, and ring-derived Bond aromaticity can disagree | multiple representations compete without an authority contract |
| derived carrier state | `_atom_pairs`, `_angles`, `_torsions`, `_rings`, `_ring_indices_cache`, `_ligand_rings`, and its signature persist on `Molecule` | algorithm results are mistaken for durable facts and require scattered invalidation |
| toolkit state | `_obmol` and `_row2idx` persist a mutable Open Babel copy; the xTB adapter manually clears them | external mutable objects can become stale second truth sources |
| hidden bonds | `_hided_metal_bonds` and `_hided_covalent_bonds` store workflow-specific topology state on the domain object | temporary algorithm policy alters the apparent molecular truth |
| conformers | current coordinates, `_conformers_index`, parallel conformer arrays, and `Molecule.__getattr__` implicitly share ownership | it is unclear whether a frame or Atom coordinates are current truth |
| MCA storage | `calculator.mca()` stores a typed prediction in `properties["mca"]`, separately writes each `Atom._mca`, and stores `Molecule._mca_sites`; accessors read the latter copies | three persistent representations lack one attachment and dependency-validity boundary; preserve the useful prediction DTO while removing duplicate value stores |
| arbitrary properties | `Molecule.properties` is used as both user annotation space and implicit result store | annotations and calculated facts cannot be governed independently |
| stereo | one 4,700-line module contains fact contracts, remapping, validation, assignment, alignment, graph semantics, and coordinate classification | carrier and rule layers are physically inseparable |
| crystal | `Crystal` stores six cell parameters, imports Open Babel, exposes `obcell`, and contains matrix conversion; `space_group` returns the `NotImplementedError` class rather than a value or raising it | carrier, mathematics, and toolkit code are mixed, with an incomplete public contract |
| import cycles | `_io/_io.py` and `core.py` inspect `sys.modules` to survive their cycle | current module boundaries are already unstable |
| eager toolkit loading | `hotpot.__init__` imports RDKit before exporting cheminfo to stabilize the current RDKit/Open Babel extension load order | importing the fact-carrier API necessarily loads a toolkit backend and preserves a global load-order workaround |
| top-level API | `hotpot.__init__` and `cheminfo.__init__` use wildcard imports; `mol_assemble` leaks typing and module names | accidental names become de facto public API |
| direct core imports | 72 production files and 81 test files directly import `hotpot.cheminfo.core` or a relative equivalent | the monolith is coupled throughout the repository |
| legacy code | `core_func.pyx` and `_clib` have no active production caller; `parse_smarts.py` is a separate incomplete parser superseded by `search`; `org_compounds.py` and `solvents.py` refer to absent spreadsheet files | dead/broken paths obscure the supported implementation |
| command layout | `hotpot/main` and `hotpot/works` overlap in conversion and workflow responsibility | the outer project tree lacks a single command/workflow boundary |
| build layout | native extension names, C++ source paths, relative includes, CMake files, stubs, and package-data rules contain current module paths | a Python-only move would produce broken wheels |

The current code, not a past implementation report, is authoritative for this
audit. In particular, `_obmol` still exists even though an earlier integration
record described its intended removal.

## 5. Fact ownership, priority, conflict, and cache contract

### 5.1 Priority is a dependency order

Priority describes which facts constrain or invalidate other facts. It is not
"last write wins" and is independent of source provenance.

| Priority | Category | Representative values | Persistence rule |
|---:|---|---|---|
| P0 (highest) | structural identity and invariants | Molecule ownership; immutable internal Atom/Bond identities; ordered Atom membership; one complete ordered Bond membership; Bond endpoints | canonical carrier facts; lower tiers never repair or override them |
| P1 | explicit structural/state facts | atomic number, isotope, radical state, bond direction/source representation, current Cartesian coordinates, unit cell, explicit/source stereo records, typed constraints, explicitly declared system charge/spin, explicitly supplied alternative coordinate frames | canonical carrier facts; a P1 change marks affected calculated P2/P3 values for recalculation/confirmation |
| P2 | committed chemical representation | formal charges, coupled Bond kind/order, coupled Atom/Bond aromatic assignment, explicit/implicit-H representation | stored only after explicit import or an atomic rule proposal is accepted |
| P3 | durable selected scientific results | generated/optimized `ConformerEnsembleResult`, partial-charge set, oxidation/electronic-state selection, energy/force records, thermo result, MCA result | typed record with subject identity, method/version, units where applicable, and provenance; selecting a result may atomically update a P1 current-state fact but never silently overwrites P0-P2 |
| P4 | task-derived findings and views | electron configuration, valence, hybridization, neighbours, components, rings, paths, angles, torsions, pair distances, formula, spectra, geometry findings, search hits, validation reports | returned to the caller; not retained on a carrier as hidden state |
| A0 (not a fact) | acceleration state | Plan 19 `_topology_index`, task-native buffers, workspaces, immutable parsed reference tables | private, reconstructible, semantically invisible, and governed by Section 5.6 |

P2 values are facts only after a deliberate commit. A valence proposal or
aromaticity finding returned by a rule is still P4 until the operation accepts
and commits the whole proposal.

### 5.2 Provenance is an orthogonal axis

Two values in the same fact slot cannot be resolved merely by their P-level.
The operation must compare source authority and either choose explicitly or
report a conflict. The existing scientific-state authority vocabulary is a
useful precedent:

```text
verified explicit input
> verified scientific source
> exact-identity trusted source
> named domain rule
> derived constraint/fallback
> unresolved
```

This is domain-specific source selection, not a universal confidence scale
or a rating attached to ordinary results. Each operation declares accepted
sources. A conflict requires incompatible statements about the same subject,
authoritative fact slot, scope, lifecycle and representation convention; valid
equivalent projections, including aromatic and coupled Kekule assignments, are
not conflicts. User mutation can replace a field through its documented
setter/operation, but it cannot violate P0 invariants or cause an unrelated
lower-priority value to repair the new input.

Factual serialization preserves supplied/source facts independently of whether
a current rule evaluation or toolkit projection is available. Strict scientific
projections still prepare and check their required facts. Registered adapters
translate quietly under documented, versioned mappings, minimizing loss without
extra runtime labels. Hotpot-initiated roundtrips retain the original Hotpot
facts' authority; independently supplied native/toolkit objects are authoritative
inputs under their documented import maps. Calculated observations remain
distinct from conversion. Rules, fallbacks, sanitization and adapters do not
silently rewrite authoritative input. Unexpected or unmapped material loss and
integrity failures require precise adapter-scoped diagnostics. Detached carriers
inspect their own contents and gates; conversion cannot reconstruct absent facts
or lost upstream relations. Adapter limitations do not establish chemical
impossibility or redefine Hotpot facts.

### 5.3 Provisional ownership of coupled facts

The following contract is recommended and requires owner confirmation in
Section 16.

| Coupled values | Sole owners and relationship |
|---|---|
| Atom/Bond membership and endpoints | `Molecule` owns ordered membership; each `Bond` owns its endpoints; Plan 19 `_topology_index` is only a projection |
| bond kind and order | one canonical P2 bond representation; exposed kind/order are synchronized projections, not independent truths. Normal kinds follow the fixed mapping; zero/dative/unknown retain their characterized representation. Direction/source transport is separate P1 |
| aromaticity | Atom and Bond local aromatic flags are committed together by one `AromaticAssignment`; Ring aromaticity is always a derived view and never writes back by itself |
| molecular and atom charge | molecular net charge and atom formal charges are distinct facts; they must sum only when formal-charge coverage is declared complete. Partial charges are a named P3 result set and never replace formal charge |
| coordinates and conformers | a `CoordinateFrame` is a value: the selected current Cartesian frame and explicitly supplied source frames are P1; generated/optimized `ConformerEnsembleResult` is P3. `conformer_load(i)` atomically copies one projected frame into current coordinates; adding a source frame does not change current coordinates |
| stereo/source transport | Stage7E installs the sole stereo-record/schema and immutable source-ledger owners. Current isotope/radical/H declarations belong only to the Atom table; display direction belongs only to Bond, separately from dative direction. `StereoPayload` is assembled on request; no retained assembled-payload mirror remains. Stage6 preserves the present whole-payload layout only until the separately gated Stage7E cutover |
| stereo and coordinates | Level-1 stereo records preserve explicit/source representation; coordinates may validate or propose Level-2 assignments but never silently overwrite explicit stereo on disagreement |
| crystal representation | public `Crystal(a, b, c, alpha, beta, gamma)` is the sole persistent cell value and preserves the current API/native ABI; “unit cell” is the concept, not a second carrier type; a lattice matrix is a calculated mathematical view |
| constraints | Molecule owns one typed `ConstraintSet`: coordinates use Atom UID, bonds use Bond UID, and angles/torsions use ordered Atom-UID tuples; Atom/Bond/Angle/Torsion `.constraint` access, if retained, is only a projection into that set |
| MCA and other predictions | a typed result record owns values and identity; `Atom.mca` and `Molecule.mca_sites` remain thin projections over the current valid record |
| MetalCluster and source structure (Plan21) | registered structure definitions and Molecule instances are distinct; a cluster references member facts and external connection mappings without duplicating Atom/Bond/coordinate truth; exact storage remains deferred |

Coordinate conventions are explicit: the Molecule current frame and all
`CoordinateFrame` values contain Cartesian coordinates in angstrom; fractional
coordinates are calculated/import-export values and are never a second Atom
fact. `Crystal` stores `(a, b, c, alpha, beta, gamma)` in angstrom/degrees in
that exact native-buffer order. Matrix row/column and angle conversion behavior
must first be characterized against the current implementation. It is either
preserved exactly or corrected in a separately approved semantic change; this
module move may not silently change it.

The public `StereoPayload` schema remains lossless and unchanged, but it is a
transport value rather than the stored Molecule layout. `set_stereo_payload()`
atomically decomposes it into stereo records, canonical current Atom/Bond
facts, and an immutable source-representation ledger. The ledger retains
presence/unknown/source values required for exact round trips, not a duplicate
`current_hydrogens`, isotope, radical, or display-direction table.
`get_stereo_payload()` reassembles the transport value from those owners.
`set(get(mol))` and supported toolkit round trips must remain byte/field,
presence/source, schema and ordering equivalent under the existing tests.
Bond display direction and the existing dative `_bond_direction` have different
meanings and must have separate canonical slots.

**Stage7E is the assigned implementation and acceptance owner of this storage
contract, and must complete before Stage8 and before Stage12/global ownership
acceptance.** Stage6 extracts rules through the access/capture seam while
preserving current storage, including `get(mol) is payload`, as a transitional
behavior test. That identity is not a permanent public storage guarantee or a
waiver of this section. Stage7E replaces those transitional assertions with
lossless canonical ownership/reassembly gates and deletes whole `_stereo_payload`
as stored truth in the same connected cutover. Its exact source owners, zero-new-
type budget, source-only ledger, H-convention entry decision, lifecycle/transport
coverage and blocking gate are specified in Stage7E below. The Stage6 decision
record is [contracts/decisions.md](stages/stage06/contracts/decisions.md).

Partial charge is likewise never duplicated in Atom storage. A selected
`PartialChargeSet` records presence, values, method/version, units, provenance,
and subject identity. `Atom.partial_charge` projects its row or reports the
reviewed no-result/stale state. A native transport request for `PARTIAL_CHARGES`
packs the selected valid set only; absence is not converted into authoritative
zero values, and no native return path writes Atom arrays directly.

### 5.4 Identity, source ownership, and finite validity state

The [common chemical-type interface protocol](chemical_type_interface_contract.md)
defines ownership, permissions and dependencies across types.
`structure/_dependencies.py` declares fact groups, their cross-type edges,
atom/component/molecule or value-specific scope, and the operations permitted
to recompute. A fact change marks affected existing calculated values
`NEEDS_RECALC`; it does not immediately execute expensive chemistry or erase
source evidence.

| Identity | Included facts | Typical dependants |
|---|---|---|
| `TopologyIdentity` | ordered internal Atom/Bond UIDs and complete endpoints | paths, rings, partition |
| `RepresentationIdentity` | topology; current elemental/isotope/radical/charge/H/aromatic/bond/stereo representation | chemical features and state/model results |
| `GeometryIdentity` | topology plus ordered current Cartesian coordinates | geometry-dependent outputs |
| `CellIdentity` | six canonical cell parameters | periodic outputs |
| `ConstraintIdentity` | topology plus typed constraint entries | constrained calculations |
| `SourceFramesIdentity` | topology plus ordered explicit source frames and provenance | source-frame consumers |

Content identity is separate from owner identity. Molecule-local UIDs can
coincide in independent Molecules; source-dependent views and updates bind to
their source owner as well as content. Copy/components explicitly remap
membership. Independent Crystal values, historical results and standalone
wrapper outputs retain their own value/lifetime contracts; the protocol does
not require them all to have a Molecule parent or mutable status.

Selected P2/P3 validity states are finite: `CURRENT`, `NEEDS_RECALC`, `ABSENT`,
`UNRESOLVED`, `CONFLICT`. They have one owner beside their value/provenance.
A stale value may remain for diagnostics but cannot be returned as current.
Explicit/source facts are not silently replaced by inference; disagreement is
reported or resolved through a complete named operation. Views project the
relevant owner's state; they do not maintain a second mutable dirty record.
Returned historical results retain the content of their execution. Current
result projections check whether the selected attachment is still applicable.

Required identities are captured once per operation/snapshot/attachment and
checked again before committing a long-running result. Ordinary controlled
scalar writes mark dependency groups without whole-molecule hashing. Valid
per-Atom projections must not each rescan all N atoms; a coarse read/iterator
validates once and reuses an immutable result within its task. Current identity
digests are not retained as a second general Molecule cache. No general
mutation revision counter is introduced.

Views fix membership and source identities. Detached membership remains
inspectable. A carrier-dependent use after its relevant source changes is
rejected; the proposed public `StaleStructureViewError` itself remains in the
public-interface review register.

### 5.5 Mutation, lazy recalculation, and complete commits

```text
fact write -> validate complete write set -> install one authoritative value
           -> mark affected dependency groups NEEDS_RECALC (cheap)

needed read -> operation ensures declared upstream inputs -> one snapshot
            -> pure rule -> validate owner/source + complete proposal
            -> atomic materialization -> CURRENT -> return
```

Rules/adapters do not write authoritative fact tables, membership, coordinates,
stereo, constraints, or result records. All private scalar/batch writes use the
same field schema and named commit boundary. Bulk numeric operations work on
operation-owned candidate columns/blocks, then install and mark once. The
owner-approved `updata_atoms` method accepts only declared fields; it does not
expose writable storage. Input/output buffers must have explicit ownership.

| Changed fact | Immediate bookkeeping | Deferred work / retained evidence |
|---|---|---|
| Atom/Bond membership/endpoints | update one topology index; atomically remap or reject table rows, frames, stereo, constraints | mark affected component representation/results; retain attributable explicit facts/provenance |
| atomic number | update sole field | mark affected chemical representation and dependent outputs; no immediate valence/H calculation |
| isotope/radical | update sole field | mark only rule-declared isotope/radical-sensitive groups |
| bond kind/order/aromatic assignment | install one complete coupled representation | mark downstream groups, excluding fields made current by this same assignment |
| formal/net charge/spin/H/stereo | commit appropriate facts/representation | mark actual dependants; no silent redistribution of explicit charge |
| current coordinates | update sole table block | mark geometry dependants; preserve source frames and explicit stereo |
| constraints/cell/source frames | update sole owner | mark only declared dependent groups |
| attach/select final result | verify owner and captured dependencies | record input provenance separately from output validity |

The companion's scenario table is normative: inferred H after element change,
explicit formal charge after element change, breaking an aromatic ring,
changing total charge, absent MCA, and stale energy are distinct cases. Stage1
expands it to every writer; Stage3/5 implement and test the complete map.

Lazy materialization applies to a named pure recomputation with sufficient
inputs and declared execution context. A getter does not download models,
start CUDA, run an external program, or launch optimization. Such results are
reported as absent/stale until the caller uses their explicit operation.
Recalculation failure preserves the previous committed facts and diagnostics;
it never produces a partially updated representation or false CURRENT state.

Optimization installs final coordinates, ensemble, and final-frame energy/force
records in one transaction. Input provenance is not the output's current
validity key. Selecting an ensemble member must not make the ensemble delete
itself merely because the starting coordinates changed. Concurrent/source
mutation before commit rejects the old proposal rather than overwriting new
facts. Technical structural validity and atomicity are tested independently of
scientific geometry/energy monitoring.

### 5.6 Allowed and forbidden retained state

Allowed:

- Plan19's one synchronously maintained `_topology_index`;
- one authoritative typed Atom fact table, required presence/source metadata,
  and finite dependency-group validity state under the approved state budget;
- explicitly selected P2 representation and P3 result records, including
  diagnostic stale records whose status is never misrepresented;
- operation/iterator-local snapshots, ring/path/geometry/native workspaces;
- immutable reference tables and compiled query IR keyed by immutable inputs;
- backend-internal acceleration during one coarse call.

Forbidden:

- retained mutable Open Babel/RDKit molecules, native Graph, or duplicate facts;
- carrier-owned ring/angle/torsion/pair view caches;
- Molecule-keyed global/weak-reference chemistry caches;
- scientific results hidden in arbitrary annotations;
- public writable aliases to canonical numeric storage outside a reviewed
  transactional batch API;
- a generic cache or status whose order changes a scientific result.

`AtomPairs` becomes an immutable task-produced table under its reviewed public
name. Dirty metadata does not authorize restoring the removed P4 view caches.
Expensive durable results have an explicit P2/P3 owner and dependency policy.

## 6. Dependency direction and execution model

### 6.1 Layer dependency

The allowed dependency direction is:

```text
                         user
                           |
                           v
                    structure facade
                           |
                           v
                       operations
                    /      |       \
                   v       v        v
                rules   adapters   reference
                   \       |        /
                    \      v       /
                     immutable snapshots

views <- returned by rules/operations; views never point back as owners
```

More precisely:

1. `structure` owns canonical fact storage, the common chemical-type
   field/interface declarations, finite dependency statuses where applicable,
   snapshot/result contracts, invariant-preserving primitives and thin user
   methods, with no scientific implementation. Each type declares its role,
   owner, permissions and lifecycle. Convenience methods enter named
   operations, including queries and authorized lazy materialization;
   structure never imports an adapter to construct its own snapshot.
2. `rules` depends on immutable structure snapshot protocols and reference data. It
   must not depend on concrete carrier internals or write a carrier. The sole
   `_SourceIdentity` declaration belongs to the pure `structure/_source_identity.py`
   leaf; snapshot contracts import it directly. Live source capture/check/digest
   helpers remain in `structure/_identity.py`, outside the rule import closure,
   as settled by [S9B-06](stages/stage09/cutover/contracts/decisions.md#s9b-06--pure-source-identity-contract-and-evidence-refresh).
3. `operations` may depend on structure contracts, rules, and adapters. It
   owns query composition, lazy-materialization sequencing, working
   copies, validation, and atomic commit.
4. `adapters` may read carriers or snapshots and call external/native systems.
   It returns a transport result or named assignment and never mutates an
   authoritative source carrier.
5. `reference` contains immutable scientific tables and schemas, not active
   calculation logic or instantiated Molecules.
6. package `__init__.py` files re-export downward; internal modules never
   import upward through a package facade. The sole Stage9B initialization
   exception is one fixed `structure._declarations._complete_type_names()` call
   in `structure/__init__.py`, after the concrete carrier/view leaves complete
   and before public import returns. It completes only the existing quoted
   annotation namespaces for default `typing.get_type_hints()` evaluation;
   it creates no class, registry, discovery loop, retained chemical state or
   alternate implementation. The [Stage9B decisions](stages/stage09/cutover/contracts/decisions.md)
   fix its finite bindings and import order.

Carrier convenience methods use function-local imports when necessary to
avoid module-initialization cycles. This is an explicit thin-facade technique,
not a plugin registry or runtime-selected implementation.

The standalone wrapper APIs retain their already approved direct-call
contracts, including any wrapper call that mutates the Molecule object passed
by its direct user. Inside Hotpot, an operation supplies `obWrapper` only an
operation-owned working copy; the adapter translates the wrapper output, and
the operation atomically commits accepted results to the authoritative
Molecule. Thus the wrapper remains independently usable without granting an
adapter a private write path to the source carrier.

The snapshot/result contracts are defined once in `structure/snapshots.py` and
`structure/results.py`. `adapters/structure/*` only constructs those values;
rules only consume them. No second snapshot dataclass is declared in an
adapter, rule, or workflow package. The data-object audit governs consolidation
of existing contracts: use scope-specific composition and shared field order,
not a universal DTO or an inheritance hierarchy created merely to reduce lines.

### 6.2 Read-only rule, selection, builder, and commit signatures

Representative target signatures are fixed conceptually as follows. Exact
type names may be refined during implementation, but rules must remain pure
and operations must remain the only commit path.

Conceptual type names below do not require one new dataclass each. First reuse
the audited record family, protocol, discriminated value or composition; count
every actual new definition in the net-consolidation budget. A shorter old
class list is not a reduction if more wrapper/update classes replace it.

```python
def calculate_valences(snapshot: MolecularSnapshot) -> ValenceAssignment: ...

def calculate_implicit_hydrogens(
    snapshot: MolecularSnapshot,
) -> HydrogenAssignment: ...

def calculate_kekule_assignment(
    snapshot: MolecularSnapshot,
) -> BondOrderAssignment: ...

def validate_stereochemistry(
    snapshot: MolecularSnapshot,
    *,
    coordinates: CoordinateFrame | None = None,
) -> StereoValidationReport: ...

def assign_stereochemistry(
    snapshot: MolecularSnapshot,
    *,
    coordinates: CoordinateFrame | None = None,
) -> StereoAssignmentReport: ...

def find_rings(topology: TopologySnapshot, *, scope: RingScope) -> tuple[RingView, ...]: ...
```

There is no `inplace` option on a rule. A mutating object method delegates to
one private named operation:

```python
def calc_implicit_hydrogens(self):
    from ..operations.chemistry import assign_implicit_hydrogens
    return assign_implicit_hydrogens(self)
```

The operation snapshots, calculates, validates, commits, and returns the
documented existing result, including None for an existing successful command.
It does not contain a second chemistry implementation. Internal proposals,
source/lifecycle records and hard-rule assessments retain the information needed
for calculation, attribution and atomicity. They do not prescribe a public
success envelope: ordinary output is the requested value, which may itself be a
candidate set, distribution or trajectory, without generic success, evidence,
admissibility or confidence fields. Method assumptions and support belong in
companion documentation and benchmark reports; Section11.1 governs findings.

Temporary topology filtering uses the accepted private Stage7B N04 contract in
`structure/_selection.py`:

```python
@dataclass(frozen=True)
class _TopologySelection:
    source: _SourceIdentity
    atom_uids: tuple[int, ...]
    bond_mask: bytes

_capture_topology_selection(mol, *, excluded_bonds=(), ligand_skeleton=False)
_check_topology_selection(mol, selection) -> None
_selected_bonds(mol, selection) -> tuple
```

The immutable 0/1 mask follows the complete Bond order; source and scope checks,
the existing XOR metal-ligand predicate, independent task projections and each
consumer's exact graph are fixed in the [accepted design](stages/stage07/topology-selection/contracts/hidden_topology_design.md).
Existing covalent-contribution and other rule scopes retain their own meanings;
there is no universal new mask inferred from those terms. Temporary work never
moves a Bond between carrier collections or changes ordinary getters through an
ambient selector. Rules/backends receive explicit snapshots or the independent
projected Molecule required by their existing algorithm. No public selector is
added.

Bulk parsers and external converters do not call public setters field by
field. They create one immutable `MoleculeRecord` and invoke one private
builder:

```python
def _build_molecule(record: MoleculeRecord) -> Molecule: ...
```

The builder installs only supplied facts, validates ownership/endpoints once,
constructs `_topology_index` once, and performs no valence, hydrogen,
aromaticity, bond-perception, force-field, or toolkit rule implicitly. Any
requested perception is a subsequent named operation.

There is deliberately no arbitrary dictionary `FactPatch`. Commit contracts
are a closed union of small domain values with named private entry points:

```python
def _commit_topology(mol: Molecule, change: TopologyChange) -> None: ...
def _commit_atom_facts(mol: Molecule, update: AtomFactUpdate) -> None: ...
def _commit_representation(mol: Molecule, assignment: RepresentationAssignment) -> None: ...
def _commit_coordinates(mol: Molecule, update: CoordinateUpdate) -> None: ...
def _commit_stereo(mol: Molecule, update: StereoUpdate) -> None: ...
def _commit_constraints(mol: Molecule, update: ConstraintUpdate) -> None: ...
def _attach_result(mol: Molecule, result: ScientificResultRecord) -> None: ...
```

`RepresentationAssignment` is a tagged union of named immutable assignments
such as `BondOrderAssignment`, `AromaticAssignment`,
`HydrogenRepresentationAssignment`, and `FormalChargeAssignment`; it is not a
free-form field map. Schema-checked scalar/batch Atom writes normalize into
the one private `AtomFactUpdate`; any proposed public keyword batch API accepts
only approved writable fields and cannot act as an arbitrary object setter.

### 6.3 Static architecture gates

The items below are final zero-exception gates. They are introduced as a
ratchet rather than falsely required on Stage 3 while legacy `core.py`,
wildcards, and eager RDKit loading still exist:

1. Stage 1 records every existing violation as an exact file/symbol allowlist
   entry and assigns the stage that must remove it.
2. From Stage 3 onward, the gate rejects every new violation and every change
   outside that allowlist.
3. Each owner stage deletes its entries and immediately turns that rule strict
   for the migrated area.
4. The final gate has an empty allowlist; no permanent exception file remains.

The ratcheted gates cover all of the following:

- structure modules have no module-level import of Open Babel, RDKit, SciPy,
  `thermo`, PubChemPy, native extension modules, or rule implementations;
- importing `hotpot` or `hotpot.cheminfo.structure` does not eagerly import
  Open Babel, RDKit, `thermo`, PubChemPy, or native execution backends;
- rules have zero writes to canonical fact storage or dependency statuses;
- adapters have zero direct carrier-private writes;
- no production module imports the deleted `cheminfo.core` path;
- no package uses wildcard exports;
- public class definitions have one physical definition and one identity;
- native graph/search code includes the one shared CSR Graph definition and
  declares no second graph storage type;
- Hotpot force-field adapters invoke the standalone `obWrapper` backend and
  contain no direct duplicate Open Babel force-field engine;
- mutable external toolkit molecules and redundant fact values are never
  stored on a carrier;
- every public chemical-type constructor, property, method, collection access,
  result and conversion has a reviewed role/owner/permission mapping;
- schema/interface projections, dependency statuses where applicable, and
  nested buffers obey the common protocol; independent values, historical
  results and views do not acquire a second mutable authority;
- task snapshots contain only requested fields; and
- a deleted or moved implementation has zero active source, setup, stub,
  documentation, and test references before its old file is removed.

## 7. Consistent public-interface protocol for chemical types

All public chemical types follow the same declaration and review process.
Each entry records its type/role, authoritative fact or dependencies, owner,
shape/dtype/unit, presence/source, read/write policy, codec, complete operation
and invalidation scope. Constructors, methods, collection access and returned
values receive the same treatment as properties. The
[companion protocol](chemical_type_interface_contract.md) defines these rules;
the API trees below apply them to specific types.

| Role | Interface obligation |
|---|---|
| fact carriers and handles: Molecule, Atom, Bond | aliases project one fact; authorized writes validate and commit a complete change |
| independent values: Crystal, coordinate/stereo values | construction, units, conversion and mutability are explicit; no mandatory Molecule parent |
| registered aggregates: future MetalCluster | a Molecule-associated unit projects member facts internally and multiple connection sites externally; type details and runtime delivery belong to Plan21 |
| collections and selections: MolBundle, conformer access | membership, sharing, addition, selection and clearing are explicit; access adds no second fact store, while explicit copy creates an independent value under its lifecycle contract |
| derived views: rings, angles, torsions, pairs | membership/source are fixed; calculations read prepared owner facts without a write API |
| scientific outputs and reports | values, provenance and guarantees are explicit; historical outputs are stable and current attachments are checked separately |
| snapshots, transport and standalone wrapper values | schema, field order, buffer ownership and lifetime are explicit; approved component API/ABI boundaries remain intact |

A writable property requires a mutable role, input that uniquely determines
the target fact or complete representation, consistency-preserving submission,
and an approved setter. Unique fact mapping alone does not make an immutable
result or a view writable. Coupled inputs use a complete named method and backend
rule rather than setter-order side effects. These are semantic obligations;
they do not introduce a universal base class, Python `Protocol`, registry,
record layout, or automatic setters.

### 7.1 Stable top-level surface

The recommended stable surface is explicit:

```text
hotpot
├── version
├── Molecule
├── Atom
├── Bond
├── BondKind
├── Crystal                         # proposed explicit export
├── MolReader / MolWriter / read_mol
├── MolBundle / to_pyg_dataset
├── is_molecule_input / to_hotpot_mol
├── Searcher / Substructure
├── Query / QueryAtom / QueryBond
├── Hit / Hits
├── SmartsSemantics
├── SmartsSyntaxError / UnsupportedSmartsError
├── ComplexStatistics              # retained only if its behavior is completed/tested
└── explicitly approved assembly names
```

`hotpot.__all__` and `hotpot.cheminfo.__all__` become reviewed snapshots.
Typing names, imported modules, implementation helpers, and wildcard contents
must not leak into them.

The identity requirement is strict:

```python
hotpot.Molecule is hotpot.cheminfo.structure.Molecule
```

There is one class definition, not an alias class, compatibility subclass, or
parallel implementation.

### 7.2 Carrier API tree and implementation disposition

Public signatures and observable semantics are preserved by default. The
trees classify implementation responsibility, not blanket write permissions.
For example, the existing writable Atom symbol/atomic_number aliases share one
fact; Bond kind/order share a coupled representation without automatically
adding a setter to today's read-only `bond_kind`.

Unknown/stale reads, method effects, result ownership and lifecycle are part
of each entry's mapping. New, removed, renamed or changed interfaces remain in
the companion's Section 8 and Section 16 here; examples and private schemas do
not publish attributes. The A1–A13 candidates describe Atom storage changes
within the common protocol, not a separate protocol limited to Atom.
MetalCluster and `Molecule.metal_clusters -> list[MetalCluster]` are
owner-specified future targets; their implementation remains in Plan21.

```text
Molecule
├── facts and membership [carrier]
│   ├── atoms / bonds / coordinates / charge
│   ├── updata_atoms(*, atoms=None, **columns) / atom_matrix (complete float64 snapshot)
│   ├── crystal / create_crystal
│   ├── conformers / conformer_add/get/load/clear (source/result projection)
│   ├── typed constraints projected through clear/constraint helpers
│   ├── properties (restricted to user annotations; see Section 16)
│   └── stereo payload access through named stereo functions
├── registered aggregates [Plan21 reservation]
│   └── metal_clusters -> list[MetalCluster] (future property)
├── topology mutation [thin operation]
│   ├── create_atom / add_atom / replace_atom / remove_atom(s)
│   ├── add_bond(s) / remove_bond(s) / add_component
│   ├── link_atoms / auto_pair_metal / build_all_pair_links
│   └── hydrogen and metal removal helpers
├── topology queries/materializations [read-only operation -> mathematics rule]
│   ├── components (independent Molecules) / simple_paths / shortest_path(s) / longest_path
│   ├── cycle_basis / all_cycle_basis
│   ├── rings / basis_rings / all_basis_rings / ligand_rings
│   ├── atom_pairs / angles / torsions
│   └── graph_spectral / canonical_tree
├── chemistry [chemistry rules + private commit operation]
│   ├── determine_rings_aromatic
│   ├── assign_bond_order
│   ├── calc_atom_valence / calc_implicit_hydrogens
│   ├── add_hydrogens / remove_hydrogens
│   ├── calc_mol_default_charge / update_mol_charge
│   └── default_unpaired_electrons / default_spin_mult
├── geometry and constraint mutation [thin operation]
│   ├── translation
│   └── constraint_bonds_angles / clear_constraints
├── geometry and descriptors [read-only query operation -> rules]
│   ├── pair_dist / dist_matrix
│   ├── bond_ring_relations / bond_ring_piercing_state
│   ├── formula / weight / element_counts
│   └── structural predicates and similarity
├── workflows [operations]
│   ├── build3d / optimize
│   ├── get_partial_charge / assign_partial_charge (Open Babel adapter)
│   ├── search_substructure
│   ├── get_thermo
│   └── mca_sites (typed result projection)
└── adapters [thin conversion]
    ├── read_one / write
    ├── to_obmol / to_pybel_mol / to_rdmol / to_pyg_data
    └── smiles / kekulize_smiles / InChi
```

The existing conformer access surface remains one interface but not one
ambiguous storage bucket. `conformer_add()` records an explicitly supplied
source frame as P1. Sampling/optimization attaches a P3
`ConformerEnsembleResult`. `conformers`/`conformer_get()` expose a deterministic
read-only projection over those owners without duplicating frames;
`conformer_load(i)` only commits the selected frame to current coordinates;
`conformer_clear()` explicitly clears both source frames and attached conformer
ensemble results. Stage 1 freezes the current return/indexing contract before
the projection is implemented.

```text
Atom
├── authoritative-fact projections [identity/row handle]
│   ├── atomic_number / symbol (two existing writable expressions of one fact)
│   ├── id / isotope (isotope public property pending A13)
│   ├── formal_charge
│   ├── coordinates (projection of the Molecule current frame)
│   ├── committed aromatic / implicit-H representation (reviewed permissions)
│   └── read-only parent Molecule reference (membership is Molecule-owned)
├── topology views [read-only query operation -> mathematics rule]
│   ├── idx / bonds / neighbours / neigh_idx
│   ├── hydrogens / heavy_neighbours / connectivity
│   └── rings / in_ring
├── chemistry views [read-only query operation -> chemistry rule]
│   ├── period / group / elemental-class predicates
│   ├── electron configuration / existing oxidation-state views
│   ├── get_formal_charge / get_valence
│   └── bond-order and missing-electron summaries
├── authorized fact writes [thin operation -> complete commit]
│   └── existing reviewed setters; constraints project Molecule's ConstraintSet
├── topology mutation [thin operation]
│   └── add_atom / add_hydrogen / link_with
└── final result projections
    ├── partial_charge (read-only selected typed-result projection)
    └── mca
```

```text
Bond
├── owned facts [carrier]
│   ├── atom1 / atom2
│   ├── bond_kind / bond_order / bond_direction
│   ├── bond_source / bond_source_metadata
│   └── source/display id
├── topology views [read-only query operation -> rule]
│   ├── idx / another_end / in_ring / rings
│   └── atoms / atoms_indices
├── chemistry views [read-only query operation -> rule]
│   ├── is_aromatic / is_covalent / is_heavy_covalent
│   ├── is_metal_ligand_bond / rotatable
│   └── ideal_bond_length
└── geometry views [read-only query operation -> mathematics rule]
    ├── constraint (projection of Molecule ConstraintSet)
    ├── length / bond_segment
    └── bond_segment_distance
```

```text
Crystal
├── canonical Crystal(a, b, c, alpha, beta, gamma) [sole cell carrier]
├── lengths in angstrom; angles in degrees
├── no public matrix addition [matrix mathematics stays in the one rule; existing from_matrix contract retained]
├── from_matrix(matrix) [constructor preserving approved convention]
├── no Molecule back-reference
└── no Open Babel object, space-group placeholder, or toolkit cache
```

MetalCluster is registered here as a separate domain type related to
Molecule, Atom and Bond. It represents single- or multi-metal units, including
oxo units, bimetallic paddlewheels and larger metal/oxo clusters. Internally it
presents a graph or point cloud; externally it acts as a superatom with
multiple attachment sites. Composition is the recommended relationship to
Atom/Bond, with member facts projected from their sole owners.

Each supported cluster kind will have a registered definition supplied or
reviewed by the owner. Template definitions and molecule-specific instances
are separate concepts. This stage records the type skeleton and relationships;
it selects no first template and creates no runtime class, registration API
or implicit recognition route. The [Plan21 draft](../21_metal_cluster/metal_cluster_implementation_plan.md)
records the framework and leaves concrete forms, matching patterns, lifecycle
and public members open. The type's public import/export path is also deferred.

### 7.3 Derived view types

`AtomSeq`, `AtomPair`, `AtomPairTable`, `Angle`, `Torsion`, `Ring`, and
`JointRing` become structurally immutable, identity-bound view/result types.
This does not pretend that referenced Atom/Bond objects are deeply immutable:
the view fixes ordered membership and source identity and has no write API or
copied chemical facts. Convenient properties delegate to rules and may not
mutate Atom, Bond, or Molecule facts. In particular:

- a view records ordered stable member identities and may retain Atom/Bond
  references solely to preserve detached membership inspection. A
  carrier-dependent calculation or later mutating operation revalidates its
  source identity before use;

- `Ring.is_aromatic` has no setter;
- `Ring.kekulize()` and `JointRing.kekulize()` are removed; mutation starts at
  the primary `Molecule` interface;
- `Angle.degrees` and `Torsion.degrees` are geometry-rule projections;
- `AtomPair.shortest_path` and distance are graph/geometry results; and
- no view instance is cached on a carrier.

### 7.4 APIs that must not be carried forward unchanged

| Current surface | Target disposition | Reason |
|---|---|---|
| `Molecule.assign_aromatic()` empty implementation | delete; retain the implemented `determine_rings_aromatic()` facade and rename it only under a separately approved API change | empty duplicate API is misleading |
| `Molecule.setattr`, `Atom.setattr`, broad ID-dict attribute updaters | replace with field-specific transactional setters or restrict to import-only private operations | generic writes bypass fact priority and validation |
| public hide/recover/clear bond family | migrate internal use to operation-local `_TopologySelection`/packed masks; retire the eight accepted methods in the connected Stage7B cutover | hidden bonds are workflow state, not chemical truth |
| `update_angles()` / `update_torsions()` | remove; views are calculated on request | their only purpose is persistent cache mutation |
| `Ring.kekulize()`, `JointRing.kekulize()`, and `determine_aromatic(inplace=True)` | remove mutating view entry points; use one Molecule operation | derived views cannot be a second write interface |
| mutable `Molecule.atom_pairs` dictionary | return an immutable table/view | generated pair data is not carrier truth |
| `Crystal.obcell` | replace with Open Babel adapter function | external object cannot be a carrier fact |
| `Crystal.space_group` current behavior | implement a real result or remove until implemented | returning an exception class is not meaningful behavior |
| `LatticeGroup.operators` current behavior | implement as a rule result or remove until implemented | permanently throwing placeholder is not an API |
| empty `calculator.Calculator` | delete | no behavior or contract |
| `kekulize_joint_ring()` placeholder | delete until a real single implementation is designed | a public function that always raises is not retained as compatibility surface |
| legacy `parse_smarts.py` classes | delete | damaged duplicate parser with no production caller |
| `ob2chem.py` compatibility conversion | delete after consumers use the one Open Babel adapter | duplicate implementation |

Historical pickles that encode `hotpot.cheminfo.core.Molecule` are not
preserved by an alias module in the recommended plan. This is an explicit
owner decision in Section 16.

### 7.5 Construction, detachment, copy, and component semantics

These are the recommended lifecycle contracts. They become normative only
after Decisions 16 and 19 are approved and must be covered by characterization
tests before moving the classes:

- `Atom()` continues to create one new Molecule and register itself;
- `Atom(mol=existing)` continues to register one isolated Atom in that
  Molecule and installs its Plan 19 empty topology-index record in O(1);
- a removed/detached Atom retains the exact inspectable `bonds`/`neighbours`
  behavior frozen by M_17-19; its parent reference is read-only and cannot be
  used to reinsert it by assignment;
- `Molecule.add_atom(foreign_atom)` and `replace_atom(..., foreign_atom)` do
  not silently steal an object from another Molecule. The recommended behavior
  is to copy its factual row into a new locally owned Atom; same-owner duplicate
  insertion is rejected;
- a Bond may be committed only when both endpoints are current members of the
  same Molecule; direct Bond construction never transfers endpoint ownership;
- `Molecule.copy()` deep-copies structural/current-state facts, the current
  coordinate frame, explicit imported/user-supplied P1 frames, Crystal,
  stereo records/source ledger, and constraints through an explicit identity
  map. It creates new internal Atom/Bond identities, preserves source/display
  IDs and source provenance, and drops runtime contexts and generated or
  optimized P3 conformer/result records. Annotation behavior follows Decision
  19; P2 presence/source/validity and pending dirty scopes are remapped too,
  never reset to CURRENT merely because the carrier was copied;
- `Molecule.components` continues to return independent Molecules in current
  deterministic component order. It subsets Atom/Bond facts, current and
  explicit P1 coordinate frames, stereo/source records, and constraints
  through the same identity map; it drops P3 results. Whole-molecule
  charge/spin and Crystal propagation follow Decision 19 because they cannot
  always be attributed to one disconnected component; and
- `add_component()` copies/remaps facts into the destination and never shares
  mutable Atom, Bond, Conformer, Constraint, Crystal, stereo, result, or
  annotation containers.

Current `_model`/`_environ` copying in force-field working copies is deleted;
the relevant runtime context is passed explicitly to the operation.

Every `to_obmol()`, `to_pybel_mol()`, and `to_rdmol()` call returns a fresh
external object. A coarse workflow may reuse that object only inside its task.
This intentionally replaces the currently observable same-object behavior of
two consecutive `to_obmol()` calls and requires Decision 18 approval.

## 8. Recommended target project tree

The names below deliberately use the owner's terms `structure` and `rules`.
`operations` is the minimal sequencing/commit layer; `adapters` is the only
external-system boundary. The MetalCluster entry is a candidate Plan21 home,
not a requirement to create an empty module or export during Plan20.

```text
hotpot/
├── __init__.py                         # explicit public exports only
├── __version__.py
├── __main__.py                         # command dispatcher
├── _cli.py
├── commands/                           # Decision 12: replaces main/works
│   ├── __init__.py
│   ├── convert.py
│   ├── optimize.py
│   ├── ml_train.py
│   ├── ff.py
│   ├── mca.py
│   ├── cbond.py
│   └── models.py
├── cheminfo/
│   ├── __init__.py                     # explicit chemistry exports
│   ├── stereo.py                       # maintained Plan 17 public facade
│   ├── structure/                      # sole durable fact carriers
│   │   ├── __init__.py
│   │   ├── molecule.py                 # Molecule
│   │   ├── atom.py                     # Atom
│   │   ├── bond.py                     # Bond
│   │   ├── bond_kind.py                # shared BondKind contract
│   │   ├── crystal.py                  # sole Crystal cell carrier
│   │   ├── metal_cluster.py            # Plan21 type reservation; no Plan20 runtime file
│   │   ├── conformer.py                # CoordinateFrame, source-frame collection
│   │   ├── stereo.py                   # records, source ledger, payload contract
│   │   ├── constraints.py              # typed ConstraintSet
│   │   ├── snapshots.py                # single immutable snapshot contracts
│   │   ├── substructure.py             # existing SMARTS profile/typed-expression contracts
│   │   ├── results.py                  # typed explicitly attached P3 records
│   │   ├── bundle.py                   # MolBundle carrier collection
│   │   ├── protocols.py                # read protocols used by rules
│   │   ├── errors.py
│   │   ├── _source_identity.py         # sole immutable owner/dependency binding
│   │   ├── _identity.py                # live source capture/check/digest helpers
│   │   ├── _state.py                   # current-version graph capture/restoration
│   │   ├── _declarations.py            # finite quoted-annotation namespace completion
│   │   ├── _fields.py                  # one MolBlock, field schema/codecs/permissions
│   │   ├── _dependencies.py            # cross-type dependency scope/status/read policy
│   │   ├── _atom_table.py              # sole typed-column Atom fact owner
│   │   ├── _topology.py                # Plan 19 topology index
│   │   ├── _mutation.py                # invariant-preserving primitives
│   │   ├── _builder.py                 # one-shot factual MoleculeRecord import
│   │   └── _commit.py                  # closed named commit operations
│   ├── views/                          # immutable derived result objects
│   │   ├── __init__.py
│   │   ├── sequence.py                 # AtomSeq; loads before Bond
│   │   ├── topology.py                 # JointRing, Ring; loads after Bond
│   │   ├── geometry.py                 # Angle, Torsion
│   │   ├── pairs.py                    # AtomPair, AtomPairs; no approved rename
│   │   └── _source.py                  # sole source/geometry/view restore helpers
│   ├── rules/
│   │   ├── __init__.py
│   │   ├── chemistry/
│   │   │   ├── __init__.py
│   │   │   ├── elements.py             # periodic classification/electrons
│   │   │   ├── bonding/
│   │   │   │   ├── aromaticity.py
│   │   │   │   ├── kekule.py
│   │   │   │   ├── valence.py
│   │   │   │   ├── hydrogen.py
│   │   │   │   ├── formal_charge.py
│   │   │   │   └── bond_order.py
│   │   │   ├── coordination_chemistry/  # chemical perception/counting
│   │   │   ├── stereochemistry/
│   │   │   │   ├── algebra.py
│   │   │   │   ├── graph.py
│   │   │   │   ├── coordinates.py
│   │   │   │   ├── validation.py
│   │   │   │   ├── assignment.py
│   │   │   │   └── alignment.py
│   │   │   ├── substructure/           # SMARTS/query chemistry semantics
│   │   │   ├── mca_sites/              # ordered site eligibility/naming/dedup
│   │   │   ├── scientific_state/
│   │   │   │   ├── contracts.py
│   │   │   │   ├── charge/
│   │   │   │   ├── oxidation_state/
│   │   │   │   ├── electronic_state/
│   │   │   │   ├── legality.py
│   │   │   │   ├── providers/
│   │   │   │   └── data/               # versioned manifests/rule data
│   │   │   └── descriptors/
│   │   │       ├── formula.py
│   │   │       ├── graph_spectrum.py
│   │   │       └── similarity.py
│   │   ├── mathematics/
│   │   │   ├── geometry/               # current geometry object/relation/native
│   │   │   ├── graph/                  # current deterministic graph algorithms
│   │   │   ├── graph_matching/         # reusable native RI core
│   │   │   └── internal_coordinates.py
│   │   └── physics/
│   │       ├── forcefields/             # acceptance/coordinates/placement kernels
│   │       └── thermodynamics/          # contracts/policies, not toolkit objects
│   ├── operations/
│   │   ├── __init__.py
│   │   ├── _carrier_signatures.py        # fixed existing eager declaration objects only
│   │   ├── topology.py                  # Atom/Bond/component transactions
│   │   ├── chemistry.py                 # assignment controllers
│   │   ├── geometry.py                  # coordinate commits/transforms
│   │   ├── constraints.py               # typed constraint transactions
│   │   ├── conversion.py                # carrier dispatch/preparation/import/export orchestration
│   │   ├── io.py                        # one MolReader/MolWriter/read_mol and carrier callbacks
│   │   ├── thermo.py                    # caller-owned Thermo and carrier getter/fallback sequencing
│   │   ├── visualization.py             # carrier/string input dispatch and drawing entry points
│   │   ├── search/                      # public search composition
│   │   ├── calculation/                 # scientific-state selection/composition
│   │   ├── conformers/                  # provider-neutral sampling workflow
│   │   ├── forcefields/                 # coordination_placement/repair/optimizer/workflow
│   │   ├── inference/
│   │   │   ├── artifacts/               # manifest resolution/install operations
│   │   │   ├── cbond/                   # runtime application/stages only
│   │   │   └── mca/                     # API/pipeline/resources/public result DTOs
│   │   └── assembly/
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── conversion.py                # external type classification only; no carrier orchestration
│   │   ├── io/                          # input/format mechanics and external parsing/serialization
│   │   ├── structure/                   # carrier -> rule snapshots
│   │   │   ├── geometry.py
│   │   │   ├── stereo.py
│   │   │   ├── substructure.py
│   │   │   ├── scientific_state.py
│   │   │   ├── forcefields.py
│   │   │   └── inference.py
│   │   ├── native/
│   │   │   ├── molecule_buffers.py
│   │   │   └── include/hotpot/native/
│   │   ├── openbabel/                   # conversion; invokes standalone obWrapper
│   │   ├── rdkit/                       # conversion/fingerprints; invokes rdWrapper
│   │   ├── thermo/                      # prepared identifiers/RD values/rows -> external values
│   │   ├── calculation/                 # scientific backend/projection contracts
│   │   ├── inference/
│   │   │   ├── artifacts.py             # filesystem/hash/integrity I/O
│   │   │   ├── runtime.py               # ONNX/PyTorch sessions/tensor bridges
│   │   │   └── serialization.py         # inference DTO encoding/decoding
│   │   ├── pubchem.py
│   │   ├── pyg.py
│   │   └── visualization/               # prepared external objects -> drawing/presentation
│   ├── reference/
│   │   ├── __init__.py
│   │   ├── elements.py                  # immutable element source values
│   │   ├── properties.py
│   │   └── data/
│   │       ├── element_properties.json
│   │       └── ideal_bond_lengths.json
│   ├── obWrappers/                      # standalone Python/C++ component retained
│   └── rdWrappers/                      # standalone Python/C/C++ component retained
├── dataset/                             # retained governed datasets
├── ds/                                  # retained dataset governance
├── pipeline/                            # retained generic pipeline engine
├── plugins/                             # retained optional integrations
└── utils/                               # non-chemical generic utilities only

tools/
└── model_deployment/cbond/              # non-runtime export/inspection tools
```

The tree is a final ownership map, not an instruction to perform one giant
filesystem move. Section 10 orders semantic extraction before physical moves.
Generic duplicate module names are avoided deliberately: molecular formula
belongs to `descriptors/formula.py`, scientific-state search belongs to
`calculation/scientific_state/composition`, chemical coordination belongs to
`coordination_chemistry`, and force-field placement belongs to an explicitly
named `coordination_placement` operation.

## 9. Complete adjustment and migration inventory

### 9.1 `core.py` symbol migration

| Current symbol/responsibility | Final owner | Adjustment |
|---|---|---|
| `Molecule` | `structure/molecule.py` | retain one class and public facade; keep facts/invariants only; make scientific methods thin delegates |
| `Atom` | `structure/atom.py` | retain one identity/row-handle facade; core facts live only in `_AtomFactTable`; schema projections remove toolkit construction and immediate setter chemistry |
| `Bond` | `structure/bond.py` | retain one class; enforce the approved kind/order contract in atomic mutations |
| `BondKind` | `structure/bond_kind.py` | retain one enum shared by carriers and lower chemical rules; C1 extracts the existing definition without changing values |
| `NotInSameMolecule`, `ObjNotInMolecule` | `structure/errors.py` | both remain active; preserve their exact names, exception payloads and native callback identities |
| `MolBlock` dynamic attribute machinery | sole `MolBlock` in `structure/_fields.py`; existing `_atom_table.py`/`_dependencies.py` factual owners | retain the current class and schema; concrete Atom/Bond dispatch imports locally; preserve efficient scalar/batch paths and descriptor/native bootstrap with no untracked alias |
| `AtomSeq`, `Angle`, `Torsion`, `Ring`, `JointRing`, `AtomPair`, `AtomPairs` | `views/sequence.py`, `geometry.py`, `topology.py`, `pairs.py` respectively | keep the current identity-bound membership/constraint projections; AtomSeq loads before Bond, whose bases remain exactly `(AtomSeq, MolBlock)`; JointRing/Ring load after Bond; keep `AtomPairs` without a rename |
| zero-reference `AtomPairKey` | none | delete; stable internal UIDs and immutable pair membership replace this unused identity wrapper |
| `Conformers` | `structure/conformer.py` plus `structure/results.py` | split current/source `CoordinateFrame` facts from generated/optimized `ConformerEnsembleResult`; preserve the public array access as a projection without storing the same frame in both owners |
| Atom/Bond identity, UID high-water marks, `id`, and `idx` | `structure/molecule.py` allocators/primitive methods; `_topology.py` index/membership checks; `_state.py` current pickle capture/restore | preserve current immutable private `_uid`, owner token, lock and non-reusing Molecule allocators; retain `id` as source/display fact and `idx` as derived current row; never use one as the other |
| Atom/Bond/Angle/Torsion constraints | `structure/constraints.py` | migrate to one Molecule-owned `ConstraintSet` keyed by internal identities; views only project membership |
| `InternalCoordinates` | `rules/mathematics/internal_coordinates.py` | pure calculation from explicit coordinates/topology; no Molecule mutation |
| `_metal_valence`, formal charge, valence, electron configuration, implicit-H logic | `rules/chemistry/` | return assignments/findings; delete carrier copies; explicit/source facts and lazy inferred values follow grouped validity policy |
| ring/cycle/path/component/tree/spectrum algorithms | `rules/mathematics/graph` and `rules/chemistry/descriptors` | retain deterministic Plan 19 semantics; return views/results only |
| aromaticity/Kekule/bond-order perception | `rules/chemistry/bonding` | one pure assignment implementation; private operation commits the patch |
| distances, angles, torsions, translation, bond/ring relation | mathematics rules plus `operations/geometry.py` for coordinate mutation | query calculations remain pure; translation is an atomic coordinate commit |
| charge, spin, partial-charge workflows | chemistry rules plus calculation operation/adapters | distinguish formal/net/partial charge fact types explicitly |
| `build3d`, `optimize` | `operations/forcefields` | retain public signatures as thin Molecule methods |
| `search_substructure` and SMARTS packing | `operations/search`, chemistry semantics, structure adapter | reuse one RI matcher; no carrier cache |
| `get_thermo` | `operations/thermo.py` plus external thermo leaves | return the complete caller-owned Thermo proxy; operations own carrier getter/fallback sequencing, adapters receive prepared external values |
| `to_obmol`, `to_rdmol`, `to_pybel_mol`, `to_pyg_data` | first three: `operations/conversion.py` plus external projection leaves; to_pyg_data retains its adapter route | operations own conversion preparation/orchestration and adapters construct fresh external values; no persisted external object; this clarification changes no to_pyg_data contract |
| `read_one`, `write`, SMILES/InChI serialization | operation orchestration plus external I/O/serialization leaves | object methods remain thin delegates; operations own carrier input/preparation/installation and adapters receive prepared external values |
| MCA result access | `structure/results.py` plus inference operation | typed explicit result replaces `properties["mca"]`; `Atom.mca`/`mca_sites` remain projections |
| `_topology_index`, `_update_topology` | `structure/_topology.py` and Molecule primitive | retain the exact Plan 19 conceptual contract; no native Graph or second index |
| hidden Bond lists | operation-local `_TopologySelection`/mask | remove from Molecule after all internal consumers migrate under accepted HT1–HT5 |
| `_model`, `_environ` | operation inputs | remove from Molecule and from working-copy metadata |
| `_obmol`, `_row2idx`, ring/angle/torsion/pair caches | none | delete, with no replacement persistent cache |

### 9.2 Current `hotpot/cheminfo` module-by-module disposition

This table accounts for every current top-level file or package rather than
only the obvious monolith. Mixed conversion/I/O/Thermo/drawing modules split by
Section6.1: operations own carrier allocation, validation, lazy preparation,
mutation and getter/fallback sequencing; external adapter leaves receive prepared
transport/assignment values and no carrier getter callbacks. Public exports point
downward to each actual owner. Until Stage9 moves the concrete Molecule class,
one function-local `_molecule_type()` import in operations.conversion uses its
real core definition; it consolidates existing debt and creates no structure alias.

| Current path | Target | Action |
|---|---|---|
| `cheminfo/__init__.py` | explicit reviewed chemistry facade | replace wildcard assembly/core imports with the approved `__all__`; import each public class/function from its sole owner |
| `AImodels/artifacts/manifest.py`, MCA/CBond manifests, model-store metadata | `operations/inference/artifacts/` plus `adapters/inference/artifacts.py` | current manifest functions return dictionaries, not an existing `ArtifactManifest` class; retain one schema/resolver and isolate filesystem/hash I/O; any new public DTO needs explicit review |
| `AImodels/artifacts/registry.py`, `resolver.py` | `operations/inference/artifacts/` | retain one artifact discovery/install/resolve workflow; no carrier or rule owns a model cache |
| `AImodels/artifacts/cli.py`, CBond/MCA CLI modules | `commands/models.py`, `commands/cbond.py`, `commands/mca.py` | preserve CLI names/arguments while removing command logic from scientific packages |
| `AImodels/cbond/apply.py`, `stage.py`, constants | `operations/inference/cbond/` | retain CBond composition/commit semantics and the `is_Nuc_site` distinction |
| `AImodels/cbond/runtime.py`; MCA `runtime.py`, model/session/tensor code | named `adapters/inference/` leaves plus operation artifact/lifetime facades | sessions/tensors/providers stay below operations; resolving MCA factories and the existing public CBondRuntime constructor retain artifact policy in operations without an upward adapter dependency or second runtime class |
| `AImodels/cbond/deploy/` model export/inspection scripts | repository `tools/model_deployment/cbond/` | keep non-runtime deployment tooling out of the importable package; no second copy remains under operations |
| MCA `api`, `batching`, `pipeline`, `resources`, `workers`, CPU/GPU predictors and GPU consumer | `operations/inference/mca/` | retain current CPU default, bounded CPU/GPU execution, GPU threshold/resource policy, rolling admission, and public workflow API; predictors invoke the runtime adapter |
| MCA `gpu_memory.py` | system/device probe in `adapters/inference/runtime.py`; resource decision in `operations/inference/mca/` | split external GPU-memory observation from batch/window/device policy |
| MCA `site_detection.py` | `rules/chemistry/mca_sites/` plus `operations/inference/mca/site_detection.py` | rules own ordered SMARTS/eligibility/naming and pure1-WL; operation owns compilation/preparation/native mapping and final exact automorphism. Preserve first-match/coordination/isotope semantics without an upward rule dependency |
| MCA `conformer.py` direct RDKit backend | `adapters/rdkit` owns the selected sole route | after Plan17 minimum MCA closure, adapt its native route without algorithm changes; if deferred, relocate the existing ETKDG+MMFF/UFF implementation intact. No ETKDG-only substitution, duplicate production route, or fabricated parity |
| MCA `featurizer`, graph input bridges, `data_extract.py` | `adapters/structure/inference.py` plus `adapters/inference/` | separate fact extraction/native packing from model tensor/session translation; never write carriers |
| MCA `MoleculePrediction` and public inference DTOs | `operations/inference/mca/results.py` | preserve the public return DTOs independently of whether a result is attached to a Molecule |
| attached MCA record | `structure/results.py` | one identity/status-bound attachment composes the existing immutable prediction payload; no copied per-Atom value arrays or second prediction schema. `MCAResultRecord` is a conceptual candidate, counted if a new class is actually required |
| MCA serialization/validation | DTO in `operations/inference/mca/results.py`; encoding in `adapters/inference/serialization.py`; structure/RDKit translation in their adapters; domain validation in `rules/chemistry/mca_sites/` | split each responsibility and delete the mixed serializer implementation |
| `ChemData/ElementProperties.json` | `reference/data/element_properties.json` | move and load through one immutable reference module |
| `ChemData/ideal_bond_length.json` | `reference/data/ideal_bond_lengths.json` | move and remove Bond class-level file-loading logic |
| active `mol_assemble/FragTemplete.json` | `operations/assembly/data/fragment_template.json` | this code-referenced file is the provisional canonical default assembly asset; rename and package/test it with the operation |
| zero-reference `ChemData/MolFragTemplate`, `MolFrameTemplate` candidates | merge into the canonical assembly schema or delete | compare schema/content during Stage 1; never move two apparent template sources forward as parallel defaults |
| `ChemData/StructQueryTemplate` | `rules/chemistry/substructure/data/` | move only if still used; otherwise delete after zero-reference audit |
| `ChemData/coformers`, `ChemData/solvent` | governed `hotpot/dataset/data/` assets | move molecule collections out of the code package with manifest/schema updates; do not instantiate all solvents at import time |
| `MolProps/` | governed dataset assets plus a tested query operation if required | split spreadsheets/docs from code; delete empty or uncalled API rather than preserve a second property system |
| `_clib/` | none | delete generated objects, SWIG wrapper, C/H/I files after build and reference audit; it has no active implementation caller |
| `_io/` | `operations/io.py` plus `adapters/io/` | one MolReader/MolWriter implementation owns carrier orchestration/plugins; external input/format leaves do not allocate/install/write carriers; remove `sys.modules` cycle workaround |
| `_native/` | `adapters/native/` | retain one generic field-selective molecule transport and shared C++ headers |
| `bundle.py` | `structure/bundle.py` | retain MolBundle as a carrier collection; move PyG conversion to adapter |
| `MolecularFacts`, `MolecularFactsIdentity`, `ComponentIdentity`, `HydrogenInventory`, `SemanticBond` | one scientific-state snapshot contract with necessary composed values | preserve the useful MolecularFacts semantic projection and component/H/semantic-bond constituents; align source dependencies with shared identities and merge only audit-confirmed synonyms. A `ScientificStateSnapshot` rename is a public proposal, not proof these five types are duplicates |
| scientific quantity/admissibility/authority enums, findings and provider values/evidence | `rules/chemistry/scientific_state/contracts.py` | A9 retains quantity semantics, exact hard-rule assessments, candidate/provenance data and all four D02 findings/reasons at their existing owners; only generic confidence production/fields and ChemicalConfidence/two exports are approved for removal, pending implementation/evidence |
| policy/provider-toggle/provider-set/selection contracts | `operations/calculation/scientific_state/contracts.py` | execution choice and orchestration configuration stay operational; A2's immutable request/budget/requirements/RuleToggle input closure stays below operations, with D03's two owners/namespaces retained |
| `BackendIdentity` frozen digest dataclass / `BackendContext` Protocol | `adapters/calculation/contracts.py` | define generic scientific backend/projection identity once at the external boundary; inference and force-field wrapper-specific contracts remain with their named standalone/adapters and do not redeclare it |
| state-composition candidates/decisions/resolution reports | `operations/calculation/scientific_state/composition/contracts.py` | retain operational search/result DTOs without declaring a second Molecule snapshot or representation identity |
| `calculator/formal_charges.py`, `molecular_charge.py`, `charge/`, `oxidation_state/`, `electronic_state/`, `legality.py` | `rules/chemistry/scientific_state/` | retain deterministic conservation, legality, charge, oxidation, electronic-state, coordination-counting, and spin rules as snapshot-to-finding functions |
| concrete `state_rules` provider modules, provider base/contracts | `rules/chemistry/scientific_state/providers/` | preserve named deterministic providers and evidence without controller or registry state |
| Immutable rule tables and neutral-atom/oxidation-state JSON | `rules/chemistry/scientific_state/data/` | keep these rule-specific scientific sources immutable and explicitly loaded; universal element facts alone belong in `reference/data` |
| `state_rules/controller.py`, executable `state_rules/manifest.py`, profile decision/sets, registry, defaults, execution/state selection, eligibility, requirements, resolver | `operations/calculation/scientific_state/` | own provider selection, policy choice, registry construction, conflict resolution, and sequencing; these are not chemistry laws |
| `calculator/state_composition/` | `operations/calculation/scientific_state/composition/` | preserve candidate expansion/search/coupling/resolution, serialization, and decision artifacts as one operation workflow; do not duplicate formula/composition rules |
| Molecule extraction embedded in resolvers/calculators | `adapters/structure/scientific_state.py` | construct the one immutable scientific-state snapshot and contain all carrier awareness |
| `calculator/mca_inference.py` | `operations/inference/mca/` | move out of general scientific-state calculation; replace arbitrary result storage |
| `calculator/base.py` | none | delete the empty `Calculator` class |
| `call_thermo.py` | `operations/thermo.py` plus `adapters/thermo/chemical.py` | sole caller-owned Thermo class/public carrier getter/fallback methods in operations; external Chemical/identifier/DataFrame work takes prepared values; preserve complete proxy surface |
| `conformers/contracts.py`, `sampler.py` | `operations/conformers/`; shared `ConformerBatch` value in `structure/conformer.py` | preserve backend-neutral seam and deterministic seed semantics; operation contracts reexport the one immutable structure value, so rule/frame owners acquire no upward operation dependency |
| `conformers/adapters.py` | `operations/conformers/` provider facades plus named RDKit/Open Babel leaves | operations own sample(Molecule,request), H/stereo preparation and coordinate commit; external leaves consume prepared values. Preserve the existing standalone route with no mixed toolkit/upward adapter module |
| `convert.py` | `operations/conversion.py` plus `adapters/conversion.py` | operation owns carrier-producing dispatch/import; leaf only classifies external types with lazy imports; preserve public names/signatures/branch priority |
| `core.py` | split per Section 9.1, then delete | no compatibility re-export module in the recommended plan |
| `core_func.pyx` | none | delete after zero-reference and build audit |
| `core_utils.py` | `operations/io.py` and `operations/topology_queries.py` | one read_mol uses the operation Reader; retain the consumed atom_idx_pair_to_bond_idx implementation at its named topology owner |
| `crystal.py` | `structure/crystal.py`, mathematics cell conversion, Open Babel adapter | keep one canonical six-parameter `Crystal` record; remove Molecule back-reference, Open Babel object, `LatticeGroup`/space-group placeholders; preserve approved matrix convention |
| `draw.py`, `fonts/`, `draw.md` | `operations/visualization.py`, `adapters/visualization/` and docs | carrier-capable draw entry/input sequencing in operations; prepared external rendering/colors/formats/resources in adapter; one implementation per symbol, no upward re-export or carrier callback |
| `elements.py` | `reference/elements.py`, `reference/properties.py`, and `rules/chemistry/elements.py` | split immutable source values from classification/electron/valence calculations; remove Open Babel dependency |
| `forcefields/coordinates.py` pure coordinate copy/clipping/addition and pure acceptance/placement criteria | `rules/physics/forcefields/` | numeric rules consume explicit displacement; mutable RNG creation/draws remain operational; no backend call or carrier write |
| `forcefields.coordinates.perturb(mol, ...)` and other coordinate write facades | `operations/forcefields/` | retain the user/workflow operation while routing its generated coordinates through one atomic coordinate commit |
| carrier extraction in `coordination.py` and `topology.py` | `adapters/structure/forcefields.py` | construct force-field/topology snapshots and operation-local `TopologySelection`; no second graph representation |
| placement/coordination sequencing in `coordination.py`; `ligand.py`; `repair.py` | `operations/forcefields/` | these choose targets, mutate working structures, retry, or repair and therefore are operations, not pure rules; delete any unimplemented placement hook |
| `forcefields/auto.py`, `attempts.py`, `ff.py`, `optimizer.py`, `stage.py`, `trajectory.py`, `workers.py`, `workflows.py`, `working_copy.py` | `operations/forcefields/` | retain workflow, transaction, trajectory, and atomic commit behavior; replace hidden-bond mutation with task-local topology masks |
| `forcefields/contracts.py`, settings, reports, and errors | rule findings in `rules/physics/forcefields/`; operation reports/errors in `operations/forcefields/`; backend protocols in `adapters/openbabel/` | split the mixed contract file by semantic owner without duplicating dataclasses; import wrapper-owned types such as `obWrappers.contracts.ConvergenceLevel` rather than redeclaring them |
| `forcefields/backend.py` | `operations/forcefields/backend.py` plus thin `adapters/openbabel/forcefields.py` | operation owns selection/currentness/coordinate publication/error control and one lifecycle lock; adapter invokes only the retained standalone `obWrapper`; no second backend implementation |
| `native_packing.py`, `native_adapters.py`, binding/translation C++ | `adapters/structure/forcefields.py`, `adapters/native/`, `operations/forcefields/native_adapters.py` | adapters own packing/raw translation only; operation retains RNG scheduling/report/trajectory ingestion/publication; never move mixed controls downward |
| native pure placement/radii/target-selection kernels | `rules/physics/forcefields/_native/` | classify by pure rule semantics even when compiled; preserve one implementation and ABI tests |
| native structure session, optimization/untangling/workflow/trajectory controllers | `operations/forcefields/_native/` | classify by orchestration semantics; build may link them into one extension but source ownership remains explicit |
| `forcefields/cli.py`, `cli_doc.md` | `commands/ff.py` and command documentation | preserve command name/arguments while removing CLI code from the operation package |
| `geometry/object.py`, `relation.py`, `settings.py`, native kernels | `rules/mathematics/geometry/` | move unchanged first; preserve conventions and native ABI behavior |
| `geometry/convert.py` | `adapters/structure/geometry.py` | it understands Molecule/Bond/Ring and therefore is not pure mathematics |
| `graph/components.py`, `cycles.py`, `matrix.py`, `paths.py`, `traversal.py`, native kernels | `rules/mathematics/graph/` | move with deterministic semantics unchanged |
| `graph/spectrum.py` | `rules/chemistry/descriptors/graph_spectrum.py` plus reusable matrix math | separate graph linear algebra from chemical descriptor policy and reuse `rules/chemistry/elements.py` rather than duplicate electron configuration logic |
| `kekulize/_protocols.py`, `aromaticity.py`, `assignment.py`, scientific settings | `rules/chemistry/bonding/` | retain pure protocol/snapshot tests and deterministic aromatic/Kekule assignment proposal code |
| `kekulize/workflow.py` and write-back paths | `operations/chemistry.py` | validate and atomically commit the one coupled Atom/Bond representation proposal; no Ring-level write path |
| `mol_assemble/` | `operations/assembly/` | retain composition workflow; replace wildcard exports with explicit surface |
| `mol_statistics.py` | none under the recommended Decision 20 outcome | delete incomplete `ComplexStatistics`; if the owner rejects that recommendation, complete/test it under a separately added operation node before migration |
| `ob2chem.py` | none | delete duplicate compatibility layer after migration to the one Open Babel adapter |
| `obconvert.py` | `operations/conversion.py`, `operations/chemistry.py` and Open Babel adapter leaves | operations own lazy prep/empty-target validation/installation and perception commits; adapters extract records/assignments or serialize prepared values |
| `obWrappers/` | retain as standalone `cheminfo/obWrappers/`; call it through `adapters/openbabel` | preserve Python/C++ standalone contracts, implementation, and direct component use; do not duplicate wrapper code inside the adapter |
| `org_compounds.py` | none | delete the broken code path; any future real molecule collection enters through governed dataset admission rather than reviving this module |
| `parse_smarts.py` | none | delete explicitly damaged, unused duplicate parser |
| `pubchem.py`, `pubchem.md` | `adapters/pubchem.py` and docs | consolidate duplicated module-level and service methods into one implementation while retaining approved convenience functions |
| `rdWrappers/` | retain as standalone `cheminfo/rdWrappers/`; call it through `adapters/rdkit` | preserve fixed RDKit version, Python/C/C++ contracts, CMake attach support, and standalone tests; do not duplicate wrapper code inside the adapter |
| `rdconvert.py` | `operations/conversion.py`, `adapters/rdkit/conversion.py` and `adapters/rdkit/stereo_transport.py` | operation owns carrier preparation/capture/remap/install; adapters own external graph/stereo translation from prepared values without carrier writes/getter callbacks |
| `rdworks.py`, `rdwork.md` | `adapters/rdkit/fingerprints.py` and docs | retain called/tested fingerprint functions; delete zero-use utilities after audit |
| `search/` public composition | `operations/search/` | retain Searcher/Substructure/Query/Hit API and fixed enumeration semantics |
| `search/smarts.py`, syntax, predicate IR, semantics | `structure/substructure.py`, `rules/chemistry/substructure/` and `operations/search/` | existing profile/typed values remain one structure contract family; pure grammar/comparisons are rules; public allocation, reached carrier reads and recursive execution are operations; remove molecular result cache |
| `search/_native` reusable RI matcher | `rules/mathematics/graph_matching/` | sole RI/domains/exact-label automorphism over shared CSR Graph; pure native chemical predicates belong to chemistry rules, Python translation/host leaves to adapters; no second graph/backend |
| `search/native.py` Molecule/query packing | `adapters/structure/substructure.py`, `substructure_query.py` and `adapters/native/substructure/` | operation owns readiness/locks and query reuse; adapters pack prepared facts/Query IR and native transport. Host containers, attribute preflight and opaque callables preserve their reached Python behavior |
| `smarts.md` | substructure operation/rule docs | update paths and test examples verbatim |
| `solvents.py` | none | delete the broken import-time singleton, missing-sheet dependency, empty methods, and semantically empty Molecule subclass; real solvent assets, if admitted, use the governed dataset/query layer |
| `stereo.py` payload schema and fact dataclasses | `structure/stereo.py`; `structure/_stereo.py` access/storage seam | Stage6 relocates the existing18 public contracts once and preserves public module/pickle identity; Stage7E removes whole-payload stored truth and installs canonical Atom/Bond decomposition |
| `stereo.py` public functions/schema exports | maintained thin `cheminfo/stereo.py` facade | preserve five Plan17 signatures, actual postponed string annotations and public class identities; delegate to one operation/rule implementation; no runtime Core annotation import or private compatibility aliases |
| `stereo.py` pure remap proposals, validation, assignment, alignment, Gate 2-7 | `rules/chemistry/stereochemistry/` | algebra owns `_bond_key`, `_axis_bond_keys`, `_remap_stereo_payload` and shared record/group transforms; graph, coordinates, validation, assignment and alignment each own one implementation; preserve all gates |
| `stereo.py` payload decomposition/remap commit | `adapters/structure/stereochemistry.py`, `structure/_stereo.py` and `operations/chemistry.py` | Stage6 capture/packing uses the structure adapter; lifecycle remap enters the chemistry operation and invokes algebra once; existing `_commit_stereo(mol, source, payload)` needs no update wrapper. Stage7E owns atomic decomposition/reassembly and connected canonical storage migration; rules never write Molecule |

### 9.3 Outer-package and build disposition

| Current area | Adjustment |
|---|---|
| `hotpot/main/conversion.py` and `hotpot/works/convert.py` | consolidate into one tested `hotpot/commands/convert.py`; delete duplicate/partial implementation |
| `hotpot/main/optimize.py`, `ml_train.py` | move command handlers to `hotpot/commands/`; scientific implementation remains in plugins/operations |
| `hotpot/works/gauss.py` | move the specialized report operation beside the Gaussian plugin or delete if unreferenced |
| `hotpot/main`, `hotpot/works` | delete after command registry and tests point to the one handler tree |
| `hotpot/__init__.py` RDKit preload | remove after adapter import-order and installed-wheel tests prove RDKit/Open Babel coexist correctly under lazy loading |
| `hotpot.__init__.py` `sys.path` mutation and global `tempdir` | remove; use normal package imports and operation-local temporary directories |
| public `hotpot.package_root` | replace the one plugin consumer with `importlib.resources`; remove from the public surface after repository-wide reference audit |
| `hotpot.__version__.py` and `version()` | retain one package-version source and the documented `hotpot.version()` callable; package metadata and runtime value must agree |
| `hotpot/pipeline`, `hotpot/ds`, `hotpot/dataset` | keep their architectural roles; update chemistry imports only |
| `hotpot/plugins` | retain as external/optional consumers; update all 72 production direct-core import sites and forbid private carrier writes |
| `setup.py`, `pyproject.toml`, package data, stubs, CMake, shell build scripts | migrate atomically with each native package; test wheel and sdist before old paths disappear |
| README/docs/examples | update only after approved API paths are fixed; test displayed examples verbatim |

### 9.4 Explicit removal list

The final tree contains none of the following:

- `hotpot/cheminfo/core.py` or a compatibility shim at that path;
- `core_func.pyx` or `_clib/`, provided the implementation-stage build audit
  confirms the present zero-use finding;
- `parse_smarts.py`, `ob2chem.py`, or the empty `calculator.Calculator`;
- persistent `_obmol`, `_row2idx`, `_angles`, `_torsions`, `_rings`,
  `_ring_indices_cache`, `_ligand_rings`, or automatically materialized
  `_atom_pairs` carrier state;
- the recursive SMARTS `WeakKeyDictionary` keyed by Molecule;
- wildcard package exports;
- package-import `sys.path` mutation, eager toolkit loading, global package
  temporary-directory state, and the legacy `package_root` convenience value;
- `sys.modules` import-cycle workarounds;
- duplicated old/new implementations, fallback backends, or compatibility
  aliases after the hard cut;
- placeholder APIs that only `pass`, return `NotImplementedError`, or always
  raise it without an implemented contract; or
- direct scientific writes to `Molecule.properties`.

### 9.5 No accidental semantic changes

The accepted Stage7B HT1–HT5 observable/private-format changes remain explicit: in
particular, H working-copy publication retains the complete original Bond prefix
then appends H, intentionally replacing old recovery-after-H ordering while
preserving chemistry, H counts, labels and original caller identity.

Physical relocation must not alter:

- RI-style substructure matching semantics, callable-predicate isolation, or
  deterministic match enumeration;
- relevant-cycle/default-cycle/all-cycle-basis definitions and ordering;
- Level-1 stereo lossless transport or Level-2 Gate 2-7 validation,
  assignment, alignment, and coordination semantics;
- CBond and MCA site recognition, including `is_Nuc_site` distinction;
- MCA CPU/GPU routing, prediction values beyond accepted numerical noise, or
  the below-256-site GPU-to-CPU threshold;
- `rdWrapper` seed behavior and fixed RDKit version;
- `obWrapper` complex force-field workflow and the owner-selected O0 default;
- force-field execution/recovery policy or the monitor definitions emitted for
  finite/non-finite energies, gradients and geometry; those observations are
  not converted into a universal scientific acceptance verdict; or
- CLI command names and command-line arguments.

### 9.6 API and implementation migration matrix

The action vocabulary is: **keep** (same owner and contract), **move** (same
contract, new physical owner), **refactor** (same supported purpose through
one new implementation), **merge** (several bodies become one), **add**, and
**delete**. Stage 1 freezes exact signatures for every retained callable before
the first move; the important contracts are fixed here rather than left to
implementation inference.

The companion data-object audit supplies the scope/contract consolidation
ledger. Before creating a new record, classify existing owners and consumers;
prefer a shared scope-specific value or composition over repeated envelopes.
Each approved family uses consistent prefix/suffix and field order. Count
current, merged/deleted, retained and new definitions under the audit's stated
method; do not claim a reduction by moving classes, replacing types with dicts,
or excluding public/wrapper contracts from the denominator.

#### Public Python surface

| Current API/family | Action | Final owner/contract |
|---|---|---|
| `hotpot.version()` | keep | reads the one runtime/package version source; no eager chemistry backend import |
| `hotpot.Molecule`, `Atom`, `Bond`, `BondKind` | move | exact same class identities at `hotpot.cheminfo.structure`; top-level exports remain the ordinary user entry points |
| `Molecule.updata_atoms`, `Molecule.atom_matrix` | add batch entry / rename existing matrix | owner-approved transactional batch method and complete19-column float64 snapshot property; directly rename atom_attr_matrix, migrate consumers and retain no old-name alias |
| `hotpot.Crystal` | add export/refactor | six-parameter value carrier described in Sections 5 and 7; no Molecule back-reference/toolkit object |
| future `MetalCluster`, `Molecule.metal_clusters -> list[MetalCluster]` | reserve for Plan21 | owner-specified names/relationship; concrete registration, construction, return semantics and export path deferred; no Plan20 runtime addition |
| `MolReader`, `MolWriter`, `read_mol` | move/merge | one `operations.io` implementation with external `adapters.io` leaves; public call signatures and supported formats are frozen in Stage 1 |
| `MolBundle`, `to_pyg_dataset`, `is_molecule_input`, `to_hotpot_mol` | move/refactor | bundle remains a carrier collection; PyG external translation stays in adapter; is_molecule_input/to_hotpot_mol belong to operations.conversion with external classification leaves; convenience signatures remain |
| `Searcher(substructure)`, `has_match(mol)`, `iter_mappings(mol, max_matches=None)`, `search(mol, max_matches=None)` | move | `operations.search`; RI semantics, callable-leaf Python behavior, truncation, and deterministic order remain exact |
| `Substructure`, `Query`, `QueryAtom`, `QueryBond`, `Hit`, `Hits`, SMARTS errors/semantics | move | public composition remains; chemistry predicates and RI execution each have one lower-level owner |
| five `hotpot.cheminfo.stereo` functions in Section 3.2 | refactor | signatures stay exact; thin maintained facade invokes extracted rule/operation code |
| `Molecule.build3d(...)`, `optimize(...)`, `search_substructure(...)`, `get_thermo(...)` | refactor | signatures remain frozen; bodies become thin operation delegates |
| Molecule graph/geometry/chemistry query properties and methods in Section 7.2 | refactor | observable return type/order remains unless this matrix or a Section 16 decision explicitly changes it; no cached second implementation |
| `Atom.partial_charge` | refactor | read-only optional-float projection of the selected typed partial-charge result; creation/replacement occurs only through the named full-set assignment operation, subject to Decision 22 |
| `Molecule.properties` | refactor | user annotations only; calculated results move to typed result records subject to Decision 7 |
| `Molecule.assign_aromatic()` (empty) | delete | `determine_rings_aromatic()` remains the single Molecule-level operation unless separately renamed |
| generic `Molecule.setattr`/`Atom.setattr` and broad attribute dictionaries | delete/refactor | no public generic mutation; private bulk import uses `_build_molecule(MoleculeRecord)` |
| hide/recover/clear-hidden-Bond family | delete after migration | root accepted Decision6/HT1–HT5; internal callers use private task-local `_TopologySelection`, with separate release and connected candidate evidence |
| `update_angles()`, `update_torsions()`, Ring/JointRing mutating Kekule methods, `determine_aromatic(inplace=True)` | delete | calculated views are read-only; one Molecule operation owns each write |
| `AtomPairs`/mutable `Molecule.atom_pairs` | refactor, optional rename | immutable task-produced pair table; public name is resolved by Decision 17, with no parallel alias |
| `Crystal.obcell` | move | Open Babel adapter function returns a fresh object |
| placeholder `Crystal.space_group`, `LatticeGroup`, `kekulize_joint_ring()` | delete until implemented | no placeholder compatibility surface |
| `ComplexStatistics` | complete-and-keep or delete | Decision 20; no incomplete top-level export |
| assembly wildcard surface | replace with explicit exports | top-level classes and the separately documented advanced generator/action functions are fixed in Decision 20; internal registration/batch helpers remain private |
| direct `cheminfo.geometry`, `graph`, `forcefields`, `search`, `AImodels.mca`, `calculator`, and `mol_assemble` imports | hard move or retain path | Decision 3; whichever choice is approved has one implementation and no compatibility alias tree |
| `hotpot.cheminfo.obWrappers` and `rdWrappers` Python APIs | keep | standalone components remain directly usable and are invoked by adapters rather than copied |
| `hotpot.package_root`, `hotpot.tempdir` | delete | resources use `importlib.resources`; temporary paths are operation-local; subject to Decision 21 |

#### Private Python contracts and state

| Current private implementation/state | Action | Final owner/contract |
|---|---|---|
| `_topology_index` and `_update_topology` concept | keep/move | sole synchronous acceleration index in `structure/_topology.py`; no public Graph object |
| internal Atom/Bond object identity derived from rows/Python `id()` | refactor | immutable Molecule-local `_uid`; public/source `id` and derived `idx` remain distinct |
| `_atoms`, `_bonds`, Atom/Bond storage | refactor/move | membership remains unique; Atom facts move into `_AtomFactTable` with `_fields.py` projections and `_dependencies.py`; Bond layout changes only where required by canonical semantics |
| `_hided_metal_bonds`, `_hided_covalent_bonds` | delete | complete Bond collection plus operation-local mask under accepted Decision6/HT1–HT5 |
| `_angles`, `_torsions`, `_rings`, `_ring_indices_cache`, `_ligand_rings`, `_atom_pairs` | delete | identity-bound views are produced on request and never carrier-owned |
| `_obmol`, `_row2idx` | delete | fresh conversion objects and direct topology lookup; no persistent external cache |
| `_model`, `_environ` | delete | explicit operation/runtime inputs |
| stored whole `_stereo_payload` | delete as stored truth in Stage7E | records/schema plus source-only ledger and canonical Atom/Bond facts; atomic lossless assembly/decomposition across all writers/consumers; completion blocks Stage8 and Stage12/global ownership acceptance |
| Angle/Torsion-local mutable constraints | merge/refactor | one Molecule-owned `ConstraintSet`; views only project entries |
| arbitrary `properties["mca"]` and partial-charge storage | refactor | dependency-bound typed records in `structure/results.py` |
| `_io`/`core` `sys.modules` cycle workaround | delete | one-way dependency and local facade imports |
| parser/converter field-by-field setter construction | refactor/add | `_build_molecule(record: MoleculeRecord) -> Molecule` validates and installs one factual batch |
| ad hoc writes from rules/backends | delete/add | closed named commits, including schema-checked `_commit_atom_facts`; complete cross-domain result/coordinate transactions validate source ownership and input dependencies |
| recursive SMARTS Molecule-keyed cache | delete | operation-local compiled query/snapshot state only |
| native packing structs copied per subsystem | merge | one field-selective transport module; each adapter requests only required fields; typed fact columns are persistent owners, not another transport schema |

#### Native C/C++ surface

| Current native surface | Action | Final contract |
|---|---|---|
| `hotpot.cheminfo.rdWrappers._rd_native` and CMake `Hotpot::rdwrappers` | keep | pinned-RDKit standalone Python/C/C++ component and signatures remain unchanged |
| `hotpot.cheminfo.obWrappers._ob_native` | keep | standalone Open Babel/native force-field component; adapters call it as the sole backend implementation |
| `_graph_native` CSR Graph/cycle kernels and `_search_native` RI matcher | move/merge ownership | one shared private CSR Graph definition; graph and RI bindings may remain separate extension entry points, but neither owns a second graph model |
| `_geometry_native` | move | pure geometry kernel ownership under mathematics; ABI/result conventions and tests remain |
| force-field C++ bindings/packing | refactor | translation belongs to adapters; no chemical decision code in bindings |
| force-field C++ placement/radii/target kernels | move | pure physics-rule source ownership, linked once into the retained native component |
| force-field C++ session/workflow/trajectory/optimization controllers | move | operation source ownership, linked once; no Python/C++ duplicate workflow |
| obsolete `_clib`/`core_func.pyx` | delete | only after build/reference zero-use gate proves no active ABI consumer |

This matrix does not authorize signature drift hidden inside a move. The Stage
1 generated signature/export snapshot is appended to the implementation log;
any symbol not covered by the class trees or this matrix is classified and
reviewed before deletion rather than silently disappearing.

## 10. Small-step implementation sequence

Every stage is independently reviewable and ends with one or more focused
commits. A stage is not complete because files were moved; it is complete only
after its named gates pass and the implementation log records the evidence.

At the start of every stage, including separately gated sub-stages, reread this
entire plan and its normative companion records. Assess the current repository
against the objectives, predecessor evidence, selected route and stage scope
before changing files or running that stage's work. Record the source commit,
document digests, worktree/environment state, completed and missing gates, and
the resulting bounded task list in the implementation log. Repeat full
alignment after material plan revisions or architectural/scientific route
changes. Within an aligned stage, bounded continuations review affected records
and actual owners/consumers and reuse valid unaffected reads/evidence; a small
change does not require the entire reread or broad validation chain again. A
changed stage number alone is not evidence that prerequisites passed.

The [integrated requirement record](integrated_requirements.json) maps Plan12
work to existing Stage7C, Stage8A/B, Stage9/10 and Stage12 owners. It is an
inventory, not current-source validation. Each owner traces actual writers and
consumers and updates its affected verification targets before closing an
assigned requirement; no extra parallel implementation or new stage is implied.

### Stage 0 — Freeze the reviewed plan

1. Record the owner-directed architecture, storage, reporting and performance
   requirements, including the consistent public-interface protocol for all
   chemical types. Resolve decisions/public surfaces needed by the next stage
   without treating documentation revision as public API approval.
2. Re-read this document in full and revise conflicting sections, not merely
   append owner comments.
3. Record the approved target tree, API decisions, fact priorities and cache
   exceptions. Register MetalCluster's type/relationships as a deferred Plan21
   target, keeping it distinct from Plan20 executable deliverables.
4. Create `implementation_log.md`; record branch, baseline, environment, and
   each subsequent commit/test result.
5. Select the Plan17-first minimum-MCA route or explicitly record its deferral;
   record independent scientific/performance baselines for both plans. Public
   audit and baseline work can proceed before unrelated API decisions.

Deliverable: plan-only commit freezing this reviewed plan and its normative
companion records together. No production code.

### Stage 1 — Capture behavioral and API baselines

1. Generate a reviewed explicit list of top-level exports, class identities,
   public signatures, CLI commands, and direct submodule paths used by Hotpot's
   own documentation.
2. Inventory every public chemical type by role: carriers/handles, independent
   values, collections, views, scientific results and transport values. Record
   constructors, properties, methods, collection access and conversions, plus
   canonical fields, private writers in plugins/tests and external buffer
   aliases. Finish data-object consumer evidence and review the consolidation/
   rename ledger before adding replacement contracts.
3. Generate the exact architecture-debt allowlist and assign every entry its
   removal stage; this list may shrink but never grow. Map every Section 9
   migration entry to its owning existing stage and verification gates,
   distinguishing semantic splits from Stage10 physical/build migration and
   the separately governed Stage11 tail work.
4. Record deterministic outputs for graph/search, stereo, chemistry
   assignments, conversion round trips, and current MCA fixtures.
5. Record the exact source digest of the historical 187 corpus and current
   rule-derived eligibility roster, including exclusions/unknowns. Deliver the
   O0 multi-seed scientific-report runner/comparator from Section11, with actual
   seeds, raw monitors and all failure/absence records; no fixed 182 assertion.
6. Run the existing default suite, installed-wheel smoke tests, native wrapper
   standalone tests, and focused scientific tests.
7. Measure the pre-refactor side of the full performance set in Section 12
   before changing code, using the frozen old/new semantic-workload mapping.
   Identify new-only mechanism measurements for later reporting without
   inventing an old baseline; older measurements are context only, not the
   executable baseline.
8. Deliver baseline manifests, raw samples/report paths, locked workload and
   comparison policy, estimated runtime/hardware budget, and the per-stage
   exact test/build/benchmark target map. Include scalar/batch fact access,
   lazy-status work, result projection, matrix export and technical test time.
9. Freeze the common interface/mutation/read table and API review register:
   type/role, entry, owner, fact/dependencies, shape/dtype/unit, presence/source,
   permissions, codec, operation, invalidation scope and lifecycle. Include
   alias mapping, coupled methods, unknown/conflict/absence behavior and
   historical-result versus current-projection semantics. Independent values
   need no artificial dirty state. No public addition proceeds unlisted.
10. Freeze a same-method net data-object budget: existing definitions minus
    verified removals/merges plus all new record/update/identity definitions.
    Track protocols, implementation classes, type aliases and C/ABI mirrors
    separately; do not claim a net reduction before the additive side is known.

Deliverable: baseline records and tests only.

### Stage 2 — Remove confirmed dead and broken implementations

1. Delete the old `parse_smarts.py` implementation and migrate any newly found
   consumer to the one supported search parser.
2. Delete the empty `calculator.Calculator` and unused `core_func.pyx/_clib`
   only after setup/source scans confirm zero use.
3. Remove `ob2chem.py` after its consumers use `obconvert`'s one implementation.
4. Remove or quarantine broken `org_compounds.py`/`solvents.py` APIs; preserve
   real assets through the approved dataset path.
5. Remove placeholder APIs only after the Section 16 public-API decision.

Gate: default suite, import smoke, source/build zero-reference checks.

### Stage 3 — Establish contracts and architecture gates

1. Create `structure`, `views`, `rules`, `operations`, `adapters`, and
   `reference` package skeletons without moving class definitions yet.
2. Reuse/consolidate audited read-only protocols, snapshots, assignment/P3
   records and the closed named commit contracts; implement the agreed naming,
   composition and common field order. Do not duplicate envelopes or add a
   generic patch dictionary. Nested numeric buffers must actually be isolated.
3. Define `MoleculeRecord` and the one-shot private import builder so readers
   and toolkit converters do not depend on public setter side effects.
4. Introduce stable private Atom/Bond UIDs, non-reusing high-water marks,
   composable source identities, and reviewed stale-view rejection semantics
   (`StaleStructureViewError` is a pending public type); freeze
   construction/detachment/copy/component behavior before Stage 4 creates
   identity-bound views.
5. Add the ratcheted dependency/import/write gates from Section 6.3.
6. Split immutable element and ideal-bond reference loading from classes.
7. Keep one implementation at all times: a new rule may be called by the old
   class facade, but its old body is deleted in the same commit.
8. Implement `_fields.py` and `_dependencies.py` as shared private declarations
   for chemical types, with type-specific groups, permissions and lifecycle.
   Migrate reviewed mappings without imposing a common base or layout; keep
   validity at the fact/result owner and isolate nested buffers.
9. Implement Atom's `_atom_table.py` against those declarations. Benchmark
   typed-column storage against current arrays and an AoS candidate, choose
   one implementation, and map approved Atom projections. Scalar/batch writes
   share one commit path; no public alias leaks authoritative RAM.

Gate: technical contract/import-cycle tests; builder installs supplied facts
only, performs no implicit chemistry and constructs topology exactly once;
UID non-reuse; exact identity retain/invalidate fixtures; ratchet tests;
default suite; cross-type interface/permission and lifecycle fixtures;
buffer isolation; dirty-propagation positive/negative cases; owner checks;
Atom schema projections and scalar/batch scaling. Migrated areas are strict;
remaining types have assigned later stages and cannot be omitted from final
protocol verification.

### Stage 4 — Extract mathematical rules and structure adapters

1. Move current pure graph and geometry implementations under mathematics
   rules using atomic path updates; preserve native module names until their
   build move is ready.
2. Split `geometry.convert` into carrier-to-geometry adapters.
3. Split graph matrix mathematics from chemical graph-spectrum policy.
4. Extract `InternalCoordinates`, distances, angles, torsions, paths, and pair
   views from `core.py`, plus the pure ring calculations. Complete Ring's
   read-only transition and ring-cache removal together with its writer
   migration in Stage5.
5. Before removing angle/torsion caches, move their persisted constraints to
   the one Molecule-owned `ConstraintSet` and update affected readers/writers
   in the same commit; views project those entries without owning the values.
6. Remove the corresponding non-Ring carrier caches and have thin properties
   calculate and return immutable views, preserving all declared constraints.
7. Extract Crystal's parameter/matrix mathematics under the characterized
   convention; its carrier/toolkit isolation is completed in Stage7A.

Gate: deterministic graph/cycle/search-order tests, geometry and Crystal matrix
parity, native geometry/graph tests, identity-bound stale-view rejection,
detached membership inspection, UID member remap, constraint persistence/remap,
default suite, and focused performance comparison.

### Stage 5 — Extract elemental, bonding, charge, and hydrogen rules

1. Split immutable element facts from elemental classifications and electron
   calculations.
2. Extract formula/mass/count descriptors.
3. Extract valence, implicit-H, formal-charge, bond-order, aromaticity, and
   Kekule calculation into read-only rules.
4. Replace `_atomic_number_setter` and `_bond_order_setter` eager chemistry with
   factual commits plus dependency marking. On-demand pure recomputation runs
   through a named operation and never silently replaces explicit-source facts.
5. Make aromatic/Kekule rules return one coupled Atom/Bond assignment. In the
   same commit, migrate all Ring write-back consumers, remove Ring write-back
   behavior and ring carrier caches, and finish identity-bound immutable Ring
   views; retain one assignment implementation throughout.
6. Enforce the approved fact-conflict/read-permission table, including unresolved
   H, breaking aromatic topology, changed total charge, and absent/stale results.

Owner-directed default-H correction (2026-10-30): absent counts receive
conventional valence-based defaults; supplied counts, including zero, are
retained. This is a bounded exception to the behavior-preservation objective.
The ordinary-nonmetal convention with explicit legacy metal0, preserved read
boundary, atomic current-H materialization and separate unapproved normalization are in
[Stage5 hydrogen defaults](stages/stage05/hydrogen-defaults/decisions.md).

Gate: technical chemistry/SMARTS/transport contracts, immutable Ring views and
coupled-assignment atomicity, lazy-status and one-time materialization tests,
complete default suite; scientific assignment and aromatic/Kekule changes
receive an execution report and explanation.

### Stage 6 — Split stereochemistry facts and rules

1. Relocate all18 existing public enums/frozen records once to
   `structure/stereo.py`, preserving schema, field order, source codes, public
   module/qualname/pickle identity and the ordered23 exports. Keep the five
   facade signatures and actual postponed string annotations unchanged.
2. Split descriptor algebra, graph semantics, coordinate classifiers, validation,
   assignment and alignment into the six named rule owners. Pure remap belongs
   to algebra; lifecycle callers enter the chemistry operation. No private facade
   alias or second implementation remains.
3. Use the exact N09 three-field `_StereoSnapshot`; capture per distinct owner,
   retain transient identity-key referents locally, and freeze each rule's inputs.
   Release every stereo-owned capture/preparation scope before user mapping or
   coordinate callbacks and native mapping work. Preserve conditional H/native
   calls, reference/target exception order and graph-before-coordinate suppression.
   Root separately key-accepted the exact tracked-outer-transaction contention
   and deferred raw-capture RuntimeErrors after revision2 review/audit; their
   triggers, messages, order and late side effects are recorded in the Stage6
   contracts. Production remains HOLD pending exact connected ownership release.
4. Rewire the connected carrier/builder/native/toolkit/conformer/working-copy
   consumers to sole owners. Keep current whole-payload storage and object
   identity behavior only as the Section5.3 transitional preservation contract;
   Stage7E owns its mandatory replacement before Stage8/global acceptance.
5. Reduce `cheminfo/stereo.py` to its maintained explicit facade after every
   Gate2–7 body moves. Require recursive type/key-order/container/scalar native
   packing equality, one pack per target, and the frozen bounded time/allocation
   adverse-disposition gate in S6-R2-09.

Before bounded measurements, freeze identical semantic work/prepared state,
fixtures/artifacts/resources and the existing Section12 sampling/noise policy;
retain all raw/adverse/missing/non-comparable observations. Correctness/count/
retention violations, unexplained scaling/repeated per-query work or unexplained
adverse cost block Stage6 pending fix/repeat. Missing/inconclusive required bounded
evidence stays open. Root must explicitly disposition every attributable adverse
time/allocation result before Stage6 acceptance, with cause, fixture/size/sample
and allocation evidence. Only measured necessary bounded O(V+E) overhead may be
accepted locally with that record; expected overhead cannot auto-pass. Local
acceptance preserves all Section12/Stage12 global performance debt and supplies
no speedup/parity claim or global threshold/waiver. Close this local gate only
with conclusive required evidence, matched compared work/state, passing
correctness/count/retention checks and every adverse result resolved or explicitly
accepted under the bounded terms; missing/inconclusive evidence cannot downgrade
an established violation or unexplained adverse result.

Gate: root-key accepted revision2 [design](stages/stage06/contracts/review.md) and
[decisions](stages/stage06/contracts/decisions.md), explicit connected ownership
release, all Level1/Level2 and connected lifecycle/H tests, exact public
annotation/signature/type-hint/pickle contracts, callback/concurrency/packing
contracts, RDKit/native round trips, the bounded cost gate and explicit adverse
result dispositions, and affected Python3.10–3.14 source and installed-wheel
selection. Static design checks and task34's historical audit
are not implementation passes. Section5.3 ownership remains open until Stage7E.

### Stage 7 — Remove second truth sources and clarify results

Each sub-stage is a separate commit/gate; failures in one risk domain do not
hold uncommitted changes from another.

#### Stage 7A — External-object freshness and adapter boundaries

1. Delete `_obmol/_row2idx`; make each conversion produce a fresh task value
   and make coarse operations reuse it locally when needed.
2. Complete the Section 9 adapter splits for `convert.py` dispatch,
   `core_utils.py` I/O and topology-utility disposition, the external thermo
   bridge/result contract, and visualization. Use the Stage3 builder for
   factual imports and keep external toolkit imports lazy.
3. Finish Crystal's sole six-parameter carrier and fresh Open Babel adapter,
   removing its Molecule back-reference, toolkit state and reviewed
   placeholders; reuse the matrix mathematics extracted in Stage4.

Gate: source/wheel conversion freshness, round trips, adapter isolation,
affected I/O/thermo/visualization technical contracts, Crystal convention and
adapter round trips, memory, and repeated-workflow conversion benchmarks.

The [2026-11-03 root completion](stages/stage07/external-adapters/completion_20261103.json)
closes Stage7A's owning extraction and applicable CP311 gates with the bounded
usable-cost/resource disposition. Historical adverse measurements and unresolved
H/isotope/stereo transport remain recorded; no later or global gate closes.

#### Stage 7B — Topology selections, views, and constraints

1. Replace hidden-bond collections with operation-local topology selections.
2. Finish the pair-table and hidden-topology consumer migration; angle/torsion
   caches removed in Stage4 and ring caches removed in Stage5 are not
   extracted a second time. Verify all remaining views have identity-bound
   immutable membership.
3. Finish migration of the remaining persisted constraints to the same
   Molecule-owned `ConstraintSet` and audit coordinate/bond/angle/torsion
   coverage. Angle/torsion ownership and preservation were established in
   Stage4; retain only view projections, with no second constraint store.

The [accepted constraint checkpoint](stages/stage07/constraints/validation/report.md)
records the completed sole-owner Atom-axis/Bond migration, preserved19/3 projections,
affected CP311 source/current-wheel evidence and root-accepted bounded usable cost.
Original failures and selective-axis/Bond-export regressions remain recorded.
The [root Stage7B completion](stages/stage07/completion_20261103.json) accepts this
constraint slice together with the connected complete-topology cutover, under
the applicable CP311 owning-stage scope. Stage7C/D/E and global gates remain open.

Root has accepted the remaining [hidden-topology cutover design](stages/stage07/topology-selection/contracts/hidden_topology_design.md),
including exact HT1–HT5 masks, private record/pickle cutover, public retirement,
file scope and prospective CP311 checks. Its C0 receipt retains previous-current
fixtures, comparable cost and the observed old H/Bond order. HT1 deliberately
retains the complete original Bond handle/UID/order prefix and appends H; old
recovery placed coordination after H. Chemistry/counts/labels remain required.
The [implementation release](stages/stage07/topology-selection/implementation_release.json)
and C1/C2/C3 freezes retain their original authoring and runtime boundaries. The
[current validation report](stages/stage07/topology-selection/validation/report.md)
records source1313/installed1313, actual installed FF11/pipeline3/MCA1,
C3 targeted source180/installed10, persistence48/current and24/old rejection,
and all23 completed cost cells. Root accepts the bounded usable costs, including
all14 adverse medians and10 adverse Python peaks; no universal parity is claimed.
The original default5385passed/2failed/129skipped and targeted repairs remain
explicit; this is not a new current full-default-suite pass. The root completion
links the durable local evidence archive and leaves other versions, clean-native,
ABI/SDK/sdist, broader science/performance and final Plan20/Plan12 acceptance open.

Gate: complete-topology and operation-graph semantics under accepted HT1–HT5,
stale-view behavior, deterministic views, constraint identity/invalidation,
force-field working-copy rollback, and comparable usable-cost/resource evidence.

#### Stage 7C — Annotations and typed calculated results

1. Split user annotations from typed scientific results using the audited
   reusable record families, not a new wrapper for every result accessor.
2. Replace `properties["mca"]` with a typed, identity-bound MCA result while
   retaining approved `Atom.mca`/`Molecule.mca_sites` behavior.
   Delete `Atom._mca` and `Molecule._mca_sites` value stores in the same change;
   their existing properties project the one selected record.
3. Replace Atom-stored partial charges with the selected identity-bound
   `PartialChargeSet`; update field-selective native packing.

The [accepted selected-result route](stages/stage07/selected-results/contracts/decisions.md)
and [current entry](stages/stage07/selected-results/entry_review.json) begin with
one MCA attachment and finite native/Python transactional invalidation, followed
by the connected partial-charge owner/import/native/model cutover. Stage1's
absence/stale/projection decisions already govern both. Root accepts the
[CP311 MCA checkpoint](stages/stage07/selected-results/validation/checkpoint.json):
fresh fact-native, affected source1721/installed1721, actual MCA1/attachment3 in
each boundary and bounded usable getter/write costs. [The report](stages/stage07/selected-results/validation/report.md)
retains original failures, scoped installed evidence and later obligations.
The connected selected-charge cutover now has an accepted
[CP311 checkpoint](stages/stage07/selected-results/validation/partial-charges/checkpoint.json):
affected source1953/installed1953, final test-only rebind8/8 and actual
FF11/pipeline3/MCA1/attachment3 per boundary. One N14 payload replaces raw Atom
charge ownership; N15 is reused and connected imports/native/model projections
retain their approved meanings. The
[Stage7C completion](stages/stage07/selected-results/completion_20261103.json)
closes only applicable CP311 owning scope, combining both slices. Original
failures, retained GFN-FF restart-read warnings, usable costs, package/native
limits and later obligations remain in the
[report](stages/stage07/selected-results/validation/partial-charges/report.md).
Stage7D and Stage7E are accepted within their CP311 owning scopes; global gates remain open. Cross-stage limitations are indexed in
[deferred_issues.md](deferred_issues.md).

Gate: exact per-mutation retain/mark/recompute table, MCA calculator/site tests,
partial-charge absence/presence/provenance and native packing, copy/component
result behavior, annotation isolation, repeated scalar/all-Atom projections and
one-check batch reads; no quadratic dependency hashing.

#### Stage 7D — Coordinate-frame ownership

Split explicit P1 source frames and generated P3 ensembles from the current
coordinates owned only by the Atom table. Frame selection and final-output
attachment form one complete transaction, with input provenance separate from
final-frame validity; keep the reviewed conformer projection deterministic.

The [F1–F7 route](stages/stage07/coordinate-frames/contracts/decisions.md) and
[entry record](stages/stage07/coordinate-frames/entry_review.json) preserve existing
public frame operations while splitting source values, generated batches and
selected observations using only N16–N18 plus reused N15. Actual caller-owned
standalone collections retain their domain; Molecule-bound Conformers is a
projection. Known Gaussian zero-padding and lost source-frame publication are
connected migration cases, not historical equality requirements. The
[completion](stages/stage07/coordinate-frames/completion_20261103.json) accepts
CP311 owning scope: affected source/installed each2105passed/1skipped, actual
FF11/pipeline3/MCA1/attachment3 per boundary and usable bounded costs. The
[report](stages/stage07/coordinate-frames/validation/report.md) preserves original
failures, final Boolean-noop/cursor repairs, warnings and delivery limits. One
factual native extension was rebuilt for the actual callback gap; five binaries
were reused. The Stage7E follow-on is now accepted within its CP311 owning scope; later/global obligations remain open.

Gate: source/generated role tests, seed behavior, load/clear invalidation,
copy/component semantics, and existing conformer array-access parity.

#### Stage 7E — Canonical stereo facts and lossless source transport

This sub-stage completes Section5.3 before entry to Stage8 and before Stage12 or
global architecture/ownership acceptance. Stage6 extraction cannot waive it.
Stage7A/7B/7D predecessor moves and the exact surviving owner paths must be
frozen at entry; this is one connected storage cutover, not an optional cleanup.

1. Own the current `structure/stereo.py` contracts and `structure/_stereo.py`
   assembly/access seam; `_fields.py`, `_atom_table.py`, `_mutation.py`,
   `_dependencies.py`, `_identity.py`, `_builder.py`, `_commit.py` and
   `structure/_native/facts.cpp`; current carrier lifecycle in `core.py`;
   `operations/chemistry.py`, `operations/stereochemistry.py`,
   `adapters/structure/stereochemistry.py` and `_native/buffers.py`; sole toolkit
   conversion/stereo owners; and the then-current conformer transport and
   force-field working-copy publication/rollback owners. At entry, record an
   exact one-to-one rename manifest for `rdconvert.py` →
   `adapters/rdkit/conversion.py` + `adapters/rdkit/stereo_transport.py`, `obconvert.py` →
   `adapters/openbabel/conversion.py`, and any already moved `_native/buffers.py`,
   `conformers/adapters.py` or `forcefields/working_copy.py`. Assign the surviving
   implementation once; never retain old/new copies. Their later Stage8 workflow
   migration and Stage9 carrier split stay outside Stage7E.
2. Delete persistent whole `_stereo_payload` and every retained assembled-payload
   mirror. Molecule owns stereo records/schema version plus an immutable
   **source-only** representation ledger. Canonical current isotope, radical and
   H declaration values/presence/source live only in the Atom table. Canonical
   display direction lives only on Bond, separately from dative `_bond_direction`.
   The ledger may retain source/unknown/unbound transport information needed for
   losslessness, never a parallel table of bound current values.
3. Freeze the exact slot/presence/source mapping at entry. In particular,
   `current_hydrogens.implicit_hydrogens` is transport output, not an extra H-rule
   input. Preserve Stage5's radical/current-bracket/no-implicit dependency tuple.
   Legacy site-total and toolkit omitted-H counts describe potentially different
   scopes; do not silently normalize imported counts or conflate them with one
   canonical quantity. Necessary distinct declared representation counts belong
   in explicit Atom-table slots, with their scope documented. A semantic H
   normalization needs its own concrete approval before code; unresolved mapping
   blocks this sub-stage and later ownership acceptance.
4. Reuse N01 `_SourceIdentity`, N02 `MoleculeRecord`, N05 `_AtomFactUpdate`,
   existing immutable public transport records and the existing transaction/undo
   mechanism. New class/dataclass/Protocol/enum/ABI-mirror budget is **zero**;
   N09 is already `_StereoSnapshot`, not a second `StereoUpdate` reservation.
   Any concrete extra type requires a separately reviewed entry decision. Preserve
   the public19-column legacy projection and native ABI prefix; private fact
   extensions stay outside that prefix, and all Python/native configuration,
   writers, batch setters, invalidation and rollback migrate together.
5. Make set/decomposition and get/reassembly atomic and lossless in field values,
   scalar types, presence/source, record/fact order and schema. Preserve opaque
   schema47/record-row81 payloads on an empty Molecule and today's accepted
   duplicate/out-of-range `AtomTransportRecord` rows. Such unbound/opaque source
   entries must remain diagnosable at the existing projection boundary; do not
   invent target facts, choose a duplicate winner, reject them earlier or use
   them to justify a second bound-current ledger. Freeze their exact source-only
   handling at entry using the existing immutable record family.
6. Migrate get/set, builder/install, H materialization, source binding and
   invalidation, copy/component/merge/remap, cold pickle, RDKit/Open Babel,
   field-selective native transport, conformer transport and working-copy
   publication/rollback in the same change. Replace transitional `is payload`
   storage assertions with canonical ownership and lossless equality tests;
   preserve the public schema classes and pickle spelling.

Gate: exact owner/rename/slot/unbound-source decisions and zero-new-type audit;
no whole-payload stored truth or mirror; lossless `set(get(mol))`, schema/row/order
and distinct H-scope fixtures; exact effective H seeds and stale-source behavior;
atomic failure/rollback across every connected writer; copy/component/merge,
cold-pickle/toolkit/native/working-copy contracts; and the complete affected
CP311 source and installed-wheel selection with the actual native build bound
to that source. Record commands/artifact identities/results and allocation/
throughput observations in the owning stage report and verification map. Missing
or failed canonical-ownership/affected CP311 evidence blocks Stage8 and
Stage12/global ownership acceptance; Stage6 cannot waive it. Adapt the other
planned Python3.10/3.12/3.13/3.14 versions later, with explicit status and complete
applicable final evidence before support/final acceptance claims. Version parity
alone is not a pre-Stage8 entry condition and does not permit permanent layout
debt or waive the final version/package/native/scientific gates.

Stage7E is **ROOT_ACCEPTED_CP311_OWNING_SCOPE** under the
[completion record](stages/stage07/canonical-stereo/completion_20261103.json).
The canonical Atom/Bond/source-only cutover removes whole-payload and sparse
isotope truth, reuses Record24/tag5/N05/existing Undo with zero new types, and
preserves public/native/H contracts. Affected source2397passed/1skipped and
composite installed2397passed/1skipped, actualFF11/pipeline3/MCA1/attachment3
per boundary, fresh factual native and bounded cost support owning acceptance.
Original failures,12installed support-only load errors and their targeted repair
remain explicit. Stage8A may enter after its own alignment; all later/global
obligations and P20-D018 consolidated optimization remain open.

### Stage 8 — Split rule policy, workflow, and backend execution

Each sub-stage is independently committed and gated:

#### Stage 8A — Scientific-state calculation

Split deterministic providers/rules, Molecule snapshot extraction, and
selection/composition/resolution controllers. Complete the assigned Plan12
providers, hard constraints, factorized candidate composition, bounded decision
search and backend-projection requirements through these same owners. Optional
scientific-profile activation retains its individual approval, applicability
artifacts and held-out validation requirements from
[Plan12 rules](../12_xtb_workflow/rules/README.md); this alignment activates none.

Gate: provider, legality, conservation, conflict/evidence, composition search,
serialization, and state-resolution tests.

The [Stage8A ownership decisions](stages/stage08/scientific-state/contracts/decisions.md)
refine this route from the actual provider dependency graph: immutable molecular
snapshot values live in structure; existing provider request, budget, requirement,
context and composition DTOs remain below operations without new types; all
configuration, registry, search and selection algorithms remain operational.
This corrects the mixed-contract table above where unchanged pure-provider
inputs would otherwise import operations. The executable runtime manifest is
operational, and BackendIdentity is a frozen dataclass, not a Protocol. Exact
paths are in the [migration](stages/stage08/scientific-state/contracts/migration.json).
A1–A6 changes no scientific algorithm, field/default or codec schema. Diagnosed
A7 corrects the extractor's input mutation by preparing an owned working copy
in operations and capturing prepared facts read-only in the adapter, retaining
the public signature and existing chemistry algorithms; its runtime evidence is
separate from the initial relocation checks. Root's
[A9 decision](stages/stage08/scientific-state/contracts/decisions.md#a9--accepted-public-calculation-reports-and-confidence-only-removal)
retains the exact public detailed-report APIs, approves confidence-only removal
from the five typed calculation records and enum/two exports, and resolves D02/D03
as NO_CONSOLIDATION. The [A9 confidence checkpoint](stages/stage08/scientific-state/validation/confidence_checkpoint.json)
accepts the connected CP311 cutover:source443focused,source/installed990affected+5skipped
and3actual pipeline/3MCA checks per boundary, with allfive current types cold-loading.
One enum retires; dataclass delta is zero. This A9 checkpoint alone did not close
wholeStage8A; later/global acceptance remains separate.
Optional activation remains held.

The [A1–A7 ownership checkpoint](stages/stage08/scientific-state/ownership_checkpoint_20261103.json)
is root accepted within CP311 source/installed scope: affected984passed/5skipped,
real CBond→FF→GFN-FF/GFN2 and MCA attachment3passed each per boundary,
144existing classes/12public signatures preserved and12assets delivered.
A7 fixes the reproduced input-mutation boundary with existing copy/H algorithms.
The [A8 H-presence correction](stages/stage08/scientific-state/validation/h_presence_checkpoint.json)
then closes P20-D020: known current H0 remains0, absent/unresolved H remainsNone,
input is unchanged and old mismatched evidence is rejected. Source/installed
affected991passed/5skipped plus real pipeline3/MCA3 each qualify this CP311
scope; no public schema or H inference algorithm changes. Corrected identities
and digest-based bounded ordering may differ from history. The then-open
unknown-H accounting and remaining scientific-state work are disposed by A10/A11
below; A8/A9 evidence alone establishes no Stage8B or global acceptance.

[A10](stages/stage08/scientific-state/contracts/decisions.md#a10--represented-inventory-and-observed-preparation-diagnostics)
now fixes the documented counted boundary: explicit atoms plus known,
nonduplicated implicit H, with unresolved inventory retained rather than a
claim of complete H assignment. It authorizes the already assigned explicit-H
and aromatic-projection diagnostics: precise actual code/subject/reason,
standard explicit-command warnings, unchanged strict failure classes and atomic
promoted-warning rollback. The current row-based ID path needs no speculative
repair. The [A10 diagnostics checkpoint](stages/stage08/scientific-state/validation/diagnostics_checkpoint.json)
accepts764focused cases and source/installed1559affected+5skipped of1564,
with3actual pipeline/3MCA checks per boundary. Source-preserving diagnostics
and warning-promotion rollback are proved within CP311. The actual inventory
probe retains the counted boundary and bounded complete/partial OS conservation
evidence. It also exposes P20-D021: the callable Lewis provider returns a
candidate outside its complete-H prerequisite. Accepted A11 enforces that
existing method domain; no arithmetic or ground-state error is asserted.
The [A11 checkpoint](stages/stage08/scientific-state/validation/lewis_checkpoint.json)
now accepts the correction:55focused, source/installed998passed/5skipped of1003,
with3actual pipeline/3MCA cases per boundary. Unknown non-H site-total-H causes
existing named abstention before enumeration; known0/positiveH and explicit-H
controls preserve values and earlier domain diagnostics. The original new-test
legacy import failure and import-only repair remain recorded. The
[Stage8A owning completion](stages/stage08/scientific-state/completion_20261103.json)
closes A1–A11 within CP311 and P20-D021; optional activation and all later/global
gates remain open. Stage8B requires its own entry alignment.


#### Stage 8B — Force fields

Split physics/acceptance kernels, structure packing, workflow controllers, and
the Open Babel/native adapter. Retain standalone `obWrapper` as the sole
backend implementation. Complete the assigned external xTB/plugin/pipeline
requirements at their operation/adapter boundaries, including finding/control
separation, sample isolation, method-specific transport and atomic coordinate
commits; retain the existing plugin and generic-pipeline ownership boundaries.

Gate: pure-policy tests, native and Python workflow parity, rollback,
trajectory, wrapper standalone tests, and FF benchmarks; O0 scientific execution
produces per-case monitoring reports rather than a fixed success-count gate.

The [Stage8B entry/B1 decision](stages/stage08/forcefields/contracts/decisions.md)
settles D04/D05/FF-CLI at current source before implementation. One validated
OptimizationStoppingCriteria belongs to pure physics contracts, the retained
BondTopology plus unchanged reference constituents belong to structure/forcefields.py,
and carrier capture belongs to the structure adapter. Retire the duplicate
Python stopping/bond/CLI option records with zero new types, retaining the real
native ABI record, scalar wrapper API and trajectory JSON. The old Stage1 paths
and upward delegated-factory proposal are superseded explicitly. B1 is the first
coherent slice; rule/operation/backend/command/native separation, connected
finding-control/sample containment and the full owning gate remain required.

The [B1 CP311 checkpoint](stages/stage08/forcefields/validation/b1/checkpoint.json)
accepts the exact three-type retirement, canonical capture/criteria and CLI
composition. Source297focused and958affected+7skipped pass; installed958+7
uses the same965orderednodes, and actual3pipeline+3MCA pass at each boundary.
The original loader collection failure is retained, six unchanged accepted natives
are reused and actual numerical variation is preserved. No clean-native, cost,
all-version or complete Stage8B acceptance follows from B1.

The [B2 CP311 checkpoint](stages/stage08/forcefields/validation/b2/checkpoint.json)
accepts existing acceptance contracts/pure rules and scoped carrier capture with
zero new classes. The current controller retains lazy check ordering, mutable
public metrics and one-scan/checkpoint reuse. Source283focused and source/installed
990affected+7skipped pass with the same997orderednodes; actual3pipeline+3MCA pass
each. Original static/checker failures, skips and numerical variation are retained.
Six unchanged natives are reused; B3/B4, full Stage8B and later/global gates remain.

The [B3a CP311 checkpoint](stages/stage08/forcefields/validation/b3a/checkpoint.json)
accepts the native input/rule/operation/reference/adapter physical cutover:36old
files retire into47final files, with20FF TUs linked once into the actual rebuilt
_ob_native and five other binaries reused. Source400focused,12native fences
(11executable/1compile-only), source/installed990affected+7skipped of997 and
actual3pipeline+3MCA per boundary pass. Original tool/stale-test-path failures and
actual numerical variation remain. This is no global ABI/performance or full
Stage8B acceptance; Python/backend/CLI B3, B4 and later/global gates stay open.


The [B3b CP311 checkpoint](stages/stage08/forcefields/validation/b3b/checkpoint.json)
accepts the sole FF command/resource move to commands/ff.py and ff.md, lazy dispatch
and package/test consumers. Source/installed each84tests pass with0skips and the
same84orderednodes; actual serial/parallel CC/CO SDF, --doc and packaged resource
checks pass. Wheel474Python bodies and sixunchanged accepted B3a natives match;
41prior non-Python omissions remain. Eight emitted-molecule observations preserve
identity/order/charge/connectivity, finite coordinates and MMFF94s energies, with
explicit low-budget nonconvergence and quality=off scope. No new native build or
whole-stage/global claim follows; remaining Python/backend B3, B4 and later gates
stay open.


The [B3c CP311 checkpoint](stages/stage08/forcefields/validation/b3c/checkpoint.json)
accepts native Python calls/results, carrier capture, shared trajectory values and
settings at their sole adapter/structure/rule owners, retiring four old modules
without shims or new types. Source46focused pass; the same931orderednodes yield
source929passed/2skipped and installed926passed/5skipped. A separate three-test
corpus supplement passes at each boundary, preserving the original full-run
counts. Actual serial/parallel MMFF94s and generic UFF pipeline CC/CO observations
retain composition, connectivity, charge, original handles, currentness, format5
trajectory and persisted lineage; low-budget budget_exhausted remains explicitly
nonconverged. Fresh wheel475Python bodies/sixreused B3a natives has41unchanged
non-Python omissions (P20-D008). Original checker failures remain. Acceptance is
this transport/capture boundary only; coupled Python operation/backend separation,
B4 controls, full scientific/standalone/FF benchmarks and later/global gates remain.

The [B3d operation/backend decision](stages/stage08/forcefields/contracts/decisions.md#b3d--accepted-coupled-python-operation-and-backend-cutover)
settles the remaining mixed Python owners, explicit facade migration, pure numeric
versus RNG boundary and removal of the unimplemented placement hook/types/keyword.
The [B3d CP311 checkpoint](stages/stage08/forcefields/validation/b3d/checkpoint.json)
accepts that connected cutover:210focused pass; source/installed each1125pass/2skip
with the same1127orderednodes across65files. Actual3CBFF/xTB and3MCA cases plus
serial/parallel MMFF94s, UFF pipeline and default conformer/build/MCA pass at each
boundary. Identity, currentness, RNG, rollback, units and format5 are preserved;
low-budget nonconvergence and actual numerical variation remain explicit.
All18old Python files retire without shims;477Python-body wheel/sixreused accepted
B3a natives has41unchanged omissions (P20-D008). D06 remains optional/unreduced.
B4 finding/control/sample containment, whole Stage8B and later/global gates remain
open; this bounded checkpoint does not accept those remaining requirements.

The [B4 failure/control decision](stages/stage08/forcefields/contracts/decisions.md#b4--accepted-failure-and-sample-control-boundary)
separates named known sample failures from unknown/protocol/source/commit
integrity errors. Its connected contract preserves complete positional sample
reports and original denominators, retains finite non-hard observations by
default, gives exact per-code sample/batch breaks, excludes FF evidence from xTB
geometry-only checks and stops exhausted cohorts after persistence. The bounded
StageResult break flag uses the existing runner; no generic-engine redesign is
introduced. The [B4 CP311 checkpoint](stages/stage08/forcefields/validation/b4/checkpoint.json)
accepts the connected stage/CLI boundary:269focused and source/installed each
1408passed/2skipped across the same1410orderednodes, actual3official pipeline
cases and two real3-input/2-output FF/xTB cohorts per boundary. The original
new-test legacy-core import failure and public-export correction are retained;
no checker waiver or scientific threshold change. Fresh478Python-body wheel,
sixunchanged B3a natives and unchanged B3d MCA evidence are rebound. Whole Stage8B
O0 scientific/standalone/FF cost and later/global gates remain open.

The [observed ring-open correction](stages/stage08/forcefields/contracts/decisions.md#observed-ring-open-aromatic-failure--operation-buffer-correction)
uses intact-source-validated Bond-buffer masks for temporary FF networks while
retaining strict ordinary aromatic rules and atomic coordinate-only publication.
It supersedes the earlier selected-Molecule FF route. The [whole Stage8B CP311
checkpoint](stages/stage08/forcefields/validation/whole-stage/checkpoint.json) accepts
source/installed1449passed+2skipped of1451, deep2/official3/mixed2cohorts per
boundary, all187source O0inputs/561slots (546complete,15ineligible,0executionerrors),
unchanged native/MCA evidence reuse and bounded usable organic FF cost. Original
failures, high/nonconverged observations, unresolved later trajectory variation
(P20-D022), finite-only phosphorus coverage (P20-D023) and the historical cost
comparator limitation remain explicit. Stage8C requires its own entry alignment;
later/global science, delivery, version and resource gates remain open.

#### Stage 8C — Search

Split SMARTS chemistry semantics, reusable RI graph matching, structure
packing, and public search orchestration while retaining the sole native CSR
Graph definition.

Gate: SMARTS conformance, callable-leaf isolation, RI mapping parity,
deterministic enumeration, native component tests, and throughput.

The [Stage8C C1 decision](stages/stage08/search/contracts/decisions.md) first
extracts the existing BondKind enum to structure/bond_kind.py so lower chemical
rules do not acquire a carrier import. Member values and ordinary public identity
remain; new pickle metadata names its physical owner while current core-path
pickles still resolve through the existing core export. The [C1 CP311 source
checkpoint](stages/stage08/search/validation/c1/checkpoint.json) accepts35files/
847passed with the same enum body, actual bond/copy behavior and captured cold
pickle contracts. C1 alone did not accept the connected cutover; its completed
C2 evidence is recorded below.

The [C2 connected decision](stages/stage08/search/contracts/decisions.md#c2--connected-search-cutover-selected-by-root)
separates the mixed Python/native bodies, keeps existing value families and
public lazy behavior, and removes molecular recursive-result caching. The
canonical binding is adapters/native/substructure/_search_native; mathematics
remains chemistry/Python-free and uses the existing shared Graph. The
[C2 CP311 checkpoint](stages/stage08/search/validation/c2/checkpoint.json) accepts
49source files with1158passed nodes/49passed subtests/1strict expected failure,
and47installed runtime files with953passed nodes/49passed subtests/the same
expected failure. The205source-only architecture/CI nodes account exactly for
the collection difference. Actual source/installed MCA each passes4cases;
3standalone native harnesses pass. The private wheel has1fresh search binary
and5reused accepted binaries; clean all-native/SDK/sdist and other versions
remain later obligations.

Search/site/SMARTS benchmarks retain22search checks,5040complete embeddings,
43ordered MCA sites on24molecules and serial/concurrent agreement. Root accepts
observed usable costs, including1.252× concurrent compilation and all233adverse
cost observations, without a significance or global speed claim (P20-D025).
P20-D024 retains the actual imported-H graft failure and chemical3/1oracle;
Stage8E must own H consumption before dependent assembly acceptance. The
[report](stages/stage08/search/validation/c2/report.md) preserves original
failures, targeted test/import corrections and qualified evidence reuse.
Stage8C closes within this CP311 scope; later/global gates remain open.

#### Stage 8D — Inference and conformer orchestration

Move CBond/MCA/conformer controllers under operations, runtime/session packing
under adapters, and artifact identity/resolution under their named owners.
Retain standalone `rdWrapper` unchanged.

The selected MCA conformer algorithm was fixed at the cross-plan boundary.
This stage relocates/adapts that one route and does not implement the deferred
Plan17 force-field/clustering features or replace it with ETKDG-only sampling.

Gate: CBond, MCA CPU/resource/site tests, `is_Nuc_site`, conformer seam/seed,
artifact resolver, runtime adapter, and `rdWrapper` standalone tests; bounded
fake-GPU routing covers the below-256 CPU fallback, rolling-window adjustment,
explicit GPU index/count, session reuse, and serialization. If the real GPU
module path, package data, or runtime-session code moves, run an additional
ai-14 one-GPU installed-wheel smoke at this stage rather than waiting for final
4-GPU validation.

The [Stage8D connected decisions](stages/stage08/inference/contracts/decisions.md)
select artifact workflow/resource ownership and the model command first (D1),
then the mixed inference/rule/runtime and conformer preparation boundaries.
Existing prediction/transfer/IPC records retain names and fields; no new
ArtifactManifest or MCA result envelope is introduced. Pointer manifests have
one owner under operations/inference/artifacts/data; effect leaves stay in the
inference adapter. Whole-stage source/installed/actual CPU/fake-GPU/one-GPU and
conformer gates remain open throughout bounded authoring.

#### Stage 8E — Assembly and statistics

Move assembly workflows and either complete or delete statistics according to
Decision 20; install explicit exports with no wildcard leakage.

Gate: documented assembly examples, export snapshot, and completed statistics
assertions if retained.

### Stage 9 — Split the carriers and hard-cut `core.py`

Stage9A completed the remaining Section9.1 semantic responsibilities; Stage9B
moved their carrier facades and deleted `core.py`. Their recorded CP311 scopes
are accepted. [Stage9C](stages/stage09/collections/validation/checkpoint.json)
completes MolBundle and the15-row migration crosswalk within the recorded CP311
boundary. Later/global native/build/version/science obligations remain open.
The [Stage9 decisions](stages/stage09/carriers/contracts/decisions.md) bind this
internal sequencing without changing the overall objective or final gate.

1. **9A — Finish semantic extraction.** Move remaining chemistry/mathematics
   calculations to their existing rule owners and query/preparation/workflow
   composition to named operations. Preserve factual row/UID/membership/commit
   primitives, lazy preparation, transactions and returned values; scientific
   changes require a concrete owning decision and affected evidence. Keep the
   one current class definition in `core.py` during this preparation; add no
   parallel carrier or compatibility route. Validate affected CP311 chemical,
   lifecycle and failure semantics before releasing the physical cutover.
2. **9B — Physical sole-owner cutover.** Once only carriers, invariant-preserving
   primitives and thin facades remain, move each definition and its once-only
   descriptor/native-dispatch/pickle initialization to its sole structure/view
   owner. Preserve approved public signatures and factual lifecycle semantics.
   The [2026-11-04 authoring entry](stages/stage09/cutover/entry_review.json) releases
   static authoring, followed by separately recorded source-ready releases. The
   [Stage9B checkpoint](stages/stage09/cutover/validation/checkpoint.json) records
   its accepted CP311 source/installed scope; whole Stage9 remains open pending
   the collection completion below. The [accepted Stage9B decisions](stages/stage09/cutover/contracts/decisions.md)
   fix the downward order: common leaves, Molecule, Atom, AtomSeq, Bond, remaining
   views, then the finite Section6.1(6) declaration completion. Keep
   `views/__init__.py` non-loading. Molecule owns its seven observation descriptors
   and 17 transaction wrappers; Atom owns its numeric descriptors/native bootstrap;
   geometry views own the existing writable constraint descriptors. Current graph
   pickle helpers belong only to `_state.py`; declaration completion is separate.
   Existing eager annotation/default objects are obtained through fixed private
   `operations/_carrier_signatures.py` accessors called by function-local binding
   helpers, preserving raw annotation/generic/default forms without future-annotation
   conversion. Those namespaces serve declarations only; scientific calls enter
   named operations and structure/views gain no direct rules/adapters import or
   eager operation import statement. Retained transitive geometry/FF startup is
   explicit Stage9B scope and does not close Stage12's final eager/lazy obligations.
3. Update every production/test direct-core import in the Stage1 inventory and
   subsequently discovered consumers, including plugins, pipelines, dataset
   bindings, typing-only imports and string entry points. Historical72/81file
   counts are context, not a completion limit.
4. Delete `core.py` in the same coherent commit that moves the last consumer.
   Add no re-export shim, alias class, fallback import, compatibility subclass
   or old-pickle module. Accepted historical core pickles remain a hard cut.
5. **9C — Complete the collection owner.** Finish rows9.2-026/9.6-006: move the
   sole MolBundle to `structure/bundle.py`, keep its ordered membership and
   convenience signatures, and put PyG export sequencing/serialization at the
   operation/adapter boundary. Remove the old module and bind existing public
   exports to the actual owners; reuse current PyG feature conversion without
   a feature/schema redesign. Validate actual source/installed collection,
   pickle and serialization behavior before whole Stage9 acceptance.

Gate: completed9A evidence, zero active old imports, sole class identity and
current-version pickle/lifecycle tests, full default suite, affected installed
CP311 evidence, plugin import smoke and no new cycles. Later native/build,
version-adaptation and global science/performance gates remain open.

### Stage 10 — Complete physical package and native build migration

The [Stage10A checkpoint](stages/stage10/native-migration/validation/checkpoint.json)
accepts the physical owner/build migration, actual clean CP311 sdist/wheel with
six newly compiled extensions, affected source/installed behavior and independent
native/SDK consumers. This is a probe-profile boundary. Other Python versions,
pinned release/manylinux delivery and global obligations remain open under the
constitution's CP311-first sequencing; this checkpoint is not whole-stage release.

Entry gate: the Stage1 migration map shows that every Section 9 semantic split
needed for the carrier/rule hard cut is complete and verified through Stage9.
This stage completes physical/build migration; separately governed Stage11
decisions retain their existing scope and deferral rules.

1. Move the approved rule/operation/adapter packages with `git mv` and update
   all consumers atomically by coherent package.
2. Update pybind11 extension names/source lists, relative C++ includes, CMake
   attach files, export maps, `.pyi` stubs, package data, and build scripts.
3. Keep C++ public namespaces and standalone wrapper contracts stable unless a
   separately approved decision says otherwise. The [native migration decisions](stages/stage10/native-migration/contracts/decisions.md)
   place generic buffer preparation at `operations/native_packing.py` and its
   sole raw transport/headers at `adapters/native`, preserving the existing
   lock, readiness, error order and one-buffer construction.
4. Build/test sdist and wheels for every supported Python version.

Gate: clean-source build, installed-wheel tests, native symbol/import checks,
standalone C/C++ consumer tests.

### Stage 11 — Clean outer commands and data ownership

This is a separately approved tail stage controlled by Decisions 12 and 13.
It is intentionally not a prerequisite for accepting the carrier/rule hard
cut, and it may be deferred without leaving duplicate chemistry
implementations.

The [Stage11A reference checkpoint](stages/stage11/reference-assets/validation/checkpoint.json)
accepts the two immutable JSON relocations and their sole loader/test/package
consumers within affected CP311 source/installed evidence. No data or public
snapshot semantics changed. Other command, legacy-template, MolProps and
dataset distribution decisions retain their separate scope and status.

1. Consolidate `main`/`works` command handlers into `commands` while retaining
   command names and arguments.
2. Move approved chemistry assets to reference, operation-owned data, or the
   governed dataset tree with manifests.
3. Delete old command/data paths and update docs/tests.

Gate: CLI loading/help/execution tests, package-data inspection, dataset
manifest checks.

### Stage 12 — Explicit exports, documentation, and final hardening

The [Stage12A lazy-import decision](stages/stage12/lazy-imports/contracts/decisions.md)
and [checkpoint](stages/stage12/lazy-imports/validation/checkpoint.json)
accept the explicit root facade and actual toolkit-load/resource consumers at
their tested CP311 source/installed boundary.
The [Stage12B H inventory decision](stages/stage12/hydrogen-representation/contracts/decisions.md)
and [checkpoint](stages/stage12/hydrogen-representation/validation/checkpoint.json)
accept the connected import/target/residual/materialization correction for P20-D026
within affected CP311 source/installed, rebuilt RDKit-native and actual model/workflow
evidence, including intentional field14 and private-pickle effects. Independent Decisions12/13,
final docs, other versions/release and global validation remain open.

1. Replace wildcard exports with the approved `__all__` lists.
2. Remove the top-level RDKit preload and verify lazy RDKit/Open Babel load
   order in both directions from an installed wheel.
3. Update README, API docs, module docs, type references, and every example.
4. Test README/docs examples verbatim.
5. Run the complete validation and performance matrix in Sections 11-12,
   including the integrated Plan12 functional and scientific requirements.
6. Write the main-scope final implementation report, final API tree, deletion
   table, benchmark tables, remaining limitations, and commit map.

No superseded implementation is retained "just in case" after this stage.
Main-scope Stage0–12 completion follows Sections 13–14. Only after its validated
delivery is committed as `B_exp` may the separate experiment in Section 17
start on a new branch, progressing through `D_exp` and `I_exp` independently.

## 11. Technical verification and scientific execution reports

Technical contracts and scientific observations have different purposes.
Ownership, transport, deterministic mathematical algorithms, API permissions,
atomicity, packaging and resource bounds have testable invariants. Named hard
rules reject the affected input, candidate, request or result within their
declared domains. Non-hard geometry, energy and chemistry gates report possible
problems while retaining defined finite observations by default; they do not
establish scientific impossibility or a quality grade. Section11.1 separates
findings from control actions and local failure from batch loss. No fixed
182/182 success requirement or unconditional historical-output equality is
introduced. Changes remain visible and explained; existing algorithms are not
silently rewritten by reporting changes.

Stage1 reruns baselines at the selected executable baseline commit. Previous
reports are input/provenance references only; final evidence comes from the
actually implemented paths. Every report states commit, input/rule/model and
wheel identities, environment, commands, enabled tiers and missing evidence.
Method assumptions and applicability belong in companion documentation;
scientific support belongs in benchmark-validation reports, with literature
where appropriate. An unavailable required run gets a versioned placeholder
with the missing protocol/evidence and blank results, leaving its gate open;
it does not produce a confidence verdict or an invented benchmark outcome.

| Area | Technical evidence / scientific report | Frequency |
|---|---|---|
| architecture/data contracts | import/write/alias/status gates; one owner/class/schema; DTO consolidation ledger and no unapproved public exports | every relevant stage; final full scan |
| chemical-type interfaces/lifecycle | role-based constructor/property/method/collection/conversion mappings, approved signatures/permissions, scalar/batch commits, independent values, views, historical/current results, remap/copy/detachment and rollback | every public/storage change; final cross-type inventory |
| graph/search | deterministic ordered graph/RI/SMARTS mathematical contracts and callable isolation | affected stages and final |
| chemistry/stereo | transport/current-fact integrity tests plus rule-versioned charge/H/aromatic/stereo findings and conflicts | affected stages and final |
| arrays/dependencies | no mutable external alias, presence, source-aware dirty propagation, only-on-demand recomputation, no implicit external execution | Stage3/5/7 and final |
| I/O/wrappers | source/wheel round trips, schema/ABI contracts, fresh toolkit values, fixed Crystal convention; independent C/C++ consumers | affected moves and final |
| conformers/force fields | seed/control/rollback/trajectory technical contracts; per-trial geometry/energy/recovery observations | affected stages and final |
| MCA/CBond | field/order/site-join/resource tests; per-site numeric and recognition reports, not a population correlation alone | inference changes and final |
| scientific corpus | fixed source snapshot/digest, rule-derived dynamic eligibility roster, per-case/per-seed reports including failures/unknowns | declared scientific tier; bounded CI evidence |
| integrated Plan12 functions | current provider/hard-rule/composition/backend/pipeline contracts, actual consumer coverage, required scientific execution and finding/control behavior from the integrated requirement record | assigned owning stages and final; historical Plan12 reports are references only |
| packaging/Python | source/installed CP311 first; declared Python3.10–3.14 adaptation, native/ABI and clean wheel/sdist evidence at their applicable delivery/final gates | affected boundaries and final; report deferred/untested versions explicitly |
| docs/performance | exact published technical examples; baseline/final runtime and resource reports | changed examples; affected benchmarks and final |

### 11.1 Scientific monitoring and comparison policy

Before evaluating a bounded change, state the expected facts, allowed differences,
rejection conditions and checks for its method/representation. Use independently
assessed positive, negative and boundary fixtures, plus useful equivalence,
round-trip or semantics-preserving checks. Historical output and two sides
sharing one implementation cannot be the sole scientific oracle. Exact equality
remains appropriate for deterministic contracts and unchanged/lossless transport;
hashes establish artifact attribution and transport integrity, not chemical truth.
Declared reactions, protonation, electron transfer and other transformations are
judged against their allowed changes, not universal conservation.

For numerical approximations and stochastic methods, declare method/state, units,
context, tolerances and invariants before observing results. Use justified
convergence, geometric equivalence, repetition, distribution or stability criteria
and record actual seeds. Retain actual values, attempts, failures and denominators;
never relax thresholds afterward to absorb an error. A validator exit code alone
establishes neither chemical meaning nor actual business success.

For every actual rule, gate, adapter boundary and workflow stage, its owner
records and tests the following decision table using existing domain contracts:

`finding code | declared domain and trigger | whether a defined value remains available | default warning/local-failure behavior | permitted per-code break scope | commit/fallback consequence`

Only an interpretation/control-relevant problem discovered during evaluation
emits a finding: a stable domain-scoped code, exact rule or phenomenon, affected
subject, and the measured value, reference, threshold, unit or reason needed to
understand it. Evaluations with no such problem stay silent. A named hard gate
rejects locally and preserves its diagnostic observation. Non-finite energy or
another violation of a declared finite-value contract makes that result
unavailable. Broken required units, mapping, attribution, schema or transaction
integrity likewise makes a result unavailable even if its numeric fields are finite.

Non-hard findings default to reporting and continuation with the finite,
attributable observation retained; a finding cannot silently erase, replace or
repair it. Finding content is independent of warning/break policy. A user may
promote a specific code to an operation, sample or batch break; that is expected
control. Independent sample failures remain attributed in stable denominators.
Unexpected termination beyond the affected operation/sample is reserved for an
objective inability to continue or integrity/infrastructure damage that makes
later results untrustworthy. Retry, repair and fallback retain the original
failure and any material change of scientific meaning. This table introduces
no generic public API, result schema or pre-Stage13 rule-interface protocol.

1. Record the 187-source corpus as a historical dataset identifier and the
   exact source digest for each execution. Eligibility is calculated by the
   current named rules; 182 is a historical derived count, not a permanent
   constant. Report newly eligible/ineligible/unknown cases and reasons.
2. Execute O0 with root seeds 42/43/44 and record each actual case/provider seed
   and its derivation. Freeze the same input snapshot for a paired comparison;
   if rules change the eligible roster, report shared/new/removed subsets with
   correct denominators instead of silently comparing different populations.
3. For every case/trial retain input facts, rule/model/backend versions, stage
   outcomes, geometry monitor details and thresholds, energy/units/finite
   state, gradient/recovery observations, last finite frame when available,
   selected final frame or explicit absence, failure/unresolved reason, seed,
   runtime, and artifact paths. Unavailable frames are reported as unavailable,
   never fabricated. Intermediate non-finite states are observations too.
4. Provide distributions and per-case changes relative to prior records.
   Classify input/rule/backend/environment changes, stochastic variation, and
   unexplained differences. Neither success-count equality nor any single
   geometry/energy threshold decides scientific correctness or completion.
   Unexplained changes require investigation and an honest unresolved record,
   not weakened monitors or a forced historical result.
5. MCA reports ordered site identity and `is_Nuc_site` changes separately from
   continuous prediction changes, including per-site/worst-case errors. A
   Plan17 algorithm cutover has its own report; Plan20 must not hide that
   scientific change inside relocation. Stable-input technical projections and
   serialization still retain exact invariant tests where justified.
6. Additional xTB/DFT/QM comparisons beyond the integrated Plan12 requirements
   remain optional independent evidence, not mandatory scientific truth. This
   does not waive required Plan12 functional/scientific execution or its existing
   activation conditions. Missing required software, corpus, hardware or reports
   leaves the applicable gate open and is recorded explicitly; a placeholder is
   not completed validation.

The full scientific test currently in
`tests/scientific_boundary/test_extractant_workflow.py` runs `hotpot-auto` and
asserts 182 quality successes. Stage1 must replace its scientific acceptance
role with a report runner/comparator; keep independent technical assertions
about serialization, execution completeness, and artifact integrity. Do not
merely rename that old test and retain its fixed scientific assertions.

Planned Stage1 private tools (not present/executed merely by this document):

```bash
python plan/20_fact_rule_architecture/run_scientific_report.py \
  --backend o0 --seeds 42 43 44 --output <caller-selected-output>
python plan/20_fact_rule_architecture/compare_scientific_reports.py \
  --baseline <baseline-report> --candidate <candidate-report> \
  --output <caller-selected-output>
```

The comparator produces JSON and a readable Markdown report. Its technical
exit status describes malformed/missing reports or execution errors; it does
not encode a fixed scientific success count or numerical agreement verdict.
Reports distinguish failed execution, a completed calculation with concerning
monitor values, and a scientifically unresolved finding.

### 11.2 Remote GPU validation

Use `ssh ai-14`; task files remain under `~/mca_test` (or the independently
allowed `~/mca_study` when explicitly selected), environment/package operations
under `~/.local/conda`. Do not modify unrelated paths or processes. Record the
installed wheel, exact commit, environment, CPU affinity, GPU indices, shared
load, commands, throughput, routing/session evidence and result paths.

Cover CPU, 1-GPU and 4-GPU execution, below-256-site CPU routing, explicit
indices/count, rolling windows, site records and numerical observations.
Shared-node functional evidence does not substitute for a comparable
performance baseline. Reserve comparable resources or classify timing as
inconclusive; do not claim a speedup from different allocations.

### 11.3 Stage-specific executable verification map

Select checks by affected behavior/risk through the maintained
[execution-unit/business mapping](functional_707_business_mapping.md), this
verification map and actual writers/consumers. Prioritize (1) Molecule/Atom/Bond
facts and lifecycle, (2) graph/aromatic/Kekulé/stereo/valence/H semantics,
(3) CBond → FF → GFN-FF → xTB and its geometry/state/attribution/local-failure
boundaries, then (4) actual MCA inference/mapping/attachment/currentness. I/O and
RDKit/Open Babel business interfaces are best effort; other independent external
issues may defer with impact and verification conditions recorded. A dependency
needed by priorities 1–4 inherits that mandatory business path's priority. This
orders the complete Plan20/Plan12 scope; it deletes no lower-priority requirement.

Between delivery/acceptance milestones, rerun affected checks and reuse valid
unaffected evidence whose source, inputs, methods and scope still apply. The
historical 707-unit inventory and 841-pass constraint selection are attribution,
not count/coverage targets. Further changes, failures or unresolved concerns
justify more verification; broad execution, reflection, fresh archives, profiling
and full-plan rereads are not a prerequisite chain for every small slice. Run
applicable default, package/native/version, backend/model, scientific and other
required tiers at their owning delivery/final gates. CP311 leads; untested or
deferred versions remain explicit and final obligations stay open.

Verify representative actual business workflows with successes and local failures,
attribution/currentness, rollback and finite-recovery checks where the method
requires them. Distinguish broken validation from broken business behavior.
Peripheral checker failures may defer with recorded limits and trustworthy direct
or alternative evidence for the required business; bypassing a checker fabricates
no pass. The defensive-programming inventory is an observation and proposal
record, not a new gate or authorization for blanket guard removal/addition.

Stage1 delivers a machine-readable list for each stage: affected contracts,
exact test/runner/build/benchmark targets, switches, expected technical scope,
report schema and artifact root. It must include newly added storage, dirty
state, permission, scalar/batch projection and DTO consolidation tests in the
Python matrix; the existing hand-selected compatibility script is a starting
point, not proof of complete coverage. Old tests asserting removed cache
identity are rewritten to preserve their original behavioral evidence, with
old→new assertion mappings logged.

Existing technical entry points:

```bash
python -m pytest -q --import-mode=importlib tests
bash tests/run_coverage.sh
bash tests/run_inference_compatibility.sh 3.10 3.11 3.12 3.13 3.14
```

Scientific/high-compute/model-artifact tiers produce the reports above and
separately record technical test results. External/optional tiers run only in
their declared environments. Record exact pass/fail/skip counts for technical
tests, report completion/errors for scientific runs, and reasons for missing
evidence; do not label an unexecuted tier as passed. Installed-wheel runs set
`HOTPOT_TEST_WHEEL`/`HOTPOT_TEST_INSTALLED_PACKAGE` and run outside the checkout.

Import tests use fresh subprocesses because tests/conftest.py preloads RDKit.
Inspect lazy `import hotpot` and both adapter-use orders in source and installed
wheels. Open Babel's adapter may lazily establish safe third-party load order.
Raw OpenBabel-first/RDKit-second import is a documented external diagnostic,
not a Hotpot acceptance requirement.

## 12. Mandatory pre-refactor efficiency baseline and global acceptance

No production refactor starts before Stage1 records an executable efficiency
baseline for the selected route. If Plan17 is performed first, keep its own
before/after records, then record Plan20 baseline anew at the post-Plan17
commit. Plan17 speed changes are not counted as Plan20 gains. This planning
revision itself contains no fabricated timings or performance certificate.

### 12.1 Reproducible workloads and artifacts

Stage1 delivers a runnable benchmark manifest and private baseline/comparison
runner, with explicit workloads, commands, repeat policy and artifact paths.
Use caller-selected output directories, not tracked test-output directories.
The baseline manifest records commit/wheel/input digests, dependency lock,
hardware, affinity/threads, process model, GPU allocation/load, cache/warm state,
model availability, test selection and semantic workload size.

Stage1 maps every comparative entry to an equivalent semantic task and records
its baseline and candidate entry points. Compare complete work, such as a write
followed by the required derived reads, when eager and lazy implementations
have different internal steps. A new mechanism with no baseline counterpart
has its cost reported separately; do not fabricate an old measurement or
include that entry in baseline/candidate speedup aggregates. Freeze the common
technical test selection and report newly added tests' execution cost
separately, while retaining the required full-suite validation results.

Baseline/final use the same workload and allocation. Give raw samples, median,
MAD, p95 where meaningful, peak RSS, and paired-bootstrap 95% confidence
intervals. Microbenchmarks: 3 warm-ups + >=10 measured samples; CPU workflows:
1 warm-up + >=5 samples; GPU workflows: 1 warm-up + >=3 samples. Cold import
uses fresh processes. Pair runs or use randomized/interleaved order where
practical; document shared-load exclusions and sampling budgets.

The manifest also freezes completion/failure/no-output classification and
actual-work accounting for the declared workloads. Report all attempts and
their total elapsed time, together with a paired comparison over the common
comparable execution set; record its membership, denominators and exclusion
reasons. Earlier failure, fewer cases or iterations, or other reduced work
does not count as an architectural speedup. Failed and no-output attempts
remain in the complete report; this accounting introduces no fixed scientific
success-rate requirement.

| Workload family | Required modes |
|---|---|
| import / empty Molecule | cold startup, lazy-loaded modules, allocations |
| factual construction / I/O | SMILES/MOL2/SDF, builder, small/typical/large molecules |
| Atom fact storage | scalar getter/setter loops, field-column batch reads/writes, N×3 transforms, insert/delete/remap, schema projection |
| neural/native export | selected-field N×F matrix, dtype/presence handling, copied bytes and conversion counts |
| result identity/projection | all-Atom MCA/partial-charge access across molecule sizes, batch reads; no N-squared full-digest scans |
| lazy dependencies | marking-only cost, first dirty read, repeated clean reads, coalesced changes, unrelated-group reads |
| graph/search/geometry | topology lookups, rings/paths/views, RI/SMARTS, geometry screening |
| values/collections | Crystal parameter/matrix conversion, MolBundle access, conformer add/get/load/clear |
| conversions | fresh OB/RD objects, one conversion per coarse workflow, snapshot/transport packing |
| conformers / force fields | same computational budget and seeds; scientific observations reported separately |
| MCA / CBond | CPU small/batch, threshold routing, 1-GPU and 4-GPU/64-CPU comparable workloads |
| test execution | fixed technical suite, collection/import time, supported-Python selected matrix; exclude dependency installation/network wait |
| memory | peak RSS, steady-state retained bytes, task buffers for representative workloads |

Status marking must be cheaper than eager recomputation. Scalar writes cannot
copy the whole table, and per-Atom reads cannot repeatedly hash the whole
molecule. Storage prototypes are measured before choosing final layout, then
only one production implementation remains.

The public `atom_matrix` property exports the complete19-column float64
projection. Field-selective snapshots/native/model tasks remain private. Keep
their frozen selected-column workloads separate from full-matrix export;
different field counts/dtypes are not interchangeable performance evidence.

### 12.2 Comparable work, usable cost and resource bounds

Chemical semantics take priority, then acceptable performance, then all-version
adaptation. Design efficient batch paths, operation-local reuse, conversion costs
and resource bounds from the outset; comprehensive optimization is not required
for every stage. Reduction of duplication should improve useful end-to-end
cost, but neither universal parity nor a fixed allowance for rare small
regressions is an acceptance rule. The main agent may accept explained, usable
degradation within authorization, recording its purpose, workload, actual
completion, absolute/relative timing and resources. The owner's approximately
fivefold example is contextual judgment, not a numerical gate.

Define runtime ratio r = candidate/baseline for comparable semantic work (convert
throughput to its inverse); r<1 means faster. Keep predeclared workload membership,
weights, methods/inputs/resources and actual-work accounting. Report each workload,
family/hardware mode and end-to-end aggregate separately, with memory/allocation
scope and uncertainty. Faster imports or reduced scientific work cannot conceal
an adverse business workflow. New-only mechanisms retain their separate cost.

The existing neutral band and confidence intervals describe measured variation;
they do not require a parity verdict to deliver usable work. Do not claim gains
from inconclusive evidence. Preserve raw timings, failed/no-output attempts,
completion and denominators, all adverse results and previous scoped dispositions.
Explain costs and scaling and record any unresolved or accepted limitations;
current policy does not relabel old measurements or failures as passes. OOM,
unbounded growth and failure to complete representative batches remain reliability
problems on affected paths and require resolution before claiming that capability.

Measurement recipes, sampling and weights stay frozen before observation. An
acceptance-policy revision does not rewrite historical manifests/reports or justify
removing benchmarks, skipping required checks, reducing work or weakening chemistry,
source ownership, integrity or approved public contracts. Where an older comparator
cannot distinguish a documented policy-only manifest change, retain that limitation
and obtain trustworthy comparable-work evidence; do not call its failed invocation
a pass. Required final performance/resource reports remain part of acceptance.

### 12.3 Unified cross-stage deferred issues

Maintain [deferred_issues.md](deferred_issues.md) as the current index of accepted
performance/resource compromises, edge/integration assessments and peripheral
validation-tool deferrals. Each stable `P20-Dnnn` records severity, state, impact,
responsible stage, treatment checkpoint, originating evidence, closure conditions
and closure evidence; stage reports retain detailed historical observations.
Update the index during affected business changes and milestone review, without
repeating unrelated passed checks or creating bookkeeping-only commits.

Consolidated treatment may cross stage boundaries after the Stage0–12 main
implementation/migration passes, within Stage12 final hardening before main-scope
acceptance and `B_exp`. Earlier required dependency/delivery gates still apply.
Routine unfinished core work is not optimization debt; in particular Stage7E
still gates Stage8. A demonstrated core semantic/integrity failure, unbounded
growth, OOM or inability to complete representative mandatory work advances the
affected item immediately. Usable costs may close with an explicit bounded final
disposition and applicable evidence; universal parity is not required. Registry
status alone cannot waive Section14 or authorize the Stage13 experiment.

## 13. Implementation log and final report contract

`plan/20_fact_rule_architecture/implementation_log.md` will be append-only for
executed work. Every entry contains:

- date/time, stage, commit, and files/responsibilities changed;
- the single implementation removed or installed;
- API/signature/path changes;
- fact ownership/cache implications;
- technical commands/tests and exact pass/fail/skip counts; scientific runs
  separately record report completeness, execution errors and observations;
- performance measurements where relevant;
- named findings, warning/break-node changes, commit/fallback consequences,
  plan impact and any decision required; and
- the full-plan/companion reread and project-state assessment at stage entry,
  with document digests, repeated full alignment for material plan/route changes,
  and the affected records/evidence reused for bounded continuations.

If implementation discovers a conflict with this plan, work pauses at the
smallest coherent checkpoint. Review affected records, edit contradictory or
redundant wording rather than merely appending, and record and communicate the
evidence, route and verification consequences. Material plan or architectural/
scientific route changes repeat full alignment; a bounded same-route correction
reuses valid unaffected alignment and evidence.
In-scope evidence-bound revisions proceed under the constitution's main-agent
authority; an unspecified public change or new scientific scope still follows
its concrete dependent decision under Section16.4.

The final implementation report below covers the main Stage0–12 delivery.
Complete its validation, report and Git submission before starting the
experiment, and record the validated baseline `B_exp` under Section 17.2.
Stage13A and Stage13B have separate definition/integration records and commits
`D_exp` and `I_exp` under Section 17; their outcomes are reported separately
from main-scope completion.

Use small validated commits for coherent business progress with related docs,
tests, evidence, checks and limitations, followed by normal pushes. Do not make
consecutive documentation-only commits without business progress or commits
devoted only to hashes, evidence packets, bookkeeping or status. Preserve unrelated
work and original failure evidence.

The main-scope final implementation report includes:

1. final route and commit map by stage;
2. final repository and API trees;
3. complete retained/moved/split/renamed/new/deleted interface table;
4. fact-ownership and cache audit, including the complete chemical-type
   interface/role/permission/lifecycle map, dirty-state transitions, buffer
   ownership and before/after data-object counts with consumer evidence;
5. test matrix with exact results and skips;
6. native/wheel/C++ component evidence;
7. scientific execution tables, dynamic eligibility, every trial's monitor
   observations, differences and unresolved findings (not fixed count verdicts);
8. performance/resource comparison tables;
9. known limitations and deferred work; and
10. confirmation that superseded duplicate and compatibility implementations
    are absent, while preserving the standalone `rdWrappers`/`obWrappers`
    components and the selected sole MCA route specified in Section 3.2.

## 14. Final acceptance criteria

The main Plan20 delivery (Stage0–12) is complete only when all of the following
are true, subject to the existing Decisions 12/13 deferral rules. Its validated
Git submission supplies `B_exp` under Section 17.2. The subsequent Stage13A/B
experiment completes separately at `D_exp`/`I_exp` under Section 17; main-scope
acceptance is established before that experiment begins. MetalCluster's type
reservation and Plan21 framework are documentation deliverables; cluster
runtime behavior is not a Plan20 gate.

1. every durable fact in Section 5 has one named carrier owner;
2. structure modules contain no chemistry/physics/mathematics implementation
   and no external/native toolkit import at module load;
3. rule modules do not write carriers and return immutable results;
4. mutating scientific workflows use result -> validate -> atomic commit;
5. all forbidden caches in Section 5.6 are absent;
6. `_topology_index` remains the one private structural acceleration index and
   retains Plan 19 semantics;
7. `core.py`, duplicate implementations, wildcard exports, and cycle hacks are
   absent;
8. approved APIs, permissions, signatures, class identity and CLI technical
   contracts pass; scientific runs provide complete versioned reports, describe
   differences and explicitly retain unresolved findings without requiring
   historical scientific equality;
9. Python 3.10-3.14, native components, wheel/sdist, documentation examples,
   scientific-report execution, and GPU technical validation are complete as
   specified, with no unexecuted tier described as passed;
10. pre-refactor baselines and final comparable-work efficiency/resource reports
    satisfy Section12: preserve actual work, completion, failures, denominators,
    adverse observations and uncertainty; explain accepted usable costs and
    unresolved limitations without a universal parity/rare-exception threshold;
    claimed gains require measurements and affected batch reliability is verified;
11. implementation log and final report are complete and internally
    consistent with the final code; and
12. Decisions 12/13 are either implemented or recorded as explicit owner-
    approved final-tree exceptions with their unchanged paths; deferral may
    not leave duplicate chemistry implementations or two active asset owners;
13. every public chemical-type constructor, property, method, collection
    access, result and conversion satisfies the common role/ownership/
    permission/lifecycle protocol; shared private declarations, source-aware
    commits and lazy dependency handling are complete where applicable;
    Atom's sole fact table supplies efficient scalar/batch access and reviewed
    schema-linked projections as its concrete implementation; and
14. audited data-contract families document actual consolidation and approved
    retention, consistent naming/field order, one owner per contract and no
    unapproved public schema changes. Claimed reductions require real consolidation;
    moves/dict substitutions do not count. A9 retains D01–D03 with zero dataclass
    delta; other owning-stage consolidation and final evidence remain required; and
15. the integrated Plan12 functional requirements are implemented and validated
    through their assigned Plan20 owners/stages; ordinary results and every actual
    rule/gate/adapter/workflow finding-control boundary satisfy Sections6.2 and11.1.
    Unavailable required execution/evidence remains an open gate, without reduced
    work, denominators or acceptance criteria.

## 15. Risks and controls

| Risk | Control |
|---|---|
| circular imports while classes move | extract read protocols/rules first; use function-local facade imports; prohibit upward package imports |
| two class identities | one physical definition; identity tests; no alias class/shim |
| hidden behavior lost from setters | baseline each writer; replace side effects with explicit coupled operations before deleting them |
| stale result after mutation | finite dependency marking, source-aware on-demand recomputation, no stale-as-current read |
| performance loss after cache deletion | one snapshot/conversion per coarse operation, operation-local reuse, before/after benchmarks |
| native wheel breakage | move Python/build/C++/stub/package-data paths atomically and test clean wheels |
| scientific monitor mistaken for a verdict | complete per-case reports, dynamic cohorts and explained observations; no fixed182 or historical equality gate |
| deterministic search/cycle order drift | exact ordered fixtures and cross-version tests |
| large data accidentally treated as code | reference-vs-dataset ownership table, manifests, package-data inspection |
| accidental public API expansion via imports | explicit `__all__` and snapshot tests |
| long-lived half-migrated tree | small coherent commits, one implementation per behavior, stage gates, no merge until hard cut completes |
| historical pickle breakage | explicit owner decision; no silent compatibility mechanism |

## 16. Owner direction and public-interface review register

### 16.1 Settled planning direction from 2026-10-28

The owner directed one consistent public-interface protocol for all chemical
types. Facts, aliases, permissions, coupled operations, invalidation, on-demand
recalculation, buffer ownership and lifecycle follow common declarations and
review requirements. Roles determine which rules apply; shared semantics do
not force common storage, a base class or mutability. Atom's unified efficient
fact table and reviewed schema projections are one concrete implementation.
MetalCluster is now a named future chemical type, with the specified
`Molecule.metal_clusters -> list[MetalCluster]` relationship. Plan20 records
its type skeleton and Plan21 owns the registered-structure/superatom work;
concrete definitions and matching forms await the owner's later direction.

Other settled requirements are cross-plan MCA sequencing; the scope-based
data-object audit with real consolidation; private dependency handling that
marks expensive values for demand-driven recalculation; scientific monitoring
reports instead of historical pass-count verdicts; and pre-refactor efficiency
records with comparable work, explained usable cost and resource bounds under
the effective goal. The 2026-11-03 direction supersedes the older universal parity
and rare-small-regression allowance, while preserving all raw/adverse evidence.

These directions authorize the documentation revision and record the
owner-specified future MetalCluster name/relationship. No production cluster
interface is established. Its constructor, members, registration/matching
methods and export path remain open, as do other proposed public additions,
removals, permission/return changes and import paths. The numbered decisions
retain references used elsewhere; recommendations are not implementation
approvals. Stage1 can collect baselines while public decisions are pending.

### 16.2 Numbered decision ledger

| Decision | Revised recommendation / scope | Status |
|---:|---|---|
| 1 | `hotpot.cheminfo.structure` as carrier package | proposed package name |
| 2 | remove direct `cheminfo.core` path while preserving ordinary top-level classes; provide migration documentation, no alias implementation | breaking import change pending review |
| 3 | move classified scientific packages, retaining the stereo facade and standalone wrapper paths | exact public import migration pending review |
| 4 | common protocol across all public chemical types: one fact per owner, mapped interfaces/permissions, source-aware coupling, lifecycle and invalidation; Atom coordinates are one concrete case | owner-directed common protocol; each public change pending review |
| 5 | replace broad arbitrary writers with field-specific commits; any batch API accepts only approved schema fields | public removals and proposed batch signature pending review |
| 6 | complete ordered Bond collection with private operation-local exclusions; retire the eight documented hide/recover/clear-hidden methods after connected consumer migration, with no replacement public topology API | root accepted exact HT1–HT5 in [Stage7B design](stages/stage07/topology-selection/contracts/hidden_topology_design.md), including intentional placement of the complete original Bond order before H; separate release and connected candidate acceptance govern delivery |
| 7 | user annotations separated from selected typed scientific results, no triple MCA storage | owner-directed consolidation; observable properties behavior pending review |
| 8 | role-specific value/view/collection/result semantics; historical outputs stable, current attachments checked; lazy recomputation for declared pure groups, no implicit GPU/external work | detailed read/attachment/lifecycle changes pending review |
| 9 | sole six-parameter Crystal, angstrom/degrees, characterize matrix convention, explicit export and placeholder removal | public additions/removals pending review |
| 10 | old core-path pickle compatibility requires an explicit migration decision; no silent compatibility module | pending historical-asset audit and owner decision |
| 11 | **replace old keep-array recommendation** with benchmarked private `_AtomFactTable` typed columns/blocks, one schema and Atom projections; no unrelated Bond rewrite | new storage requirement owner-directed; public access signatures pending |
| 12 | consolidate main/works command tree preserving commands/arguments; may defer independently | tail-stage scope pending |
| 13 | consolidate chemistry asset ownership with manifests; may defer independently | tail-stage scope pending |
| 14 | documented advanced pure-rule imports may be public; mutating controllers stay private | any advanced public exports pending review |
| 15 | amend Plan19 budget for sole Atom table, identity infrastructure, finite dirty metadata, typed facts/results; no second topology index/Graph/general cache | new private requirements owner-directed; full budget recorded for review |
| 16 | preserve characterized constructor/detachment semantics; foreign-Atom insertion copies/remaps rather than steals | behavioral changes pending review |
| 17 | immutable identity-bound views and one ConstraintSet; proposed `StaleStructureViewError`, `AtomPairs`→`AtomPairTable`, removed view write APIs | every new/renamed/removed public symbol pending review |
| 18 | conversion calls return fresh external objects; task-local reuse | observable identity change pending review |
| 19 | copy/components/merge remap facts, constraints, source frames/stereo/cell; drop generated results and runtime contexts | exact lifecycle behavior below remains a review proposal |
| 20 | remove incomplete statistics and use explicit assembly exports | exact exports/removals below pending review |
| 21 | lazy backend import; remove sys.path/tempdir/package_root side effects, resource consumers use importlib.resources | root-reviewed exact route and CP311 source/installed evidence accepted in [Stage12A](stages/stage12/lazy-imports/validation/checkpoint.json); remaining final/version obligations stay open |
| 22 | selected PartialChargeSet is the sole value owner; proposed Optional[float] read-only property and full-set assignment | public setter/return-type change pending review |
| 23 | **replace old blocking success/failure-pair rule** with O0 seeds42/43/44 scientific reports, dynamic eligibility, all monitors/outcomes and explained changes; no fixed182 or historical equality verdict | reporting direction owner-directed; technical invariants remain independently tested |
| 24 | MetalCluster type, Molecule association and future `Molecule.metal_clusters -> list[MetalCluster]`; internal graph/point cloud, external multi-attachment superatom, registered kinds | owner-specified planning target; Plan20 design registration only, Plan21 implementation/details deferred |

Decision19 detail: proposed copy preserves factual state, constraints, Crystal,
source stereo/provenance/current and explicit P1 frames; annotation mapping
container is copied with existing value references; P3 generated results and
runtime state are dropped. Components retain attributable facts and cell,
drop whole-molecule annotations/results, and carry charge/spin only where
component-resolved facts exist. Add-component does not share mutable state.
Empty destination adopts source charge/spin/cell; a populated destination
accepts compatible/equal cell or no source cell, otherwise rejects atomically.
Two declared charges add; an undeclared input leaves combined charge
undeclared. Combined spin requires a separately supplied valid selection.
These are lifecycle proposals, not automatic outcomes of a filesystem move.

Decision20 detail: proposed top-level assembly names are Fragment,
EdgeShoulder, AtomLink, AlkylGraft, BondAdding, AtomReplace, RingWedge and
AssembleFactory. Existing documented advanced functions alkyl_generator,
atom_link_atom_action, shoulder_bond_action, bond_order_add, atom_replace,
ring_wedge remain at the selected assembly path. ComplexStatistics deletion,
other helper visibility and those exact exports require review.

### 16.3 Cross-type interface review and concrete candidates

The [common protocol's Section 8](chemical_type_interface_contract.md#8-公开接口审阅清单)
organizes the review scope by role/type. Each proposed change supplies its
exact type, name/signature, permissions, input/output shape and units, fact
owner/dependencies, lifecycle, old-entry disposition and consumer/ABI impact.
Review covers Molecule/Atom, Bond, Crystal, collections, views, results,
transport/wrapper values and the future MetalCluster. It includes existing
setters or return semantics that would change. MetalCluster's name and future
Molecule property are specified by the owner; constructor, other members,
registration methods and public paths await later review. The domain-specific
cluster registration policy does not create a public generic rule/schema
registry. Shared declaration machinery approves no additional interfaces.

The companion's A1–A13 are the concrete Atom-storage entries within this
review process. During Stage3 the owner approved these exact public names:

```python
Molecule.updata_atoms(*, atoms=None, **columns) -> None
Molecule.atom_matrix  # read-only property: independent float64 array, shape (N, 19)
```

Keep the owner's exact `updata_atoms` spelling. `atom_matrix` takes no fields
or dtype argument; it reuses the existing complete19-column projection and
never stores a second matrix. Optional result absence follows A2's NaN policy
at Stage7C; it does not start a model or add MCA as a twentieth column.
The owner's later rename/no-duplication instruction supersedes A2's earlier
matrix spelling: directly rename `atom_attr_matrix`, migrate its consumers and
retain no old-name alias. Implementation remains subject to the Stage3 gates.
No public fact-table class, schema registry,
dirty-state method, `new_attrs` example property, universal base or parallel
batch API is added. Stage1 A2/A6/A8 and the accepted Stage7C contracts now settle partial-charge
absence/returns, read-only assignment boundaries and stale reads. Other existing
attrs/constructor arguments, coordinate aliases and inferred-property setters
retain their specific review boundaries. For a required multi-input operation, first reuse an
existing complete method; otherwise list its exact signature for approval.
Neither Atom candidates nor invertibility authorize corresponding setters or
batch methods on other chemical types.

The data-object audit's result/tag/naming changes follow the same process and
ordinary-result boundary in Section6.2. On 2026-11-03 at
`7952d1f14994e88892dd094cd775e92970c6c0e1`, root explicitly accepted
[A9](stages/stage08/scientific-state/contracts/decisions.md#a9--accepted-public-calculation-reports-and-confidence-only-removal)
under the existing full-authority mandate and effective goal, before product
authoring. Retain the five typed calculation records with six required fields
`identity, facts_identity, candidates, admissibility, resolution, findings`;
remove only generic confidence production/fields and ChemicalConfidence/two
exports. The obsolete StateInferenceResult/common-rating proposal is rejected.

The exact existing `resolve_scientific_state`, `project_state` and
`resolve_and_project_state` signatures/owners remain public requested detailed
calculation/decision/projection reports. ScientificStateBundle retains all current
fields, including ordered provider outputs for explicit inspection, candidate
space, resolution, decision, trace and source binding. BackendProjectionResult
and ScientificStateWorkflowResult retain their exact projection/control and
state-plus-projection records. Ordinary Molecule/Atom charge/spin/structure values
receive no new envelope or rating. D02 retains all four finding/reason types and
exact codecs; D03 retains both input types, namespaces, configuration/digest
semantics and dependency direction, each as NO_CONSOLIDATION.

A9 accepts the seven-field/retired-enum Python-object pickle hard cut without
compatibility classes, aliases, migration loaders or dual runtime. Known versioned
JSON schemas/digests remain unchanged, so no wire-version bump is authorized.
Exact constructor/export effects are in A9; its [current CP311 checkpoint](stages/stage08/scientific-state/validation/confidence_checkpoint.json)
accepts implementation, source/installed validation and current/legacy-cut pickle evidence. Historical 5→1/4→1/2→1 estimates
are not current commitments: dataclass delta is zero, with measured enum-class
delta −1 recorded separately. Other public changes retain their own review scope.
Standalone wrappers keep their reviewed component identities; inheritance or
aliases cannot conceal a second implementation.

### 16.4 Ready-to-execute decision record

Before a production step, record the relevant approved decision, exact public
signature/permission changes (or explicitly none), owning stage, baseline
commit, technical verification targets, scientific-report targets and measured
performance baseline. Unresolved unrelated tail-stage choices do not block
private audit/baseline work. Any new public proposal pauses only its dependent
step until reviewed; do not silently publish it or ask for approval of an
unspecified interface.

## 17. 最后阶段：kekulize/aromatic 实验性规则接口协议

状态：依据本轮负责人指示新增的实施尾阶段；当前仅修订规划，尚未定义生产协议或
接入算法。本节记为 **Stage13**，严格排在既有 Stage0–12 完成、验证并提交之后。

### 17.1 范围与既有规划的关系

本实验创建一个最小的私有规则接口协议，且仅将 **kekulize/aromatic** 接入该协议。
公开的 Molecule 方法继续作为普通使用入口，协议连接领域 operation 与具体规则实现。

第1–16节及其现有配套文档规定的实施范围、事实归属、接口审阅、存储、依赖、
测试与基准要求全部维持。前文的化学类型事实/权限/生命周期共同协议，与本节新增的
实验性规则接入协议是不同范围；本节不把新增实验协议推广到全部化学类型或规则。

charge、spin、oxidation、clusters、stereo、搜索、力场及其他领域继续按既有规划
实施，无需为本实验增加注册点、通用基类、结果外壳、状态枚举或扩展字段。前置阶段
也无需为尚未实施的实验预留运行时结构。既有芳香/Kekulé 提取与迁移仍按原阶段完成，
不得推迟到实验中，或在实验中重复实施。

### 17.2 启动前置：完成、验证、报告、Git 提交

在定义或实现本实验协议之前，必须依次完成以下事项：

1. 完成 Stage0–12 的既定交付，并满足第14节原有验收条件。Decision12/13 等
   条件范围只采用前文已经规定的决策与延期规则；本节不新增延期或豁免渠道。
2. 执行第11节规定的技术验证与科学报告，以及第12节规定的基线/最终效率比较；
   保留实际命令、输入/环境身份、测试结果、科学观测和原始基准记录。未执行的必要
   验证不得记作通过，尚未解决的阻断项不得借实验绕过。
3. 完成实施日志与主范围最终报告，将主范围代码、测试及可提交的证据记录提交 Git。
   在日志中记录这一已验证提交的完整 SHA，作为实验基线 `B_exp`。外置大型产物
   沿用原有管理方式，在报告中记录其路径和摘要，不为实验改变数据治理规则。
4. 确认主范围交付已提交、实验工作区无混入的前置实施修改，再从 `B_exp` 创建新的
   实验分支，计划名为 `experiment/plan20-kekulize-aromatic-protocol`。

`B_exp` 必须是将来实际完成验证的提交，不能使用当前规划提交或编造的 SHA。
当前文档分支不代表实验已经启动。主范围交付与实验结果分别报告，实验不得反向
改变前置验收结论或使主范围提前依赖一个尚不存在的协议。

### 17.3 Stage13A — 定义最小实验协议，独立验证并提交

第一步在实验分支上完成协议定义及独立契约验证，**不改接生产调用链**。先读取
`B_exp` 中已经完成迁移的芳香/Kekulé operation、规则、快照和赋值结果，以这些
真实消费者为依据确定确切私有签名；不得按迁移前的 `core.py` 路径重建旧接口。

协议仅声明下列连接内容，优先复用已有领域定义：

| 连接处 | 最小约定 |
|---|---|
| 实现绑定 | 芳香/Kekulé 领域内的显式私有函数绑定；确有调度需要时才使用小型调用表，记录实现身份，默认仍为 `B_exp` 的唯一现有算法 |
| 输入 | 当前操作所需的只读事实、行映射、拓扑选择及已有参数；复用现有快照与字段选择，不新增全量 Molecule 镜像 |
| 输出 | 已有完整赋值提案、结果及诊断的明确形态、单位和覆盖范围；结果不直接写载体，不新增跨领域万能 DTO |
| operation 响应 | 根据领域结果及当前工作流策略决定提交、保留部分进展、恢复或报告；结构完整性与科学不确定性分别处理 |
| 实现迭代 | 保持连接的输入/输出语义，允许领域支持的数值、候选和覆盖范围演进；超出现有消费者支持范围时明确报告原因 |

不确定、多候选或经验性选择不自动构成错误，不新增“必须证明唯一科学解”才能
建模的条件。已有执行假设、默认值和恢复策略沿用 `B_exp`，依据与诊断由现有领域
结果承载；本实验不借接口接入调整科学算法。程序错误、错误行映射和不完整事务
仍按既有技术契约处理，不以占位值或吞异常伪装成可用结果。

协议保持私有，不新增公开 RuleRegistry、Operator、通用 apply_rule、运行时插件
发现或所有对象必须实现的基类。新定义按既有数据对象账本计数，复用现有结果和
函数优先于新增包装类。公开签名、返回方式、权限及用户配置保持 `B_exp` 的已审版本。

交付：确切私有协议定义、输入/输出/响应说明、隔离的契约测试与记录。验证成功后
单独 Git 提交，记录定义提交 `D_exp`。没有这项提交及验证，不进入 Stage13B；
定义协议的提交不得夹带生产接入、其他领域改造或新的科学实现。

### 17.4 Stage13B — 仅接入 kekulize/aromatic，验证并提交

第二步以已提交的 `D_exp` 为起点，在同一实验分支上将现有唯一芳香/Kekulé 算法
接入协议。调用关系为：

```text
现有 Molecule 公开方法
    -> 现有芳香/Kekulé operation
    -> 实验协议的领域绑定
    -> 现有只读规则与赋值结果
    -> operation 按既有事务边界提交
```

保持 `B_exp` 的公开方法、参数、返回语义、默认算法和建模策略；规则仍只读，
operation 仍负责完整写入与恢复。接入替换一个调用连接，不复制算法，不保留供用户
选择的旧/新生产路线，不恢复前置阶段已删除的隐藏键状态、视图写入口或持久缓存。
其他领域即使调用芳香性功能，也只消费原有操作接口，不接入实验协议或获得新义务。

验证至少覆盖普通芳香环、稠合环、含配位键体系、领域已有的未支持/未决情况，
以及执行失败路径。检查公开调用、原子/键映射、完整赋值、拓扑选择与恢复、失败
后的已提交状态和诊断传递。复用实际规则检验接入，不以只返回固定值的替身代替
全部集成证据；协议绑定自身可用隔离测试检查。

按第11节执行受影响的契约/集成检查及默认测试集；受影响的构建或安装包验证仍
按原有规则执行。科学运行保留逐例观测和差异解释，不新增固定成功数量或科学唯一性
断言。同输入、同算法、同环境下与 `B_exp` 配对比较芳香/Kekulé 及其实际调用流程的
时间、分配与内存，沿用第12节的可比性及效率标准，不用减少工作量制造收益。

验证与比较完成后，单独 Git 提交接入变更及报告，记录接入提交 `I_exp`。结果说明
协议增加的实际维护/运行成本、对建模连续性的影响与可保留价值；发现问题时保留
真实结论和独立实验分支，不改写 `B_exp` 的已验证主范围。

### 17.5 两步推进与结束记录

| 里程碑 | 必须具备的证据 | 允许推进的范围 |
|---|---|---|
| `B_exp` | 主范围完成、原有测试/科学报告/基准验证、最终报告和 Git 提交 | 创建新的实验分支 |
| `D_exp` | 协议定义、独立契约验证、单独 Git 提交；生产调用未改接 | 开始 kekulize/aromatic 接入 |
| `I_exp` | 接入验证、与 `B_exp` 的基准比较、实验报告、单独 Git 提交 | 评估本实验的保留或合并 |

任何实验结果均不自动授权向其他领域推广接口协议。进一步推广作为独立后续工作
重新提出；本轮其他实施内容完整沿用当前文档。
