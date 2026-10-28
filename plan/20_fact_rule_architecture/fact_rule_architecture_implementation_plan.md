# Fact-Carrier and Rule Architecture Refactor

Status: **draft architecture audit and implementation plan for owner review**.
No production implementation is authorized by this document until the owner
approves the plan and resolves the decisions in Section 16.

Planning branch: `refactor/fact-rule-architecture`

Planning worktree: `/home/zhangzhiyuan/hotpot` (no additional worktree)

Planning baseline: `0a4b0ee`

Audit date: 2026-10-27

## 1. Objective

Reorganize Hotpot's chemistry kernel around one explicit architectural rule:
domain objects own durable facts, while chemical, physical, and mathematical
rules only calculate from those facts.

`Molecule`, `Atom`, `Bond`, `Crystal`, and their closely related carriers
remain the ordinary user interface. A user should continue to write code such
as `mol.optimize(...)`, `mol.assign_bond_order()`, or `bond.length`; ordinary
use must not require a rule registry, rule handle, operator object, backend
profile, or dependency-injection setup. Those object methods become thin,
readable entry points into named rule or operation functions.

The refactor has eight simultaneous goals:

1. make each durable molecular fact have exactly one authoritative owner;
2. separate fact storage and user-facing interfaces from chemistry, physics,
   mathematics, toolkit integration, and workflow execution;
3. make rule evaluation read-only and return immutable findings, proposals,
   or result objects;
4. permit mutation only through a small private operation/commit boundary that
   validates and applies one complete change atomically;
5. remove persistent derived chemistry, toolkit molecules, and algorithm
   workspaces from the carriers;
6. preserve the useful top-level Hotpot API and scientific behavior while
   deleting obsolete internal paths and duplicate implementations;
7. classify the full chemistry package by responsibility rather than merely
   splitting the current `core.py`; and
8. retain the performance benefits of the native graph, native geometry,
   `rdWrapper`, `obWrapper`, and field-selective molecule buffers without
   allowing any of them to become a second source of truth.

This is an architectural refactor, not a new aromaticity model, force field,
stereochemistry convention, graph-search algorithm, or MCA model. Scientific
algorithm changes require separate owner approval and separate evidence.

## 2. Scope and non-goals

### 2.1 In scope

- `hotpot/cheminfo` fact carriers, views, rules, operations, adapters,
  reference data, native bindings, public exports, and documentation;
- every production, plugin, pipeline, dataset-provider, CLI, and test import
  affected by the chemistry-kernel move;
- conditionally, `hotpot/main` and `hotpot/works`, whose overlapping
  command/workflow roles currently obscure the outer architecture (Decision
  12); this cleanup is not interleaved with carrier extraction;
- setup, wheel, sdist, CMake, C/C++ include, package-data, and typing-stub paths;
- conditionally, relocation of misowned chemistry assets (Decision 13), as a
  separate coherent data-governance commit; and
- removal of dead or superseded chemistry implementations discovered by the
  zero-reference/build audit; and
- behavior, packaging, scientific-boundary, and performance validation.

### 2.2 Out of scope

- changing the contents or governance model of `hotpot/dataset` and
  `hotpot/ds`, except for moving chemistry-package assets that are currently in
  the wrong ownership layer;
- redesigning the generic `hotpot.pipeline` engine;
- restructuring independent plugin internals that do not violate the new
  carrier/rule boundary;
- changing trained model weights or scientific acceptance thresholds;
- adding compatibility profiles or selectable old/new implementations; and
- changing the underlying Atom/Bond storage layout solely for stylistic
  reasons. Section 16 asks whether the owner wants a separate storage-layout
  change; the recommended answer is no for this stage.

## 3. Architectural vocabulary and fixed boundaries

### 3.1 Vocabulary

The following terms are normative for this plan.

| Term | Meaning | Lifetime and ownership |
|---|---|---|
| fact | a value deliberately committed as part of a molecular structure, representation, selected state, or durable final result | owned by exactly one carrier field or typed result record |
| carrier | a domain object that stores facts and exposes the primary user API | durable; authoritative |
| view | a structurally immutable projection such as a ring, path, angle, torsion, or atom pair | caller-owned result; never authoritative |
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

After explicit owner approval, Plan 20 proposes two narrow superseding
amendments to Plan 19 rather than treating them as ordinary internal moves:

1. replace persistent active/hidden Bond collections and public hide/recover
   workflow state with one complete authoritative Bond collection plus
   operation-local topology masks; and
2. reopen Plan 19's closed Core state/change budget solely to split Core and
   add the carrier facts in Section 3.3. No second topology index, native
   Graph, or general cache is authorized.

Until the owner approves Decisions 6 and 15, Plan 19's existing hidden-bond
and Core state-budget constraints continue to govern those areas.

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
| monotonic Atom/Bond UID high-water marks | P0 identity-allocation infrastructure; never decremented or reused within a Molecule lifetime |
| Plan 19 `_topology_index` | sole reconstructible structural acceleration index |
| molecular net charge/spin declarations | explicit P1 facts where present |
| current Cartesian coordinate frame | P1 current geometry |
| optional `Crystal` value | P1 six-parameter unit-cell fact; no Molecule back-reference |
| source coordinate-frame collection | explicitly imported/user-supplied P1 frames with source provenance; distinct from generated results |
| stereo records plus source-representation ledger | canonical stereo facts and lossless source provenance; never a duplicate current Atom/Bond table |
| `ConstraintSet` | coordinate constraints keyed by Atom UID, bond constraints by Bond UID, and angle/torsion constraints by ordered Atom-UID tuples |
| user annotation mapping | user-owned metadata only |
| typed result collection | explicitly attached P3 results, including generated/optimized conformer ensembles, with declared dependency identities |

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
| `forcefields/` | contracts, validation, working-copy isolation, trajectory, backend, and workflow responsibilities are substantially separated | split physics policy from execution/adapters; preserve atomic commit and scientific gates |
| `conformers/` | backend-neutral request/result contracts and a provider seam exist | move under operations; keep provider results immutable |
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
| MCA storage | `calculator.mca()` writes an untyped array to `mol.properties["mca"]`, while `Atom.mca` and `Molecule.mca_sites` project from it | a final scientific result has no explicit identity, provenance, or invalidation contract |
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
| P1 | explicit structural/state facts | atomic number, isotope, radical state, bond direction/source representation, current Cartesian coordinates, unit cell, explicit/source stereo records, typed constraints, explicitly declared system charge/spin, explicitly supplied alternative coordinate frames | canonical carrier facts; a P1 change invalidates dependent P2/P3 values |
| P2 | committed chemical representation | formal charges, coupled Bond kind/order, coupled Atom/Bond aromatic assignment, explicit/implicit-H representation | stored only after explicit import or an atomic rule proposal is accepted |
| P3 | durable selected scientific results | generated/optimized `ConformerEnsembleResult`, partial-charge set, oxidation/electronic-state selection, energy/force records, thermo result, MCA result | typed record with subject identity, method/version, units where applicable, and provenance; selecting a result may atomically update a P1 current-state fact but never silently overwrites P0-P2 |
| P4 | derived findings and views | electron configuration, valence, hybridization, neighbours, components, rings, paths, angles, torsions, pair distances, formula, spectra, geometry findings, search hits, validation reports | returned to the caller; not retained on a carrier as hidden state |
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

This ordering is not applied globally by a magic generic resolver. Each named
operation states which sources it accepts and produces a typed conflict when
same-authority values disagree. User mutation can replace a field through its
documented setter/operation, but it cannot violate P0 invariants or cause an
unrelated lower-priority value to repair the new input.

### 5.3 Provisional ownership of coupled facts

The following contract is recommended and requires owner confirmation in
Section 16.

| Coupled values | Sole owners and relationship |
|---|---|
| Atom/Bond membership and endpoints | `Molecule` owns ordered membership; each `Bond` owns its endpoints; Plan 19 `_topology_index` is only a projection |
| bond kind and order | `Bond.kind` and `Bond.order` are one coupled P2 representation owned by Bond. Normal single/double/triple/aromatic kinds must agree with the fixed mapping; zero/dative/unknown follow their explicitly tested mapping. Direction/source transport is a separate P1 fact |
| aromaticity | Atom and Bond local aromatic flags are committed together by one `AromaticAssignment`; Ring aromaticity is always a derived view and never writes back by itself |
| molecular and atom charge | molecular net charge and atom formal charges are distinct facts; they must sum only when formal-charge coverage is declared complete. Partial charges are a named P3 result set and never replace formal charge |
| coordinates and conformers | a `CoordinateFrame` is a value: the selected current Cartesian frame and explicitly supplied source frames are P1; generated/optimized `ConformerEnsembleResult` is P3. `conformer_load(i)` atomically copies one projected frame into current coordinates; adding a source frame does not change current coordinates |
| stereo/source transport | Molecule stores stereo records and an immutable source-representation ledger. Current isotope/radical/H/direction values are owned by Atom/Bond and are assembled into `StereoPayload` on request; they are never stored a second time in the ledger |
| stereo and coordinates | Level-1 stereo records preserve explicit/source representation; coordinates may validate or propose Level-2 assignments but never silently overwrite explicit stereo on disagreement |
| crystal representation | public `Crystal(a, b, c, alpha, beta, gamma)` is the sole persistent cell value and preserves the current API/native ABI; “unit cell” is the concept, not a second carrier type; a lattice matrix is a calculated mathematical view |
| constraints | Molecule owns one typed `ConstraintSet`: coordinates use Atom UID, bonds use Bond UID, and angles/torsions use ordered Atom-UID tuples; Atom/Bond/Angle/Torsion `.constraint` access, if retained, is only a projection into that set |
| MCA and other predictions | a typed result record owns values and identity; `Atom.mca` and `Molecule.mca_sites` remain thin projections over the current valid record |

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
`set(get(mol))` and supported toolkit round trips must remain byte/field
equivalent under the existing schema tests.

Partial charge is likewise never duplicated in Atom storage. A selected
`PartialChargeSet` records presence, values, method/version, units, provenance,
and subject identity. `Atom.partial_charge` projects its row or reports the
existing no-result state. A native transport request for `PARTIAL_CHARGES`
packs the selected valid set only; absence is not converted into authoritative
zero values, and no native return path writes Atom arrays directly.

### 5.4 Identity and result validity without revisions

Result validity is mechanical even though no mutation revision counter is
added. Structure adapters calculate composable deterministic identities from
current facts when attaching or reading a result:

| Identity | Included facts | Typical dependants |
|---|---|---|
| `TopologyIdentity` | ordered internal Atom/Bond UIDs and complete Bond endpoints only | paths, rings, component partition, topology-dependent rules |
| `RepresentationIdentity` | topology identity; atomic number/isotope/radical/formal charge; system charge/spin; Bond kind/order/direction/source representation; aromatic/H state; stereo records | search features, MCA chemical features, state results |
| `GeometryIdentity` | topology identity plus ordered current Cartesian coordinates | distances, geometry gates, geometry-dependent model input |
| `CellIdentity` | canonical six unit-cell parameters | periodic/crystal results |
| `ConstraintIdentity` | topology identity plus coordinate/bond/angle/torsion constraint entries and values | constrained optimization and geometry operations |
| `SourceFramesIdentity` | topology identity plus ordered explicit P1 source frames and their source provenance | workflows that explicitly consume a source-frame collection |

An energy result, for example, may declare representation, geometry,
constraint, and cell dependencies; a ring view declares topology only. Every
typed P3 record stores the dependency identities captured when it was
produced. A commit proactively removes records affected by its named
dependency class; a named result accessor additionally recomputes and compares
the declared identities before returning it. The Molecule does not cache a
second copy of its *current* identities: current digests are calculated at
task/access boundaries and may be reused only in an operation-local context.

Views carry the required source identity and stable internal member identities.
Their ordered membership remains inspectable, but a carrier-dependent
calculation or mutation after the corresponding source identity changes raises
`StaleStructureViewError`. Detached Atom/Bond inspection retains the approved
Plan 19 behavior and is covered by its existing lifecycle tests.

### 5.5 Mutation and invalidation

All multi-field changes follow one sequence:

```text
carrier facts -> immutable snapshot -> pure rule/backend result
              -> validate complete patch -> atomic commit or no commit
```

Rules and adapters never write `_atoms`, `_bonds`, Atom/Bond storage arrays,
coordinates, stereo payload, conformers, or result records directly. A private
operation may call carrier-private primitive mutation methods, but only after
the complete patch passes structural and scientific validation.

Invalidation is dependency-driven:

| Changed fact | Mandatory invalidation/action |
|---|---|
| atom membership/UID | rebuild topology index; atomically remap or reject current coordinates, source frames, stereo/source ledger, and constraints; discard dependent P2 assignments and every topology-bound result |
| Bond membership/endpoints | rebuild topology index; remap or reject Bond stereo/source records and Bond/angle/torsion constraints; discard dependent P2 assignments and every topology-bound result |
| atomic number/isotope/radical | keep topology index; atomically reject or replace affected P2 representation and invalidate representation-dependent results |
| Bond kind/order | keep topology index; commit the coupled P2 pair once and invalidate representation-dependent results; kind and order never invalidate one another mid-commit |
| Bond direction/source representation | keep topology index; update the P1 transport fact and invalidate representation-dependent results |
| formal charge, system charge/spin, implicit H, aromaticity, or stereo | keep topology index; invalidate representation-dependent results only |
| current coordinates | keep topology and source stereo; invalidate coordinate-dependent results only |
| explicit source-frame collection | keep current coordinates/topology; invalidate only `SourceFramesIdentity` dependants |
| attach/replace a generated conformer ensemble | no fact invalidation unless the documented operation also loads a frame; replace only the named P3 record |
| typed constraints | keep topology/geometry; invalidate `ConstraintIdentity` dependants |
| unit cell | invalidate crystal-coordinate/periodic results only |

No general persistent revision counter is introduced merely to support caches.
This preserves Plan 19's closed topology contract. If a later measured cache
requires revisions, it needs a separate amendment rather than entering this
refactor as convenience state.

### 5.6 Allowed and forbidden caching

Allowed:

- the existing Plan 19 `_topology_index`, because it contains only Atom row and
  incident Bond references, is rebuilt synchronously from P0 owners, and has no
  independent chemical values;
- operation-local ring/path/geometry/native workspaces;
- iterator-owned snapshots for the iterator lifetime;
- immutable reference tables and compiled immutable query IR keyed only by
  immutable inputs; and
- backend-internal acceleration during one coarse call.

Forbidden:

- `_obmol`, `_row2idx`, RDKit molecule, native Graph, or other mutable toolkit
  object retained on `Molecule`;
- carrier-owned `_rings`, `_ring_indices_cache`, `_ligand_rings`, `_angles`,
  `_torsions`, or equivalent derived chemistry caches;
- molecule-keyed global or weak-reference caches such as the current SMARTS
  atom-state cache;
- hidden writes to an arbitrary `properties` dictionary; and
- a cache whose presence, stale state, or construction order can change a
  scientific result or enumeration order.

`AtomPairs` requires migration rather than blind deletion because current data
providers consume it. It becomes an immutable, task-produced pair view/table;
consumers receive or construct that view explicitly instead of mutating a
carrier-owned dictionary.

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

1. `structure` owns facts, snapshot/result contracts, primitive
   invariant-preserving mutations, thin user methods, and no scientific
   implementation. Every convenience method enters a named operation, even
   for a read-only query; structure never constructs a snapshot by importing
   an adapter itself.
2. `rules` depends on immutable structure snapshot protocols and reference data. It
   must not depend on concrete carrier internals or write a carrier.
3. `operations` may depend on structure contracts, rules, and adapters. It
   owns read-only query composition as well as mutation sequencing, working
   copies, validation, and atomic commit.
4. `adapters` may read carriers or snapshots and call external/native systems.
   It returns a transport result or named assignment and never mutates an
   authoritative source carrier.
5. `reference` contains immutable scientific tables and schemas, not active
   calculation logic or instantiated Molecules.
6. package `__init__.py` files only re-export downward; internal modules never
   import upward through a package facade.

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
adapter, rule, or workflow package.

### 6.2 Read-only rule, selection, builder, and commit signatures

Representative target signatures are fixed conceptually as follows. Exact
type names may be refined during implementation, but rules must remain pure
and operations must remain the only commit path.

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
documented existing result. It does not contain a second chemistry
implementation.

Temporary topology filtering has one private, non-owning contract:

```python
@dataclass(frozen=True)
class TopologySelection:
    source_identity: TopologyIdentity
    atom_uids: tuple[int, ...]
    bond_mask: tuple[bool, ...]  # aligned to complete Molecule Bond order

def select_topology(
    mol: Molecule,
    *,
    scope: TopologyScope,
    excluded_bond_uids: Collection[int] = (),
) -> TopologySelection: ...
```

`FULL`, ligand-skeleton, covalent-only, metal-excluded, and temporary ring-open
work all compose this selection or its packed snapshot. They never move a Bond
between carrier collections. Open Babel bond perception, hydrogen rules,
Kekule assignment, and force fields receive the explicit selection they need.

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
def _commit_representation(mol: Molecule, assignment: RepresentationAssignment) -> None: ...
def _commit_coordinates(mol: Molecule, update: CoordinateUpdate) -> None: ...
def _commit_stereo(mol: Molecule, update: StereoUpdate) -> None: ...
def _commit_constraints(mol: Molecule, update: ConstraintUpdate) -> None: ...
def _attach_result(mol: Molecule, result: ScientificResultRecord) -> None: ...
```

`RepresentationAssignment` is a tagged union of named immutable assignments
such as `BondOrderAssignment`, `AromaticAssignment`,
`HydrogenRepresentationAssignment`, and `FormalChargeAssignment`; it is not a
free-form field map.

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
- rules have zero writes to carrier-private storage;
- adapters have zero direct carrier-private writes;
- no production module imports the deleted `cheminfo.core` path;
- no package uses wildcard exports;
- public class definitions have one physical definition and one identity;
- native graph/search code includes the one shared CSR Graph definition and
  declares no second graph storage type;
- Hotpot force-field adapters invoke the standalone `obWrapper` backend and
  contain no direct duplicate Open Babel force-field engine;
- mutable external toolkit molecules are never stored on a carrier;
- task snapshots contain only requested fields; and
- a deleted or moved implementation has zero active source, setup, stub,
  documentation, and test references before its old file is removed.

## 7. User-facing API contract

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

Public signatures and observable semantics are preserved by default. Their
implementations are classified below.

```text
Molecule
├── facts and membership [carrier]
│   ├── atoms / bonds / coordinates / charge
│   ├── crystal / create_crystal
│   ├── conformers / conformer_add/get/load/clear (source/result projection)
│   ├── typed constraints projected through clear/constraint helpers
│   ├── properties (restricted to user annotations; see Section 16)
│   └── stereo payload access through named stereo functions
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
├── owned facts [carrier]
│   ├── atomic_number / isotope / id
│   ├── formal_charge
│   ├── coordinates (projection of the Molecule current frame)
│   ├── committed aromatic / implicit-H representation
│   └── read-only parent Molecule reference (membership is Molecule-owned)
├── topology views [read-only query operation -> mathematics rule]
│   ├── idx / bonds / neighbours / neigh_idx
│   ├── hydrogens / heavy_neighbours / connectivity
│   └── rings / in_ring
├── chemistry views [read-only query operation -> chemistry rule]
│   ├── symbol / period / group / elemental-class predicates
│   ├── electron configuration / oxidation state / hybridization
│   ├── get_formal_charge / get_valence
│   └── bond-order and missing-electron summaries
├── topology mutation [thin operation]
│   ├── add_atom / add_hydrogen / link_with
│   └── symbol / coordinates / constraint setters (constraint commits to Molecule)
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
├── matrix [calculated mathematics view]
├── from_matrix(matrix) [constructor preserving approved convention]
├── no Molecule back-reference
└── no Open Babel object, space-group placeholder, or toolkit cache
```

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
| public hide/recover/clear bond family | migrate internal use to operation-local `TopologyView`/bond masks; remove or explicitly retain only by owner decision | hidden bonds are workflow state, not chemical truth |
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
  19;
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
external-system boundary.

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
│   │   ├── bond.py                     # Bond, BondKind
│   │   ├── crystal.py                  # sole Crystal cell carrier
│   │   ├── conformer.py                # CoordinateFrame, source-frame collection
│   │   ├── stereo.py                   # records, source ledger, payload contract
│   │   ├── constraints.py              # typed ConstraintSet
│   │   ├── snapshots.py                # single immutable snapshot contracts
│   │   ├── results.py                  # typed explicitly attached P3 records
│   │   ├── bundle.py                   # MolBundle carrier collection
│   │   ├── protocols.py                # read protocols used by rules
│   │   ├── errors.py
│   │   ├── _state.py                   # named canonical field schemas
│   │   ├── _topology.py                # Plan 19 topology index
│   │   ├── _mutation.py                # invariant-preserving primitives
│   │   ├── _builder.py                 # one-shot factual MoleculeRecord import
│   │   └── _commit.py                  # closed named commit operations
│   ├── views/                          # immutable derived result objects
│   │   ├── __init__.py
│   │   ├── topology.py                 # AtomSeq, Ring, JointRing
│   │   ├── geometry.py                 # Angle, Torsion
│   │   └── pairs.py                    # AtomPair, AtomPairTable
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
│   │   ├── topology.py                  # Atom/Bond/component transactions
│   │   ├── chemistry.py                 # assignment controllers
│   │   ├── geometry.py                  # coordinate commits/transforms
│   │   ├── constraints.py               # typed constraint transactions
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
│   │   ├── conversion.py                # supported-object dispatcher
│   │   ├── io/                          # MolReader/MolWriter/formats
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
│   │   ├── thermo/                      # external thermo/RDKit bridge
│   │   ├── calculation/                 # scientific backend/projection contracts
│   │   ├── inference/
│   │   │   ├── artifacts.py             # filesystem/hash/integrity I/O
│   │   │   ├── runtime.py               # ONNX/PyTorch sessions/tensor bridges
│   │   │   └── serialization.py         # inference DTO encoding/decoding
│   │   ├── pubchem.py
│   │   ├── pyg.py
│   │   └── visualization/               # drawing/presentation
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
| `Atom` | `structure/atom.py` | retain one class; remove Open Babel construction and implicit chemistry from constructor/setters |
| `Bond`, `BondKind` | `structure/bond.py` | retain one class/enum; enforce the approved kind/order contract in atomic mutations |
| `NotInSameMolecule`, `ObjNotInMolecule` | `structure/errors.py` or none | preserve their exact names if active/public; otherwise delete after zero-reference audit; do not rename them as part of this refactor |
| `MolBlock` dynamic attribute machinery | `structure/_state.py`, or delete | replace magic attribute dispatch with a named field schema; do not expose a general arbitrary setter |
| `AtomSeq`, `Angle`, `Torsion`, `Ring`, `JointRing`, `AtomPair`, `AtomPairs` | `views/` | make membership immutable/identity-bound; rename `AtomPairs` to `AtomPairTable` only if owner approves; project constraints from Molecule; remove carrier cache ownership |
| zero-reference `AtomPairKey` | none | delete; stable internal UIDs and immutable pair membership replace this unused identity wrapper |
| `Conformers` | `structure/conformer.py` plus `structure/results.py` | split current/source `CoordinateFrame` facts from generated/optimized `ConformerEnsembleResult`; preserve the public array access as a projection without storing the same frame in both owners |
| Atom/Bond identity, UID high-water marks, `id`, and `idx` | `structure/_state.py` and `_topology.py` | introduce immutable private `_uid` plus non-reusing Molecule allocators; retain `id` as source/display fact and `idx` as derived current row; never use one as the other |
| Atom/Bond/Angle/Torsion constraints | `structure/constraints.py` | migrate to one Molecule-owned `ConstraintSet` keyed by internal identities; views only project membership |
| `InternalCoordinates` | `rules/mathematics/internal_coordinates.py` | pure calculation from explicit coordinates/topology; no Molecule mutation |
| `_metal_valence`, formal charge, valence, electron configuration, implicit-H logic | `rules/chemistry/` | return assignments/findings; delete duplicate bodies from carriers |
| ring/cycle/path/component/tree/spectrum algorithms | `rules/mathematics/graph` and `rules/chemistry/descriptors` | retain deterministic Plan 19 semantics; return views/results only |
| aromaticity/Kekule/bond-order perception | `rules/chemistry/bonding` | one pure assignment implementation; private operation commits the patch |
| distances, angles, torsions, translation, bond/ring relation | mathematics rules plus `operations/geometry.py` for coordinate mutation | query calculations remain pure; translation is an atomic coordinate commit |
| charge, spin, partial-charge workflows | chemistry rules plus calculation operation/adapters | distinguish formal/net/partial charge fact types explicitly |
| `build3d`, `optimize` | `operations/forcefields` | retain public signatures as thin Molecule methods |
| `search_substructure` and SMARTS packing | `operations/search`, chemistry semantics, structure adapter | reuse one RI matcher; no carrier cache |
| `get_thermo` | physics contract + external thermo adapter | return a typed result; no RDKit/thermo import in carrier |
| `to_obmol`, `to_rdmol`, `to_pybel_mol`, `to_pyg_data` | adapters | each call builds a fresh external value; no persisted external object |
| `read_one`, `write`, SMILES/InChI serialization | adapters/I/O | object methods remain thin entry points if their public contract is retained |
| MCA result access | `structure/results.py` plus inference operation | typed explicit result replaces `properties["mca"]`; `Atom.mca`/`mca_sites` remain projections |
| `_topology_index`, `_update_topology` | `structure/_topology.py` and Molecule primitive | retain the exact Plan 19 conceptual contract; no native Graph or second index |
| hidden Bond lists | operation-local `TopologyView`/mask | remove from Molecule after all internal consumers migrate |
| `_model`, `_environ` | operation inputs | remove from Molecule and from working-copy metadata |
| `_obmol`, `_row2idx`, ring/angle/torsion/pair caches | none | delete, with no replacement persistent cache |

### 9.2 Current `hotpot/cheminfo` module-by-module disposition

This table accounts for every current top-level file or package rather than
only the obvious monolith.

| Current path | Target | Action |
|---|---|---|
| `cheminfo/__init__.py` | explicit reviewed chemistry facade | replace wildcard assembly/core imports with the approved `__all__`; import each public class/function from its sole owner |
| `AImodels/artifacts/manifest.py`, MCA/CBond manifests, model-store metadata | `operations/inference/artifacts/` plus `adapters/inference/artifacts.py` | keep `ArtifactManifest` and model identity metadata beside the one resolver; isolate file reading/hash/integrity I/O in the adapter; model cards/licences remain artifact documentation, not chemistry reference data |
| `AImodels/artifacts/registry.py`, `resolver.py` | `operations/inference/artifacts/` | retain one artifact discovery/install/resolve workflow; no carrier or rule owns a model cache |
| `AImodels/artifacts/cli.py`, CBond/MCA CLI modules | `commands/models.py`, `commands/cbond.py`, `commands/mca.py` | preserve CLI names/arguments while removing command logic from scientific packages |
| `AImodels/cbond/apply.py`, `stage.py`, constants | `operations/inference/cbond/` | retain CBond composition/commit semantics and the `is_Nuc_site` distinction |
| `AImodels/cbond/runtime.py`; MCA `runtime.py`, model/session/tensor code | `adapters/inference/runtime.py` | own ONNX/PyTorch/device sessions, tensor conversion, and external runtime errors; operations select and invoke them |
| `AImodels/cbond/deploy/` model export/inspection scripts | repository `tools/model_deployment/cbond/` | keep non-runtime deployment tooling out of the importable package; no second copy remains under operations |
| MCA `api`, `batching`, `pipeline`, `resources`, `workers`, CPU/GPU predictors and GPU consumer | `operations/inference/mca/` | retain current CPU default, bounded CPU/GPU execution, GPU threshold/resource policy, rolling admission, and public workflow API; predictors invoke the runtime adapter |
| MCA `gpu_memory.py` | system/device probe in `adapters/inference/runtime.py`; resource decision in `operations/inference/mca/` | split external GPU-memory observation from batch/window/device policy |
| MCA `site_detection.py` | `rules/chemistry/mca_sites/` | retain ordered SMARTS priority, first-match naming, site eligibility, `is_Nuc_site`, and symmetry/dedup semantics as deterministic chemical rules |
| MCA `conformer.py` direct RDKit backend | none; use `adapters/rdkit` -> standalone native `rdWrapper` | delete the second RDKit conformer implementation after parity/performance gates; one native wrapper path remains |
| MCA `featurizer`, graph input bridges, `data_extract.py` | `adapters/structure/inference.py` plus `adapters/inference/` | separate fact extraction/native packing from model tensor/session translation; never write carriers |
| MCA `MoleculePrediction` and public inference DTOs | `operations/inference/mca/results.py` | preserve the public return DTOs independently of whether a result is attached to a Molecule |
| attached MCA record | `structure/results.py` | define one typed identity-bound `MCAResultRecord`; operations attach it and Atom/Molecule only project it |
| MCA serialization/validation | DTO in `operations/inference/mca/results.py`; encoding in `adapters/inference/serialization.py`; structure/RDKit translation in their adapters; domain validation in `rules/chemistry/mca_sites/` | split each responsibility and delete the mixed serializer implementation |
| `ChemData/ElementProperties.json` | `reference/data/element_properties.json` | move and load through one immutable reference module |
| `ChemData/ideal_bond_length.json` | `reference/data/ideal_bond_lengths.json` | move and remove Bond class-level file-loading logic |
| active `mol_assemble/FragTemplete.json` | `operations/assembly/data/fragment_template.json` | this code-referenced file is the provisional canonical default assembly asset; rename and package/test it with the operation |
| zero-reference `ChemData/MolFragTemplate`, `MolFrameTemplate` candidates | merge into the canonical assembly schema or delete | compare schema/content during Stage 1; never move two apparent template sources forward as parallel defaults |
| `ChemData/StructQueryTemplate` | `rules/chemistry/substructure/data/` | move only if still used; otherwise delete after zero-reference audit |
| `ChemData/coformers`, `ChemData/solvent` | governed `hotpot/dataset/data/` assets | move molecule collections out of the code package with manifest/schema updates; do not instantiate all solvents at import time |
| `MolProps/` | governed dataset assets plus a tested query operation if required | split spreadsheets/docs from code; delete empty or uncalled API rather than preserve a second property system |
| `_clib/` | none | delete generated objects, SWIG wrapper, C/H/I files after build and reference audit; it has no active implementation caller |
| `_io/` | `adapters/io/` | split reader, writer, and format handlers; remove `sys.modules` cycle workaround |
| `_native/` | `adapters/native/` | retain one generic field-selective molecule transport and shared C++ headers |
| `bundle.py` | `structure/bundle.py` | retain MolBundle as a carrier collection; move PyG conversion to adapter |
| `MolecularFacts`, `MolecularFactsIdentity`, `ComponentIdentity`, `HydrogenInventory`, `SemanticBond` | `structure/snapshots.py` `ScientificStateSnapshot` using the one `RepresentationIdentity` | merge the duplicate fact/identity model into the shared snapshot contract; the scientific-state adapter constructs it and rules consume it |
| scientific quantity/admissibility/authority/confidence enums, findings and provider values/evidence | `rules/chemistry/scientific_state/contracts.py` | immutable rule vocabulary and findings have this one owner |
| policy/toggle/provider-set/budget/requirements/selection contracts | `operations/calculation/scientific_state/contracts.py` | execution choice and orchestration configuration are operation contracts, not facts or rules |
| `BackendIdentity`/`BackendContext` protocols | `adapters/calculation/contracts.py` | define generic scientific backend/projection identity once at the external boundary; inference and force-field wrapper-specific contracts remain with their named standalone/adapters and do not redeclare it |
| state-composition candidates/decisions/resolution reports | `operations/calculation/scientific_state/composition/contracts.py` | retain operational search/result DTOs without declaring a second Molecule snapshot or representation identity |
| `calculator/formal_charges.py`, `molecular_charge.py`, `charge/`, `oxidation_state/`, `electronic_state/`, `legality.py` | `rules/chemistry/scientific_state/` | retain deterministic conservation, legality, charge, oxidation, electronic-state, coordination-counting, and spin rules as snapshot-to-finding functions |
| concrete `state_rules` provider modules, provider base/contracts | `rules/chemistry/scientific_state/providers/` | preserve named deterministic providers and evidence without controller or registry state |
| `state_rules/manifest.py`, immutable rule tables, neutral-atom/oxidation-state JSON | `rules/chemistry/scientific_state/data/` | keep these rule-specific scientific sources immutable and explicitly loaded; universal element facts alone belong in `reference/data` |
| `state_rules/controller.py`, profile decision/sets, registry, defaults, execution/state selection, eligibility, requirements, resolver | `operations/calculation/scientific_state/` | own provider selection, policy choice, registry construction, conflict resolution, and sequencing; these are not chemistry laws |
| `calculator/state_composition/` | `operations/calculation/scientific_state/composition/` | preserve candidate expansion/search/coupling/resolution, serialization, and decision artifacts as one operation workflow; do not duplicate formula/composition rules |
| Molecule extraction embedded in resolvers/calculators | `adapters/structure/scientific_state.py` | construct the one immutable scientific-state snapshot and contain all carrier awareness |
| `calculator/mca_inference.py` | `operations/inference/mca/` | move out of general scientific-state calculation; replace arbitrary result storage |
| `calculator/base.py` | none | delete the empty `Calculator` class |
| `call_thermo.py` | `adapters/thermo/chemical.py` plus physics result contract | current module is an external `thermo`/RDKit bridge, not a native Hotpot rule |
| `conformers/contracts.py`, `sampler.py` | `operations/conformers/` | preserve backend-neutral seam and deterministic seed semantics |
| `conformers/adapters.py` | RDKit/Open Babel adapter subpackages | split backend-specific code; no mixed toolkit adapter module |
| `convert.py` | `adapters/conversion.py` | preserve supported-object dispatch with lazy external imports |
| `core.py` | split per Section 9.1, then delete | no compatibility re-export module in the recommended plan |
| `core_func.pyx` | none | delete after zero-reference and build audit |
| `core_utils.py` | `adapters/io` and named topology utilities | move `read_mol`; relocate or remove `atom_idx_pair_to_bond_idx` according to consumers |
| `crystal.py` | `structure/crystal.py`, mathematics cell conversion, Open Babel adapter | keep one canonical six-parameter `Crystal` record; remove Molecule back-reference, Open Babel object, `LatticeGroup`/space-group placeholders; preserve approved matrix convention |
| `draw.py`, `fonts/`, `draw.md` | `adapters/visualization/` and docs | retain presentation capability outside domain carriers/rules |
| `elements.py` | `reference/elements.py`, `reference/properties.py`, and `rules/chemistry/elements.py` | split immutable source values from classification/electron/valence calculations; remove Open Babel dependency |
| `forcefields/coordinates.py` pure coordinate proposal helpers and pure acceptance/placement criteria | `rules/physics/forcefields/` | move `_perturbed_coordinates(...)`-style snapshot-to-value/evidence kernels only; no backend call or carrier write |
| `forcefields.coordinates.perturb(mol, ...)` and other coordinate write facades | `operations/forcefields/` | retain the user/workflow operation while routing its generated coordinates through one atomic coordinate commit |
| carrier extraction in `coordination.py` and `topology.py` | `adapters/structure/forcefields.py` | construct force-field/topology snapshots and operation-local `TopologySelection`; no second graph representation |
| placement/coordination sequencing in `coordination.py`; `ligand.py`; `repair.py` | `operations/forcefields/` | these choose targets, mutate working structures, retry, or repair and therefore are operations, not pure rules; delete any unimplemented placement hook |
| `forcefields/auto.py`, `attempts.py`, `ff.py`, `optimizer.py`, `stage.py`, `trajectory.py`, `workers.py`, `workflows.py`, `working_copy.py` | `operations/forcefields/` | retain workflow, transaction, trajectory, and atomic commit behavior; replace hidden-bond mutation with task-local topology masks |
| `forcefields/contracts.py`, settings, reports, and errors | rule findings in `rules/physics/forcefields/`; operation reports/errors in `operations/forcefields/`; backend protocols in `adapters/openbabel/` | split the mixed contract file by semantic owner without duplicating dataclasses; import wrapper-owned types such as `obWrappers.contracts.ConvergenceLevel` rather than redeclaring them |
| `forcefields/backend.py` | thin `adapters/openbabel/forcefields.py` facade | invoke the retained standalone `obWrapper` only; migrate/delete direct Open Babel force-field bodies so Hotpot has no second backend implementation |
| `native_packing.py`, `native_adapters.py`, binding/translation C++ | `adapters/structure/forcefields.py`, `adapters/native/` | own field-selective packing, Python/C++ conversion, and binding code only |
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
| `obconvert.py` | `adapters/openbabel/conversion.py` and a named bond-perception operation | split lossless translation from `ConnectTheDots`/bond-order behavior |
| `obWrappers/` | retain as standalone `cheminfo/obWrappers/`; call it through `adapters/openbabel` | preserve Python/C++ standalone contracts, implementation, and direct component use; do not duplicate wrapper code inside the adapter |
| `org_compounds.py` | none | delete the broken code path; any future real molecule collection enters through governed dataset admission rather than reviving this module |
| `parse_smarts.py` | none | delete explicitly damaged, unused duplicate parser |
| `pubchem.py`, `pubchem.md` | `adapters/pubchem.py` and docs | consolidate duplicated module-level and service methods into one implementation while retaining approved convenience functions |
| `rdWrappers/` | retain as standalone `cheminfo/rdWrappers/`; call it through `adapters/rdkit` | preserve fixed RDKit version, Python/C/C++ contracts, CMake attach support, and standalone tests; do not duplicate wrapper code inside the adapter |
| `rdconvert.py` | `adapters/rdkit/conversion.py` and `adapters/rdkit/stereo.py` | separate molecule conversion from stereo transport translation |
| `rdworks.py`, `rdwork.md` | `adapters/rdkit/fingerprints.py` and docs | retain called/tested fingerprint functions; delete zero-use utilities after audit |
| `search/` public composition | `operations/search/` | retain Searcher/Substructure/Query/Hit API and fixed enumeration semantics |
| `search/smarts.py`, syntax, predicate IR, semantics | `rules/chemistry/substructure/` | retain one SMARTS parser and chemical predicate implementation |
| `search/_native` reusable RI matcher | `rules/mathematics/graph_matching/` | keep one RI-style native algorithm over the sole shared native CSR Graph; no VF2/backend selector and no search-private graph representation |
| `search/native.py` Molecule packing | `adapters/structure/substructure.py` | pack current facts into task-local native input; callable predicates alone return to Python |
| `smarts.md` | substructure operation/rule docs | update paths and test examples verbatim |
| `solvents.py` | none | delete the broken import-time singleton, missing-sheet dependency, empty methods, and semantically empty Molecule subclass; real solvent assets, if admitted, use the governed dataset/query layer |
| `stereo.py` payload schema and fact dataclasses | `structure/stereo.py` | retain the public transport schema; split stored stereo records/source ledger from current Atom/Bond facts |
| `stereo.py` public functions/schema exports | maintained thin `cheminfo/stereo.py` facade | preserve the five Plan 17 public signatures and complete public schema class identities exactly; facade delegates to one operation/rule implementation |
| `stereo.py` pure remap proposals, validation, assignment, alignment, Gate 2-7 | `rules/chemistry/stereochemistry/` | split by algebra, topology, coordinate classification, validation, assignment, and alignment; preserve all gates |
| `stereo.py` payload decomposition/remap commit | `adapters/structure/stereo.py` plus `operations/chemistry.py` | build immutable updates, validate once, and commit; rules never write Molecule |

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
- force-field acceptance meaning, especially finite optimal energy and ability
  to continue through recoverable intermediate non-finite states; or
- CLI command names and command-line arguments.

### 9.6 API and implementation migration matrix

The action vocabulary is: **keep** (same owner and contract), **move** (same
contract, new physical owner), **refactor** (same supported purpose through
one new implementation), **merge** (several bodies become one), **add**, and
**delete**. Stage 1 freezes exact signatures for every retained callable before
the first move; the important contracts are fixed here rather than left to
implementation inference.

#### Public Python surface

| Current API/family | Action | Final owner/contract |
|---|---|---|
| `hotpot.version()` | keep | reads the one runtime/package version source; no eager chemistry backend import |
| `hotpot.Molecule`, `Atom`, `Bond`, `BondKind` | move | exact same class identities at `hotpot.cheminfo.structure`; top-level exports remain the ordinary user entry points |
| `hotpot.Crystal` | add export/refactor | six-parameter value carrier described in Sections 5 and 7; no Molecule back-reference/toolkit object |
| `MolReader`, `MolWriter`, `read_mol` | move/merge | one `adapters.io` implementation; public call signatures and supported formats are frozen in Stage 1 |
| `MolBundle`, `to_pyg_dataset`, `is_molecule_input`, `to_hotpot_mol` | move/refactor | bundle remains a carrier collection; PyG/external conversion moves to adapters; convenience signatures remain |
| `Searcher(substructure)`, `has_match(mol)`, `iter_mappings(mol, max_matches=None)`, `search(mol, max_matches=None)` | move | `operations.search`; RI semantics, callable-leaf Python behavior, truncation, and deterministic order remain exact |
| `Substructure`, `Query`, `QueryAtom`, `QueryBond`, `Hit`, `Hits`, SMARTS errors/semantics | move | public composition remains; chemistry predicates and RI execution each have one lower-level owner |
| five `hotpot.cheminfo.stereo` functions in Section 3.2 | refactor | signatures stay exact; thin maintained facade invokes extracted rule/operation code |
| `Molecule.build3d(...)`, `optimize(...)`, `search_substructure(...)`, `get_thermo(...)` | refactor | signatures remain frozen; bodies become thin operation delegates |
| Molecule graph/geometry/chemistry query properties and methods in Section 7.2 | refactor | observable return type/order remains unless this matrix or a Section 16 decision explicitly changes it; no cached second implementation |
| `Atom.partial_charge` | refactor | read-only optional-float projection of the selected typed partial-charge result; creation/replacement occurs only through the named full-set assignment operation, subject to Decision 22 |
| `Molecule.properties` | refactor | user annotations only; calculated results move to typed result records subject to Decision 7 |
| `Molecule.assign_aromatic()` (empty) | delete | `determine_rings_aromatic()` remains the single Molecule-level operation unless separately renamed |
| generic `Molecule.setattr`/`Atom.setattr` and broad attribute dictionaries | delete/refactor | no public generic mutation; private bulk import uses `_build_molecule(MoleculeRecord)` |
| hide/recover/clear-hidden-Bond family | delete after migration | supersedes Plan 19 only with Decision 6 approval; internal callers use private task-local `TopologySelection` |
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
| `_atoms`, `_bonds`, Atom/Bond storage arrays | move | named carrier schema in `structure/_state.py`; initial physical layout remains unchanged under Decision 11 |
| `_hided_metal_bonds`, `_hided_covalent_bonds` | delete | complete Bond collection plus operation-local mask, subject to Decision 6 |
| `_angles`, `_torsions`, `_rings`, `_ring_indices_cache`, `_ligand_rings`, `_atom_pairs` | delete | identity-bound views are produced on request and never carrier-owned |
| `_obmol`, `_row2idx` | delete | fresh conversion objects and direct topology lookup; no persistent external cache |
| `_model`, `_environ` | delete | explicit operation/runtime inputs |
| stored whole `_stereo_payload` | refactor | stereo records/source ledger plus canonical Atom/Bond facts; payload assembled/decomposed losslessly |
| Angle/Torsion-local mutable constraints | merge/refactor | one Molecule-owned `ConstraintSet`; views only project entries |
| arbitrary `properties["mca"]` and partial-charge storage | refactor | dependency-bound typed records in `structure/results.py` |
| `_io`/`core` `sys.modules` cycle workaround | delete | one-way dependency and local facade imports |
| parser/converter field-by-field setter construction | refactor/add | `_build_molecule(record: MoleculeRecord) -> Molecule` validates and installs one factual batch |
| ad hoc writes from rules/backends | delete/add | closed `_commit_topology`, `_commit_representation`, `_commit_coordinates`, `_commit_stereo`, `_commit_constraints`, `_attach_result` entry points |
| recursive SMARTS Molecule-keyed cache | delete | operation-local compiled query/snapshot state only |
| native packing structs copied per subsystem | merge | one field-selective transport module; each adapter requests only required fields |

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

### Stage 0 — Freeze the reviewed plan

1. Resolve all Section 16 decisions.
2. Re-read this document in full and revise conflicting sections, not merely
   append owner comments.
3. Record the approved target tree, API decisions, fact priorities, and cache
   exceptions as the normative contract.
4. Create `implementation_log.md`; record branch, baseline, environment, and
   each subsequent commit/test result.

Deliverable: plan-only commit. No production code.

### Stage 1 — Capture behavioral and API baselines

1. Generate a reviewed explicit list of top-level exports, class identities,
   public signatures, CLI commands, and direct submodule paths used by Hotpot's
   own documentation.
2. Record every current carrier field and writer, including private writes in
   plugins and tests.
3. Generate the exact architecture-debt allowlist and assign every entry its
   removal stage; this list may shrink but never grow.
4. Record deterministic outputs for graph/search, stereo, chemistry
   assignments, conversion round trips, and current MCA fixtures.
5. Freeze the tracked 187-source digest, derived 182 eligibility manifest, and
   paired O0 seed-42/43/44 per-case baseline required by Decision 23.
6. Run the existing default suite, installed-wheel smoke tests, native wrapper
   standalone tests, and focused scientific tests.
7. Measure the full performance set in Section 12 before changing code; older
   measurements are context only, not the executable baseline.

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
2. Add read-only structure protocols, immutable snapshots, typed assignment
   results, typed P3 result records, and the closed named commit contracts from
   Section 6.2; do not add a generic patch dictionary.
3. Define `MoleculeRecord` and the one-shot private import builder so readers
   and toolkit converters do not depend on public setter side effects.
4. Introduce stable private Atom/Bond UIDs, non-reusing high-water marks,
   composable source identities, and `StaleStructureViewError`; freeze
   construction/detachment/copy/component behavior before Stage 4 creates
   identity-bound views.
5. Add the ratcheted dependency/import/write gates from Section 6.3.
6. Split immutable element and ideal-bond reference loading from classes.
7. Keep one implementation at all times: a new rule may be called by the old
   class facade, but its old body is deleted in the same commit.

Gate: contract/import-cycle tests; builder installs supplied facts only,
performs no implicit chemistry, and constructs topology exactly once; UID
non-reuse; exact identity retain/invalidate fixtures; ratchet tests; default
suite.

### Stage 4 — Extract mathematical rules and structure adapters

1. Move current pure graph and geometry implementations under mathematics
   rules using atomic path updates; preserve native module names until their
   build move is ready.
2. Split `geometry.convert` into carrier-to-geometry adapters.
3. Split graph matrix mathematics from chemical graph-spectrum policy.
4. Extract `InternalCoordinates`, distances, angles, torsions, paths, rings,
   and pair views from `core.py`.
5. Remove corresponding carrier caches and have thin properties calculate and
   return immutable views.

Gate: deterministic graph/cycle/search-order tests, geometry parity, native
geometry/graph tests, identity-bound stale-view rejection, detached membership
inspection, UID member remap, and focused performance comparison.

### Stage 5 — Extract elemental, bonding, charge, and hydrogen rules

1. Split immutable element facts from elemental classifications and electron
   calculations.
2. Extract formula/mass/count descriptors.
3. Extract valence, implicit-H, formal-charge, bond-order, aromaticity, and
   Kekule calculation into read-only rules.
4. Replace `_atomic_number_setter` and `_bond_order_setter` side effects with
   explicit complete operations.
5. Make aromatic/Kekule rules return one coupled Atom/Bond assignment; remove
   Ring write-back behavior.
6. Enforce the approved fact-conflict table.

Gate: chemistry fixtures, SMARTS chemical predicates, import/export round
trips, aromatic/Kekule parity, complete default suite.

### Stage 6 — Split stereochemistry facts and rules

1. Move Level-1 payload/transport dataclasses into `structure/stereo.py`
   without changing schema, field order, source codes, or native packing.
2. Split descriptor algebra, graph semantics, coordinate classifiers,
   validation, assignment, and alignment into named rule modules.
3. Route remapping and commits through structure mutation/operation code.
4. Update RDKit/Open Babel translation to depend on transport contracts rather
   than the former monolith.
5. Reduce `cheminfo/stereo.py` to the maintained Plan 17 public facade after
   every Gate 2-7 implementation body moves; delete the old monolithic bodies,
   not the facade module.

Gate: all Level-1/Level-2 tests, native packing, RDKit round trips, public
stereo signatures, Python 3.10-3.14 focused matrix.

### Stage 7 — Remove second truth sources and clarify results

Each sub-stage is a separate commit/gate; failures in one risk domain do not
hold uncommitted changes from another.

#### Stage 7A — External-object freshness

Delete `_obmol/_row2idx`; make each conversion produce a fresh task value and
make coarse operations reuse it locally when needed.

Gate: source/wheel conversion freshness, round trips, adapter isolation,
memory, and repeated-workflow conversion benchmarks.

#### Stage 7B — Topology selections, views, and constraints

1. Replace hidden-bond collections with operation-local topology selections.
2. Replace mutable carrier `AtomPairs` and ring/angle/torsion caches with
   identity-bound returned views.
3. Move every persisted coordinate/bond/angle/torsion constraint to the one
   Molecule-owned `ConstraintSet` and retain only view projections.

Gate: hidden-bond workflow parity, stale-view behavior, deterministic views,
constraint identity/invalidation, force-field working-copy rollback, and
view-performance benchmarks.

#### Stage 7C — Annotations and typed calculated results

1. Split user annotations from typed scientific results.
2. Replace `properties["mca"]` with a typed, identity-bound MCA result while
   retaining approved `Atom.mca`/`Molecule.mca_sites` behavior.
3. Replace Atom-stored partial charges with the selected identity-bound
   `PartialChargeSet`; update field-selective native packing.

Gate: exact per-mutation retain/invalidate table, MCA calculator/site tests,
partial-charge absence/presence/provenance and native packing, copy/component
result behavior, and annotation isolation.

#### Stage 7D — Coordinate-frame ownership

Split current coordinates, explicit P1 source frames, and generated P3
conformer ensembles; make frame loading one atomic coordinate commit and keep
the public conformer projection deterministic.

Gate: source/generated role tests, seed behavior, load/clear invalidation,
copy/component semantics, and existing conformer array-access parity.

### Stage 8 — Split rule policy, workflow, and backend execution

Each sub-stage is independently committed and gated:

#### Stage 8A — Scientific-state calculation

Split deterministic providers/rules, Molecule snapshot extraction, and
selection/composition/resolution controllers.

Gate: provider, legality, conservation, conflict/evidence, composition search,
serialization, and state-resolution tests.

#### Stage 8B — Force fields

Split physics/acceptance kernels, structure packing, workflow controllers, and
the Open Babel/native adapter. Retain standalone `obWrapper` as the sole
backend implementation.

Gate: pure-policy tests, native and Python workflow parity, rollback,
trajectory, O0 complex behavior, wrapper standalone tests, and FF benchmarks.

#### Stage 8C — Search

Split SMARTS chemistry semantics, reusable RI graph matching, structure
packing, and public search orchestration while retaining the sole native CSR
Graph definition.

Gate: SMARTS conformance, callable-leaf isolation, RI mapping parity,
deterministic enumeration, native component tests, and throughput.

#### Stage 8D — Inference and conformer orchestration

Move CBond/MCA/conformer controllers under operations, runtime/session packing
under adapters, and artifact identity/resolution under their named owners.
Retain standalone `rdWrapper` unchanged.

Gate: CBond, MCA CPU/resource/site tests, `is_Nuc_site`, conformer seam/seed,
artifact resolver, runtime adapter, and `rdWrapper` standalone tests; bounded
fake-GPU routing covers the below-256 CPU fallback, rolling-window adjustment,
explicit GPU index/count, session reuse, and serialization. If the real GPU
module path, package data, or runtime-session code moves, run an additional
ai-14 one-GPU installed-wheel smoke at this stage rather than waiting for final
4-GPU validation.

#### Stage 8E — Assembly and statistics

Move assembly workflows and either complete or delete statistics according to
Decision 20; install explicit exports with no wildcard leakage.

Gate: documented assembly examples, export snapshot, and completed statistics
assertions if retained.

### Stage 9 — Split the carriers and hard-cut `core.py`

1. Once `core.py` contains only carriers/facades, move each class definition
   to its one structure/view module.
2. Update all 72 production and 81 test direct-core import files.
3. Update plugins, pipeline, dataset providers, typing-only imports, and string
   entry points.
4. Delete `core.py` in the same commit that moves the last consumer.
5. Do not add a re-export shim, alias class, fallback import, or compatibility
   subclass.

Gate: zero old imports, class identity tests, full default suite, plugin import
smoke, no new cycles.

### Stage 10 — Complete physical package and native build migration

1. Move the approved rule/operation/adapter packages with `git mv` and update
   all consumers atomically by coherent package.
2. Update pybind11 extension names/source lists, relative C++ includes, CMake
   attach files, export maps, `.pyi` stubs, package data, and build scripts.
3. Keep C++ public namespaces and standalone wrapper contracts stable unless a
   separately approved decision says otherwise.
4. Build/test sdist and wheels for every supported Python version.

Gate: clean-source build, installed-wheel tests, native symbol/import checks,
standalone C/C++ consumer tests.

### Stage 11 — Clean outer commands and data ownership

This is a separately approved tail stage controlled by Decisions 12 and 13.
It is intentionally not a prerequisite for accepting the carrier/rule hard
cut, and it may be deferred without leaving duplicate chemistry
implementations.

1. Consolidate `main`/`works` command handlers into `commands` while retaining
   command names and arguments.
2. Move approved chemistry assets to reference, operation-owned data, or the
   governed dataset tree with manifests.
3. Delete old command/data paths and update docs/tests.

Gate: CLI loading/help/execution tests, package-data inspection, dataset
manifest checks.

### Stage 12 — Explicit exports, documentation, and final hardening

1. Replace wildcard exports with the approved `__all__` lists.
2. Remove the top-level RDKit preload and verify lazy RDKit/Open Babel load
   order in both directions from an installed wheel.
3. Update README, API docs, module docs, type references, and every example.
4. Test README/docs examples verbatim.
5. Run the complete validation and performance matrix in Sections 11-12.
6. Write the final implementation report, final API tree, deletion table,
   benchmark tables, remaining limitations, and commit map.

No old implementation is retained "just in case" after this stage.

## 11. Validation matrix

The refactor is accepted on behavior and architecture, not merely import
success. Stage 1 reruns the behavioral and performance baseline at the planning
baseline commit. Final validation reruns every execution path touched by the
refactor. Existing reliable records may supply only reference inputs, expected
scientific evidence, and corpus manifests; they may not substitute for an
execution result unless the final report records the source commit/digest,
environment, and proof that the path was not touched.

| Validation area | Required evidence | Frequency |
|---|---|---|
| architecture | static import/write/cache gates; one class identity; zero deleted-path references | every stage, final full scan |
| public API | approved export snapshot; signature comparison; CLI command/argument comparison | baseline and every public move |
| carrier lifecycle | add/remove/replace Atom/Bond, component merge, coordinates, copy, rollback, stereo remap, result invalidation | every carrier/operation stage |
| graph | components, paths, deterministic default/all bases, relevant cycles, spectra, topology index invariants | graph stages and final |
| search | SMARTS conformance, RI parity, callable predicates, coordination semantics, deterministic enumeration | search changes and final |
| chemistry | valence, implicit H, charge, bond kind/order, aromaticity, Kekule, formula, descriptors | chemistry stages and final |
| stereo | Level-1 storage/transport/remap; all Level-2 Gate 2-7 validation, assignment, projection, alignment | stereo stage and final |
| adapters/I/O | OB/RD/native round trips, current coordinate/stereo/cell preservation, six-parameter Crystal packing and frozen matrix convention, no stale external object | each adapter move and final |
| wrappers | independent Python and C/C++ `rdWrapper`/`obWrapper` builds and operation tests | native move and final |
| conformers/FF | seed determinism, conformer seam, topology preservation, working-copy rollback, validation/trajectory/workflow | FF/conformer stages and final |
| MCA/CBond | site detection including `is_Nuc_site`, CPU values, GPU routing/threshold, CPU/GPU numerical tolerance | inference move and final |
| scientific boundary | frozen 187-source-corpus digest and 182 eligible-cohort manifest; O0 per-case geometry gate, finite optimal energy, continuability, last finite frame, failure reason, and runtime | toggled full validation; representative default subset in CI |
| packaging | sdist, wheel, package data, stubs, clean installed-package smoke | native/package move and final |
| Python matrix | Python 3.10, 3.11, 3.12, 3.13, and 3.14 | focused during stages; complete before merge |
| docs | exact README and documentation examples | every doc update and final |

### 11.1 Numerical and stochastic comparison policy

- Integer, categorical, topology, ordering, transport, and serialization facts
  require exact equality unless the approved API explicitly changes.
- Floating calculations use scientifically justified absolute/relative
  tolerances from the owning test; exact float equality is not introduced.
- MCA comparisons retain current numerical-tolerance and correlation checks
  and must preserve site labels/`is_Nuc_site` exactly.
- Conformer and force-field stochastic workflows are not required to reproduce
  identical coordinates, and the historical O0 output is a reference rather
  than a structural ground truth. Intermediate non-finite energy is allowed.
  A case fails for gradient behavior only when a non-finite gradient prevents
  further optimization and every recovery mechanism already defined by the
  standard workflow (including applicable coordinate/forbidden-angle
  perturbation) is exhausted without recovering an optimization direction.
  An accepted optimal structure must pass the established geometry gate and
  have finite optimal energy.
- Stage 1 records the tracked 187-molecule source digest and the exact 182-case
  eligibility manifest. Final O0 reporting is per case and includes success or
  failure classification, reason/stage, last finite frame even for failures,
  gate evidence, optimal energy, recovery outcome, seed, and runtime. The
  refactor must introduce no deterministic per-case regression; stochastic
  paths use the approved repeated-seed comparison rather than an invented
  182/182 success requirement.
- xTB/DFT/QM is not made a mandatory acceptance criterion for this structural
  refactor; existing optional plugin integration tests remain toggled.

### 11.2 Remote GPU validation

Final MCA GPU validation runs on `ai-14` under the existing `hp` environment,
using only `~/mca_test` for remote task files. It must cover representative
CPU, 1-GPU, and 4-GPU execution, the below-256-site CPU route, explicit GPU
indices/count, dynamic rolling-window behavior, and prediction/site parity.
The implementation log records exact commit, installed wheel, command,
hardware allocation, throughput, and result paths.

### 11.3 Concrete local and matrix entry points

The implementation log records exact pass/fail/skip counts, enabled switches,
and the reason for every skipped tier. The standard entry points are:

```bash
python -m pytest -q --import-mode=importlib tests
bash tests/run_coverage.sh
bash tests/run_inference_compatibility.sh 3.10 3.11 3.12 3.13 3.14
python -m pytest -q --import-mode=importlib \
  --run-scientific-boundary --run-high-compute --run-model-artifacts tests
```

`--run-external` and `--run-optional` are executed as separate declared tiers
when their external programs/dependency groups are installed; they are not
silently folded into a default pass. Clean-wheel runs set
`HOTPOT_TEST_WHEEL`/`HOTPOT_TEST_INSTALLED_PACKAGE` according to the existing
scripts and execute outside the source tree.

Import-boundary tests use fresh subprocesses because `tests/conftest.py`
currently preloads RDKit. Both source-tree and installed-wheel tests inspect
`sys.modules` after: `import hotpot`; first use of the RDKit adapter followed
by the Open Babel adapter; first use of the Open Babel adapter followed by the
RDKit adapter; and isolated import/use of each adapter. The Open Babel adapter
may establish the known safe third-party library load order locally before
loading Open Babel; this is still lazy because it happens only on adapter use.
Raw `import openbabel` followed by `import rdkit` is a non-blocking diagnostic:
Hotpot cannot repair the known PyPI Open Babel 3.2.1 process-level crash when
users bypass Hotpot adapters. Ordinary pytest-process import state is not
accepted as lazy-load evidence.

## 12. Performance and resource validation

Removing persistent derived/toolkit caches must not turn one logical
operation into repeated conversions or Python/native crossings. Each coarse
operation should build one required snapshot/backend object and reuse it only
for that operation.

Baseline and final measurements use the same workload digest, build mode,
dependency lock, host class, CPU affinity/thread counts, process model, and
GPU indices/allocation. Microbenchmarks use at least 3 warm-ups and 10 measured
repeats; CPU workflows use at least 1 warm-up and 5 measured repeats; remote
GPU workflows use at least 1 warm-up and 3 measured repeats. The report gives
raw samples, median, MAD, and a paired bootstrap 95% confidence interval for
the final/baseline ratio, plus workload size and resource allocation, for:

| Benchmark | What it protects |
|---|---|
| import `hotpot` and construct an empty Molecule | lazy dependency boundary and startup cost |
| read/write representative SMILES/MOL2/SDF | adapter overhead and round-trip behavior |
| add/remove/replace atoms and bonds | carrier/commit overhead |
| neighbor/bond lookup | retained Plan 19 topology-index performance |
| rings, paths, angles, torsions, pair table | cost of removing persistent derived caches |
| native SMARTS/RI matching | snapshot and matcher throughput |
| geometry screening | structure-adapter and native-kernel throughput |
| OB/RD conversion once and repeated within one workflow | fresh-object cost and required operation-local reuse |
| conformer proposal and force-field workflow | workflow/backend split overhead |
| MCA CPU small and batched workloads | inference orchestration and conformer path |
| MCA 4-GPU/64-CPU workload on `ai-14` | end-to-end throughput and GPU feed |
| peak RSS for representative batch work | removal of persistent external/derived state |

A regression is not hidden by relaxing a test. A relocated unchanged kernel
blocks merge when the 95% interval is wholly above 1.05 runtime ratio. A
carrier microbenchmark or CPU end-to-end workflow blocks when it is wholly
above 1.10. GPU feed/end-to-end throughput blocks when its throughput-ratio
interval is wholly below 0.95. Median peak RSS above 1.10 of baseline blocks.
If noise prevents a conclusion, repetitions increase; the result is not
declared a pass by explanation alone. A threshold may be waived only by an
explicit owner decision that cites the measured table. Scientific correctness
and single-source ownership are never sacrificed to retain an unsafe cache.

## 13. Implementation log and final report contract

`plan/20_fact_rule_architecture/implementation_log.md` will be append-only for
executed work. Every entry contains:

- date/time, stage, commit, and files/responsibilities changed;
- the single implementation removed or installed;
- API/signature/path changes;
- fact ownership/cache implications;
- commands/tests and exact pass/fail/skip counts;
- performance measurements where relevant;
- unexpected findings, plan impact, and decision required; and
- whether the complete plan was re-read before any architectural revision.

If implementation discovers a conflict with this plan, work pauses at the
smallest coherent checkpoint. The plan is reread in full, contradictory or
redundant wording is edited rather than merely appended, and the owner is
notified before the revised route is implemented.

The final implementation report includes:

1. final route and commit map by stage;
2. final repository and API trees;
3. complete retained/moved/split/renamed/new/deleted interface table;
4. fact-ownership and cache audit;
5. test matrix with exact results and skips;
6. native/wheel/C++ component evidence;
7. scientific benchmark tables and per-case failure analysis where relevant;
8. performance/resource comparison tables;
9. known limitations and deferred work; and
10. confirmation that no compatibility implementation or old backend remains.

## 14. Final acceptance criteria

Implementation is complete only when all of the following are true:

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
8. approved top-level APIs, signatures, class identity, CLI behavior, and
   scientific semantics pass their baselines;
9. Python 3.10-3.14, native components, wheel/sdist, documentation examples,
   scientific boundaries, and GPU validation pass as specified;
10. unexplained performance regressions are absent; and
11. implementation log and final report are complete and internally
    consistent with the final code; and
12. Decisions 12/13 are either implemented or recorded as explicit owner-
    approved final-tree exceptions with their unchanged paths; deferral may
    not leave duplicate chemistry implementations or two active asset owners.

## 15. Risks and controls

| Risk | Control |
|---|---|
| circular imports while classes move | extract read protocols/rules first; use function-local facade imports; prohibit upward package imports |
| two class identities | one physical definition; identity tests; no alias class/shim |
| hidden behavior lost from setters | baseline each writer; replace side effects with explicit coupled operations before deleting them |
| stale result after mutation | typed identity-bound results plus dependency-specific removal on commit |
| performance loss after cache deletion | one snapshot/conversion per coarse operation, operation-local reuse, before/after benchmarks |
| native wheel breakage | move Python/build/C++/stub/package-data paths atomically and test clean wheels |
| stochastic scientific false failures | compare gates, finite optimal energy, continuability, and toleranced metrics rather than exact coordinates |
| deterministic search/cycle order drift | exact ordered fixtures and cross-version tests |
| large data accidentally treated as code | reference-vs-dataset ownership table, manifests, package-data inspection |
| accidental public API expansion via imports | explicit `__all__` and snapshot tests |
| long-lived half-migrated tree | small coherent commits, one implementation per behavior, stage gates, no merge until hard cut completes |
| historical pickle breakage | explicit owner decision; no silent compatibility mechanism |

## 16. Owner decisions required before implementation

The recommendations below are part of the proposed design but remain
unapproved until the owner responds.

1. **Carrier package name.** Use `hotpot.cheminfo.structure` rather than
   `model`, `data`, or `carriers`. Recommendation: approve `structure`; it is
   clear to chemistry users and does not collide with ML models or datasets.

2. **Legacy `hotpot.cheminfo.core` path.** Decide whether direct imports from
   this internal path are a supported public contract. Recommendation: no;
   preserve `hotpot.Molecule/Atom/Bond/BondKind`, migrate all repository
   consumers, delete `core.py`, and create no shim.

3. **Existing scientific package paths.** Decide whether direct paths such as
   `hotpot.cheminfo.geometry`, `graph`, `forcefields`, `search`,
   `AImodels.mca`, `calculator` (including MCA integration contracts), and
   `mol_assemble` must remain. Recommendation: allow one hard move to the
   approved classified tree and update every documented import; retain only
   `hotpot.cheminfo.stereo`, `obWrappers`, and `rdWrappers` package paths plus
   wrapper C++ component names because those are explicitly maintained
   standalone/facade contracts. No old package-path shim is created.

4. **Fact coupling contract.** Approve or amend Section 5.3, especially Bond
   kind/order, coupled Atom/Bond aromatic assignments, molecular versus formal
   charge, current coordinates versus conformer frames, stereo versus
   coordinates, and matrix-owned crystal cells. Recommendation: approve the
   table as written.

5. **Generic mutation APIs.** Decide whether `Molecule.setattr`, `Atom.setattr`,
   and broad ID-dictionary writers are public requirements. Recommendation:
   remove them from the public surface and replace internal import use with
   field-specific atomic operations.

6. **Hidden-bond API.** Decide whether users must directly call
   hide/recover/clear Bond methods. Recommendation: remove them after force
   fields/Kekule use operation-local `TopologyView` masks; hidden topology is
   workflow state, not molecule truth.

7. **Annotations and calculated results.** Decide the fate of
   `Molecule.properties`. Recommendation: retain it only as an explicitly
   user-owned annotation mapping, never as a rule/result store; add a private
   typed result collection projected through named APIs such as `mca_sites`.

8. **Result attachment behavior.** Decide whether calculations attach final
   results automatically. Recommendation: queries return results without
   attachment; MCA, partial-charge assignment, conformer sampling, and
   optimization attach typed P3 results as part of their documented mutating
   operation and invalidate them by dependency. `conformer_add()` writes an
   explicit/source P1 frame, while `conformer_load()` only atomically selects
   current P1 coordinates; neither operation reclassifies that frame as a P3
   result.

9. **Crystal public surface and coordinate convention.** Approve canonical
   six-parameter `Crystal(a, b, c, alpha, beta, gamma)` storage in
   angstrom/degrees, Cartesian Atom coordinates in angstrom, a calculated
   lattice matrix, explicit `hotpot.Crystal` export, and removal of
   unimplemented space-group placeholders. Recommendation: approve, preserve
   the characterized current matrix convention during this refactor, and
   handle any convention correction as a separate reviewed semantic change.

10. **Historical pickle compatibility.** Old pickles record
    `hotpot.cheminfo.core.<Class>`. Recommendation: do not support them in this
    refactor; supporting them requires a compatibility path that conflicts
    with the single-interface requirement. A one-off external migration tool
    can be planned separately if real assets require it.

11. **Atom/Bond physical storage layout.** Decide whether this stage should
    replace the current compact arrays with dataclasses, slots, structured
    arrays, or molecule-owned columns. Recommendation: do not combine that
    performance-sensitive change with the semantic/module refactor. Introduce
    a named schema in `_state.py`, retain the proven layout initially, and
    evaluate a storage rewrite with a separate benchmarked plan.

12. **Outer command cleanup.** Approve replacing overlapping `hotpot.main` and
    `hotpot.works` with one `hotpot.commands` tree while preserving CLI names
    and arguments. Recommendation: approve; otherwise those two packages remain
    an explicit exception to the complete project-tree cleanup.

13. **Chemistry asset ownership.** Approve moving immutable tables/templates
    to reference/operation data and molecule collections such as coformers and
    solvents to governed datasets with manifests. Recommendation: approve,
    but perform large asset moves in their own reviewable commit.

14. **Advanced rule imports.** Decide whether named pure rule functions are a
    documented advanced Python API. Recommendation: permit explicit imports
    from `hotpot.cheminfo.rules...` for advanced users, but keep every mutating
    operation private and keep the object method as the sole ordinary write
    interface.

15. **Plan 19 Core state-budget amendment.** Approve the exact final state
    budget in Section 3.3: stable private Atom/Bond identities and monotonic
    non-reusing UID high-water marks, current/source
    coordinate facts, stereo records/source ledger, `ConstraintSet`, user
    annotations, and typed P3 records (including generated conformer
    ensembles), while retaining only
    `_topology_index` as the only general carrier acceleration state.
    Recommendation:
    approve as the explicit replacement for Plan 19's closed change budget;
    do not authorize revisions, native Graph storage, or general caches.
    Public `Atom.id`/`Bond.id` retain their characterized source/display and
    refresh behavior, `idx` remains the current derived row, and neither is
    silently replaced by private `_uid`; Stage 3 freezes these distinctions in
    tests.

16. **Construction and ownership.** Approve retaining `Atom()` and
    `Atom(mol=...)` auto-registration, read-only detached-object behavior, and
    same-Molecule Bond endpoints, while changing foreign-Atom insertion from
    silent ownership theft to fact-copy/remap. Recommendation: approve the
    Section 7.5 contract and freeze it with characterization tests first.

17. **Derived views and constraints.** Approve identity-bound read-only views,
    `StaleStructureViewError`, one Molecule-owned `ConstraintSet`, deletion of
    Ring/JointRing write methods and angle/torsion update caches, and the hard
    rename `AtomPairs -> AtomPairTable` with no alias. Recommendation: approve;
    every mutation then has one Molecule-level interface and one
    implementation.

18. **External conversion object identity.** Decide whether repeated
    `to_obmol()`, `to_pybel_mol()`, and `to_rdmol()` calls may stop returning a
    cached mutable object. Recommendation: each call returns a fresh external
    object; an operation may reuse one conversion only within its own task.

19. **Copy, component, and metadata semantics.** Approve the lifecycle contract
    in Section 7.5. Recommendation: `copy()` preserves P0-P2 facts,
    constraints, Crystal, source stereo/provenance, current and explicit P1
    frames, and copies the annotation mapping container while retaining its
    values; it drops P3 results/generated conformers/runtime state.
    `components` drops P3 results and whole-molecule annotations, retains an
    immutable Crystal value for periodic sources, and carries charge/spin only
    when component-resolved facts exist. `add_component` never shares mutable
    carrier state or imports component-level annotations/results. For
    `add_component`: an empty destination adopts the source Crystal and
    charge/spin; a populated destination retains its Crystal when the source
    has none, accepts an equal Crystal, and atomically rejects either a
    different explicit Crystal or a Crystal-bearing source added to a
    non-crystal destination. Two declared net charges are added; an undeclared
    input produces undeclared combined charge. Combined spin is unset unless
    the caller supplies a separately validated state, because multiplicities
    are not generally additive.

20. **Top-level statistics and assembly exports.** Decide the exact surface
    currently leaked by wildcards. Recommendation: delete incomplete
    `ComplexStatistics`; explicitly export only `Fragment`, `EdgeShoulder`,
    `AtomLink`, `AlkylGraft`, `BondAdding`, `AtomReplace`, `RingWedge`, and
    `AssembleFactory` from top-level `hotpot`. Preserve the already documented
    `alkyl_generator`, `atom_link_atom_action`, `shoulder_bond_action`,
    `bond_order_add`, `atom_replace`, and `ring_wedge` as the advanced public
    assembly-package surface at its Decision 3 final path, but do not export
    them top-level. Keep `MolBatch`, decorators, registration helpers, and
    predicates private.

21. **Package-import side effects.** Approve removing the global RDKit preload,
    `sys.path` mutation, `tempdir`, and `package_root` convenience value from
    `hotpot.__init__`. Recommendation: approve after both Hotpot adapter-use
    orders pass fresh-process source-tree and installed-wheel tests; permit the
    Open Babel adapter to establish its required safe third-party load order
    locally. Replace the sole known `package_root` consumer with
    `importlib.resources` and use task-local temporary directories. Raw
    third-party Open Babel-first import remains a documented external
    diagnostic, not a Hotpot acceptance requirement.

22. **Partial-charge public semantics.** Approve replacing writable,
    default-zero `Atom.partial_charge` storage with a read-only projection over
    the current valid `PartialChargeSet`. Recommendation: return
    `Optional[float]` (`None` means no selected value), remove the Atom setter, keep
    `Molecule.get_partial_charge(model, model_args)` as a non-mutating tuple
    calculation, and make `Molecule.assign_partial_charge(...)` the sole
    public full-set commit while retaining its tuple return value. No
    per-Atom partial update or implicit zero is carried forward.

23. **187/182 stochastic non-regression rule.** Approve a paired baseline/final
    O0 run on the frozen 182 cohort with explicit seeds 42, 43, and 44 in the
    same environment/resource allocation. Recommendation: every
    baseline-success/final-failure pair is a blocking regression requiring
    case analysis and explicit owner waiver; historical baseline failures do
    not become a fabricated 182/182 requirement. Report every trial and retain
    the last finite frame for failed cases under Section 11.1.

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
