# Native geometry and coordination-force-field refactor plan

Date: 2026-09-29

Status: approved; phased implementation in progress

Planning baseline: `9c70ef63f337`

## 1. Decisions and scope

Two architectural requirements are fixed for this refactor:

1. Every public computational function in migration scope retains both a C++
   source-level entry and a Python entry.  Both entries reach one canonical C++
   implementation; a Python reimplementation, silent fallback, or second
   scientific backend is not permitted.
2. The existing three scientific stages remain independent:
   Stage 1 ligand construction, Stage 2 coordination-bond construction and
   restoration, and Stage 3 full-complex optimization.  Each stage keeps a
   named controller, explicit input/output contract, stage-specific report,
   focused tests, and a callable entry.  A workflow-scoped native session may
   carry data between stages, but it must not collapse them into one
   inseparable algorithm.

The target therefore combines explicit stage boundaries with native state
reuse.  It is not a monolithic one-call replacement:

```text
Stage 1: ligand construction
    input molecule -> ligand-build result/report
                         |
                         | explicit stage artifact
                         v
              create one native structure session
                         |
Stage 2: coordination construction/restoration
    placement -> bond restoration -> Stage 2 result/report
                         |
                         | explicit stage transition; session may be reused
                         v
Stage 3: full-complex optimization
    topology gate -> untangling -> optimization -> Stage 3 result/report
                         |
                         v
final structure + combined report + optional batched trajectory
```

The following boundaries are mandatory:

1. `geometry` computes mathematical facts only. It must not know what a metal,
   donor, covalent radius, chemically reasonable distance, or acceptable
   structure is.
2. Native force-field code selects radii and targets, interprets mathematical
   facts, ranks candidates, mutates the native working structure, executes
   repairs and invokes the Open Babel backend.
3. Python supplies the public API, converts `Molecule` data to contiguous typed
   buffers at an explicit stage/session boundary, maps the returned native
   contracts, writes trajectories and atomically commits the selected result
   to the caller's molecule.
4. The hot loops must not call Python or cross a pybind boundary. Forcefields
   C++ calls geometry C++ directly.
5. No persistent `HpMol` is introduced. The native structure is a
   workflow-scoped value/session with explicit lifetime and stage transitions;
   it never becomes Hotpot's persistent molecule model.
6. This stage does not solve case 61 and does not introduce per-bond
   transactional coordination restoration. Those behaviours remain explicitly
   deferred. The native migration must preserve the current Stage 2 semantics
   while leaving a clean transaction seam for later work.
7. No silent Python fallback is retained after cut-over. A missing native
   extension is an explicit installation error.
8. Existing public Python names, signatures and result meanings remain stable.
   Private implementation helpers are not promoted merely because they are
   technically importable; independently meaningful mathematical operations
   are reviewed and promoted deliberately.
9. Sharing a native session is an implementation optimization, not a licence
   for a stage to inspect or mutate another stage's private control state.

The placement component remains independently testable and belongs to Stage 2.
The retained Python Stage 1 controller and the native Stage 2/3 controllers
remain independently testable.  Migrated native subcomponents are callable
either from Python bindings or directly by C++ composition.

## 2. Current implementation baseline

This section records the current code as fact.  It is the behavioural and
boundary baseline for the refactor, not the target architecture.

### 2.1 Current three-stage ownership

| Stage | Current controller | Process and state ownership | Current output |
|---|---|---|---|
| 1. Ligand construction | `_build_ligand_proxies_worker()` -> `_build_ligand_proxies()` | A spawned Python child receives a worker molecule and performs ligand-component build, candidate relaxation and ring untangling | One `BuildWorkerResult` containing coordinates, diagnostics and optional ligand trajectories |
| 2. Coordination construction/restoration | `_restore_coordination_bonds_incrementally()` | The parent Python process mutates its `working_mol`; placement, bond screening and restoration control are Python | A `CoordinationBondRestorationReport`; the coordinates/topology remain on the same Python molecule |
| 3. Full-complex optimization | `_optimize_complex_working_mol()` | The parent Python process passes the Stage 2-mutated `working_mol` through topology gates, repair and numerical optimization | A `ForceFieldRunReport`; the selected coordinates remain on the same Python molecule |

Stage 1 and Stage 2 are currently grouped by
`_prepare_complex_working_mol()`.  Stage 3 is invoked separately by
`_complexes_build_workflow()`.  The functions are logically separate, but
their runtime state is not fully isolated: Stage 2 and Stage 3 successively
mutate the same parent-process `working_mol` and append to the same trajectory.
Only Stage 1 has process-level isolation.

At the existing public layer, `build_complex3d()` composes Stages 1 and 2,
`optimize_complex()` exposes Stage 3 independently, and `complexes_build()`
composes all three.  There is no public Stage-1-only or Stage-2-only entry.

### 2.2 Current communication graph

```text
parent Python working_mol
        |
        | spawn/pickle worker_mol
        v
Stage 1 Python child
        |-- repeated Python -> pybind -> C++ Open Babel calls
        |
        `-- one Pipe result:
            coordinates + diagnostics + optional trajectories
        |
        v
parent writes coordinates into working_mol
        |
        v
Stage 2 Python controller
        |-- Python geometry/placement/restoration control
        `-- repeated short Python -> pybind -> C++ optimizations
        |
        | same Python working_mol and trajectory
        v
Stage 3 Python controller
        |-- Python geometry checkpoints and ring repair
        `-- Python -> pybind -> C++ optimizer segments
```

Stage 1 does not return or preserve its child-process `OBMol`.  Stage 1 to
Stage 2 communication is a coordinate-array result applied to the parent
`Molecule`.  Stage 2 to Stage 3 communication is a direct Python object handoff,
not C++-to-C++ communication and not a pybind transfer.

### 2.3 Current Python/C++ boundary cost

Every current `build`, `single_optimize` or `optimize` call independently does:

```text
Python Molecule
  -> traverse atoms/bonds and create typed NumPy arrays
  -> pybind copies the arrays into C++ vectors
  -> C++ constructs a new OBMol
  -> run one build or optimization segment
  -> copy coordinates/report back to Python
```

There is no native molecule or `OBMol` retained across stages or across short
repair segments.  The exception is the epoch loop inside one Stage 3
`optimize()` call: all of its epochs already execute in C++ on the same
temporary `OBMol`, without a Python callback per epoch.

For the nominal path in which the first candidate succeeds and no ring
piercing occurs, let $L$ be the number of non-metal ligand components and $K$
the number of intended coordination bonds.  Current native execution calls are:

| Operation | Current nominal count |
|---|---:|
| Stage 1 `build` | $L$ |
| Stage 1 `single_optimize` (warmup, settling, refinement) | $3L$ |
| Stage 2 `single_optimize` | $K$ |
| Stage 3 `optimize` | $1$ |
| Total | $4L+K+1$ |

For one ligand component the nominal total is $K+5$.  Every ring-repair short
optimization or Stage 2 stalled relaxation adds a `single_optimize` call; a
Stage 3 repair followed by renewed numerical optimization adds another
`optimize` call.  Each call currently repeats full packing and `OBMol`
construction.

### 2.4 Current Geometry boundary

`geometry/relation.py` and `geometry/convert.py` currently perform relation
work in Python/NumPy.  Geometry screening itself therefore does not cross a
geometry pybind boundary.  A ring-cache miss may call the separate graph C++
Relevant Cycles implementation through `Molecule.rings_for_scope()`; that is
graph-topology enumeration, not a native geometry-relation calculation.
Hiding or restoring a bond invalidates the ring cache, so the next scan may
repeat that graph-native call.

The stable Python geometry boundary is the package `__all__`.  Public objects,
settings, result records, scalar relation functions, batch relation functions
and chemical-object conversion functions exported there are the compatibility
surface.  Underscore-prefixed helpers in `relation.py` are implementation
details even though Python currently allows an explicit private import.

### 2.5 Baseline source evidence

| Fact | Current source |
|---|---|
| Three-stage workflow coordinator | `forcefields/workflows.py:599-705` |
| Stage 1 spawn, Pipe receive and coordinate application | `forcefields/workflows.py:211-243` |
| Stage 1 worker envelope | `forcefields/workers.py:34-84` |
| Stage 1 ligand controller | `forcefields/ligand.py:106-500` |
| Stage 2 invocation and controller | `forcefields/workflows.py:261-294`; `forcefields/repair.py:945-1195` |
| Stage 3 controller | `forcefields/workflows.py:366-596` |
| Stage 2/3 handoff through `_PreparedComplex` | `forcefields/workflows.py:295-300,637-676` |
| Per-call Python molecule packing | `obWrappers/native.py:30-45`; `obWrappers/packing.py:55-147` |
| Per-call NumPy-to-vector copy | `obWrappers/_native/native_bindings.cpp:291-328` |
| Per-call `OBMol` construction | `obWrappers/_native/openbabel_adapter.cpp:17-68`; `obWrappers/_native/native_engine.cpp:635-719` |
| Native Stage 3 epoch loop | `obWrappers/_native/native_engine.cpp:704-1036` |
| Relevant Cycles cache path | `core.py:2379-2388`; `graph/cycles.py:155-161` |

## 3. Located performance hotspots

### 3.1 Measured evidence

The current `geometry` package has no native implementation. Its approximately
2,500-line relation kernel repeatedly combines Python loops with very small
NumPy arrays. A read-only `cProfile` spot check on the planning baseline found:

| Workload | Runtime and calls | Dominant path |
|---|---:|---|
| Prepare one non-planar eight-membered cycle 20 times | 6.183 s; 8,660,917 calls | surface enumeration, triangle-pair embedding and segment-triangle predicates |
| Screen 1,000 segments against one prepared non-planar eight-membered cycle | 1.554 s; 2,225,011 calls | `closest_cycle_edge()` and repeated segment-distance/local-scale work |
| Compute all distances for 200 points | 0.377 s; 577,703 calls | Python pair enumeration and repeated finiteness checks |

For the existing 187-complex benchmark, 406,103 bond--ring candidate pairs were
formed. AABB rejected 363,777 pairs, but 42,326 pairs still entered the exact
Python kernel. These figures are sufficient to justify native migration, but
the first implementation commit must add reproducible profile scripts before
claiming final speedups.

### 3.2 Computationally intensive units

| Priority | Work unit | Present cost mechanism | Native migration unit |
|---:|---|---|---|
| P0 | Non-planar cycle preparation | triangulation enumeration, embedded-surface tests and tens of thousands of tiny `np.cross`/dot operations | prepare all requested cycles in one C++ batch |
| P0 | Segment--cycle classification | candidate-pair Python iteration, exact triangle tests, evidence merging and a second closest-edge pass | one prepared-workspace batch returning state and requested evidence level |
| P0 | Metal placement evaluation | candidate × donor × atom × bond × cycle Python loops in `coordination.py` | one C++ placement engine operating on the whole structure |
| P0 | Repeated short relaxation | every repair step repacks the molecule, copies NumPy into vectors and rebuilds `OBMol` | one persistent native session and one `OBMol` lifecycle |
| P1 | Point-pair quality checks | Python $O(N^2)$ pair creation followed by scalar small-array operations | batched point-pair distance/threshold query |
| P1 | Point--segment and segment--segment clearances | repeated Python object construction and function dispatch | bulk primitive kernels plus spatial broad phase |
| P1 | Candidate generation/refinement | Python generators, small NumPy allocations and repeated deduplication | bounded C++ seed generation and three-variable refinement |
| P2 | General line relations, point-in-cycle and angle utilities | individually modest but frequently composed | native primitives exposed through the existing Python API |

The baseline functions that define these migration seams are:

| Current source | Current responsibility | Target owner |
|---|---|---|
| `geometry/relation.py:448-533` | SVD plane fitting and planarity evidence | `geometry/_native/cycle_surface.*` |
| `geometry/relation.py:901-997` | segment--triangle exact relation | `geometry/_native/segment_cycle.*` |
| `geometry/relation.py:1000-1379` | triangulation and embedded-surface proof | `geometry/_native/cycle_surface.*` |
| `geometry/relation.py:1839-2244` | prepared non-planar surface and batched segment--cycle relation | `geometry/_native/{cycle_surface,segment_cycle,batch}.*` |
| `geometry/relation.py:2388-2419` | point-pair distance enumeration | `geometry/_native/batch.*` |
| `geometry/relation.py:2504-2540` | closest cycle edge | the same native relation pass, not a second scan |
| `geometry/convert.py:532-611` | Python ring/bond batch loop and report aggregation | native batch plus a thin source-object mapper |
| `forcefields/coordination.py:137-278` | radius-normalized clearance and nested candidate checks | geometry facts plus forcefields-native policy |
| `forcefields/coordination.py:281-380` | proposal generation, ranking and immediate mutation | `forcefields/_native/placement_engine.*` |
| `forcefields/repair.py:422,546,1131` | repeated short native optimization calls | one `structure_session` lifecycle |
| `obWrappers/packing.py:34-147` | full molecule repacking before each call | one session-input packing after Stage 1 for composed Stage 2/3 execution |

For $C$ candidates, $D$ donors, $A$ atoms, $B$ bonds and $R$ rings, the current
placement path is approximately

$$
O\left(CD\left(A+B+R K_{surface}\right)\right),
$$

where $K_{surface}$ is the non-planar surface-classification cost. Moving only
`point_segment_distance()` behind pybind would not solve the problem: it would
replace Python arithmetic overhead with millions of Python/C++ calls. The
minimum useful native unit is the complete candidate batch and its prepared
geometry workspace.

## 4. Target source and runtime structure

```text
hotpot/cheminfo/
├── geometry/
│   ├── __init__.py                 # unchanged public exports
│   ├── object.py                   # public immutable value objects
│   ├── relation.py                 # public contracts and thin native adapters
│   ├── convert.py                  # chemical-object/source mapping and packing
│   ├── settings.py                 # public mathematical tolerances
│   ├── native.py                   # native loader and result conversion
│   ├── _geometry_native.pyi
│   └── _native/
│       ├── types.hpp
│       ├── tolerances.hpp
│       ├── primitives.hpp/.cpp     # distance, angle, projection, closest point
│       ├── spatial.hpp/.cpp        # AABB and batched neighbour queries
│       ├── cycle_surface.hpp/.cpp  # planar/non-planar prepared surfaces
│       ├── segment_cycle.hpp/.cpp  # factual three-state relation
│       ├── batch.hpp/.cpp          # coarse-grained workspaces and scans
│       └── bindings.cpp
│
├── forcefields/
│   ├── ligand.py                   # independent Stage 1 Python controller
│   ├── coordination.py             # stable Python Stage 2 facade
│   ├── native.py                   # independent stage facades and coordinator
│   ├── native_packing.py           # explicit stage/session boundary packing
│   ├── native_reports.py           # native stage results -> public contracts
│   ├── settings.py                 # public force-field policy configuration
│   └── _native/
│       ├── contracts.hpp
│       ├── stage_contracts.hpp
│       ├── structure_session.hpp/.cpp
│       ├── radii.hpp/.cpp
│       ├── target_selection.hpp/.cpp
│       ├── placement_evidence.hpp/.cpp
│       ├── placement_candidates.hpp/.cpp
│       ├── placement_policy.hpp/.cpp
│       ├── placement_engine.hpp/.cpp
│       ├── coordination_stage.hpp/.cpp
│       ├── untangling_engine.hpp/.cpp
│       ├── optimization_stage.hpp/.cpp
│       ├── workflow_coordinator.hpp/.cpp
│       ├── trajectory.hpp/.cpp
│       └── bindings.cpp
│
├── graph/_native/
│   └── relevant_cycles.*           # existing C++ topology implementation
│
└── obWrappers/_native/
    ├── molecule_data.*             # existing buffer value object, generalized
    ├── openbabel_adapter.*         # one native structure -> one OBMol
    ├── native_engine.*             # existing numerical engine
    └── rules and registry sources
```

Source ownership and binary packaging are deliberately distinguished:

- mathematical sources live under `geometry/_native`;
- scientific policy and workflow sources live under `forcefields/_native`;
- Open Babel-specific adapter code remains under `obWrappers/_native`;
- the force-field pipeline and Open Babel adapter must be linked into one
  Open Babel-containing extension and share one process-wide mutex/plugin
  state.

The first implementation should retain the current `_ob_native` binary as that
single Open Babel runtime and add the `forcefields/_native` sources to it. A
second independently linked `_ff_native` must not be introduced, because it
would create two Open Babel plugin states and two unrelated mutexes in one
process. Renaming the sole binary later is a packaging-only decision.

`_geometry_native` is a separate Open-Babel-free extension. Its stateless C++
geometry sources are also compiled into the force-field extension so the
force-field engine calls them directly. The initial implementation may compile
the same canonical geometry sources into both extensions; it must not fork or
copy the algorithms into a second source implementation. Changing the build
backend to create a shared native library is not coupled to this scientific
refactor.

## 5. Python/C++ boundary

### 5.1 Dual-entry, single-implementation contract

Every public computational operation migrated by this plan has one canonical
C++ implementation and two supported call paths:

```text
C++ geometry or forcefields caller ---------+
                                             v
                                  canonical C++ function
                                             ^
Python public function -> pybind adapter ----+
```

The following rules are acceptance requirements:

- the C++ function is declared in a reusable header and is directly callable
  by another C++ module without Python or pybind;
- the Python function binds or wraps that same C++ function and preserves its
  documented name, signature, return type and semantics;
- scalar and batch public operations remain callable from Python, while native
  hot loops use the batch/workspace form directly;
- no migrated operation retains a second Python numerical implementation,
  silent fallback or duplicated C++ algorithm;
- Python object adapters in `convert.py` may remain Python-only facades because
  they accept duck-typed `Molecule`, `Atom`, `Bond` and `Ring` objects, but all
  mathematical computation below those adapters uses a dual-entry native
  operation;
- underscore-prefixed algorithm steps do not automatically become public APIs.
  A private helper with an independently meaningful mathematical result must
  first be promoted deliberately; triangulation bookkeeping, barycentric
  branches, tolerance derivation and evidence-merging micro-helpers remain C++
  implementation details.

The stable Python geometry surface is the complete `geometry.__all__`, not only
the high-level bond--ring scan.  In particular, the existing scalar relation
functions and iterator/batch functions remain independently callable after the
cut-over.  The same rule applies to public force-field computations introduced
or migrated by this plan.

This is a source-level C++ API contract within Hotpot.  Installing headers and
guaranteeing a third-party binary ABI is outside this refactor unless approved
separately.

### 5.2 Responsibility boundary

| Responsibility | Python | C++ |
|---|---|---|
| Public `Molecule`, `Atom`, CLI and user-facing functions | Owns and preserves | Does not implement a persistent Hotpot molecule |
| Input conversion | Traverses Hotpot objects once and creates contiguous typed arrays | Validates shapes and creates a short-lived structure/session |
| User settings | Typed dataclasses and enums | Matching strongly typed POD structs |
| Relevant Cycle implementation | Public adapter only | Reuses the existing graph C++ implementation directly |
| Public geometry objects | Keeps `Point`, `Line`, `Segment`, `Plane`, `Triangle`, `Cycle` | Does not expose them inside hot loops |
| Distances, projections, angles and AABB | Thin batch/scalar facade | Sole numerical implementation |
| Planar/non-planar segment--cycle relation | Maps native state/evidence to public dataclasses | Sole numerical implementation |
| Radius and target-distance model | Exposes policy selection only | Selects the model and values |
| Atoms, bonds, rings and donor groups to inspect | Supplies declared donor identities/topology | Selects the actual check targets from the native structure |
| Placement candidates | No candidate loop | Generates, refines, deduplicates and evaluates all candidates |
| Chemical classification/ranking | Presents results | Produces `FULLY_FEASIBLE/PARTIAL/INFEASIBLE` and lexicographic rank |
| Metal translation and topology actions | Commits only the final returned state | Mutates the native working state |
| Ring opening, perturbation and restoration | No inner-loop control | Executes current bounded workflow |
| Open Babel setup/relaxation | Stable facade only | Uses the existing native engine in the same session |
| Trial-frame capture | Requests a retention mode | Records coordinates and scalar evidence without Python calls |
| Trajectory persistence and rendering | Writes NPZ/JSON/SDF and images | Never performs disk I/O |
| Warnings and public exceptions | Maps stable native status/error codes | Returns structured codes and preserves final finite state |
| Atomic update of the caller's molecule | Applies selected result after a successful native return | Never holds a Python object reference after return |

Stage independence and boundary frequency are separate concerns:

- a standalone Python stage call crosses pybind once in and once out and invokes
  the same C++ stage function available to native callers;
- the composed fast path may enter a native workflow coordinator once after
  Stage 1; that coordinator contains no scientific implementation of its own
  and calls the independent Stage 2 and Stage 3 C++ controllers directly;
- alternatively, Python may call Stage 2 and Stage 3 separately while both use
  one opaque workflow-scoped session, so the molecule is still packed once and
  `OBMol` is not rebuilt merely because a stage boundary was observed;
- no native hot loop calls back into Python.  Calls release the GIL while C++
  work is running.

The composed fast path is therefore an optimization over the same stage APIs,
not a separate backend and not the only supported route.

## 6. Mathematical geometry versus force-field science

| Subject | Geometry: mathematical fact | Forcefields native: scientific judgement/action |
|---|---|---|
| Point--point | distance, squared distance and finite-state flag | which atom pairs are relevant and whether their separation is acceptable |
| Point--segment | closest point, distance and segment parameter $t$ | whether a metal occupies a ligand-bond neighbourhood |
| Segment--segment | closest points, distance and two parameters | which chemical bonds obstruct a proposed coordination path |
| Angle | cosine, angle and degeneracy | whether an angle is a soft ranking feature or a declared-template constraint |
| Ratio | numerator divided by a supplied positive scale | which radius/target supplies that scale and what range is acceptable |
| AABB/spatial query | bounds, overlap and candidate indices | search radius, skin and selected chemical targets |
| Plane/cycle | planarity measurement, polygon position and surface evidence | whether a particular ring set is actionable in the workflow |
| Segment--cycle | `PIERCES`, `DOES_NOT_PIERCE`, `UNDETERMINED` plus evidence | reject, downgrade, warn or initiate untangling |
| Sphere/sphere-intersection/trilateration | candidate coordinates and residuals | target sphere radii and which candidate strategy is enabled |
| Relevant Cycles | graph-topological result, owned by `graph` | `ligand_skeleton`/`full_graph` scope and maximum actionable size (16) |
| Atomic/ionic/covalent radii | prohibited | source, unit, version and fallback policy |
| Donor identity | prohibited | supplied by the intended topology/CBond; FF selects groups and obstacles |
| Target M--D distance | prohibited | $d_i^*=s(r_M+r_{D_i})$ or a future versioned parameter provider |
| Chemical reasonableness | prohibited | placement and final-structure status |
| Candidate rank/selected frame | prohibited | explicit lexicographic policy |
| Coordinate or topology mutation | prohibited | placement, bond-state changes, perturbation and restoration |
| Force-field energy/gradient | prohibited | Open Babel backend execution and convergence policy |

Geometry C++ must therefore not contain identifiers or contracts named
`Metal`, `Donor`, `CovalentRadius`, `Reasonable`, `Accepted` or equivalent
scientific concepts. It operates on coordinates, indices, scales and numerical
tolerances only.

## 7. Native geometry API shape

The public Python geometry API remains stable, but it must dispatch to
coarse-grained native functions. The internal C++ API should be equivalent to:

```cpp
PreparedCycleBatch prepare_cycles(
    CoordinateView coordinates,
    IndexView cycle_indices,
    OffsetView cycle_offsets,
    const GeometrySettings& settings
);

SegmentCycleBatch screen_segments(
    const PreparedCycleBatch& cycles,
    SegmentView segments,
    PairView candidate_pairs,
    DetailLevel detail
);

PairDistanceBatch point_pair_distances(
    CoordinateView coordinates,
    OptionalPairView pairs
);
```

The public relation functions covered by the dual-entry contract are:

| Existing Python entry | Canonical C++ source entry or kernel |
|---|---|
| `measure_planarity()` | `measure_planarity(...)` |
| `determine_line_relation()` | `determine_line_relation(...)` |
| `line_distance()` | `line_distance(...)` |
| `point_segment_distance()` | `point_segment_measurement(...).distance` |
| `segment_segment_distance()` | `segment_segment_measurement(...).distance` |
| `point_pair_distances()` | `point_pair_distances(...)` |
| `find_point_pairs_below_distance()` | `find_point_pairs_below_distance(...)` |
| `locate_point_in_planar_cycle()` | `locate_point_in_planar_cycle(...)` |
| `iter_segment_cycle_screenings()` | `screen_segments(..., STATE_ONLY/ACTIONABLE)` |
| `iter_segment_cycle_relations()` | `screen_segments(..., FULL)` |
| `determine_segment_cycle_relation()` | scalar view of `screen_segments(..., FULL)` |
| `closest_cycle_edge()` | closest-edge evidence from the same prepared relation kernel |

The C++ names are source-contract names and may be finalized before Phase 1,
but the one-to-one semantic mapping is mandatory.  `Point`, `Line`, `Segment`,
`Plane`, `Triangle`, `Cycle`, enums and result dataclasses remain the Python
value layer.  The public `convert.py` functions remain callable and preserve
source-object association while delegating their numerical work to the same
native kernels.

`DetailLevel` prevents diagnostic materialization from consuming the speedup:

- `STATE_ONLY`: states, counts and AABB flags;
- `ACTIONABLE`: additionally features, indeterminacy causes and closest edge for
  non-`DOES_NOT_PIERCE` pairs;
- `FULL`: complete intersection points and surface-family evidence for public
  diagnostic APIs.

The force-field fast path uses `STATE_ONLY` or `ACTIONABLE`; the public
`determine_*` geometry functions request `FULL` only when their contract
requires it. Non-actionable pairs must not create Python objects.

Because the project remains C++17, views use a small internal `ArrayView<T>`;
they must not require C++20 `std::span`. Input arrays remain alive for the
duration of the call. Mutable coordinates are copied exactly once into the
native session; topology may remain a read-only view plus a native active-bond
mask.

## 8. Shared native state and explicit stage contracts

### 8.1 Shared session input

Stage 1 produces an explicit ligand-build artifact.  When Stage 2 is to run,
Python flattens the built ligand, metal specification and intended coordination
topology into one stable atom order and creates a workflow-scoped native
session:

```text
ComplexSessionInput
├── schema_version: int32
├── atomic_numbers: int32[N]
├── formal_charges: int32[N]
├── partial_charges: float64[N]
├── coordinates: float64[N, 3]
├── atom_aromatic: uint8[N]
├── ligand_bond_indices: int32[B, 2]
├── ligand_bond_orders: float64[B]
├── ligand_bond_kinds: uint8[B]
├── ligand_bond_aromatic: uint8[B]
├── metal_indices: int32[K]
├── intended_coordination_bonds: int32[C, 2]
├── intended_coordination_orders: float64[C]
├── intended_coordination_kinds: uint8[C]
├── optional_unit_cell: float64[6]
└── session and stage options
```

Even if the first public API accepts one metal, the native schema uses
`metal_indices[K]` so multi-metal support does not require another ABI.
Ligand covalent bonds and intended coordination bonds must be separate:

- Stage 2 controls when coordination bonds become active;
- candidate rings must be perceived from the pre-addition topology;
- the current Open Babel `MoleculeData` cannot silently encode every dative
  bond as an ordinary bond without losing Hotpot semantics.

Ligand component membership is derived once inside C++ from the ligand-bond
graph.  It is not accepted as a second caller-supplied source of graph truth.
The intended coordination-bond records preserve directed metal/donor indices,
bond order and bond kind; trajectory topology may canonicalize endpoint order
only when it serializes a topology revision.

The native session obtains Relevant Cycles directly from the existing C++ graph
implementation. It must not substitute Open Babel ring perception.

### 8.2 Independent and composed entries

The target controller contracts are conceptually:

```cpp
CoordinationStageResult restore_coordination(
    StructureSession&, const CoordinationStageOptions&
);
ComplexOptimizationResult optimize_complex(
    StructureSession&, const ComplexOptimizationOptions&
);
```

Two explicit factories establish the initial topology instead of inferring it
from an ambiguous mask: `create_coordination_session(...)` starts with all
intended coordination bonds inactive, while
`create_optimization_session(...)` starts from an already assembled complex
with those bonds active. The session is non-copyable and exposes neither its
`OBMol` nor writable coordinate storage through Python. Component membership
is derived from the ligand covalent graph when the session is created.

Stage 1 retains its spawn-safe Python controller and calls the sole native
Open Babel `build` and `single_optimize` backends.  Each migrated Stage 2/3
computational entry is bound to Python.  Exact new names are frozen before
Phase 5, but the independent contracts are not optional.  Existing public
facades retain their behaviour:

```python
build_report = ff.build_complex3d(mol, ...)  # existing Stage 1 + Stage 2 facade
optimization_report = ff.optimize_complex(mol, ...)  # existing Stage 3 facade
workflow_report = ff.complexes_build(mol, ...)  # explicit three-stage composition
```

Phase 5 introduces only real input, session, snapshot, option, result and
trajectory round trips. It must not publish placeholder functions named
`restore_coordination()` or `optimize_complex()` before their scientific
controllers exist in Phase 7.

The full-workflow fast path may enter one native coordinator after Stage 1.
That coordinator only calls the independent Stage 2 and Stage 3 C++ functions
on a shared `StructureSession`; it contains no duplicate placement, repair or
optimization logic.  A caller may instead invoke either stage independently,
in which case the Python facade creates or accepts a session and calls the same
C++ function.

Stage 1 retains its current spawned-worker timeout boundary unless a later plan
explicitly changes it.  Its Open Babel computational operations remain native;
this refactor does not merge the Stage 1 controller into the Stage 2/3 session.
For a ligand-plus-metal facade, the Python result contains a newly constructed
complex `Molecule`; the input ligand is not mutated.  An already assembled
complex preserves the current working-copy and atomic-commit behaviour.

### 8.3 Stage and workflow outputs

```text
BuildWorkerResult (existing Python Stage 1 contract)
├── coordinates
├── ligand_build_report
└── optional ligand trajectory

CoordinationStageResult
├── selected and terminal coordinates
├── final_active_coordination_mask
├── placement_report
├── coordination_restoration_report
└── optional Stage 2 trajectory

ComplexOptimizationResult
├── selected and terminal coordinates
├── ring_untangling_report
├── optimization_report
├── final_acceptance_report
├── warning_codes
└── optional Stage 3 trajectory

ComplexWorkflowResult
├── Stage 1, Stage 2 and Stage 3 reports without flattening ownership
├── final selected/terminal structure
└── optional combined NativeTrajectoryBatch
    ├── coordinates: float64[F, N, 3]
    ├── stage/event/attempt/step: integer arrays
    ├── energy/rms_gradient/max_gradient: float64 arrays
    ├── topology_revision: int32[F]
    └── compact topology revisions or active-bond masks
```

The native frame-detail request is an enum rather than several booleans:

- `NONE`: no additional diagnostic attempt frames beyond the facts required by
  the existing trajectory contract;
- `OPTIMIZATION`: additionally retain optimizer epoch coordinates where the
  existing `retain_epoch_history` contract requests them;
- `ALL_ATTEMPTS`: additionally include the input structure and every placement
  proposal selected for exact evaluation.

The existing trajectory always records the events selected by
`TrajectoryStart`; `save_movie` does not turn that factual record on or off.
It currently controls conformer materialization and whether optimizer epoch
history is requested.  Native batching must preserve this distinction.  The
existing public `save_movie=False/True` option therefore controls epoch-frame
detail and final materialization, while a separate explicit diagnostic option
may request `ALL_ATTEMPTS` without changing which structure is selected.

When `retain_frames=False`, C++ keeps only the current, selected, terminal and
minimal rollback states. When enabled, raw coordinate storage is approximately

$$
8\times 3\times N\times F\ \text{bytes},
$$

before metadata. Frames cross pybind once as batched arrays. Python adds
`ForceFieldTrajectory.ingest_native_batch(...)` and remains responsible for
coordinate/topology deduplication, compressed persistence and rendering.

## 9. Metal-placement checks and actions

All geometry columns below are computed by `geometry/_native`. All decisions
and actions are made by `forcefields/_native`.

| Check | Geometry fact | Force-field rule | Native action |
|---|---|---|---|
| Finite candidate | finite mask for the proposed point and relevant coordinates | non-finite candidates are unusable | discard candidate and count the reason |
| Metal--non-donor atom | $d_{MA}=\|M-A\|$ | hard clash when $d_{MA}<\max(0.50\,\text{Å},0.55(r_M+r_A))$ | discard candidate |
| Metal--ligand bond | point--segment distance, closest point and parameter $t$ | an interior contact with an unrelated covalent bond is a hard obstruction | discard candidate |
| Other metal | point--point distance | apply the configured metal--metal exclusion model unless explicitly bridged | discard or downgrade according to declared topology |
| Local crowding | raw neighbour distances inside a broad-phase radius | compute radius-normalized crowding; it is not called an energy | use only for ranking |
| Metal--donor distance | $d_i=\|M-D_i\|$ and ratio to a supplied scale | target provider chooses $d_i^*$; current broad reachability remains $0.65\le d_i/d_i^*\le1.60$ | mark each donor reachable/unreachable |
| Two-donor feasibility | sphere separation/intersection facts | required target shells must be mutually reachable in the frozen ligand frame | choose exact intersection seeds or mark the pair partial |
| Multi-donor fit | residuals from sphere/trilateration candidates | evaluate required donor and chelate-group coverage | refine promising seeds and classify coverage |
| M--D path versus atom | point--segment distance, closest point and $t$ | unrelated atoms obstruct only the interior path; the intended endpoint is excluded | reject that donor path or downgrade the candidate |
| M--D path versus bond | segment--segment distance and both parameters | expected shared endpoint contact is allowed; an unrelated interior crossing is not | reject that donor path or downgrade the candidate |
| M--D path versus ring | `PIERCES/DOES_NOT_PIERCE/UNDETERMINED` and evidence | `PIERCES` rejects the path; `UNDETERMINED` cannot be called safe | reject or classify as `PARTIAL`; record warning evidence |
| Donor approach direction | $\angle(M-D_i-X)$ for donor neighbours | no universal N/O/S/π-donor hard threshold in this stage | soft ranking only |
| Donor angular distribution | $\angle(D_i-M-D_j)$ and Gram/eigenvalue facts | without an explicit geometry template, do not impose tetrahedral/octahedral ideals | soft ranking; template policy remains optional |
| Ligand/chelate-group coverage | index membership and reachable counts | a candidate covering one donor must not hide failure of another required group | determine `FULLY_FEASIBLE` versus `PARTIAL` |
| Ring scope | relevant-cycle indices and sizes | placement uses ligand-skeleton rings; rings over 16 are reported but not repaired | select the native cycle workspace and warning codes |
| Displacement | $\|M_{candidate}-M_{input}\|$ | equal-quality candidates should move the metal less | final tie-break |

The broad-phase radius is not a chemical acceptance threshold. A spatial query
uses the largest pair-specific exclusion cutoff plus a configurable skin; every
survivor is checked with its own elemental cutoff.

Candidates are classified as:

- `FULLY_FEASIBLE`: no hard center collision and every required donor/group has
  a non-piercing reachable path;
- `PARTIAL`: the center itself is usable and at least one required path is
  usable, but coverage is incomplete or mathematically undetermined;
- `INFEASIBLE`: hard center collision remains, no donor is reachable, or every
  candidate path is definitely blocked.

Ranking is lexicographic, never a single opaque weighted score:

1. placement status;
2. covered ligand/chelate groups;
3. reachable donor count;
4. definite piercing and hard-collision counts;
5. minimum normalized atom/bond clearance;
6. worst donor-distance deviation;
7. RMS donor-distance deviation;
8. undetermined-relation count;
9. angular dispersion as a tie-break only;
10. displacement from the input position;
11. deterministic proposal sequence.

### 9.1 Candidate generation and action sequence

```text
evaluate input metal coordinate as candidate 0
    |
    +-- FULLY_FEASIBLE --> retain it without movement
    |
    `-- otherwise
          |
          +-- one donor: target sphere and local open directions
          +-- two donors: target-sphere intersection circle
          +-- three or more: deterministic multi-start distance least squares
          +-- centroid/outward directions and Fibonacci fallback
          |
          v
       batch broad phase
          |
          v
       exact geometry facts for survivors
          |
          v
       force-field classification and lexicographic ranking
          |
          +-- best FULLY_FEASIBLE --> apply once
          +-- no full result but PARTIAL exists --> apply best partial and warn
          `-- no usable result --> retain best finite evidence, return infeasible
```

Candidate evaluation is immutable. Only the selected coordinate is committed
to the native working structure. Placement does not hide/restore bonds, invoke
Open Babel, control retries or write trajectory files; the enclosing native
Stage 2 controller performs the subsequent restoration and relaxation.  The
workflow coordinator never reaches into placement policy or duplicates it.

## 10. End-to-end staged workflow

The final target preserves three explicit scientific stages.  The composed
workflow may reuse native state, but no stage implicitly enters the next one:

```text
Stage 1 -- ligand construction
    spawned worker boundary retained
    build -> candidate relaxation -> ligand ring untangling
    returns the existing Python BuildWorkerResult
        |
        | explicit result; workflow may stop and inspect here
        v
create ComplexSessionInput and one native StructureSession
        |
        v
Stage 2 -- coordination construction/restoration
    MetalPlacementEngine
      -> incremental restoration with current approved semantics
      -> Stage 2 validation
    returns CoordinationStageResult
        |
        | explicit result; workflow may stop and inspect here
        | composed path retains the same StructureSession/OBMol
        v
Stage 3 -- full-complex optimization
    full-graph ring checkpoint
      -> bounded ring-opening/perturb/relax/restore when required
      -> full numerical optimization
      -> final checkpoint and current acceptance policy
    returns ComplexOptimizationResult
        |
        v
workflow coordinator combines reports only
        |
        v
Python maps reports, writes warnings/files and atomically updates Molecule
```

Each stage owns its control loop, options, result, report, warnings, trajectory
events and termination policy.  The coordinator may sequence stages and pass an
explicit artifact or `StructureSession&`; it must not implement placement,
restoration, untangling or optimization decisions itself.

There are two supported execution forms backed by the same stage functions:

- **independent stage call:** Python crosses pybind for that stage and receives
  its stage result;
- **composed fast path:** after Stage 1, one native coordinator calls the Stage
  2 and Stage 3 C++ functions directly on one session and returns both unmerged
  stage reports together.

The composed form does not replace or weaken the independent entries.  Stage 2
and Stage 3 may also be called separately against one explicit opaque session,
avoiding repacking while preserving a visible boundary.

For Stage 2 and Stage 3 composition, Open Babel must be constructed once per
native session and updated in place where its API safely permits.  Stage 1
retains its separate worker lifetime.  Any necessary topology rebuild must be
explicit in the owning stage report; it must not silently re-enter Python and
repack the whole molecule.

Failures preserve the last finite coordinates and any requested frames. Python
emits warnings and returns inspectable structures according to the existing
business contract; native numerical failure must not erase the trajectory.

## 11. Implementation phases and commit boundaries

Each numbered phase is a separate, revertible commit or short commit series.
No phase may mix a mathematical semantic change with a performance rewrite.

### Phase 0: characterization fence

- Add reproducible cProfile/native timing scripts under tests or benchmarks.
- Freeze the current three-stage call graph, public API inventory, stage reports
  and the nominal $4L+K+1$ native-call baseline.
- Freeze geometry golden results including state, features, causes, evidence
  counters, intersection points and closest-edge tie-breaks.
- Record current Python/C++ crossing counts, OBMol rebuild counts, Stage 2/3
  times and peak memory on selected small/large/long-tail cases.

### Phase 1: native geometry primitives

- Add C++ value types, dimension-aware tolerances, distances, projections,
  angles and AABB batch kernels.
- Add direct C++ unit tests, `_geometry_native` Python tests and cross-entry
  differential tests for every migrated public operation.
- Keep public Python signatures and return dataclasses unchanged.

### Phase 2: native planar cycle relations

- Port plane fitting while preserving the documented non-canonical normal-sign
  contract, together with 2D projection, polygon simplicity/contact and planar
  segment--cycle relations.
- Switch only after exact state/evidence parity.

### Phase 3: native non-planar cycle relations

- Port triangulation order, embedded-surface proof, triangle-pair and
  segment-triangle kernels, budgets and consensus state.
- Preserve enumeration and comparison order exactly.
- Add batched workspace and detail-level outputs.

### Phase 4: geometry Python cut-over

- Convert `relation.py` to public wrappers and result mapping.
- Convert `convert.py` to one-shot packed batch calls while retaining source
  object association.
- Verify that every item in `geometry.__all__` remains importable and that every
  migrated public computation reaches its unique C++ backend.
- Remove replaced Python numerical kernels after parity; do not retain a hidden
  fallback.

### Phase 5: native session and stage-contract seams

- Add `ComplexSessionInput`, `CoordinationStageResult`,
  `ComplexOptimizationResult` and composed-result contracts.  Stage 1 retains
  its existing Python `BuildWorkerResult`/`ComplexBuildDiagnostics` contracts.
- Reuse the existing molecule buffer fields and add separated intended
  coordination bonds, metal indices and component IDs.
- Expose direct C++ and Python-bound entries for each migrated stage
  computation, all backed by the same C++ functions.
- Prove each stage round-trip independently, then prove shared-session
  composition and optional batched-frame transport without changing scientific
  behaviour.

### Phase 6: native metal placement

- Implement target/radius policy, obstacle selection and the full check matrix.
- Implement current-position assessment, single/two/multi-donor candidate
  generation, bounded refinement, classification and lexicographic ranking.
- Keep placement as an isolated engine; integrate it before the first Stage 2
  bond is restored.

### Phase 7: independent native stages with session reuse

- Link force-field orchestration sources into the sole Open Babel extension.
- Move current Stage 2 control and short relaxation into its own controller and
  commit/test it independently, without adding deferred transaction behaviour.
- Move the full-complex ring checkpoint/untangling and Stage 3 optimization
  into a separate controller and commit/test it independently.
- Add a thin workflow coordinator that sequences those controllers and may pass
  the same `StructureSession&`; it must contain no duplicated stage logic.
- Eliminate repeated molecule packing and avoidable OBMol reconstruction.

### Phase 8: trajectory and result integration

- Add native stage/event/evidence contracts and batch frame export.
- Add `ForceFieldTrajectory.ingest_native_batch(...)`.
- Preserve current selected-versus-terminal semantics, topology revisions,
  warning codes and failure artifacts.

### Phase 9: public workflow switch and cleanup

- Make existing independent and composed complex APIs use the same native stage
  controllers.  The composed facade must not replace the independent entries.
- Delete replaced Python candidate loops, numerical geometry kernels, repeated
  native-call adapters and duplicate thresholds.
- Keep no unconditional `try/except` or alternative compatibility path.

### Phase 10: validation and performance report

- Run focused unit/property/differential tests, the complete project suite and
  Python 3.9--3.14 build matrix.
- Run the full 187-complex standard benchmark with retained trajectories and
  final PNGs.
- Publish an ablation report for every phase: absolute time, relative speedup,
  native call count, OBMol rebuild count, exact kernel count and memory.

## 12. Verification gates

### 12.1 Geometry correctness

- scalar and batch results are identical;
- every migrated public computation is directly testable through C++ and
  Python, and both paths reach the same implementation and agree;
- Python baseline and C++ agree on every discrete state, feature and
  indeterminacy cause;
- numeric fields agree within the documented dimension-aware tolerance;
- rigid translation/rotation and uniform scaling preserve the required states;
- AABB broad phase has zero false negatives;
- planar, non-planar, coplanar, endpoint, boundary, line-extension, degenerate,
  non-finite and budget-exhausted cases are covered;
- ring sizes 3--8 exercise complete surface enumeration; 9--16 exercise the
  documented incomplete/undetermined path.

The C++ build must not use `-ffast-math`: IEEE NaN/Inf behaviour and numerical
guard bands are part of the geometry contract.  The existing public contract
does not fix the sign of the best-fit-plane normal, so differential checks
compare $\mathbf{n}$ and $-\mathbf{n}$ as equivalent and the migration must not
introduce a new sign promise. Non-planar triangulation and budget order, and
closest-edge tie-breaking, must match the current definitions.

### 12.2 Placement correctness

- an already fully feasible input position remains unchanged;
- case 109's initial Eu--C collision is detected before any coordination bond
  is restored and the selected position removes it;
- single-, double- and multi-donor generators are deterministic under a seed;
- all intended donor and chelate-group failures remain visible in evidence;
- `PIERCES` rejects a path and `UNDETERMINED` produces `PARTIAL`, not success;
- donor-incident bonds permit their shared endpoint but not an interior
  crossing;
- other metals and hydrogen atoms participate in collision checks;
- `retain_frames` changes only retained data, never the selected coordinates;
- no acceptable candidate returns the best finite partial evidence and a
  structured warning code.

### 12.3 Scientific regression

- the complete existing test suite passes;
- geometry public API names and value-object types remain available;
- Stage 1, Stage 2 and Stage 3 remain independently invocable, testable and
  reportable, while the composed workflow produces the same staged results;
- the Stage 1 spawned-worker timeout boundary remains intact;
- Relevant Cycles remain Hotpot's native graph result, not Open Babel rings;
- current 175 passing structures among 178 CBond-success cases do not regress;
- case 61 remains a declared non-goal and must not be made to pass by weakening
  thresholds;
- case 54 is monitored but not claimed fixed: its observed collapse happens
  after bond restoration, and this stage intentionally does not add the
  deferred post-relaxation transaction gate;
- all returned final structures and retained trajectories remain readable and
  have the intended topology.

### 12.4 Performance evidence

The report must separately measure:

- geometry packing, cycle preparation, AABB, exact planar/non-planar kernels
  and Python materialization;
- placement candidate generation, broad phase, exact checks and ranking;
- Stage 2, ring untangling, Stage 3 and final validation;
- Python/C++ call count and OBMol construction count per complex;
- P50, P95, maximum and total 187-case wall time;
- peak resident memory with and without `retain_frames`.

Required structural performance properties are:

- the composed fast path performs one Python-to-native entry and one return
  after Stage 1, while retaining independent Stage 2 and Stage 3 entries;
- an independently invoked stage performs one bounded request/return and never
  invokes a Python scalar kernel from its native hot loop;
- no Python objects or NumPy allocations in candidate/geometry inner loops;
- no Python callbacks while the GIL is released;
- Stage 2 and Stage 3 can share one native molecular session, with every
  unavoidable `OBMol` rebuild counted;
- no nested default threading; the standard benchmark continues using its
  existing case-level process parallelism.

Numeric speedup targets are recorded only after Phase 0 establishes stable
baselines. Correctness and chemical pass/fail equivalence take precedence over
an arbitrary target multiplier.

## 13. Principal risks and controls

| Risk | Control |
|---|---|
| Non-planar three-state semantics drift during C++ rewrite | golden differential corpus; preserve enumeration/comparison order; separate planar and non-planar commits |
| AABB incorrectly proves a relation for an invalid surface | retain the current rule that AABB proves non-piercing only after surface validity/completeness is known |
| Native and Python defaults diverge | Python settings are the sole public configuration and are passed once into native POD values |
| Radius policy becomes an undocumented table | version the provider, source every value, record provider ID in results; do not use the currently unverified ionic-radius table |
| One total score hides a collision | retain hard gates and a named lexicographic rank tuple |
| Two Open Babel extensions create independent locks/plugin states | ship one Open Babel-linked binary and one process-wide recursive mutex |
| Full frame retention consumes excessive memory | default off; batch allocation only when requested; report frame count and estimated bytes |
| Frozen ligand geometry cannot satisfy all donors | return `PARTIAL` with complete evidence; do not call it chemically impossible |
| Native rewrite changes case61 implicitly | preserve current Stage 2 semantics and keep case61 as an explicit non-goal |
| Shared session silently merges Stage 2 and Stage 3 | separate controllers, contracts, reports and tests; coordinator contains sequencing only |
| Python and C++ public paths become two implementations | bind the canonical C++ function directly and enforce cross-entry differential tests |
| Python compatibility code survives indefinitely | remove the replaced implementation in the same cut-over phase; no silent fallback |

## 14. Completion definition

This refactor is complete only when all of the following are true:

1. Every migrated public computational function has a direct C++ source entry
   and a Python entry backed by that same sole C++ implementation; existing
   public Python contracts remain intact.
2. Stage 1 ligand construction, Stage 2 coordination restoration and Stage 3
   full-complex optimization retain separate controllers, contracts, reports,
   callable entries and focused tests.
3. The composed fast path reuses the independent Stage 2 and Stage 3 functions
   and may share one native session; the coordinator contains no scientific
   implementation of its own.
4. Metal placement is a named, independently testable native component rather
   than logic mixed into restoration or optimization loops.
5. Radius/target selection, candidate classification and all coordinate or
   topology actions are confined to forcefields native code.
6. Geometry native contains no chemical value judgement.
7. The final structure, selected and terminal coordinates, structured reports
   and optional complete frame batch are returned even for inspectable failure
   outcomes.
8. Replaced Python hot loops and duplicate policy constants are removed.
9. Python 3.9--3.14 native builds, the complete test suite and the 187-case
   benchmark pass the stated gates.
10. The implementation report provides per-phase profile evidence rather than
   attributing all improvement to the rewrite as a whole.
