# Native Geometry and Force-Field Implementation Report

Date: 2026-09-30

Status: production implementation complete; in-scope Phase 10 validation
completed, with declared project-global and historical-ablation limits

Approved plan: [`native_geometry_forcefield_refactor.md`](native_geometry_forcefield_refactor.md)

Planning baseline: `9c70ef63f337`

This report records what was implemented, how the runtime boundary changed,
which commits belong to each implementation phase, and which performance
statements are supported by the final validation. Detailed commands, raw-result
summaries and case-level analysis are in
[`native_geometry_forcefield_test_report.md`](native_geometry_forcefield_test_report.md),
and supported entry points are documented in
[`native_api_and_usage.md`](native_api_and_usage.md).

## 1. Scope and fixed decisions

The implementation covers two related migrations:

1. move high-frequency Euclidean geometry calculations from Python and small
   NumPy operations to canonical C++ kernels while preserving the public
   `hotpot.cheminfo.geometry` API;
2. move coordination placement, incremental coordination restoration,
   ring--bond untangling, full-complex optimization and their shared runtime
   state into coarse-grained native force-field stages.

The following design decisions were preserved:

- `geometry` reports mathematical facts only. Chemical acceptance, radius
  policy, donor targets, repair choices and structure mutation remain owned by
  `forcefields`.
- Stage 1 ligand construction, Stage 2 coordination restoration and Stage 3
  full-complex optimization remain separate controllers with separate
  contracts and testable entries.
- Stage 1 remains in its spawned Python worker. After Stage 1, the composed
  path packs one typed session input and enters one native Stage 2 + Stage 3
  workflow call.
- Independent native Stage 2 and Stage 3 Python entries remain available.
  The composed coordinator invokes those same C++ stage functions; it is not a
  second scientific implementation.
- The native working structure is workflow-scoped. No persistent `HpMol` or
  second Hotpot molecule model was introduced.
- Public Python geometry calls and direct C++ calls use the same C++ source
  implementation. The force-field extension compiles those geometry sources
  directly so its hot loops do not call Python or cross into another Python
  extension.
- The Open Babel-linked native force-field code is part of the existing
  `hotpot.cheminfo.obWrappers._ob_native` extension. It does not introduce a
  second Open Babel runtime.
- Case 61 and transactional post-restoration coordination-bond acceptance are
  outside this migration's scientific scope. No threshold was weakened to
  make that case pass.

## 2. Resulting source layout

The implemented source boundary is:

```text
hotpot/cheminfo/
├── geometry/
│   ├── __init__.py                 # stable public Python exports
│   ├── object.py                   # public geometry value objects
│   ├── settings.py                 # public numerical settings
│   ├── relation.py                 # thin result mapping/public wrappers
│   ├── convert.py                  # chemical-object adapters
│   ├── native.py                   # explicit Python/native facade
│   ├── _geometry_native.pyi        # Python typing surface
│   └── _native/
│       ├── primitives.*            # vector, distance and angle facts
│       ├── spatial.*               # AABB and spatial broad phase
│       ├── planar_predicates.*     # planar predicates
│       ├── triangle_predicates.*   # triangle facts
│       ├── cycle_surface.*         # planarity and cycle surface entry
│       ├── nonplanar_surface.*     # embedded-surface enumeration
│       ├── nonplanar_segment.*     # segment/nonplanar-surface relation
│       ├── prepared_cycle.*        # prepared-cycle dispatch
│       ├── segment_cycle.*         # scalar segment--cycle relation
│       ├── batch.*                 # packed batch screening
│       ├── construction.*          # native construction utilities
│       └── bindings.cpp            # Python entry to the canonical kernels
├── graph/
│   └── _native/
│       └── relevant_cycles.*        # Relevant Cycles C++ kernel and binding
├── forcefields/
│   ├── workflows.py                # public workflow composition/commit
│   ├── native.py                   # independent native stage facade
│   ├── native_packing.py           # one typed session-input boundary
│   ├── native_reports.py           # native fact/result conversion
│   ├── native_adapters.py          # public force-field report adaptation
│   ├── trajectory.py               # Python persistence and native ingestion
│   ├── ligand.py                   # retained Stage 1 controller
│   └── _native/
│       ├── structure_session.*      # workflow-scoped native structure/OBMol
│       ├── topology_workspace.*     # Relevant Cycle and ring workspace
│       ├── placement_policy.*       # force-field scientific policy
│       ├── placement_candidates.*   # candidate generation/refinement
│       ├── placement_engine.*       # placement checks, ranking and action
│       ├── coordination_stage.*     # independent Stage 2 controller
│       ├── untangling_engine.*      # native targeted ring repair
│       ├── optimization_stage.*     # independent Stage 3 controller
│       ├── workflow_stage.*         # thin Stage 2 -> Stage 3 coordinator
│       ├── trajectory.*             # batched native frame/evidence records
│       ├── stage_contracts.*        # explicit native result contracts
│       └── bindings.*               # bindings compiled into `_ob_native`
└── obWrappers/
    ├── _ob_native.pyi               # complete native typing surface
    └── _native/
        ├── native_engine.*           # ordinary Open Babel operations
        ├── openbabel_adapter.*       # typed buffers -> OBMol
        ├── registry.*                # Open Babel corrections registry
        └── native_bindings.cpp       # sole Open Babel extension module
```

`setup.py` builds three extension modules:

- `hotpot.cheminfo.geometry._geometry_native`, which exposes the public
  geometry Python entry;
- `hotpot.cheminfo.graph._relevant_cycles`, which exposes the Relevant Cycles
  graph kernel;
- `hotpot.cheminfo.obWrappers._ob_native`, which contains the Open Babel
  engine, the native force-field stages and the same canonical geometry C++
  sources needed by the native hot loops.

The geometry sources are one mathematical implementation compiled into the
geometry extension and again inside the Open Babel/force-field extension for
two entry boundaries; they are not two independently maintained geometry
algorithms. The third extension is the separate Relevant Cycles graph kernel.

## 3. Architecture before and after

### 3.1 Planning baseline

At `9c70ef63f337`, the nominal complex workflow was:

```text
Python caller
  |
  +-- Stage 1 Python child
  |     repeated pack -> pybind -> construct OBMol -> build/short optimize
  |     returns coordinates, diagnostics and optional trajectories
  |
  +-- Stage 2 Python controller
  |     Python placement + Python geometry + Python bond-restoration loop
  |     repeated pack -> pybind -> construct OBMol -> short optimize
  |
  `-- Stage 3 Python controller
        Python checkpoints + Python untangling orchestration
        pack -> pybind -> construct OBMol -> native optimizer segment(s)
```

For a nominal first-candidate success with no piercing, (L) ligand
components and (K) intended coordination bonds produced the documented
native-call model:

\[
N_{\mathrm{native,baseline}}=4L+K+1.
\]

Each call independently repacked the Python molecule and constructed an
`OBMol`. Repair and stalled-relaxation paths added further calls.

### 3.2 Implemented architecture

The implemented composed path is:

```text
Python caller
  |
  +-- Stage 1 Python child
  |     ligand build remains isolated and returns its explicit artifact
  |
  `-- parent Python
        pack one ComplexSessionInput
        |
        `-- one Python -> `_ob_native.run_complex_workflow(...)` call
              |
              +-- construct one StructureSession / working OBMol
              |
              +-- Stage 2 C++ controller
              |     placement -> candidate screening -> restoration -> relax
              |     returns an independent CoordinationStageResult
              |
              `-- Stage 3 C++ controller on the same session
                    checkpoint -> untangle -> optimize -> final checkpoint
                    returns an independent ComplexOptimizationResult
        |
        +-- map native reports and ingest one batched trajectory
        `-- atomically commit the selected structure to the caller molecule
```

The direct entries remain:

```text
Python restore_coordination(...) -> canonical C++ Stage 2 function
Python optimize_complex(...)     -> canonical C++ Stage 3 function
C++ workflow coordinator         -> the same Stage 2 and Stage 3 functions
```

The architecture therefore reduces boundary crossings without merging the
scientific stages. Stage reports, selected and terminal coordinates, warnings,
checkpoint evidence and trajectory events remain separate and inspectable.

### 3.3 Structural comparison

| Concern | Planning baseline | Implemented structure | Evidence type |
|---|---|---|---|
| Geometry arithmetic | Python loops and many small NumPy operations | canonical C++ scalar/prepared/batch kernels | source and differential tests |
| Geometry use inside FF | Python calls | direct C++ calls to compiled geometry sources | build/source structure |
| Stage 2 placement | Python policy and candidate loop | isolated native placement engine | source and focused tests |
| Stage 2 restoration | Python loop with repeated native calls | one native stage controller | source and focused tests |
| Stage 3 ring repair | Python orchestration | native checkpoint/untangling controller | source and focused tests |
| Stage 3 numerical loop | already native per optimizer call | native stage on shared session | source and focused tests |
| Stage 2 -> Stage 3 state | Python `Molecule` handoff | explicit results over one workflow session | source and composition tests |
| Composed boundary | repeated calls | one native request/return after Stage 1 | routing/source tests; no production counter emitted |
| Trajectory transfer | Python frame-by-frame production | native batch, one Python ingestion step | source and ingestion tests |
| Failure after completed Stage 2 | generic setup failure could lose Stage 2 context | typed failure carries completed Stage 2 diagnostics/frames | focused failure tests |
| Legacy Python Stage 2/3 implementations | active | removed after public cut-over | source and stale-symbol tests |

The final column distinguishes structural verification from timed performance.
A source-level one-call architecture does not by itself quantify speedup.

## 4. Python/C++ responsibility boundary

| Responsibility | Python owner | C++ owner |
|---|---|---|
| Public `Molecule`, Atom identity and user-facing API | yes | no persistent Hotpot object |
| Stage 1 child-process lifecycle and timeout | yes | Open Babel operations only |
| One-shot session input packing | traverses Hotpot objects and creates contiguous typed arrays | validates and copies/owns native values |
| Geometry settings | public dataclasses/enums | matching typed values and sole calculations |
| Point, line, segment, cycle facts | public value/result mapping | sole numerical predicates |
| Relevant Cycle topology | public Hotpot graph contract | consumes native cycle results/workspace |
| Radius and target policy | selects public option | owns values, checks and ranking |
| Metal placement candidates | no hot-loop candidate work | generation, refinement, classification and selection |
| Coordination restoration | public invocation/result mapping | topology mutation and bounded relaxation loop |
| Ring--bond untangling | public invocation/result mapping | checkpoint, watch set, topology actions and relaxation |
| Full-complex optimization | public options and report mapping | persistent-session Open Babel execution |
| Stage 2/3 composition | invokes coordinator | sequences independent native stage functions |
| Optional frame collection | requests detail/retention | records native batch without Python callbacks |
| Trajectory serialization/rendering | NPZ/JSON/SDF/image I/O | no disk I/O |
| Warning and exception presentation | maps stable native evidence | returns typed codes and preserved finite state |
| Caller mutation | atomically applies selected result | never retains Python object references |

The hot-loop boundary is intentionally coarse. Public small geometry functions
remain callable from Python, but native forcefields call their C++ sources
directly instead of making millions of Python or pybind calls.

## 5. Phase 0--9 implementation commits

The grouping below is functional. Some test fences and plan corrections were
committed immediately before the production code they constrain, so commit
order across phase headings is not a claim that phases were strictly linear.
Every listed commit is independently revertible in the Git history.

The plan itself was approved in `4a8cd3c` (`docs(plan): define native geometry
and staged force-field refactor`).

### Phase 0 -- characterization fence

| Commit | Change |
|---|---|
| `5931d44` | froze the geometry migration golden corpus and added the reusable public-boundary profile script |
| `49d89a0` | recorded the non-canonical best-fit-plane normal-sign contract before migration |
| `f3ab247` | expanded the frozen nonplanar surface semantics before native implementation |

The committed Phase 0 assets establish correctness and provide a profiling
entry, but no Phase 0 profile JSON was committed. Boundary-call, `OBMol`
construction and RSS counters requested by the plan were not captured as a
machine-readable Phase 0 artifact.

### Phase 1 -- native geometry primitives

| Commit | Change |
|---|---|
| `6806276` | added native vector, distance, tolerance and spatial primitive kernels |
| `6d5a03b` | expanded the planar migration characterization fence |

### Phase 2 -- native planar cycle relations

| Commit | Change |
|---|---|
| `1ddd7eb` | implemented native planar cycle kernels |
| `ddad5cb` | verified parity with the frozen Python planar source semantics |

### Phase 3 -- native nonplanar and prepared/batch relations

| Commit | Change |
|---|---|
| `5d286dd` | implemented native nonplanar surface preparation |
| `a688123` | added surface-preparation parity tests |
| `756307e` | implemented native nonplanar segment classification |
| `7d8556e` | added nonplanar relation parity tests |
| `565a06f` | fixed finite-segment boundary handling so extension-only hits are excluded |
| `722c2f1` | aligned boundary-contact characterization with the corrected finite-segment semantics |
| `e0bdf69` | added general prepared-cycle dispatch |
| `4b46846` | verified prepared-cycle dispatch |
| `cb73b99` | added packed cycle batch screening |
| `f5bd251` | verified packed batch semantics |

The finite-segment correction is a mathematical bug fix, not a performance
claim. Its tests prevent the migration from preserving an incorrect extension
line result merely for byte-for-byte compatibility.

### Phase 4 -- public geometry cut-over

| Commit | Change |
|---|---|
| `7cbbd16` | routed public geometry relation functions through native kernels |
| `9ce8ab6` | enforced the native public source boundary and rejected hidden Python fallbacks |
| `bf4084a` | added native construction kernels required by downstream placement/session work |
| `a967002` | documented the canonical adapter boundary in the geometry API reference |

The Python public objects and result dataclasses remain stable; numerical work
is delegated to native functions and mapped back to those contracts.

### Phase 5 -- native session and low-level stage seams

| Commit | Change |
|---|---|
| `cd0c12c` | corrected the planned native session contracts before implementation |
| `8437d44` | added typed workflow-session input and native session contracts |
| `659eedd` | verified Python/native session packing and round trip |
| `c53a09f` | added in-place Open Babel optimization on the native session |
| `b7f3a00` | verified in-place optimization semantics |

This phase establishes a workflow-scoped `StructureSession`; it does not
introduce a persistent replacement for `Molecule`.

### Phase 6 -- native metal placement

| Commit | Change |
|---|---|
| `f531f9e` | preserved the approved force-field placement policy boundaries in the plan |
| `eb1a65a` | implemented the isolated native metal-placement engine |
| `083fbac` | verified placement outcomes and evidence |
| `73e9fba` | shared the native ring workspace used by placement/restoration checks |
| `49ba19a` | verified shared workspace semantics |

Placement remains a named component inside Stage 2. Geometry supplies
distances, clearances and intersection facts; forcefields selects radii,
targets, classifications and coordinate actions.

### Phase 7 -- independent native stages and shared-session composition

| Commit | Change |
|---|---|
| `afe1a95` | added the independent native coordination-restoration stage |
| `af74b27` | verified the coordination stage |
| `e3e026e` | exposed detailed native ring-checkpoint evidence |
| `522b1a9` | verified detailed checkpoints |
| `2ff2d46` | added the native ring-untangling engine |
| `d55c4cd` | verified native untangling |
| `16babd5` | completed native optimization-stage contracts |
| `6466d90` | verified optimization contracts |
| `2805942` | preserved detailed ring-checkpoint evidence through native optimization |
| `059afe2` | verified checkpoint evidence |
| `8d4a360` | implemented the native full-complex optimization stage |
| `b6a67c8` | verified Stage 3 behavior |
| `5209db3` | compiled the optimization stage into the sole Open Babel extension |
| `1f10700` | bound the independent native optimization entry |
| `bb264e5` | exposed the independent Python optimization facade |
| `be1e59a` | declared the native optimization typing API |
| `e2cbced` | verified the Python/native Stage 3 boundary |
| `9df551e` | shared native preflight checks across stage entries |
| `accffa5` | composed Stage 2 and Stage 3 over one native session |
| `c69b445` | bound the composed native workflow |
| `fc59d5f` | exposed the composed Python facade |
| `d9b35b1` | declared the composed typing API |
| `9029159` | verified workflow composition and independent-stage equivalence |
| `b9761f9` | normalized native bond--ring facts consumed by public acceptance |
| `2f362dd` | verified native-checkpoint acceptance without a Python rescan |

### Phase 8 -- trajectory, timing and public-result integration

| Commit | Change |
|---|---|
| `8562706` | added batched native trajectory ingestion |
| `92ed7ca` | verified native trajectory ingestion |
| `167f548` | measured native Stage 2 and Stage 3 elapsed time |
| `a9307e2` | exposed native stage runtime fields |
| `44e0d16` | carried runtime fields through the typed API |
| `5f02390` | verified native runtime metrics are finite and stage-local |
| `2554c5d` | adapted native stage results to public force-field contracts |
| `f976638` | verified native result adapters |
| `d1888f6` | split perturbation streams so Stage 2 and Stage 3 retain independent deterministic inputs |
| `98c67c4` | verified stage-specific perturbation streams |

Native `elapsed_seconds` fields existed at this point. Phase 10 commit
`b497b78` subsequently exposed the Stage 1, Stage 2 and Stage 3 timings in the
reusable 187-case report.

### Phase 9 -- public switch, cleanup and failure-contract hardening

| Commit | Change |
|---|---|
| `0bf9b71` | switched public complex workflows to the native Stage 2/3 controllers |
| `e728295` | verified public routing and the absence of Python geometry rescans |
| `40ac6fa` | retired legacy Python stage-control tests |
| `94cbce3` | aligned blocked native setup state with the public report |
| `4fad47d` | covered blocked native tail reports |
| `75f9685` | removed legacy Python metal placement |
| `153a8b0` | removed legacy Python coordination restoration |
| `9a9125f` | removed the legacy stage-report merger |
| `9099980` | retired legacy merger tests |
| `cd828b7` | retired legacy Python placement tests |
| `3763eb2` | retired legacy Python coordination tests |
| `3966e79` | preserved typed setup failures at the native stage boundary |
| `a76d27e` | verified structured native setup failures |
| `2cf6828` | exposed workflow-stage setup context |
| `1d70d01` | removed a redundant setup-stage alias |
| `a329c77` | completed setup-failure coverage |
| `40c4335` | narrowed public setup-stage types |
| `ecb062a` | retained completed Stage 2 evidence when later setup fails |
| `d2fc139` | verified preservation of failed-workflow evidence |
| `1fb8c17` | exposed complex-build failure diagnostics through the CLI |
| `a7c8221` | covered CLI diagnostics |
| `fdb0181` | removed placement settings made obsolete by the native policy boundary |

`efe9572` (`test(compat): track native forcefield suite`) is Phase 10
preparation. It adds compatibility-suite coverage but is not itself a completed
Python 3.9--3.14 matrix result.

### Phase 10 -- measurement, integration hardening and final validation

| Commit | Change |
|---|---|
| `5fb99f1` | isolated Open Babel system headers for the compatibility build |
| `b497b78` | exported Stage 1, Stage 2 and Stage 3 benchmark timings |
| `ee60315` | added the fresh-process trajectory-retention/RSS profiler |
| `9a0bb6b` | batched lazy segment iterators in the native geometry path |
| `443ea8a` | centralized native force-field helpers |
| `b60ae94` | verified Stage 1 timing attribution |
| `74431a0` | reduced the composed Stage 2 + Stage 3 path to one Python/native entry |
| `1b4a770` | centralized Python ring-screening defaults |
| `601e4f6` | centralized native computational defaults |
| `6c80213` | enforced Python/C++ default-value parity |
| `fb20ec3` | targeted the workflow seam owned by the caller's input molecule |
| `0219e64` | removed cycle-order dependence from ring-containment tests |
| `2518a0c` | rejected Open Babel's premature optimizer stop unless the global maximum atom gradient also meets its intended threshold |

`59e5741` adds the API guide only. The clean final benchmark manifest names
that commit; `2518a0c` is its production-code ancestor and there are no source
or test changes between them.

## 6. Correctness evidence already present

The repository contains the following committed verification mechanisms:

- a frozen geometry characterization corpus at
  `tests/fixtures/geometry/native_migration_golden.json`;
- direct public-boundary characterization in
  `tests/geometry_characterization.py`;
- native primitive, planar, nonplanar, prepared-cycle, batch and public
  cut-over tests;
- C++ source-entry and Python-entry tests for migrated native stages;
- session packing, in-place optimization, placement, restoration, untangling,
  composition and trajectory-ingestion tests;
- public workflow routing tests that reject re-entry into retired Python stage
  implementations;
- structured failure tests, including preservation of a completed Stage 2
  result when Stage 3 setup fails.

These tests support implementation and behavior statements. They do not
replace the final compatibility matrix or 187-case result requested by Phase
10.

## 7. Evidence classification

Performance evidence in the workspace has different audit strength. This
report uses four levels:

| Level | Meaning |
|---|---|
| A -- repository reproducible | command/script, input identity and comparison data are committed or generated by a committed stable protocol |
| B -- locally auditable | machine-readable artifacts are present in this workspace, but ignored by Git and therefore not portable with a clone |
| C -- committed summary only | numerical summary is committed, but its raw run directory is absent or not retained |
| D -- planning observation | number came from an exploratory spot check without a retained exact command/output artifact |

No single-run timing is treated as a statistically stable speedup. A single
controlled run can show absence of a large regression and an observed delta;
stable publication claims require repeated interleaved measurements.

## 8. Historical performance evidence

### 8.1 Earlier three-stage workflow optimization

Source: `plan/08_forcefields_performance/improve_ff_efficient.md`.

Evidence level: **C -- committed summary only**. The referenced complete
artifact directory
`movie/extractants_eu_three_stage_refactor_16c_20260924/` is not present in
the current workspace.

| Metric | `b55ae69` baseline | `f0a6e7c` result | Observed change |
|---|---:|---:|---:|
| Quality pass / fail / CBond fail | 171 / 7 / 9 | 171 / 7 / 9 | identical |
| Wall time | 380.590 s | 157.776 s | -222.814 s (-58.54%, 2.41x) |
| Aggregate case time | 5669.518 s | 2365.233 s | -3304.285 s (-58.28%, 2.40x) |
| Median case time | 20.699 s | 11.672 s | -9.027 s (-43.61%) |
| P95 case time | 81.496 s | 23.587 s | -57.909 s (-71.06%) |
| Maximum case time | 207.612 s | 71.749 s | -135.863 s (-65.44%) |

This was a multi-change workflow refactor. It is historical context, not an
ablation of the present native migration.

The same committed report contains isolated historical measurements:

| Optimization | Historical observation | Evidence level |
|---|---:|---|
| strict AABB broad phase | 2.59x and 4.96x on cases 0001 and 0048 | C |
| fixed ring-pair watch set | 6.95x for one watched pair versus 101 full pairs | C |
| shared Stage 2 workspace | 0.562 s -> 0.334 s (1.68x) | C |
| one OBBuilder call | 0.04095 s/call; avoiding 49 calls estimates 2.01 s | C |
| immutable topology + frame kernels | 30.210 s -> 21.840 s (1.38x) | C |

These microbenchmarks must not be multiplied to predict end-to-end speedup.

### 8.2 Native Open Babel wrapper migration

Sources:

- `plan/09_native_openbabel_forcefields/native_openbabel_forcefields_test_report.md`;
- local directories `movie/extractants_eu_obwrappers_16c_20260928/` and
  `movie/extractants_eu_native_cpp_16c_20260928/`.

Evidence level: **B -- locally auditable**. Summaries, per-case results and the
comparison are present, but all `movie/` artifacts are ignored by Git.

| Metric | Python/SWIG `503f2a5` | native backend `2d42ef6` | Observed change |
|---|---:|---:|---:|
| Passed / quality failed / CBond failed | 175 / 3 / 9 | 175 / 3 / 9 | identical |
| Wall time | 138.753 s | 135.842 s | -2.911 s (-2.10%, 1.021x) |
| Aggregate case time | 2100.904 s | 2062.645 s | -38.259 s (-1.82%, 1.019x) |
| Aggregate force-field time | 2066.643 s | 2029.281 s | -37.362 s |
| Median force-field time | baseline value in local comparison | 0.192 s lower | paired delta only |
| Faster / slower paired FF cases | -- | 139 / 39 | 178 paired cases |

This is one paired full run. It supports an observed approximately 2% change
without a chemistry-status regression; it does not establish a stable 2%
speedup distribution.

### 8.3 Reusable pre-refactor 187-case baseline

Source directory:
`movie/benchmarks/extractants_eu_187_1e14f28_20260929/`.

Evidence level: **B -- locally auditable**, with the strongest local protocol
identity. It contains a clean-worktree manifest, corpus SHA-256, settings,
187 per-case reports, trajectories, structures, renders, `results.csv`,
`summary.json`, `integrity.json` and `report.md`. The artifacts remain ignored
by Git.

No production-code changes exist between benchmark commit `1e14f28` and the
planning baseline `9c70ef6`; the intervening commits are documentation and CI
records. This makes it the preferred end-to-end baseline for the present
refactor.

| Metric | Baseline result |
|---|---:|
| Commit | `1e14f28671b4b1cb22ca3e7d1d76e8fe5b2f769d` |
| Inputs | 187 |
| Passed / quality failed / CBond failed | 175 / 3 / 9 |
| Overall success | 93.5829% |
| Success after CBond | 98.3146% |
| Wall time | 135.986209 s |
| Aggregate case time | 2060.754837 s |
| Aggregate CBond time | 23.868330 s |
| Aggregate force-field time | 2028.690725 s |
| Median case time | 10.058320 s |
| P95 case time | 20.606408 s |
| Maximum case time | 69.294549 s |
| Throughput | 1.375139 cases/s |
| Complete trajectory archives | 178 |
| Main trajectory frames | 8545 |

Its comparison with the prior `2d42ef6` run is consistent with run-to-run
noise: wall time changed by +0.106%, aggregate case time by -0.092%, and median
case time by -0.956%.

### 8.4 Geometry timing evidence

The committed `tests/performance/README.md` records this initial 2026-09-18
baseline:

| Workload | Median | P95 | Evidence level |
|---|---:|---:|---|
| planar 6-membered ring, one segment | 1.334 ms | 1.401 ms | C |
| planar 8-membered ring, one segment | 1.698 ms | 1.710 ms | C |
| nonplanar 6-membered ring, one segment | 21.183 ms | 21.523 ms | C |
| nonplanar 8-membered ring, one segment | 312.748 ms | 354.966 ms | C |
| lazy frame relation gate | 8.027 ms | 8.142 ms | C |
| dense 16-pair frame scan | 16.186 ms | 16.262 ms | C |

The command is repository reproducible:

```bash
python -m tests.performance.test_segment_cycle_relation_benchmark \
  --repeats 20 \
  --output /tmp/hotpot-segment-cycle.json
```

The original raw JSON was not committed, so the historical numbers are level
C rather than level A.

The approved plan also records exploratory `cProfile` observations:

| Workload | Observation | Evidence level |
|---|---:|---|
| prepare one nonplanar 8-cycle 20 times | 6.183 s; 8,660,917 calls | D |
| screen 1000 segments against one prepared nonplanar 8-cycle | 1.554 s; 2,225,011 calls | D |
| all pair distances for 200 points | 0.377 s; 577,703 calls | D |

Those three measurements justify hotspot selection, but their exact ad-hoc
driver and raw output were not retained. They must not be presented as a
strict before/after benchmark.

## 9. Final evidence and remaining limits

Phase 10 produced the required in-scope functional evidence:

1. a clean-worktree 187-case standard run with complete trajectories and
   renders;
2. strict reconstruction and integrity checks over every case artifact;
3. final native geometry profiles and a three-repeat trajectory-retention/RSS
   profile;
4. a successful Python 3.9--3.14 native/inference compatibility matrix,
   including Open Babel
   3.1 on Python 3.9 and Open Babel 3.2 on newer interpreters.

The following measurement limits remain and are reported explicitly rather
than filled with estimates:

- no machine-readable Phase 0 output from the present profiler was committed;
  the locally retained Python baseline is usable for descriptive comparison;
- native-call, `OBMol` rebuild and aggregated AABB/exact-kernel counters are
  structurally tested but are not emitted by the 187 reporter;
- several early phase commits were not connected to the public workflow, so a
  per-phase end-to-end timing ablation would compare different executable
  paths rather than isolate one change;
- the baseline and final 187 measurements are one complete run each. Their
  difference is an observed same-machine result, not a confidence interval;
- `movie/` and `/tmp` evidence are locally auditable but intentionally ignored
  by Git. The compact numerical conclusions are retained in the test report;
- PNG validation checks file integrity and render metadata. Cases 54, 61, 109
  and 125 were visually spot-checked, but the full set did not receive an
  independent human assessment of chemical plausibility.

## 10. Phase 10 commands

Record the exact final commit and a clean worktree before running any command.
Use the same machine, environment and CPU-affinity policy for baseline/current
comparisons.

### 10.1 Public geometry boundary profile

```bash
LD_LIBRARY_PATH=/home/zhangzhiyuan/usr/conda3/envs/hp-usage/lib \
/tmp/hotpot-native-env/bin/python -m tests.performance.profile_geometry_boundary \
  --repeats 20 \
  --output /tmp/59e5741.geometry-profile.json
```

### 10.2 Segment--cycle benchmark

```bash
LD_LIBRARY_PATH=/home/zhangzhiyuan/usr/conda3/envs/hp-usage/lib \
/tmp/hotpot-native-env/bin/python \
  -m tests.performance.test_segment_cycle_relation_benchmark \
  --repeats 20 \
  --output /tmp/59e5741.segment-cycle.json
```

### 10.3 Trajectory-retention/RSS profile

```bash
LD_LIBRARY_PATH=/home/zhangzhiyuan/usr/conda3/envs/hp-usage/lib \
/tmp/hotpot-native-env/bin/python \
  -m tests.performance.profile_forcefield_retention \
  --cases 1,54,61,109,125 \
  --repeats 3 \
  --profile standard \
  --output /tmp/59e5741.retention-rss.json
```

### 10.4 Standard 187-case benchmark

```bash
LD_LIBRARY_PATH=/home/zhangzhiyuan/usr/conda3/envs/hp-usage/lib \
/tmp/hotpot-native-env/bin/python -m tests.benchmarks.coordination_complexes \
  --suite extractants-eu-187 \
  --backend hotpot \
  --profile standard \
  --workers 16 \
  --render required \
  --render-workers 16 \
  --output movie/benchmarks/extractants_eu_187_59e5741_20260930
```

The 187-case run must not use `--resume` when reporting total wall time. A
resumed partial run intentionally marks complete-invocation wall time as
unavailable.

### 10.5 Python compatibility matrix

```bash
UV_CACHE_DIR=/tmp/hotpot-uv-cache \
bash tests/run_inference_compatibility.sh 3.9 3.10 3.11 3.12 3.13 3.14
```

## 11. Final profile results

### 11.1 Run identity

| Field | Final value |
|---|---|
| Git commit | `59e5741b9cac26b8dc787edd481a40ba6ad41d38` (`2518a0c` production code plus API documentation only) |
| Worktree clean | yes, recorded as `dirty: false` in the benchmark manifest |
| Date/time and timezone | 2026-09-30, Asia/Shanghai (manifest UTC timestamp `2026-09-30T09:09:48.171801+00:00`) |
| Host/CPU model | `chemlex-ai-2`; AMD Ryzen Threadripper PRO 3995WX |
| Logical/physical CPU count | 128 / 64 |
| CPU affinity/governor policy | CPUs 0--127 / `ondemand` |
| RAM | 503 GiB |
| OS/kernel | Linux 5.15.0-139-generic, x86-64 |
| Compiler and flags | g++ 9.4.0; C++17, optimized build, no `-ffast-math` |
| Python | 3.11.16 |
| NumPy | 1.26.4 |
| Open Babel build/runtime | 3.2.1 / 3.2.1 |
| ONNX Runtime/provider | 1.30.0 / CPU and Azure providers |
| Input SHA-256 | `77ef4913363bb10f0d134d6150c4f9e7214ddfaf03f8e6fc23f57308d5d3443f` |

### 11.2 Public geometry characterization

| Metric | Phase 0 baseline | Final HEAD | Absolute change | Relative change |
|---|---:|---:|---:|---:|
| Median workload time (ms) | 54.363675 | 2.639308 | -51.724367 | -95.14% (20.60x) |
| P95 workload time (ms) | 54.580590 | 3.025497 | -51.555094 | -94.46% (18.04x) |
| Minimum / maximum (ms) | 52.980962 / 54.604692 | 2.602829 / 3.148385 | -- | -- |
| Total profiled calls | 36,568 | 3,672 | -32,896 | -89.96% |
| Python geometry calls | 9,265 | 1,169 | -8,096 | -87.38% |
| Native geometry calls | 0 | 36 | +36 | expected cut-over |
| Peak Python traced bytes | 110,141 | 199,823 | +89,682 | +81.42% |

The higher `tracemalloc` peak is about 88 KiB and measures Python-visible
temporary allocations only. It is not total process RSS and does not offset
the approximately 20.6x workload-time reduction.

### 11.3 Segment--cycle workloads

| Workload | Historical median | Final median | Historical P95 | Final P95 | Median speedup |
|---|---:|---:|---:|---:|---:|
| planar 6-membered pair | 1.334 ms | 0.040307 ms | 1.401 ms | 0.050062 ms | 33.10x |
| planar 8-membered pair | 1.698 ms | 0.040547 ms | 1.710 ms | 0.044250 ms | 41.88x |
| nonplanar 6-membered pair | 21.183 ms | 0.063601 ms | 21.523 ms | 0.070634 ms | 333.06x |
| nonplanar 8-membered pair | 312.748 ms | 0.287997 ms | 354.966 ms | 0.310284 ms | 1,085.94x |
| lazy frame relation gate | 8.027 ms | 0.281224 ms | 8.142 ms | 0.321435 ms | 28.54x |
| dense 16-pair frame scan | 16.186 ms | 0.660328 ms | 16.262 ms | 0.700197 ms | 24.51x |

The historical run used CPython 3.11.15 and NumPy 2.3.5; the final run used
CPython 3.11.16 and NumPy 1.26.4. The large reductions are consistent with the
native cut-over, but the environment difference makes them descriptive rather
than a perfectly controlled A/B comparison. Full details are in the test
report.

### 11.4 Native stage and memory evidence

The final 187 run records 543.905 s in Stage 1, 59.167 s in Stage 2 and
958.042 s in Stage 3 across the 178 CBond-successful cases. These are summed
case CPU/wall durations under 16-way case parallelism; they are not additive
to the benchmark wall clock.

The three-repeat fresh-worker retention profile over cases 1, 54, 61, 109 and
125 produced:

| Metric | `save_movie=False` | `save_movie=True` | Observed difference |
|---|---:|---:|---:|
| Samples | 15 | 15 | -- |
| Median retained frames | 26 | 122 | +96 |
| Median explicitly retained coordinate bytes | 14,400 | 294,240 | +279,840 bytes |
| Median process peak RSS | 270,296 KiB | 270,380 KiB | +84 KiB (+0.031%) |
| Median workflow wall time | 3.755380 s | 3.756492 s | +0.001112 s (+0.030%) |

The explicit coordinate increase is real. The process-RSS delta is smaller
than run-to-run noise in the paired samples, so it supports “no material RSS
regression detected,” not a precise 84 KiB memory claim. Runtime counters for
native entries, `OBMol` reconstruction, AABB candidates and exact kernel calls
were not added solely for this report; their structural invariants remain
covered by source and routing tests.

## 12. Final 187-case comparison

Preferred baseline: `1e14f28`, described in Section 8.3.

| Metric | `1e14f28` baseline | Final HEAD | Absolute change | Relative change |
|---|---:|---:|---:|---:|
| Passed / quality failed / CBond failed | 175 / 3 / 9 | 177 / 1 / 9 | +2 / -2 / 0 | -- |
| Overall success rate | 93.5829% | 94.6524% | +1.0695 percentage points | -- |
| Success after CBond | 98.3146% | 99.4382% | +1.1236 percentage points | -- |
| Wall time | 135.986209 s | 125.233353 s | -10.752856 s | -7.91% |
| Aggregate case time | 2060.754837 s | 1849.082543 s | -211.672294 s | -10.27% |
| Aggregate CBond time | 23.868330 s | 23.583082 s | -0.285248 s | -1.20% |
| Aggregate force-field time | 2028.690725 s | 1817.255579 s | -211.435146 s | -10.42% |
| Median case time | 10.058320 s | 9.083435 s | -0.974885 s | -9.69% |
| P95 case time | 20.606408 s | 20.071277 s | -0.535131 s | -2.60% |
| Maximum case time | 69.294549 s | 42.413686 s | -26.880863 s | -38.79% |
| Throughput | 1.375139 cases/s | 1.493212 cases/s | +0.118073 cases/s | +8.59% |
| Complete trajectories | 178 | 178 | 0 | -- |
| Main trajectory frames | 8,545 | 20,291 | +11,746 | +137.46% |
| Missing/corrupt artifacts | 0 | 0 | 0 | -- |

Required scientific comparison beyond aggregate counts:

| Check | Final result |
|---|---|
| Per-case status transition table | cases 54 and 109 changed `failed_quality -> passed`; case 61 remained `failed_quality`; no new failures |
| Newly failing cases | none |
| Newly passing cases with reason | 54 and 109: native metal placement removed Eu collision/collapse; 125 also passes relative to pre-fix `0219e64` after false convergence was corrected |
| Case 61 remains a declared failure without relaxed thresholds | yes; two Eu--N distances remain 1.5073 and 1.5012 Angstrom, failing the unchanged short-bond and radius-ratio checks |
| Case 54 outcome and evidence | passed; Eu--S distances 2.754/2.639 Angstrom, no final piercing, RMS/max gradient 0.458/1.648 |
| Case 109 placement/collision outcome and evidence | passed and truly converged; both Eu--N distances 2.34291 Angstrom, RMS/max gradient 0.0495/0.0928 |
| Case 125 outcome after convergence fix | passed after 100 epochs/9,968 submitted steps; quality checks pass, but correctly reports budget exhaustion rather than convergence |
| Final confirmed ring--bond piercing count | zero for the highlighted cases; all 177 accepted cases pass the terminal hard gate |
| Atom order and intended topology preserved | all 178 outputs match the authoritative selected frame; Eu adjacency exactly matches the reported CBond donor set |
| Every CBond-success case has readable trajectory and output structure | yes, 178/178 |
| Required per-case and contact-sheet PNG files present | yes; 178 molecular renders, 9 explicit CBond-failure placeholders, and one root contact sheet |

The intermediate `0219e64` build completed in 95.751 s but inherited Open
Babel's premature stopping behavior. After `2518a0c`, 162 cases truthfully
report budget exhaustion instead of 19, Stage 3 aggregate time increases from
496.684 s to 958.042 s, and main trajectory frames increase from 9,142 to
20,291. That is the cost of executing the requested optimizer budget rather
than accepting a false global convergence result. The final implementation is
still 7.91% faster on wall time than the original baseline and improves the
chemical pass count from 175 to 177.

## 13. Per-phase ablation disposition

A strict wall-time value for every Phase 1--9 commit is **not recoverable**.
Several commits introduced native kernels before the public path called them;
later commits changed evidence collection and convergence correctness. Running
the 187 workflow at every commit would therefore compare inactive code or
different scientific contracts, not isolate one implementation change.

The retained evidence instead separates three defensible scopes:

| Scope | Evidence | Result |
|---|---|---|
| Geometry migration | identical public workload before/after | 20.60x median boundary speedup; 24.51x--1,085.94x across focused segment--cycle workloads |
| Integrated native workflow before convergence correction | `1e14f28` vs `0219e64` | 29.59% lower wall time, but one false-convergence regression (case 125) |
| Final chemically corrected workflow | `1e14f28` vs `59e5741` | 7.91% lower wall time and two additional passing structures |

Commit-level source and differential tests remain the evidence for individual
phase correctness. No timing number is assigned to a phase that did not have
an executable, contract-equivalent before/after pair.

## 14. Python 3.9--3.14 compatibility matrix

The isolated compatibility runner rebuilt all native extensions for every
interpreter before running the same selected project suite and SMARTS suite.

| Python | Environment identity | NumPy | Open Babel | Native build | Focused geometry | Focused forcefields | Compatibility suite | Notes |
|---|---|---|---|---|---|---|---|---|
| 3.9 | isolated `uv --no-project` | 2.0.2 | 3.1.0 | pass | included/pass | included/pass | 1,189 passed, 4 skipped, 3 xfailed; SMARTS 255 passed | 34 warnings |
| 3.10 | isolated `uv --no-project` | 2.2.6 | 3.2.1 | pass | included/pass | included/pass | 1,189 passed, 4 skipped, 3 xfailed; SMARTS 255 passed | 49 subtests; 5 warnings |
| 3.11 | isolated `uv --no-project` | 1.26.4 | 3.2.1 | pass | included/pass | included/pass | 1,189 passed, 4 skipped, 3 xfailed; SMARTS 255 passed | 49 subtests; 5 warnings |
| 3.12 | isolated `uv --no-project` | 2.5.3 | 3.2.1 | pass | included/pass | included/pass | 1,189 passed, 4 skipped, 3 xfailed; SMARTS 255 passed | 49 subtests; 44 warnings |
| 3.13 | isolated `uv --no-project` | 2.5.3 | 3.2.1 | pass | included/pass | included/pass | 1,189 passed, 4 skipped, 3 xfailed; SMARTS 255 passed | 49 subtests; 44 warnings |
| 3.14 | isolated `uv --no-project` | 2.5.3 | 3.2.1 | pass | included/pass | included/pass | 1,189 passed, 4 skipped, 3 xfailed; SMARTS 255 passed | 49 subtests; 44 warnings |

The matrix must distinguish “interpreter unavailable” from a build or test
failure. Open Babel build/runtime versions and C++ ABI must be recorded because
Python version alone does not characterize this extension.

## 15. Compatibility-suite and artifact result

| Item | Final result |
|---|---|
| Exact command | `UV_CACHE_DIR=/tmp/hotpot-uv-cache bash tests/run_inference_compatibility.sh 3.9 3.10 3.11 3.12 3.13 3.14` |
| Collected tests per interpreter | 1,196 main outcomes plus 255 SMARTS conformance tests |
| Passed | 1,189 main + 255 SMARTS on every interpreter |
| Failed | 0 on every interpreter |
| Skipped / xfailed | 4 / 3 on every interpreter |
| Warnings | 34 on Python 3.9; 5 on 3.10--3.11; 44 on 3.12--3.14 |
| Main-suite wall time | 219.57--235.99 s per interpreter; complete six-version runner about 35 minutes |
| Coverage command/result | not run in Phase 10; this phase validates behavior and compatibility, not a coverage target |
| Failure analysis | case 61 and the Open Babel false-convergence correction are detailed in the test report |

This is the repository's defined native/inference compatibility selection,
not every test file under the checkout. A project-global `pytest --collect-only`
audit found 1,535 tests but cannot complete in the lean inference environment:
unrelated example/plugin tests require optional `torch`, `torch_geometric`,
`numba`, `requests`, and `scikit-learn` dependencies, and default pytest import
mode collides on duplicate test basenames. No project-global pass is claimed.

The final artifact audit additionally established:

- all 733 JSON files parse strictly without `NaN` or `Infinity`;
- all 178 trajectory archives deserialize through
  `ForceFieldTrajectoryArchive.read()` and contain 20,291 valid main frames
  plus 186 valid ligand-build branches;
- all selected coordinates, coordinate/topology revisions, energies and bond
  indices are finite and internally consistent;
- 178 MOL2 outputs match their authoritative selected-frame atom order,
  coordinates and undirected bond set; each has one Eu and its Eu adjacency
  exactly matches the CBond donor indices;
- aggregate SDF counts are 178 for all outputs and 177 for passing outputs;
- 178 molecular PNGs, 9 explicit failure placeholders and the root contact
  sheet are valid; no staging, backup or temporary artifacts remain.

## 16. Completion statement

The production work in Phases 0--9 and the defined in-scope Phase 10 validation
are complete. The project-global optional test collection and irrecoverable
historical per-phase timing ablations are explicit exceptions, not silently
claimed results. The refactor preserves the three independent scientific
stages and the stable Python API while providing direct C++ and Python access
to one numerical implementation. The final standard corpus has no new
failures, fixes cases 54 and 109 relative to the reusable baseline, and
corrects case 125's false convergence without weakening a chemistry threshold.
Case 61 remains the one declared force-field quality failure.

The measured final workflow is 7.91% faster in wall time than the original
same-machine baseline. More importantly, it reports convergence truthfully:
budget exhaustion is no longer accepted merely because Open Babel stopped on
one atom's small gradient. The detailed evidence and reproducibility limits
are part of the test report and must accompany any performance claim.
