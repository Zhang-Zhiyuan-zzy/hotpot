# Native geometry and force-field API and usage

Date: 2026-09-30

This document describes the API implemented by the native geometry and
force-field refactor.  It distinguishes the stable user-facing Python API,
the advanced native-session API, and the repository-level C++ composition
surface.

## 1. API boundaries

| Layer | Import or include path | Intended audience | Stability |
|---|---|---|---|
| Stable force-field API | `hotpot.cheminfo.forcefields` | ordinary Hotpot users | public |
| Stable geometry API | `hotpot.cheminfo.geometry` | ordinary Hotpot users | public |
| Native force-field session API | `hotpot.cheminfo.forcefields.native` | advanced workflows requiring explicit Stage 2/3 control | public advanced API |
| Direct native geometry adapters | `hotpot.cheminfo.geometry.native` | advanced numerical callers | public advanced API |
| Native extension bindings | `hotpot.cheminfo.obWrappers._ob_native`, `hotpot.cheminfo.geometry._geometry_native` | implementation only | private |
| C++ headers | `hotpot/cheminfo/{geometry,forcefields}/_native/*.hpp` | Hotpot source development and in-tree composition | source-level API; not an installed SDK |

Migrated geometry operations and native Stage 2/3 functions reach the same
canonical C++ implementations from Python and C++.  There is no Python
numerical fallback for those operations.  Failure to import a required native
extension is an installation error.

The native structure is workflow-scoped.  It is not a second persistent
`Molecule` implementation and it does not retain references to Python objects.

## 2. Build and runtime requirements

The native/inference stack is validated on Python 3.9 through 3.14. Building
from source requires:

- a C++17 compiler;
- `setuptools`, `wheel`, and `pybind11`;
- an Open Babel Python package that contains its C++ headers and shared
  library;
- `openbabel-wheel>=3.1.1.23,<3.2` on Python 3.9, or
  `openbabel>=3.2.1,<3.3` on Python 3.10 and newer.

Normal installation builds all native extensions:

```bash
$ python -m pip install .
```

For an editable source checkout:

```bash
$ python -m pip install -e .
```

The build produces the geometry extension, Relevant Cycles extension, and the
Open Babel/force-field extension.  A package or wheel missing any required
extension is incomplete; Hotpot does not silently switch to a second
implementation.

## 3. Stable force-field Python API

Use the package facade rather than implementation modules:

```python
from hotpot.cheminfo import forcefields as ff
```

### 3.1 Workflow functions

| Function | Operation | Return type |
|---|---|---|
| `ff.build3d(mol, *, add_hydrogens=True, seed=None, timeout=1000.0)` | Build initial organic 3D coordinates without force-field optimization | `Build3DReport` |
| `ff.optimize(mol, forcefield="UFF", *, ...)` | Optimize existing coordinates with the ordinary Open Babel path | `ForceFieldRunReport` |
| `ff.build_complex3d(mol, forcefield=None, *, ...)` | Run Stage 1 ligand construction and Stage 2 coordination restoration | `ComplexBuildReport` |
| `ff.optimize_complex(mol, forcefield=None, *, ...)` | Run Stage 3 on an existing explicit complex | `ForceFieldRunReport` |
| `ff.complexes_build(mol, forcefield=None, *, ...)` | Run all three complex stages | `ComplexBuildReport` |
| `ff.build_and_optimize(mol, forcefield="UFF", *, ...)` | Select the organic or complex build-and-optimize workflow from `mol.has_metal` | `ForceFieldWorkflowReport` |
| `ff.auto_optimize(mol, forcefield=None, *, ...)` | Select ordinary or complex optimization for existing coordinates | `ForceFieldRunReport` |

Important keyword groups are:

- optimization: `algorithm`, `epochs`, `steps_per_epoch`,
  `stopping_criteria`, `increasing_vdw`, and van der Waals cutoff limits;
- perturbation: `seed`, `perturb_interval`, and `perturb_sigma`;
- structure acceptance: `quality_level` and `quality_thresholds`;
- trajectory: `save_movie`, `trajectory_start`, and `trajectory_path`;
- complex construction: `max_attempts`, ligand and coordination attempt
  limits, relaxation-step counts, and `coordination_geometry`.

`candidate_count` is reserved on the relevant high-level functions and
currently has no effect.  It must not be interpreted as active multi-conformer
search.

### 3.2 Supporting public contracts

The facade also exports the following supporting API.  Names in one row form a
related contract group; they are not separate numerical backends.

| Group | Public names |
|---|---|
| Trajectory selection | `TrajectoryPath`, `TrajectoryStart`, `TrajectoryStage`, `TrajectoryEvent` |
| Trajectory records | `AtomIdentity`, `BondTopology`, `BondTopologyRevision`, `RingFrameEvidence`, `CoordinationFrameEvidence`, `OptimizationFrameEvidence`, `FrameEvidence`, `ForceFieldFrame`, `ForceFieldTrajectory`, `ForceFieldTrajectoryArchive` |
| Optimization options/status | `OptimizationAlgorithm`, `OptimizationStoppingCriteria`, `TerminationReason` |
| Reports | `ForceFieldRunReport`, `Build3DReport`, `RingUntanglingReport`, `CoordinationBondRestorationReport`, `ComplexBuildDiagnostics`, `ForceFieldWorkflowReport`, `BuildAndOptimizeReport`, `ComplexBuildReport`, `ForceFieldSetupReport` |
| Report fields and validation | `ForceFieldDiagnosticValue`, `ForceFieldSetupStage`, `ForceFieldWorkflowStage`, `AcceptanceCheck`, `StructureAcceptanceThresholds`, `ForceFieldAcceptanceEvidence`, `ForceFieldValidationReport` |
| Coordination geometry | `CoordinationEnvironment`, `CoordinationGeometryCandidate`, `CoordinationGeometryResult`, `collect_coordination_environments`, `prepare_coordination_geometry` |
| Topology | `AtomTopologySignature`, `BondTopologySignature`, `TopologyReference`, `capture_topology` |
| Acceptance | `evaluate_structure_acceptance`, `is_structure_accepted` |
| Coordinate operation | `perturb` |
| Errors and warnings | `ForceFieldError`, `ForceFieldSetupError`, `BuildWorkerError`, `BuildTimeoutError`, `ComplexBuildError`, `ComplexBuildWarning`, `ComplexBuildWorkerError`, `ComplexBuildTimeoutError`, `GeometryQualityError`, `GeometryQualityWarning` |

`CandidateRejection` is also exported as the immutable record used by complex
build diagnostics.

### 3.3 Ordinary molecule example

```python
from hotpot import read_mol
from hotpot.cheminfo import forcefields as ff

mol = read_mol("CCO", fmt="smi")
build_report = ff.build3d(mol, seed=7)
optimization_report = ff.optimize(
    mol,
    forcefield="UFF",
    epochs=20,
    steps_per_epoch=100,
    quality_level="standard",
    seed=7,
)

print(optimization_report.best_energy, optimization_report.energy_unit)
```

Both functions operate through a working copy and commit the selected result
to `mol` only after the workflow has produced a usable result.

### 3.4 Complex examples

The input must already identify the metal and the intended metal--ligand
bonds, for example after the cbond workflow.

Run all three stages:

```python
from hotpot.cheminfo import forcefields as ff

report = ff.complexes_build(
    complex_mol,
    forcefield="UFF",
    epochs=100,
    steps_per_epoch=100,
    seed=7,
    save_movie=True,
    trajectory_path="trajectory/eu_complex",
)
print(report.quality_report.passed)
```

Keep construction and final optimization explicit:

```python
build_report = ff.build_complex3d(
    complex_mol,
    forcefield="UFF",
    seed=7,
)
optimization_report = ff.optimize_complex(
    complex_mol,
    forcefield="UFF",
    epochs=100,
    steps_per_epoch=100,
    seed=7,
)
```

`build_complex3d()` is Stage 1 + Stage 2.  `optimize_complex()` is Stage 3.
The composed `complexes_build()` workflow uses the same stage implementations;
it is not a separate scientific algorithm.

## 4. Three-stage contract

| Stage | Responsibility | Public high-level access | Native-session access |
|---|---|---|---|
| 1. Ligand construction | add hydrogens, build ligand coordinates, repair ligand ring--bond piercing | first part of `build_complex3d()` and `complexes_build()` | none; retained Python worker/controller |
| 2. Coordination restoration | place metals, restore intended coordination bonds, perform bounded relaxation | second part of `build_complex3d()` and `complexes_build()` | `create_coordination_session()` + `restore_coordination()` |
| 3. Full-complex optimization | full-graph topology gate, ring untangling, numerical optimization and final checkpoint | `optimize_complex()` and final part of `complexes_build()` | `create_optimization_session()` + `optimize_complex()` |

Stages remain independently testable.  Stage 2 and Stage 3 may share one
native session to avoid reconstructing the `OBMol`, but neither stage consumes
the other's private controller state.  Their results and trajectories remain
separate.

## 5. Advanced native force-field session API

Import this layer explicitly:

```python
from hotpot.cheminfo.forcefields import native
```

### 5.1 Exported option and state types

| Type | Purpose |
|---|---|
| `FrameDetail` | optional native diagnostic-frame detail: `NONE`, `OPTIMIZATION`, or `ALL_ATTEMPTS` |
| `MetalPlacementOptions` | Stage 2 metal candidate generation, geometric clearance, distance-ratio and evidence policy |
| `RingScreeningOptions` | Stage 3 actionable-ring and Relevant Cycle limits plus geometry settings |
| `OptimizationStoppingOptions` | optional energy, displacement and gradient stability thresholds |
| `CoordinationStageOptions` | Stage 2 force field, attempts, relaxation, perturbation, frame and placement policy |
| `ComplexOptimizationOptions` | Stage 3 optimizer, untangling, perturbation, stopping, frame and ring-screening policy |
| `StructureSnapshot` | detached coordinates, active masks, component IDs and session revision counters |

`FrameDetail.NONE` disables optional diagnostic frames only.  Required stage
boundaries, selected frames, terminal frames, and other contractually required
facts are still recorded.

### 5.2 Exported session and stage functions

```python
native.create_coordination_session(source)
native.create_optimization_session(source)
native.snapshot_structure(session)
native.update_structure_coordinates(session, coordinates)
native.set_coordination_active_mask(session, active_mask)
native.set_ligand_bond_active_mask(session, active_mask)

native.assess_metal_position(session, metal_index, coordinates, *, options=...)
native.place_metal(session, metal_index, *, options=...)
native.place_metals(session, *, options=...)

native.restore_coordination(
    session, perturbation_offsets, *, options=CoordinationStageOptions()
)
native.optimize_complex(
    session,
    untangling_offsets,
    optimization_offsets,
    *,
    options=ComplexOptimizationOptions(),
)
native.run_complex_workflow(
    session,
    coordination_offsets,
    untangling_offsets,
    optimization_offsets,
    *,
    coordination_options=CoordinationStageOptions(),
    optimization_options=ComplexOptimizationOptions(),
)
native.run_complex_workflow_from_input(
    source,
    coordination_offsets,
    untangling_offsets,
    optimization_offsets,
    *,
    coordination_options=CoordinationStageOptions(),
    optimization_options=ComplexOptimizationOptions(),
)
```

`source` may be a Hotpot `Molecule` or a packed `ComplexSessionInput`.
`ComplexSessionInput` and `pack_complex_session_input()` live in
`hotpot.cheminfo.forcefields.native_packing`; they are typed boundary helpers,
not exports of `forcefields.native`.

Placement functions currently return native result records.  Although the
functions are exported by the advanced facade, callers should consume named
attributes and must not depend on the private pybind classes' module path.

### 5.3 Deterministic offset contracts

The advanced stage functions do not create hidden random perturbations.  They
consume contiguous `float64` arrays with shape `(frame_count, atom_count, 3)`:

| Stream | Required frame count |
|---|---:|
| Stage 2 `coordination_offsets` | `max(attempt_limit - 1, 0)` |
| Stage 3 `untangling_offsets` | `untangling_attempt_limit` |
| Stage 3 `optimization_offsets` | `0` when `perturb_interval is None`; otherwise `(epochs - 1) // perturb_interval` |

Use zero-length arrays when a stream has no frames:

```python
import numpy as np
from hotpot.cheminfo.forcefields import native

atom_count = len(complex_mol.atoms)
coordination_options = native.CoordinationStageOptions(attempt_limit=20)
optimization_options = native.ComplexOptimizationOptions(
    epochs=100,
    untangling_attempt_limit=30,
    perturb_interval=None,
)
rng = np.random.default_rng(7)

coordination_offsets = rng.normal(
    0.0,
    coordination_options.perturb_sigma,
    (coordination_options.attempt_limit - 1, atom_count, 3),
).astype(np.float64)
untangling_offsets = rng.normal(
    0.0,
    optimization_options.perturb_sigma,
    (optimization_options.untangling_attempt_limit, atom_count, 3),
).astype(np.float64)
optimization_offsets = np.empty((0, atom_count, 3), dtype=np.float64)
```

### 5.4 Independent and composed native calls

Independent Stage 2 followed by Stage 3 on one session:

```python
session = native.create_coordination_session(complex_mol)
stage2 = native.restore_coordination(
    session,
    coordination_offsets,
    options=coordination_options,
)
stage3 = native.optimize_complex(
    session,
    untangling_offsets,
    optimization_offsets,
    options=optimization_options,
)
snapshot = native.snapshot_structure(session)
```

Combined Stage 2 + Stage 3 on the same explicit session:

```python
session = native.create_coordination_session(complex_mol)
result = native.run_complex_workflow(
    session,
    coordination_offsets,
    untangling_offsets,
    optimization_offsets,
    coordination_options=coordination_options,
    optimization_options=optimization_options,
)
```

Combined Stage 2 + Stage 3 with session ownership internal to the call:

```python
result = native.run_complex_workflow_from_input(
    complex_mol,
    coordination_offsets,
    untangling_offsets,
    optimization_offsets,
    coordination_options=coordination_options,
    optimization_options=optimization_options,
)
```

`restore_coordination()`, `optimize_complex()`, and
`run_complex_workflow()` mutate their explicit native `StructureSession`.
`run_complex_workflow_from_input()` owns and mutates an internal session.
None of them mutates a source `Molecule`. Their stage results expose selected
and terminal coordinates, masks, reports, warning codes, and a native
trajectory batch. Snapshot, mask-setter, assessment, and placement functions
instead return their own operation-specific contracts. The stable workflow
layer maps reports, ingests trajectory frames, and atomically commits the
selected result to the caller's molecule.

## 6. Geometry Python/native boundary

The stable geometry facade is:

```python
from hotpot.cheminfo import geometry as geo
```

It exports value objects (`Point`, `Line`, `Segment`, `Plane`, `Triangle`, and
`Cycle`), relation/result records, numerical settings, mathematical relation
functions, and chemical-object conversion/scanning functions.  See
`hotpot/cheminfo/geometry/README.md` for the complete geometry API and formulas.

The direct native adapter is:

```python
from hotpot.cheminfo.geometry import native as native_geo

distance = native_geo.point_segment_distance(point, segment)
```

`geometry.relation` maps native numerical results into the stable public
geometry records.  `geometry.convert` maps Hotpot chemical objects to geometry
objects and maps relation facts back to their chemical source objects.  The
canonical arithmetic and predicates execute in C++ in both cases.

The geometry layer reports mathematical facts only.  It does not choose metal
radii, define chemically acceptable distances, decide whether a structure is
physically reasonable, or mutate molecular topology.  Those decisions belong
to force-field or chemistry policy.

Do not import `hotpot.cheminfo.geometry._geometry_native` directly.  Its
pybind types and signatures are implementation details.

## 7. C++ source-level composition API

The C++ implementation has independently callable headers.  Principal
geometry headers include:

- `geometry/_native/primitives.hpp`;
- `geometry/_native/prepared_cycle.hpp`;
- `geometry/_native/batch.hpp`;
- `geometry/_native/spatial.hpp`.

Principal force-field headers include:

- `forcefields/_native/contracts.hpp` and `stage_contracts.hpp`;
- `forcefields/_native/structure_session.hpp`;
- `forcefields/_native/placement_engine.hpp`;
- `forcefields/_native/coordination_stage.hpp`;
- `forcefields/_native/optimization_stage.hpp`;
- `forcefields/_native/workflow_stage.hpp`.

For example, in the `hotpot::forcefields` namespace the source-level stage
composition is:

```cpp
auto session = create_coordination_session(input);
auto stage2 = restore_coordination(*session, coordination_options,
                                   coordination_offsets);
auto stage3 = optimize_complex(*session, optimization_options,
                               untangling_offsets, optimization_offsets);
```

`run_complex_workflow()` composes those same C++ stage functions.  Native
force-field code calls the geometry C++ functions directly; it does not call
Python and does not cross a pybind boundary inside hot loops.

These headers are a repository/source-level composition surface.  The project
does not currently install a standalone C++ SDK, exported CMake targets, or an
ABI-stability contract.  External users should use the public Python facades
unless they build against the Hotpot source tree and accept source-level
compatibility.

## 8. Trajectory semantics

Optimization and complex high-level workflows return a
`ForceFieldTrajectoryArchive` through their report. `build3d()` is the
exception: its `Build3DReport` has no trajectory. An archive contains one main
trajectory and optional Stage 1 ligand-build attempts. Every frame records:

- stage and event;
- a pooled coordinate revision;
- a pooled bond-topology revision;
- optional energy in kJ/mol;
- optional ring, coordination, or optimization evidence.

The selected frame is the workflow result.  The terminal frame is the last
state reached.  They are intentionally distinct and may differ after a
rollback or failed terminal attempt.

`save_movie=False` materializes only the selected frame in
`Molecule.conformers`; `save_movie=True` materializes every retained frame.
This switch does not change the workflow's selected structure.  A supplied
`trajectory_path` serializes the archive independently of `save_movie`.

```python
from hotpot.cheminfo.forcefields import ForceFieldTrajectoryArchive

archive = ForceFieldTrajectoryArchive.read("trajectory/eu_complex")
print(len(archive.main.frames))
print(archive.main.selected_index, archive.main.terminal_index)
archive.main.write_sdf("trajectory/eu_complex.sdf")
```

The directory archive contains JSON metadata, compressed coordinate arrays,
and SDF records by default.  Publication uses a sibling staging directory and
replacement so an ordinary serialization/publication failure does not expose
a partially written archive.  This is single-writer failure atomicity, not a
concurrent-reader or crash-durability protocol.

`Molecule.conformers` is coordinate-only and cannot represent changing bond
topology.  The trajectory archive is authoritative for topology-changing
coordination and ring-untangling frames.

## 9. Warnings and failures

The stable workflow distinguishes diagnostic quality failure from inability to
produce a usable structure:

- a finite, topology-preserving selected frame that fails the requested
  acceptance gate is retained and returned with `GeometryQualityWarning`;
- absence of a usable finite-topology frame raises `GeometryQualityError`;
- missing force fields or backend setup/preflight failures raise
  `ForceFieldSetupError`;
- worker and timeout failures use the corresponding build/complex exception
  classes exported by `hotpot.cheminfo.forcefields`.

When a `ForceFieldError` occurs after trajectory recording began, the partial
`ForceFieldTrajectoryArchive` is attached to `error.trajectory`.  If
`trajectory_path` was supplied, that partial archive is also written for
inspection.  Warnings and native result `warning_codes` remain explicit; they
are not converted into silent fallback behavior.

Native Stage 2/3 results expose both `selected_coordinates` and
`terminal_coordinates`.  Native optimizer `converged` means that the Open
Babel stop was verified against a global maximum atom-gradient threshold;
budget exhaustion is reported separately instead of being mislabeled as
convergence.

## 10. Which entry should be used?

| Requirement | Recommended entry |
|---|---|
| Build and optimize an organic molecule | `ff.build_and_optimize()` |
| Optimize existing organic coordinates | `ff.optimize()` |
| Build and optimize a complex end to end | `ff.complexes_build()` |
| Build a complex now and optimize later | `ff.build_complex3d()`, then `ff.optimize_complex()` |
| Automatically select organic/complex optimization | `ff.auto_optimize()` |
| Control Stage 2 and Stage 3 independently on one native session | `forcefields.native` |
| Compute geometry facts from Python | `hotpot.cheminfo.geometry` |
| Compose geometry or force-field kernels inside Hotpot C++ | in-tree `_native/*.hpp` headers |

Use the stable facades by default.  The advanced session API is intended for
callers that require deterministic perturbation streams, explicit active-bond
masks, placement evidence, native snapshots, or separate Stage 2/3 result
contracts.
