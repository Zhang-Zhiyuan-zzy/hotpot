# Stage 17 extension — fully native RDKit organic conformer proposal plan

Status: proposed for owner review; implementation is not authorized by this
document alone.

This is a continuation of Stage 17, not a new work objective. The former
Plan-19 O0/MCA conformer-engine study is retained in this directory because it
measures the native RDKit component, its integration boundary, and the exact
organic conformer behavior required by this extension.

## 1. Objective and fixed decisions

The objective is to make the complete organic conformer-proposal operation run
inside the pinned native RDKit component:

```text
Hotpot graph/stereo buffers
  -> native RDKit molecule
  -> native hydrogen materialization
  -> ETKDGv3 embedding
  -> optional native MMFF94/MMFF94s/UFF optimization
  -> optional post-optimization conformer clustering
  -> mapped Hotpot atom-row coordinates and per-conformer evidence
```

The following decisions govern the implementation:

1. RDKit remains pinned to exactly 2026.09.1. No version profile, dynamic
   compatibility branch, or automatic upgrade is introduced.
2. There is one native execution implementation and one public proposal
   function, `rdWrappers.generate_conformers()`. Python and C++ are facades over
   the same C/native core.
3. No Python RDKit call may perform embedding, hydrogen expansion, force-field
   parameter selection, force-field optimization, or conformer clustering in
   the production proposal path.
4. The existing field-selective `NativeMoleculeBuffers` transport is reused.
   A second RDKit-specific Hotpot buffer schema is prohibited.
5. `MMFF94` is the intended name; there is no `MBFF94` backend. Native support
   covers `MMFF94`, `MMFF94s`, and `UFF`.
6. The MCA route explicitly passes seed 42. The standalone rdWrapper default
   remains `seed=None`; workflow reproducibility must not depend on that
   component default.
7. The current MCA routing rule is reproduced exactly: use MMFF94 only when
   `MMFFHasAllMoleculeParams()` is true, otherwise use UFF. UFF is not a retry
   after an MMFF optimization failure.
8. A negative RDKit optimization status is a failed conformer. Status zero is
   converged. A positive status records that the iteration limit was reached
   and is not fatal, matching the current MCA behavior.
9. Open Babel remains available through `obWrappers`. Explicit GAFF or
   Ghemical optimization is not reimplemented in RDKit.
10. Complete complexes keep the owner-selected O0 workflow. This plan neither
    changes the complex default nor activates the reserved complex
    `candidate_count` parameter.
11. ETKDG-only generation remains supported by passing no force-field options.
    It is not retained through a second algorithm or a second public function.
12. The current Python-RDKit Uni-Mol feature builder is outside the conformer-
    proposal boundary. It may remain temporarily after the proposal cutover;
    removing it requires a separate native feature-builder decision.

## 2. Terminology and current-state answer

This document distinguishes three operations that must not be conflated:

- **proposal batch**: one or more coordinate sets returned by an embedding
  engine;
- **optimized ensemble**: proposed coordinate sets independently minimized by
  a force field; and
- **conformer cluster**: an optimized ensemble from which geometrically
  equivalent members have been deterministically removed.

The current repository has the following capabilities:

There is no public package or module literally named `hotpotc` in the current
tree. If `hotpotc` denotes Hotpot's C/C++ surface, that surface currently means
the standalone `Hotpot::rdwrappers` SDK described below; it is not a C++ facade
over the complete Python force-field workflows.

| Capability | Current status | Exact limitation |
|---|---|---|
| Organic native proposal batch | Available | `rdWrappers.generate_conformers()` returns K ETKDG coordinates |
| Organic native UFF/MMFF optimization | Missing | current native rdWrapper stops after ETKDG |
| Organic optimized conformer cluster | Missing | no post-optimization native clustering/ranking workflow |
| `Molecule.conformers` storage | Available | storage only; it does not generate an ensemble |
| Ordinary `build3d()` | Single structure | uses OBBuilder, not rdWrapper |
| Ordinary `build_and_optimize()` | Single structure | OBBuilder followed by the ordinary optimizer |
| Complete-complex build | Single selected structure | O0 internally explores attempts but does not expose them as an ensemble |
| Complex `candidate_count` | API placeholder | explicitly ignored by `build_complex3d()`, `complexes_build()`, and `build_and_optimize()` |
| C++ organic proposal batch | Available | standalone `Hotpot::rdwrappers` exposes native ETKDG only |
| C++ complete-complex ensemble | Missing | no public C++ O0 workflow or ensemble API exists |

Therefore Hotpot currently exposes a low-level ordinary-organic proposal-batch
interface, but it does not expose a complete ordinary-organic or complex
conformer-cluster workflow. Repeating the complete O0 workflow externally is a
valid way to obtain several accepted complex structures, but it is not a
current public ensemble operation.

## 3. Scope and non-goals

### 3.1 In scope

- add native RDKit UFF, MMFF94, and MMFF94s optimization to the existing
  conformer generator;
- materialize force-field hydrogens and project original atom rows in C++;
- optionally cluster optimized conformers in native code;
- preserve the standalone Python, C, and C++ rdWrapper components;
- make the ordinary organic coordinate proposal used by `build3d()` native;
- use the fused native route for supported force fields in the ordinary
  organic build-and-optimize workflow;
- cut MCA conformer proposal over from Python RDKit to the fused native route;
- retain exact site search and `is_Nuc_site` semantics; and
- run component, scientific, prediction, performance, resource, packaging,
  and wheel validation.

### 3.2 Explicit non-goals

- replacing O0 for complete complexes;
- describing O0 internal recovery attempts as independent conformers;
- implementing a thermodynamic population model;
- adding new geometry gates to standalone ligand generation;
- implementing GAFF or Ghemical inside rdWrapper;
- automatically updating RDKit;
- keeping a second Python-RDKit embedding/optimization fallback;
- nativeizing the Uni-Mol feature builder in this change; or
- changing native substructure search or MCA site-recognition semantics.

## 4. Public API tree and signatures

The existing `generate_conformers()` function remains the sole public organic
proposal interface. Force-field work is an option of that operation, not a new
parallel API.

### 4.1 Python API

```text
hotpot.cheminfo.rdWrappers
|- enum GenerationStatus
|- enum ForceFieldKind
|  |- AUTO
|  |- MMFF94
|  |- MMFF94S
|  `- UFF
|- enum OptimizationState
|  |- CONVERGED
|  `- MAX_ITERATIONS
|- class ForceFieldOptions
|  `- __init__(*, forcefield=ForceFieldKind.AUTO, max_iterations=200,
|              add_hydrogens=True)
|- class ETKDGOptions
|  `- __init__(*, conformer_count=1, seed=None,
|              prune_rms_threshold=None, use_random_coordinates=False,
|              optimization=None,
|              post_optimization_rms_threshold=None)
|- class OptimizationRecord
|  |- forcefield: ForceFieldKind
|  |- state: OptimizationState
|  |- rdkit_status: int
|  `- final_energy_kcal_mol: float
|- class ConformerRecord
|  |- ordinal: int
|  |- engine_conformer_id: int
|  `- optimization: OptimizationRecord | None
|- class ConformerBatch
|  |- status: GenerationStatus
|  |- coordinates: NDArray[np.float64]       # (K, input_atom_count, 3)
|  |- records: tuple[ConformerRecord, ...]
|  |- engine: str
|  |- engine_version: str
|  `- diagnostic: str | None
|- class ETKDGSampler
|  |- __init__(options: ETKDGOptions | None = None)
|  `- generate(mol: Molecule, *, constraints=None) -> ConformerBatch
|- generate_conformers(mol: Molecule, *, options=None, constraints=None)
|    -> ConformerBatch
`- runtime_info() -> RuntimeInfo
```

The signatures to freeze are:

```python
class ForceFieldKind(str, Enum):
    AUTO = "auto"
    MMFF94 = "MMFF94"
    MMFF94S = "MMFF94s"
    UFF = "UFF"

class OptimizationState(IntEnum):
    CONVERGED = 0
    MAX_ITERATIONS = 1

@dataclass(frozen=True, kw_only=True)
class ForceFieldOptions:
    forcefield: ForceFieldKind = ForceFieldKind.AUTO
    max_iterations: int = 200
    add_hydrogens: bool = True

@dataclass(frozen=True, kw_only=True)
class ETKDGOptions:
    conformer_count: int = 1
    seed: int | None = None
    prune_rms_threshold: float | None = None
    use_random_coordinates: bool = False
    optimization: ForceFieldOptions | None = None
    post_optimization_rms_threshold: float | None = None

@dataclass(frozen=True)
class OptimizationRecord:
    forcefield: ForceFieldKind
    state: OptimizationState
    rdkit_status: int
    final_energy_kcal_mol: float

@dataclass(frozen=True)
class ConformerRecord:
    ordinal: int
    engine_conformer_id: int
    optimization: OptimizationRecord | None = None

def generate_conformers(
    mol: Molecule,
    *,
    options: ETKDGOptions | None = None,
    constraints: CoordinationEmbeddingConstraints | None = None,
) -> ConformerBatch: ...
```

`optimization=None` retains coordinate-only ETKDG behavior. A non-`None`
`post_optimization_rms_threshold` requires optimization; it is not silently
applied to raw ETKDG coordinates.

The component remains nonmutating. It never writes coordinates into `mol` or
`mol.conformers`. Callers deliberately store or commit returned coordinates.

### 4.2 Deterministic ensemble and clustering semantics

The native implementation performs these operations in order:

1. embed candidate conformers in RDKit engine-conformer order;
2. optimize each candidate serially with RDKit internal thread count one;
3. discard a candidate only when embedding failed, force-field setup failed,
   optimization returned a negative status, coordinates became non-finite, or
   final energy became non-finite;
4. if clustering is disabled, return surviving conformers in original ordinal
   order;
5. if clustering is enabled, sort candidates by
   `(final_energy_kcal_mol, original_ordinal)`;
6. greedily retain a candidate only when its symmetry-aware heavy-atom RMSD to
   every retained representative is at least the configured threshold; and
7. return representatives in that same deterministic energy/ordinal order.

RDKit `MolAlign` symmetry-aware RMSD is used; fixed atom-row Kabsch RMSD is not
chemically sufficient for cluster membership. The already locked ETKDG closure
contains `MolAlign_static`, so this adds no dependency family.

Clustering is an intentional cardinality reduction and does not by itself make
a batch partial. A batch is `PARTIAL` only when embedding or optimization
failed for at least one requested candidate and at least one candidate still
survived. If none survive, the status is `FAILED`. The diagnostic separately
reports requested, embedded, optimized, failed, clustered, and returned counts
without inventing replacement coordinates.

### 4.3 C ABI evolution

The native library keeps one `hotpot_rd_generate()` function and one core
pipeline. **A Hotpot release accepts exactly its installed ABI version and
full structure sizes.** Extending the component replaces the option structure
atomically; it does not accept an older prefix, translate a legacy request,
publish parallel v1/v2 entry points, or retain a compatibility parser.

- add force-field and clustering fields to the one current option structure;
- require its exact full `struct_size`;
- retain one conformer-record representation for the release; and
- add one result accessor returning parallel immutable optimization rows.

```c
typedef enum hotpot_rd_forcefield_kind {
    HOTPOT_RD_FORCEFIELD_NONE = 0,
    HOTPOT_RD_FORCEFIELD_AUTO = 1,
    HOTPOT_RD_FORCEFIELD_MMFF94 = 2,
    HOTPOT_RD_FORCEFIELD_MMFF94S = 3,
    HOTPOT_RD_FORCEFIELD_UFF = 4
} hotpot_rd_forcefield_kind;

typedef struct hotpot_rd_options_v1 {
    /* Existing current-release fields remain first. */
    uint32_t has_optimization;
    uint32_t forcefield_kind;
    uint64_t max_iterations;
    uint32_t add_hydrogens;
    uint32_t has_post_optimization_rms_threshold;
    double post_optimization_rms_threshold;
} hotpot_rd_options_v1;

typedef struct hotpot_rd_optimization_record_v1 {
    uint64_t ordinal;
    uint32_t forcefield_kind;
    int32_t rdkit_status;
    double final_energy_kcal_mol;
} hotpot_rd_optimization_record_v1;

HOTPOT_RD_API const hotpot_rd_optimization_record_v1 *
hotpot_rd_result_optimization_records(
    const hotpot_rd_result *result,
    uint64_t *record_count);
```

The fields are stored inline in the one installed option structure, not passed
through an extension pointer or exposed through a second create/generate
function. Flag fields are restricted to zero or one, and `max_iterations` must
fit the pinned RDKit helper's supported integer range. Git history is the only
old-ABI recovery mechanism.

### 4.4 C++ API

```text
namespace hotpot::rdwrappers
|- enum class ForceFieldKind
|- enum class OptimizationState
|- struct ForceFieldOptions
|- struct OptimizationRecord
|- struct ETKDGOptions                 # optional ForceFieldOptions + cluster RMS
|- struct ConformerRecord              # optional OptimizationRecord
|- struct ConformerGenerationResult
|- class ETKDGSampler
|  `- generate(MoleculeBufferView, optional<constraints>) const
`- generate_conformers(MoleculeBufferView, options, optional<constraints>)
```

The installed C++ facade continues to copy C result data into containers owned
by the consumer translation unit. RDKit, Boost, STL containers, and exceptions
do not cross the shared-library boundary.

The corresponding signatures are frozen as:

```cpp
struct ForceFieldOptions {
    ForceFieldKind forcefield = ForceFieldKind::Auto;
    std::uint64_t max_iterations = 200;
    bool add_hydrogens = true;
};

struct ETKDGOptions {
    std::uint64_t conformer_count = 1;
    std::optional<std::int64_t> seed = std::nullopt;
    std::optional<double> prune_rms_threshold = std::nullopt;
    bool use_random_coordinates = false;
    std::optional<ForceFieldOptions> optimization = std::nullopt;
    std::optional<double> post_optimization_rms_threshold = std::nullopt;
};

class ETKDGSampler final {
public:
    explicit ETKDGSampler(ETKDGOptions options = {});
    [[nodiscard]] ConformerGenerationResult generate(
        native::MoleculeBufferView molecule,
        std::optional<CoordinationEmbeddingConstraintsView> constraints =
            std::nullopt
    ) const;
};

[[nodiscard]] ConformerGenerationResult generate_conformers(
    native::MoleculeBufferView molecule,
    ETKDGOptions options = {},
    std::optional<CoordinationEmbeddingConstraintsView> constraints =
        std::nullopt
);
```

## 5. Native scientific implementation

### 5.1 Molecule construction and hydrogen lifecycle

The native core reconstructs one RDKit `RWMol` from the common molecule view.
The input atom rows are recorded before hydrogen expansion. When optimization
requests `add_hydrogens=True`, RDKit materializes only hydrogens implied by the
transported graph/hydrogen state. Existing explicit graph hydrogens remain
ordinary input rows and are not duplicated.

ETKDG and the force field operate on the same hydrogen-expanded native
molecule. Returned coordinates contain exactly the original input rows in the
original order; native-only appended hydrogen rows are projected out. This
removes the current Python `_materialized_hydrogen_projection` requirement for
ordinary organic proposals while preserving its atom-mapping guarantee.

### 5.2 Embedding

The implementation retains ETKDGv3, `numThreads=1`, the explicit caller seed,
the current pruning option, and the current `useRandomCoords` option. A
one-conformer request uses RDKit's one-conformer embedding operation so the MCA
reference is not assumed to be equivalent to `EmbedMultipleConfs(count=1)`.
Multi-conformer requests use the existing multiple-conformer operation.

The ordinary `build3d(add_hydrogens=True)` workflow continues to create its
explicit Hotpot working-copy hydrogen rows before calling coordinate-only
ETKDG, because those graph rows are part of the public committed molecule.
This is Hotpot graph preparation, not a Python-RDKit conformer operation.

### 5.3 Force-field routing

`ForceFieldKind.AUTO` implements only this rule:

```text
MMFFHasAllMoleculeParams(molecule)
  ? MMFF94
  : UFF
```

Explicit `MMFF94`, `MMFF94s`, or `UFF` requests do not silently change backend.
Every returned conformer records the effective force field, raw RDKit status,
interpreted convergence state, and final RDKit energy in kcal/mol.

The native code uses the official pinned RDKit force-field helpers. It does not
copy or fork RDKit parameter tables. `ForceFieldHelpers_static`,
`ForceField_static`, `Optimizer_static`, and `Trajectory_static` are already in
the locked 2026.09.1 link closure; the target/link proof and binary-size record
are refreshed after direct use.

## 6. Workflow integration

### 6.1 Backend-neutral adapter

`RDKitConformerAdapter` is changed from a hard-coded one-conformer selector to
a faithful adapter of `ConformerSamplingRequest.conformer_count`. It is
configured with optional native `ForceFieldOptions` and returns the complete
native batch in stable order. It does not implement embedding or optimization.

`OpenBabelConformerAdapter` remains available for O0 complex construction. No
automatic fallback from RDKit to Open Babel is added to an organic request.

### 6.2 Ordinary Hotpot organic workflows

The cutover is deliberately limited by operation:

| Operation | Planned behavior |
|---|---|
| `rdWrappers.generate_conformers()` | sole public batch/cluster proposal interface |
| organic `forcefields.build3d()` | native ETKDG proposal, one conformer, no FF |
| organic `forcefields.build_and_optimize()` with UFF/MMFF | fused native ETKDG + requested RDKit FF |
| organic `build_and_optimize()` with GAFF/Ghemical | native ETKDG proposal followed by the existing explicitly requested optimizer |
| `forcefields.optimize()` on existing coordinates | unchanged; it is not a proposal operation |
| any complete-complex operation | unchanged O0 behavior |

No new public registry, profile, or backend selector is introduced. The
existing public function decides from molecular scope and the already supplied
force-field name. A returned conformer cluster is not automatically written to
`Molecule.conformers`; the standalone caller performs that explicit storage
operation. Single-structure workflows commit only their selected conformer.

### 6.3 MCA cutover

MCA uses this exact native request:

```python
ETKDGOptions(
    conformer_count=1,
    seed=42,
    optimization=ForceFieldOptions(
        forcefield=ForceFieldKind.AUTO,
        max_iterations=200,
        add_hydrogens=True,
    ),
)
```

Normalization packs the already-created Hotpot molecule into the common
field-selective `NativeMoleculeBuffers` once and carries those spawn-safe arrays
in `MoleculeDTO`. The feature worker calls the native binding with those
buffers directly. It must not reconstruct an intermediate Python RDKit
molecule merely to call rdWrapper.

During this extension the returned heavy-atom coordinates may still be
attached to the current Python RDKit feature molecule because
`mol_to_unimolv2()` remains Python-RDKit based. That attachment is outside the
proposal operation and must be measured separately. A later native feature
builder can remove `MoleculeDTO.rdkit_binary`; this plan makes no such claim.

Site detection remains on the same Hotpot DTO and native search path. Conformer
replacement cannot alter site records or `is_Nuc_site`.

## 7. Small-step implementation sequence

Every completed step is appended to `implementation_log.md` with changed
files, commands, results, unexpected findings, and the next gate. If a finding
changes the plan, the complete plan is reread before revision.

### Step 0 — Freeze extension baselines

- record the current branch and source identities;
- bind the existing R0, 182-case O0, 256-molecule MCA, CPU, and GPU artifacts;
- rerun only lightweight contract tests needed to confirm the local baseline;
- record current wheel size, extension size, exported symbols, and link map.

### Step 1 — Freeze the replacement API contract

- add the Python enums and immutable option/evidence records;
- replace `ETKDGOptions` and `ConformerRecord` at their existing public names;
- add C and C++ declarations;
- replace the C option structure and keep exact-size validation;
- add API-shape and wrong-size rejection tests before implementation.

### Step 2 — Prove the direct force-field closure

- include the pinned RDKit UFF/MMFF helper headers;
- link direct force-field use through the existing static closure;
- update the expected target link map only if the measured target graph changes;
- fail if a new shared or unpinned dependency appears.

### Step 3 — Implement native hydrogen expansion and projection

- preserve original atom-row identities;
- materialize implied hydrogens inside native RDKit;
- embed on the expanded molecule;
- project returned coordinate rows back to the input graph;
- test explicit, implicit, bracket-explicit, isotopic, and stereo-carrier H cases.

### Step 4 — Implement native UFF/MMFF optimization

- implement AUTO/MMFF94/MMFF94s/UFF routing;
- optimize each conformer serially;
- capture raw status, convergence state, effective force field, and finite final
  energy;
- implement partial/failed batch status without substitute coordinates.

### Step 5 — Implement native post-optimization clustering

- use symmetry-aware heavy-atom RMSD;
- freeze energy/ordinal representative selection and output order;
- test symmetric substituents, enantiomer preservation, duplicates, and threshold
  boundary behavior.

### Step 6 — Publish evidence through C++, C, and Python

- add the result accessor and facade copies;
- update pybind construction and type stubs;
- retain the no-optimization output contract;
- prove Python/C/C++ coordinate and metadata equality on shared fixtures.

### Step 7 — Execute the optimized R0 matrix

- run all 187 extractants at requested counts 1, 8, and 32;
- execute AUTO, explicit MMFF94/MMFF94s coverage fixtures, and UFF fallback
  fixtures;
- record completion, partial results, finite energy/coordinates, diversity,
  clustering, determinism, and timing.

### Step 8 — Cut over ordinary organic workflows

- route organic `build3d()` through native ETKDG;
- route supported organic build-and-optimize requests through the fused path;
- retain explicit GAFF/Ghemical optimizer behavior after native proposal;
- leave `optimize()` and every complex path unchanged;
- update public documentation without adding a backend/profile option.

### Step 9 — Cut over MCA conformer proposal

- add shared native buffers to the DTO;
- pass seed 42 and AUTO/200 explicitly;
- remove Python embedding and force-field calls from production MCA proposal;
- remove superseded proposal glue only after parity tests pass;
- retain the current Python feature builder in this step.

### Step 10 — Run integration, scientific, and performance gates

- execute all matrices in Sections 8 and 9;
- diagnose every failed molecule individually;
- do not average away missing or failed cases;
- compare resource usage and installed artifacts.

### Step 11 — Documentation and final report

- update Stage-17 implementation report and README;
- report the final implementation route, API, deviations from plan, tests,
  scientific results, performance tables, package audit, and remaining limits;
- do not claim the Python-RDKit MCA dependency is gone while the feature builder
  still imports RDKit.

## 8. Test plan

### 8.1 Contract and component tests

| ID | Test | Required result |
|---|---|---|
| C01 | Existing ETKDG-only Python calls | unchanged output contract |
| C02 | Non-current C option size/version | rejected without translation |
| C03 | Python/C/C++ shared fixture | identical coordinates and records |
| C04 | Same seed repeated three times | bit-identical native batch |
| C05 | Seeds 42/43/44 | different batches allowed; each reproducible |
| C06 | Input molecule snapshot | topology, coordinates, stereo, conformer storage unchanged |
| C07 | Original-row projection after native H expansion | exact shape and atom order |
| C08 | AUTO with complete MMFF coverage | records MMFF94 |
| C09 | AUTO without complete MMFF coverage | records UFF |
| C10 | Explicit MMFF94/MMFF94s/UFF | no silent backend substitution |
| C11 | RDKit status 0 / positive / negative | converged / accepted-limit / failed semantics |
| C12 | Non-finite coordinate or energy fixture | rejected, never returned as successful |
| C13 | Multi-conformer partial failure | surviving conformers returned as `PARTIAL` |
| C14 | Symmetry-aware cluster fixture | deterministic chemically equivalent collapse |
| C15 | Stereo and enhanced-group fixtures | represented semantics unchanged |
| C16 | Explicit and implicit hydrogen fixtures | no duplication or row remapping |
| C17 | Clustering on detached coordinate copies | returned coordinates are not changed by RMSD alignment |

### 8.2 Workflow tests

| ID | Test | Required result |
|---|---|---|
| W01 | organic `build3d()` | native ETKDG called once; one structure committed |
| W02 | organic UFF/MMFF build-and-optimize | fused native route used |
| W03 | explicit GAFF/Ghemical | native proposal plus requested existing optimizer |
| W04 | existing-coordinate `optimize()` | unchanged routing |
| W05 | molecule containing a metal | O0 path unchanged |
| W06 | complex `candidate_count` | remains reserved and behavior-neutral |
| W07 | MCA CPU preprocessing | seed 42 and AUTO/200 observed |
| W08 | MCA GPU feature workers | same native request and stable ordered join |
| W09 | MCA site output | ordered sites and `is_Nuc_site` exact |
| W10 | feature failure | original native diagnostic reaches the user |

### 8.3 Release tests

- locked offline source build;
- standalone C and dual-libstdc++-ABI C++ consumers;
- CPython 3.10-3.14 wheel matrix;
- installed-wheel smoke with no source tree on `PYTHONPATH`;
- ELF dependency, RPATH/RUNPATH, build-ID, exported-symbol, and archive-content
  audits;
- license and notice verification; and
- wheel and `_rd_native` size comparison with the frozen baseline.

## 9. Scientific validation and benchmark matrix

### 9.1 Fixed corpora

| Corpus | Purpose |
|---|---|
| canonical 187 extractants | standalone organic generation, FF coverage, 1/8/32 ensemble behavior |
| fixed 182 CBond-positive cases | prove all complete-complex behavior remains O0 and unchanged |
| stratified 256 MCA molecules | paired predictions, seed-42 parity, CPU/GPU throughput |
| focused stereo/H/symmetry fixtures | semantic transport and clustering correctness |

The existing corpus hashes and case order remain authoritative. A changed
corpus requires a new benchmark identity, not an overwritten result.

### 9.2 Standalone R0-native-FF matrix

For every one of the 187 extractants, request 1, 8, and 32 conformers with the
same per-case seed convention used by R0. Record:

- requested, embedded, optimized, failed, clustered, and returned counts;
- effective force field per conformer;
- RDKit status and finite final energy;
- input atom-row and topology preservation;
- fixed-seed repeat equality;
- symmetry-aware pairwise RMSD and cluster membership;
- wall time per phase and peak RSS; and
- every failure diagnostic and retained visualizable frame.

The component passes only if every returned conformer has finite coordinates,
finite final energy, exact original-row mapping, and a recorded effective force
field. Partial results are reported rather than promoted to full success.

### 9.3 Ordinary-organic workflow validation

Run the public single-structure `build3d()` and `build_and_optimize()` workflows
over all 187 extractants. The former uses native coordinate-only ETKDG; the
latter is tested with the public default UFF plus the supported MMFF routes.

The old Open Babel structure is a paired reference, not a coordinate answer.
The native route is accepted per molecule only when:

- the workflow returns a final structure with exact source heavy-atom identity
  and topology;
- all committed coordinates are finite;
- the selected force-field energy is finite;
- optimization can obtain a direction or recover through an already
  established recovery action;
- the existing ordinary-organic geometry acceptance gate passes; and
- no new geometry inspection or perturbation stage has been added solely to
  favor the native route.

Every failed case retains its final or last useful frame for diagnosis. A
gallery is generated for paired human inspection, but a missing active report
of nonphysical appearance is the only manual condition; the Open Babel
coordinates are not treated as ground truth.

### 9.4 MCA paired numerical validation

Run the current pinned Python RDKit reference and native candidate with the
same 256 molecules, seed 42, model artifacts, site search, feature code,
batching, and backend. Required comparisons are:

- molecule completion count;
- exact ordered `(atom_index, site_type)` records;
- exact `is_Nuc_site` for all atoms;
- per-molecule force-field routing;
- heavy-atom coordinate hashes and maximum absolute coordinate deviation;
- all-atom and nucleophilic-site Pearson, Spearman, signed error, MAE, RMSE,
  P95 absolute error, and maximum absolute error; and
- per-failure molecule analysis.

Because both routes use the same pinned RDKit implementation and call policy,
bit-identical coordinates and predictions are the target. Correlation alone is
not sufficient to authorize default cutover. Any non-bit-identical result is
classified and brought back to the owner with atom-level evidence before the
default changes.

### 9.5 Performance configurations

Measure after one warm-up and report at least three timed repetitions:

| Layer | Baseline | Candidate | Resources |
|---|---|---|---|
| proposal microbenchmark | Python RDKit ETKDG+MMFF/UFF | fused native rdWrapper | 1 CPU and 8 CPUs |
| R0 187 x 1/8/32 | native ETKDG only plus observation | native ETKDG+FF+cluster | same fixed CPUs |
| MCA CPU | current Python RDKit | native fused proposal | 8 CPUs, batch 64 |
| MCA GPU | current Python RDKit | native fused proposal | one A800, 1 site CPU + 3 feature CPUs, batch 2048 |
| ordinary organic build | OBBuilder/current optimizer | native proposal/fused supported FF | same 187 corpus and CPU set |

Report median wall time, molecules/s, conformers/s, phase timing, CPU
utilization, peak RSS, worker count, extension size, and wheel size. No result
from concurrently loaded CPUs or GPUs is used as formal throughput evidence.

The provisional non-regression gate is candidate median end-to-end wall time no
more than 2% above the paired baseline on both MCA configurations. A larger
regression keeps the component interface but blocks the default workflow
cutover pending owner review. Speedup is reported only when it exceeds run-to-
run noise; native implementation alone is not described as a performance win.

### 9.6 Default-cutover gates

The native organic proposal becomes the default only when all applicable gates
pass:

1. one native implementation serves Python and C++;
2. exact input atom-row, topology, hydrogen, and represented-stereo semantics;
3. correct force-field route and optimization-status semantics;
4. finite coordinates and final energy for every returned conformer;
5. same-seed deterministic output;
6. exact MCA ordered sites and `is_Nuc_site`;
7. native-versus-reference MCA numerical differences resolved and owner
   accepted;
8. no change to the 182-case O0 complex path;
9. CPU and GPU non-regression gate met; and
10. wheel, SDK, binary, license, and interpreter matrices pass.

## 10. Expected file ownership

Implementation is expected to remain within these existing boundaries:

```text
hotpot/cheminfo/_native/buffers.py
hotpot/cheminfo/rdWrappers/
  contracts.py
  sampler.py
  native.py
  _rd_native.pyi
  CMakeLists.txt
  _native/include/hotpot/rdWrappers/{c_api.h,rdwrapper.hpp}
  _native/src/{core.hpp,validation.cpp,sampler.cpp,c_api.cpp,python_binding.cpp}
hotpot/cheminfo/conformers/{contracts.py,adapters.py}
hotpot/cheminfo/forcefields/workflows.py
hotpot/cheminfo/AImodels/mca/{serialization.py,conformer.py,pipeline.py,workers.py}
third_party/rdkit/probe/
tests/test_cheminfo/rdWrappers/
tests/test_cheminfo/forcefields/
tests/test_cheminfo/test_rdwrapper_api.py
tests/test_packaging/
plan/17_rdkit_wrapper/
```

The list is a boundary, not permission for unrelated edits. Production code is
changed in small reviewable steps, and existing unrelated worktree changes are
preserved.

## 11. Risks and stopping conditions

| Risk | Required response |
|---|---|
| Native graph differs from current MCA RDKit graph | stop cutover; fix transport rather than loosen numerical gates |
| `EmbedMolecule` and one-count multi-embed differ | use the exact one-conformer call for MCA |
| MMFF/UFF status differs from Python helper behavior | reproduce helper semantics before workflow integration |
| Post-FF clustering collapses stereoisomers | block clustering and fix symmetry/stereo mapping |
| Static closure or wheel gains an unplanned dependency | reject build and update no lock until owner review |
| MCA throughput regresses beyond the gate | retain component, do not make it the default |
| O0 complex behavior changes | revert the integration step; complex path is out of scope |
| Python RDKit is still imported by featurization | report truthfully; do not claim full MCA RDKit-Python removal |

Git history is the rollback mechanism. No duplicate legacy production backend,
hidden profile, or silent fallback is retained in the source tree.
