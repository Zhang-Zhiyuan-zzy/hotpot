# Native RDKit wrapper and pluggable conformer-sampling implementation plan

Status: owner-approved on 2026-10-22; implementation is in progress.

Planning branch: `feature/rdWrapper`

Planning baseline: `9640d906a9d13e969152d0f3a4d92a3a16b9e0ce`

## 1. Objective and governing decisions

This stage will provide a standalone native RDKit conformer component and then
determine, independently, how much that component can contribute to Hotpot's
existing force-field workflow. Native ETKDG is useful as a public component
even if the force-field benchmark does not justify making it the production
ligand builder.

The intended production responsibility split is:

```text
conformer sampler
    -> coordinates only
Open Babel force field
    -> ligand and/or complete-complex numerical relaxation
existing Hotpot coordination and geometry workflow
    -> topology, placement, recovery, acceptance, and reporting
```

The following are normative owner requirements or direct consequences of
them:

1. Planning precedes implementation. No production path is changed before the
   owner approves this document.
2. **One Hotpot version supports exactly one RDKit version.** The baseline for
   this stage is the current stable **RDKit 2026.09.1** release. Hotpot's Python
   dependency, vendored native source, compatibility layer, build metadata,
   and runtime version report must all identify that exact release.
3. **RDKit upgrades are deliberately lazy and owner-controlled.** Dependency
   ranges, a resolver, CI against `latest`, or routine Hotpot release work must
   never advance the version. It changes only after an explicit owner
   instruction and a complete compatibility/benchmark review.
4. The fixed 187-ligand corpus and its 182 CBond-positive complex cohort decide
   whether ETKDG replaces a component in the force-field path. ETKDG's value as
   an independent conformer-generation component does not depend on that
   cutover decision.
5. A sampler performs coordinate generation only. It does not mutate Hotpot
   topology, optimize with a force field, apply Hotpot geometry acceptance, or
   perturb coordinates.
6. The standalone RDKit sampling path adds no Hotpot geometry scan,
   perturbation, or acceptance gate. Diagnostic measurements in an offline
   benchmark are allowed, but must not influence the generated conformers.
7. **This stage does not change the order, checks, recovery policy, or overall
   semantics of Hotpot's force-field optimization workflow.** Force-field work
   is limited to substituting the coordinate-generation component and testing
   the result. Existing ligand, placement, coordination-restoration,
   ring-untangling, optimization, and final-acceptance stages remain in place.
8. Conformer sampling remains replaceable. Open Babel and RDKit are current
   adapters, not permanent definitions of Hotpot conformer science.
9. **`rdWrappers` exposes both a Python API and a C++ API.** The C++ library has
   no Python dependency; the Python extension is a thin binding over the same
   library. Either interface can run as an independent component.
10. **`rdWrappers` remains available and is tested as a standalone conformer
    interface regardless of its eventual role in `hotpot.ff`. `obWrappers`
    also remains available even if a force-field workflow stops using its
    builder.** This is component retention, not two competing production
    force-field routes.
11. Hotpot keeps one composed default force-field route. This stage adds no
    force-field `backend=`, named profile, compatibility flag, or user-facing
    switch between complete workflows.
12. Native force-field execution must not construct or translate through a
    Python `rdkit.Chem.Mol`. RDKit objects, when used, are private C++
    implementation details.
13. The existing Hotpot atom order, explicit topology, formal charges,
    aromaticity, bond kinds, coordination direction, and represented
    stereochemistry are invariant across sampling.
14. RDKit 2026.09.1 publishes Python wheels for CPython 3.10-3.15 but not 3.9
    and publishes no source distribution. **To preserve the exact-version
    decision without taking ownership of RDKit's complete Python bindings,
    this Hotpot release supports Python 3.10-3.14.**
15. Existing Hotpot/Open Babel native modules remain C++17. The public
    `rdWrappers` C/C++ boundary is C++17-consumable, while only the private
    pinned RDKit implementation and its dependency closure compile as C++20.
16. Enhanced-stereo group identity includes an explicit persistent lineage
    scope. Copying or splitting a molecule preserves that scope; independently
    imported molecules receive distinct scopes so coincident toolkit group
    numbers cannot merge accidentally during component assembly.
17. Hydrogen transport separates the immutable source-representation ledger
    from the current graph state used for native construction. Materializing
    explicit hydrogen atoms updates only current state and must never cause
    native RDKit to count the source implicit/bracket hydrogen values again.
18. The exact offline native-source closure is locked, not only RDKit and
    Boost. Any upstream archive fetched by the pinned minimal CMake graph is
    identified and SHA-verified before configuration; network fallback and
    moving Git references are forbidden.

## 2. Confirmed current-state audit

### 2.1 The current native molecule boundary is not `MoleculeView`

The closest current types are:

- Python `_MoleculeBuffers` in
  `hotpot/cheminfo/obWrappers/packing.py`;
- C++ `hotpot::obwrappers::MoleculeData` in
  `hotpot/cheminfo/obWrappers/_native/molecule_data.hpp`; and
- Python/C++ `ComplexSessionInput` in the force-field package.

`_MoleculeBuffers` already stores contiguous atomic numbers, formal charges,
partial charges, coordinates, atom aromatic flags, bond endpoints, bond
orders, bond kinds, bond aromatic flags, and an optional unit cell. The
pybind boundary then copies every supplied array into owning C++ vectors.

The schema is chemically general in shape but not in ownership or policy:

- every field is mandatory even when an operation does not consume it;
- its Python and C++ names live under `obWrappers`;
- its base validation rejects dative bonds because Open Babel cannot represent
  them losslessly; and
- `ComplexSessionInput` duplicates most base fields instead of composing a
  common molecule payload.

These restrictions are adapter policy, not a Hotpot-wide native molecule
contract.

### 2.2 Current complex construction stages

The production complete-complex workflow is:

```text
Hydrogenated working copy
    -> hide metal-ligand bonds and split components
    -> Stage 1 ligand proxy construction
    -> optional reserved coordination-geometry hook (not implemented)
    -> native metal placement
    -> incremental coordination-bond restoration
    -> complete-complex ring checkpoint/repair
    -> complete-complex Open Babel optimization
    -> post-optimization ring checkpoint/repair
    -> terminal structure acceptance
    -> atomic commit to the caller molecule
```

Stage 1 currently calls OBBuilder once per metal-free component. It then uses
the first built geometry as the center of up to `max_attempts` Cartesian
Gaussian perturbations. Every attempted geometry is relaxed by Open Babel.
The loop stops at the first candidate that passes the existing basic ligand
gate; it is not torsion-space conformer sampling.

`candidate_count` is present on `build_complex3d()`, `complexes_build()`, and
`build_and_optimize()`, but is documented as reserved and ignored.

### 2.3 Existing geometry-review inventory

The table below distinguishes already executed geometry work from a proposed
new check. “Retain” means this plan must not duplicate, widen, or move the
check merely because RDKit is introduced.

| Stage | Current geometry work | Current effect | Plan |
|---|---|---|---|
| OBBuilder output | No independent acceptance immediately after the single build call | Coordinates become the perturbation center | The sampler itself also has no new Hotpot gate |
| Ligand candidate warm-up | Full ligand-skeleton bond-ring scan | Supplies entry evidence for ring repair | **Retain unchanged in the force-field integration test** |
| Ligand ring recovery | Watched/full ring scans, ring opening, Cartesian perturbation, Open Babel settling | Repairs confirmed ligand bond-ring piercing | **Retain unchanged in the force-field integration test** |
| Ligand candidate acceptance | Existing checkpoint plus `basic` acceptance: coordinate/topology/force-field facts, all-pair distances, explicit-bond geometry, and ring facts | Accepts first passing candidate or records a usable fallback | **Retain unchanged in the force-field integration test** |
| Best ligand refinement | Long Open Babel refinement followed by another full ligand scan, optional untangling, and another `basic` acceptance | Commits or rolls back refined ligand coordinates | **Retain unchanged in the force-field integration test** |
| Metal placement | Candidate metal-center collision, donor-distance, atom-path, bond-path, ring-path, donor-pair, and clearance evaluation | Selects a feasible metal position | Retain; intrinsic to existing placement semantics |
| Coordination restoration | Each proposed metal-donor segment is screened against prepared rings; stalled restoration may perturb and relax | Accepts, retries, or ultimately forces coordination bonds | Retain; no extra screening added |
| Complex entry | Full-graph bond-ring checkpoint | Runs native untangling or blocks numerical optimization | Retain |
| Complex optimization exit | Full-graph checkpoint after numerical optimization and after stabilization | Repairs optimizer-created piercing and selects evidence | Retain |
| Final acceptance | Reuses the native final ring checkpoint and computes the selected acceptance level's remaining checks | Produces the public quality report | Retain; never repeat the ring scan |
| `build_complex3d()` terminal `off` acceptance | Shape, finite-coordinate, topology, and metric collection; no standard geometry gate | Rejects an unreturnable structure | Retain |

The expensive ligand-only work is therefore not hypothetical: the current
Stage 1 can perform a full ring scan, all-pair distance measurement, explicit
bond checks, targeted perturbation/untangling, and a second scan after long
refinement. Introducing ETKDG without restructuring Stage 1 would retain all
of that work automatically.

### 2.4 Standalone generation versus force-field integration

These are deliberately separate test surfaces:

```text
standalone rdWrappers
    ETKDG conformer generation -> returned conformer batch
    (no Hotpot geometry gate, perturbation, or force-field optimization)

hotpot.ff component-substitution test
    native ETKDG coordinates
        -> the same existing ligand warm-up/recovery/acceptance/refinement
        -> the same placement/restoration/complex optimization/final acceptance
```

The standalone interface directly returns RDKit's conformers because adding a
Hotpot geometry policy inside `rdWrappers` would destroy component
independence. The force-field test, by contrast, changes only the initial
coordinate provider. **It does not remove, reorder, or weaken any current
Hotpot force-field stage in this implementation round.**

Benchmark code may measure standalone output after generation has completed.
That post hoc observation is timed separately and cannot reject, repair,
select, or perturb the returned conformers. Any future proposal to delete an
existing force-field geometry stage requires a separate plan and owner
decision after this component-substitution evidence exists.

### 2.5 Molecule lifecycle prerequisites discovered during implementation

Level-1 stereo cannot be correct if the underlying structural lifecycle loses
or aliases graph state. The implementation audit found that current
`Molecule.copy()` and `add_component()` omit hidden bonds, atom removal can
leave incident bonds in hidden-bond lists, and `components` can share the
source atom NumPy attribute arrays. Those are existing structural defects, not
stereo semantics.

Phase 2 repairs and tests those behaviors in isolated checkpoints before
stereo remapping is connected. Copy/component extraction must own independent
atom arrays; active and hidden bonds must be cloned/merged with stable endpoint
mapping; deleting an atom must remove every active or hidden incident bond.
Hide/recover itself does not change Level-1 payload bytes. Permanently clearing
or deleting a bond filters only records and bond facts owned by that bond.

## 3. Scientific hypotheses and routes under test

### 3.1 H1: ligand-only ETKDG

The first hypothesis is that ETKDG supplies more chemically plausible and
diverse organic ligand starting conformers than one OBBuilder geometry plus
Cartesian perturbations.

The test route is:

```text
hide metal-ligand bonds
    -> sample each metal-free component with ETKDGv3
    -> supply selected initial coordinates at the current builder seam
    -> execute the unchanged ligand warm-up/recovery/acceptance/refinement
    -> restore the component coordinates in stable Hotpot atom order
    -> continue the existing complex workflow
```

If several RDKit conformers are generated for a component-substitution
experiment, the adapter selects one deterministically by RDKit conformer
ordinal before entering the unchanged force-field workflow. Activating a new
energy-selection loop inside production would change that workflow and is
outside this stage. RDKit's own embedding-time conformer pruning belongs to
the standalone sampler; it is not a Hotpot acceptance policy.

### 3.2 H2: stock whole-complex ETKDG as coordinate generation only

The existing 30/182 RDKit result combines whole-complex ETKDG with RDKit
MMFF/UFF optimization, so it does not isolate the embedder. A coordinate-only
route is required to report whether graph construction and ETKDG itself
succeed before any force field is involved.

A separate **standalone research arm** therefore constructs the exact Hotpot
complex graph inside native RDKit for the experiment, represents directed
coordination bonds explicitly, embeds coordinates, and then discards the
RDKit molecule. It performs no RDKit force-field optimization.

```text
whole-complex ETKDGv3
    -> coordinates only
    -> detached benchmark observation
```

This route characterizes whether ETKDG can embed the graph. It is not connected
to `hotpot.ff` in this implementation round because doing so would replace
Hotpot metal placement and therefore change the overall workflow.

### 3.3 H3: coordination-aware ETKDG extension

ETKDG can potentially be extended for coordination complexes without adding
metal UFF/MMFF parameters because distance-geometry embedding and molecular
mechanics parameterization are separate operations.

The first extension must use supported RDKit distance-geometry hooks rather
than forking RDKit's embedder:

1. build the complete graph in stable Hotpot atom order;
2. retain ligand covalent/aromatic topology and ETKDG torsion knowledge;
3. overlay lower/upper metal-donor distance bounds derived from the same
   Hotpot coordination-distance policy already used by native placement;
4. when an explicit coordination geometry is available, add compatible
   donor-donor bounds rather than inventing a new geometry model;
5. triangle-smooth and embed the augmented bounds matrix;
6. optionally anchor metal coordinates through the RDKit coordinate-map
   mechanism when that is meaningful for the selected version;
7. return only original-atom coordinate rows; and
8. perform no force-field relaxation inside `rdWrappers`.

Two constructions must be characterized independently as standalone research:

- **metal-node embedding**: the real metal remains in the distance graph and
  metal-donor bounds constrain its shell;
- **ligand preorganization**: donor-donor constraints preorganize a ligand,
  while existing Hotpot metal placement remains authoritative.

Ligand preorganization is architecturally safer because ETKDG remains an
organic-ligand sampler and Hotpot retains metal placement. Metal-node
embedding may be useful for multiple ligand components but is more exposed to
unsupported sanitization, bound smoothing, and dative-bond behavior.

If supported bounds/coordinate hooks cannot express the required constraints,
this stage records that limitation. It must not begin an invasive RDKit source
fork. A later Hotpot-native scientific sampler should implement the same
backend-neutral sampler contract. No H2/H3 result changes the current
force-field stage order in this round.

## 4. Public API and pluggable conformer architecture

### 4.1 Existing force-field API is unchanged

**No existing public force-field function gains a `backend=`, `sampler=`, or
profile parameter in this stage.** The relevant current public functions and
their behavior remain authoritative:

```text
hotpot.cheminfo.forcefields
|- build3d(mol, *, add_hydrogens=True, seed=None, timeout=1000.0)
|- build_complex3d(mol, forcefield=None, *, candidate_count=None, ..., seed=None)
|- complexes_build(mol, forcefield=None, *, candidate_count=None, ..., seed=None)
`- build_and_optimize(mol, forcefield="UFF", *, candidate_count=None, ..., seed=None)
```

`candidate_count` remains reserved in the force-field interface during this
round. Making it affect force-field control flow requires a later plan. The
integration adapter selects the deterministic first successful RDKit
conformer and supplies its coordinates at the current builder seam.

### 4.2 Public Python `rdWrappers` API

The independent public Python component is concrete at class/function level:

```text
hotpot.cheminfo.rdWrappers
|- enum GenerationStatus
|- class ETKDGOptions
|  `- __init__(*, conformer_count=1, seed=None,
|              prune_rms_threshold=None, use_random_coordinates=False)
|- class CoordinationEmbeddingConstraints
|  `- __init__(coordination_pairs, metal_donor_bounds,
|              donor_pair_indices=None, donor_pair_bounds=None)
|- class ConformerRecord
|  |- ordinal: int
|  `- engine_conformer_id: int
|- class ConformerBatch
|  |- status: GenerationStatus
|  |- coordinates: NDArray[np.float64]          # (K, N, 3)
|  |- records: tuple[ConformerRecord, ...]
|  |- engine: str
|  |- engine_version: str
|  `- diagnostic: str | None
|- class RuntimeInfo
|  |- hotpot_version: str
|  |- rdkit_version: str
|  `- native: bool
|- class ETKDGSampler
|  |- __init__(options: ETKDGOptions | None = None)
|  `- generate(mol: Molecule, *, constraints=None) -> ConformerBatch
|- generate_conformers(mol: Molecule, *, options=None, constraints=None)
|    -> ConformerBatch
`- runtime_info() -> RuntimeInfo
```

The signatures to freeze are:

```python
@dataclass(frozen=True)
class ETKDGOptions:
    conformer_count: int = 1
    seed: int | None = None
    prune_rms_threshold: float | None = None
    use_random_coordinates: bool = False

@dataclass(frozen=True)
class CoordinationEmbeddingConstraints:
    coordination_pairs: NDArray[np.int32]
    metal_donor_bounds: NDArray[np.float64]
    donor_pair_indices: NDArray[np.int32] | None = None
    donor_pair_bounds: NDArray[np.float64] | None = None

@dataclass(frozen=True)
class ConformerRecord:
    ordinal: int
    engine_conformer_id: int

@dataclass(frozen=True)
class ConformerBatch:
    status: GenerationStatus
    coordinates: NDArray[np.float64]
    records: tuple[ConformerRecord, ...]
    engine: str
    engine_version: str
    diagnostic: str | None

@dataclass(frozen=True)
class RuntimeInfo:
    hotpot_version: str
    rdkit_version: str
    native: bool

class ETKDGSampler:
    def __init__(self, options: ETKDGOptions | None = None): ...

    def generate(
        self,
        mol: Molecule,
        *,
        constraints: CoordinationEmbeddingConstraints | None = None,
    ) -> ConformerBatch: ...

def generate_conformers(
    mol: Molecule,
    *,
    options: ETKDGOptions | None = None,
    constraints: CoordinationEmbeddingConstraints | None = None,
) -> ConformerBatch: ...

def runtime_info() -> RuntimeInfo: ...
```

`generate_conformers()` and `ETKDGSampler.generate()` do not mutate `mol`, its
topology, current coordinates, or `mol.conformers`. Callers explicitly choose
whether and where to store a returned coordinate set. The public API exposes
Hotpot and NumPy types only; it never exposes a Python or C++ RDKit object.

The public component deliberately excludes force-field energy, Hotpot geometry
acceptance, perturbation/recovery, and final-candidate selection. These
omissions keep standalone generation useful even if no `hotpot.ff` workflow
ever adopts it.

### 4.3 Public standalone C++ API

The C++ component is not an implementation detail of pybind. The repository
currently has no exported CMake target or installed C++ SDK, so this is a new
build capability. It uses a narrow C ABI plus a header-only C++17 facade:

```text
include/hotpot/native/array_view.hpp
`- template<class T> class ArrayView

include/hotpot/native/molecule_buffer_view.hpp
|- enum class MoleculeField
|- struct MoleculeBufferView
`- struct OwnedMoleculeData

include/hotpot/rdWrappers/c_api.h
|- enum hotpot_rd_status
|- struct hotpot_rd_options_v1
|- struct hotpot_rd_molecule_view_v1
|- struct hotpot_rd_constraints_view_v1
|- opaque hotpot_rd_sampler
|- opaque hotpot_rd_result
|- hotpot_rd_sampler_create/destroy(...)
|- hotpot_rd_generate(...)
|- hotpot_rd_result_* accessors(...)
`- hotpot_rd_runtime_* accessors(...)

include/hotpot/rdWrappers/rdwrapper.hpp
|- enum class GenerationStatus
|- struct ETKDGOptions
|- struct CoordinationEmbeddingConstraintsView
|- struct ConformerRecord
|- struct ConformerGenerationResult
|- struct RuntimeInfo
|- class ETKDGSampler
|  |- explicit ETKDGSampler(ETKDGOptions options = {})
|  `- generate(MoleculeBufferView,
|              optional<CoordinationEmbeddingConstraintsView>) const
|- generate_conformers(MoleculeBufferView, ETKDGOptions,
|                      optional<CoordinationEmbeddingConstraintsView>)
`- runtime_info()

CMake target: Hotpot::rdwrappers
```

The stable binary boundary is the versioned C API. Its principal signatures
are:

```c
typedef struct hotpot_rd_sampler hotpot_rd_sampler;
typedef struct hotpot_rd_result hotpot_rd_result;

hotpot_rd_status hotpot_rd_sampler_create(
    const hotpot_rd_options_v1 *options,
    hotpot_rd_sampler **out_sampler);

void hotpot_rd_sampler_destroy(hotpot_rd_sampler *sampler);

hotpot_rd_status hotpot_rd_generate(
    hotpot_rd_sampler *sampler,
    const hotpot_rd_molecule_view_v1 *molecule,
    const hotpot_rd_constraints_view_v1 *constraints,
    hotpot_rd_result **out_result);

const double *hotpot_rd_result_coordinates(
    const hotpot_rd_result *result,
    uint64_t *conformer_count,
    uint64_t *atom_count);

hotpot_rd_string_view_v1 hotpot_rd_result_diagnostic(
    const hotpot_rd_result *result);
void hotpot_rd_result_destroy(hotpot_rd_result *result);
```

The C structs contain only fixed-width scalars and pointer/length pairs. No
STL container, exception, RDKit type, or Boost type crosses the shared-library
boundary. Every returned pointer is borrowed from its opaque result and
remains valid only until `hotpot_rd_result_destroy()`.

C ABI v1 also contains flattened Level-1 stereo record/carrier arrays,
enhanced-group lineage scopes, atom transport facts with separate source and
current hydrogen states, and bond display-direction facts. The v1 headers are
still implementation-local until Phase 3 release closure, so audit-driven
field corrections occur before—not after—the ABI is published. Runtime
accessors report the C API version, Hotpot version, canonical RDKit version,
and exact pinned RDKit commit.

The public C++17 facade retains the reviewed class/function shape:

```cpp
namespace hotpot::rdwrappers {

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

[[nodiscard]] RuntimeInfo runtime_info();

}  // namespace hotpot::rdwrappers
```

These C++ functions are inline/header-only wrappers around the C API. They
construct `std::vector` and `std::string` in the consumer translation unit,
which keeps both `_GLIBCXX_USE_CXX11_ABI=0` and `=1` consumers away from an STL
DSO boundary. `native::ArrayView<T>` is a C++17 pointer/length view and replaces
the previously proposed C++20 `std::span`.

The private C++20 `hotpot_rdkit_core` contains the only ETKDG implementation.
The installed `libhotpot_rdwrappers.so.0` and `_rd_native` reuse that core and
the same C API implementation; neither contains a second sampler algorithm.
The C++ SDK installs the public headers, shared library, CMake package
configuration, and licenses. The Python wheel is tested separately and is not
advertised as the C++ SDK.

### 4.4 Internal backend-neutral seam

The public RDKit component is specific by design. Separately, force-field
composition depends on a small private backend-neutral contract:

```text
hotpot.cheminfo.conformers
|- contracts.py
|  |- ConformerSamplingRequest
|  `- ConformerBatch
|- sampler.py
|  `- class ConformerSampler(Protocol)
|     `- sample(molecule, request) -> ConformerBatch
`- adapters.py
   |- class RDKitConformerAdapter
   `- class OpenBabelConformerAdapter
```

“Pluggable” means dependency inversion at this private seam, not a public
string registry. The force-field composition root uses exactly one selected
adapter, while tests may inject another. A future Hotpot-native scientific
sampler implements the same private protocol without changing `rdWrappers`.

**Neither public wrapper component is deleted by a force-field cutover.**
`hotpot.cheminfo.obWrappers` continues to expose its current build/optimization
API, and `hotpot.cheminfo.rdWrappers` continues to expose native ETKDG. Only
superseded workflow glue may be removed when it has no independent component
value.

## 5. Generic Python/C++ molecule communication

### 5.1 Common buffers, consumer-specific requirements

The Open Babel-specific `_MoleculeBuffers` will move to a private common
module and become `NativeMoleculeBuffers`. It remains a typed dataclass, not a
dynamic dictionary.

Fields may be absent, but absence and an empty molecular dimension are
different states. The buffer therefore carries explicit atom/bond counts and
optional arrays or an internal presence mask.

```python
class NativeMoleculeField(Enum): ...

@dataclass(frozen=True)
class NativeMoleculeBuffers:
    schema_version: int
    atom_count: int
    bond_count: int
    atomic_numbers: NDArray[np.int32] | None = None
    formal_charges: NDArray[np.int32] | None = None
    coordinates: NDArray[np.float64] | None = None
    bond_indices: NDArray[np.int32] | None = None
    bond_kinds: NDArray[np.uint8] | None = None
    stereo_records: ... | None = None
    atom_transport: ... | None = None
    bond_transport: ... | None = None
    ...

def pack_molecule(
    mol: Molecule,
    fields: frozenset[NativeMoleculeField],
) -> NativeMoleculeBuffers: ...
```

Field sets are private constants owned by consumers, for example
`RDKIT_ETKDG_FIELDS`, `OPENBABEL_BUILD_FIELDS`, and
`OPENBABEL_OPTIMIZATION_FIELDS`. They are not user profiles.

The stereo fields use flat record/carrier/index/string tables at the native
boundary. They preserve record and carrier enumeration order, group scope,
source codes, and the distinction between absent values and explicit zero or
`UNSPECIFIED`. They are requested by RDKit but are not forced into an Open
Babel operation that cannot consume them. Python dataclasses remain the
authoritative readable representation; the arrays are only a transport form.

### 5.2 Topology and conformer coordinates are separate

Static molecular identity is packed once. A `ConformerBatch` carries changing
coordinates separately. Standalone RDKit can therefore return `(K, N, 3)`
without repeating topology. In the `F1` test, one deterministic coordinate set
is passed into the unchanged Open Babel flow; this stage does not introduce
Open Babel batch minimization.

The RDKit ETKDG request does not require partial charges, unit-cell values, or
input coordinates unless constrained embedding explicitly requests them.
Open Babel optimization requires coordinates. Complex-only fields remain in a
composition around the common molecule payload rather than expanding the base
payload for every consumer.

### 5.3 C++ views and ownership

The common C++ boundary provides:

```cpp
struct MoleculeBufferView;       // non-owning, one synchronous call
struct OwnedMoleculeData;        // explicit session ownership
struct ConformerBatchView;       // non-owning K x N x 3 coordinates
```

Each adapter validates its complete required field set once and then receives
a strong adapter-specific input. Missing optional data must not silently
become zero charges, non-aromatic bonds, unspecified stereochemistry, or zero
coordinates.

`MoleculeBufferView` is source-shared between extensions, but no pybind-bound
C++ object is passed from `_ob_native` to `_rd_native`. Both extensions accept
the common NumPy/POD schema independently. Long-lived force-field sessions
make one deliberate owning copy; synchronous embedding may borrow the array
memory for the duration of the call.

### 5.4 Backend policy leaves the common schema

The generic bond-kind enum includes dative and zero-order bonds. Open Babel's
lossless-representation checks move into the Open Babel adapter. RDKit's
sanitization, property-cache, and bond-direction requirements remain in its
adapter. The base native schema must not reject a chemical feature merely
because one consumer cannot represent it.

`ComplexSessionInput` is currently a widely consumed flat aggregate on both
the Python and C++ sides. Physically nesting `OwnedMoleculeData` in this stage
would force broad aggregate/source-API churn unrelated to RDKit. It therefore
retains its flat layout and gains zero-copy accessors such as
`ligand_molecule_view(const ComplexSessionInput&)`. A later physical-layout
change requires a separate parity-proved decision.

The refactor first introduces the common view without production consumers,
then migrates Open Babel through a compatibility layer, and only then admits
RDKit. Generic structural validation is separated from Open Babel
compatibility validation; in particular, the existing lossy-dative rejection
must still occur before `OBMol` construction rather than polluting the common
schema.

The migration preserves existing Open Babel error timing as well as error
text. Its adapter validator retains the former per-bond validation order
(index, self-bond, order, then dative) even though the consumer-neutral
validator checks columns by structural category. A mixed-invalid input must
therefore produce the same first Open Babel error as before the refactor.

### 5.5 Level-1 storage and structural remapping

`Molecule._stereo_payload` stores the immutable versioned payload outside the
Atom/Bond numeric attribute layouts and outside user-mutable `properties`.
Level-1 remapping is implemented once as a pure helper over stable atom rows:

- records are never sorted; carrier, record, atom-fact, and bond-fact order is
  retained;
- a record is removed when its anchor or any atom carrier is not mapped;
- non-atom carriers remain unchanged;
- bond existence is checked with an undirected endpoint key while the stored
  ordered endpoints are preserved;
- partial enhanced groups are not collapsed, renumbered, or chemically
  reinterpreted; and
- descriptors, source namespace/code, source ledger, and group lineage are
  copied byte-for-byte except for the required atom-row remapping.

Copy and component extraction retain group scopes. Independently imported
molecules receive distinct opaque scopes; therefore `add_component()` can
concatenate payloads without guessing whether equal toolkit group numbers are
related. Copying a molecule deliberately preserves lineage, so recombining a
copy with its source retains their shared group identity. Treating a copy as a
new independent provenance domain would require a future explicit detach/
rebase operation, not implicit behavior in assembly.

## 6. Native RDKit subset and compatibility boundary

### 6.1 Build source

Headers alone cannot provide ETKDG. The production extension requires the
matching RDKit implementation libraries. **The authoritative baseline is the
RDKit 2026.09.1 stable release**, upstream tag `Release_2026_09_1`, commit
`fece8caa860bdf6c9c82bdb9253f22a0cf87150c`, obtained from an exact archive
whose URL and SHA-256 are recorded in `third_party/rdkit/rdkit.lock.json`.

Conda headers/libraries may be used to inspect dependencies and prototype a
build. They are not a wheel runtime dependency and are not copied as the
authoritative source. A Python RDKit wheel is also not a suitable native SDK:
it does not ship the complete public headers and its bundled shared-library
names and ABI are wheel-specific.

The initial private build enables only the real dependency closure needed by
the official `DistGeomHelpers` target. The source-configuration audit found
that the complete offline closure includes exact archives for RDKit, Boost,
Eigen, better-enums, Catch2, and `RingDecomposerLib`; upstream CMake would
otherwise fetch several of them by a moving Git/download path, and its
`RingDecomposerLib` path verifies only MD5. Hotpot pre-populates every source
from its SHA-256 lock and permits no configure-time download. Catch2 remains a
build/configuration dependency only when the pinned upstream graph requires
it; it is not linked into the production sampler. Unrelated FreeType/font and
other optional downloads are disabled.

The compiled target graph includes the actual transitive targets rather than
only the first-level names: at minimum RDGeneral, DataStructs, Optimizer,
EigenSolvers, RDGeometryLib, Alignment, GraphMol, SmilesParse, DistGeometry,
DistGeomHelpers, MolAlign, SubstructMatch, ForceField, ForceFieldHelpers, and
RingDecomposerLib. The generated link map is the authority if it reveals a
deeper required target. **ForceFieldHelpers is compiled only because ETKDG
depends on it; Stage 17 never exposes or invokes RDKit UFF/MMFF as the business
optimizer.** Python wrappers, drawing, descriptors, reactions, database
integration, general file I/O, and unrelated targets remain disabled.

The dependency closure is determined from the pinned release's CMake targets
and a link map, not from a manually guessed list of copied headers.

The native build freezes Boost 1.86.0 (upstream minimum is 1.81), CMake,
Ninja, compiler identity, CMake flags, and CPU baseline. The
implementation probe uses GCC 14.3.0, CMake 3.31.8, Ninja 1.12.1,
`-march=x86-64 -mtune=generic`, and disables POPCNT-specific optimization.
RDKit 2026.09.1 builds as C++20/PIC; the existing Hotpot extensions remain
C++17. The current host's CMake 3.16.3 and GCC 9.4.0 are rejected probe inputs,
not accepted release identities. Wheel builds use a separately recorded fixed
manylinux toolchain rather than treating a temporary local Conda prefix as a
release contract.

### 6.2 Extension isolation

`hotpot.cheminfo.rdWrappers._rd_native` remains separate from `_ob_native`.
The private `hotpot_rdkit_core` and pinned native dependency closure are built
as position-independent static libraries with hidden visibility. A
version-script controlled SDK shared library exports only `hotpot_rd_*` C
symbols. No RDKit, Boost, STL, or C++ exception crosses the C or pybind
surface.

This isolation is required because Open Babel and RDKit have independent ABI
and build requirements and because a user may also import Python RDKit in the
same process. The supported Python process order is RDKit/Hotpot first, then
Open Babel, matching current `hotpot.__init__` policy. OpenBabel-first followed
by Python RDKit currently segfaults in the unmodified third-party wheels; that
reverse order remains an isolated known-unsupported diagnostic rather than a
false Stage 17 release gate.

### 6.3 Exact version lock and thin compatibility layer

Only the private compatibility implementation includes RDKit headers. It maps
the stable public API from Section 4.3 to the pinned release's `RWMol`,
bond/stereo values, bounds matrix, and embedding parameters.

**The version relation is one-to-one:**

```text
one Hotpot release line
    -> exactly RDKit 2026.09.1 Python package metadata
    -> exactly RDKit 2026.09.1 native source archive
    -> exactly one rdkit_2026_09 compatibility implementation
```

Implementation must replace the current `rdkit>=2025.9,<2027` declarations in
`pyproject.toml` and `requirements.txt` with **`rdkit==2026.9.1`**, the package
version corresponding to canonical release 2026.09.1. Configuration fails if
the native source identity differs; runtime reporting exposes the compiled
identity.

There is no version range, automatic fallback, multi-version preprocessor
matrix, runtime dispatch, or CI job that silently moves to a newer RDKit.
**An RDKit upgrade starts only after an explicit owner instruction.** That work
uses a dedicated branch, updates the one thin compatibility implementation,
reruns standalone C++/Python tests plus the complete 187/182 and wheel matrix,
and then replaces the old pin. Routine Hotpot maintenance leaves the pin
untouched.

Because that exact Python package has no CPython 3.9 wheel or sdist, project
metadata, CI, compatibility scripts, release jobs, and Conda recipes are
updated together from Python 3.9-3.14 to **Python 3.10-3.14**. Stage 17 does not
silently compile or distribute a separate full Python RDKit for 3.9.

### 6.4 No Python RDKit translation in production force fields

The native adapter necessarily translates a Hotpot graph into an RDKit C++
graph once. It must not:

- instantiate Python `Chem.Atom`, `Chem.RWMol`, or `Chem.Conformer` objects;
- round-trip through SMILES, SDF, or MOL2;
- call RDKit `AddHs()` and change atom count/order; or
- return an RDKit molecule to Python.

Hotpot creates the explicit-H working copy. Native RDKit creates atoms in
input row order and returns only `(K, N, 3)` coordinates and compact status
metadata.

Removing the Python `rdkit` dependency from all of `import hotpot` is outside
this force-field stage. The current top-level package, drawing, conversion,
and ML descriptor features still use Python RDKit. They are nevertheless
pinned to the same 2026.09.1 release so a Hotpot version never claims support
for two different RDKit versions.

## 7. Two-level Hotpot stereochemistry plan

ETKDG is stereochemistry-sensitive, while Hotpot's current `stereo.py` is only
a placeholder and `rdconvert.py` leaves RDKit stereochemistry assignment as a
TODO. The solution is intentionally split so that lossless transport is not
confused with chemical interpretation.

### 7.1 Level 1: transport only

Level 1 stores and transfers stereo facts without asserting that they are
chemically valid or geometrically consistent. Its public Python API tree is:

```text
hotpot.cheminfo.stereo
|- enum StereoAnchorKind             # ATOM, BOND
|- enum StereoCarrierKind            # ATOM, IMPLICIT_H, LONE_PAIR, VACANCY
|- enum StereoGroupKind              # ABS, AND, OR
|- class StereoGroupRef
|  |- scope: str
|  |- kind: StereoGroupKind
|  `- id: int
|- class StereoCarrier
|  |- kind: StereoCarrierKind
|  `- atom: int | None
|- class StereoAnchor
|  |- kind: StereoAnchorKind
|  `- atom_indices: tuple[int, ...]   # one atom or ordered bond endpoints
|- class StereoRecord
|  |- anchor: StereoAnchor
|  |- carriers: tuple[StereoCarrier, ...]
|  |- descriptor: str
|  |- group: StereoGroupRef | None
|  |- source_namespace: str
|  `- source_code: str
|- class HydrogenTransportState
|  |- implicit_hydrogens: int | None
|  |- bracket_explicit_hydrogens: int | None
|  `- no_implicit: bool | None
|- class AtomTransportRecord
|  |- atom: int
|  |- isotope: int | None
|  |- radical_electrons: int | None
|  |- current_hydrogens: HydrogenTransportState | None
|  |- source_hydrogens: HydrogenTransportState | None
|  `- source_namespace: str | None
|- class BondTransportRecord
|  |- atom_indices: tuple[int, int]
|  `- display_direction: str | None
|- class StereoPayload
|  |- schema_version: int
|  |- records: tuple[StereoRecord, ...]
|  |- atom_facts: tuple[AtomTransportRecord, ...]
|  `- bond_facts: tuple[BondTransportRecord, ...]
|- get_stereo_payload(mol: Molecule) -> StereoPayload
`- set_stereo_payload(mol: Molecule, payload: StereoPayload) -> None
```

The signatures to freeze are:

```python
def get_stereo_payload(mol: Molecule) -> StereoPayload: ...

def set_stereo_payload(
    mol: Molecule,
    payload: StereoPayload,
) -> None: ...
```

`set_stereo_payload()` performs structural storage only: it does not calculate
CIP labels, verify that carriers are bonded or unique, prove that an anchor is
stereogenic, compare descriptors with coordinates, perform cross-molecule
chemical alignment, or repair a contradiction. It may only remap atom indices
during known structural operations such as copy/component extraction.

The generic `StereoRecord` is intentionally extensible enough to preserve
tetrahedral, alkene, allene/cumulene, atropisomeric, square-planar,
trigonal-bipyramidal, square-pyramidal, octahedral, enhanced-group, and unknown
toolkit records without interpreting them. Non-atom carriers explicitly cover
implicit H, lone pairs, and vacancies. Double-bond anchors use stable ordered
endpoint pairs, never mutable bond-list indices. `source_namespace/source_code`
preserve unsupported future values instead of flattening them.

Enhanced groups use `StereoGroupRef(scope, kind, id)`. `id` remains the source
group number and `scope` is an opaque persistent provenance domain, not a
toolkit name. Independent toolkit imports allocate different scopes; copy and
component extraction retain the original scope. This prevents accidental
group merging without breaking a real group that spans components of one
source molecule.

Copying, components, component assembly, atom/bond removal, hide/recover,
working-copy commit, field-selective Python/C++ packing, native return, and
toolkit adapters preserve the payload and reference order exactly. The
authoritative lossless serialization boundary is the versioned Hotpot payload
and generic molecule/native DTO. SMILES, SDF, and MOL2 are tested only for the
subset each format/adapter can represent; an unsupported value is reported
explicitly and never silently erased.

Atom transport facts distinguish isotope and radical electrons without
changing the legacy `Atom` numeric layout used by search/MCA. Hydrogen facts
are split into `source_hydrogens`, an immutable ledger of how the importing
representation encoded H, and `current_hydrogens`, the only state consumed by
native graph construction. Toolkit import initializes both from the source;
explicit-H materialization changes only current state, and working-copy commit
or rollback transfers it transactionally. Bond transport facts preserve
display direction independently of chemical bond kind. For all optional facts,
absence remains distinct from an explicit zero or `UNSPECIFIED` value.

Level 1 does not decide whether the current and source hydrogen encodings are
chemically valid. The update caused by materializing or deleting explicit H is
a mechanical representation transform. Newly explicit H atoms receive current
transport state but no fabricated source ledger. Native RDKit must never read
`source_hydrogens` when constructing the current graph.

Lifecycle behavior is fixed before toolkit mapping:

- `Molecule.copy()` preserves the payload and clones active and hidden bonds;
- `components` and `AtomSeq.to_mol()` filter/remap by selected atom and bond
  identity while owning independent atom arrays;
- `add_component()` offsets incoming atom rows, keeps target records first,
  and relies on group scope rather than renumbering source group IDs;
- atom deletion removes every record/fact containing the deleted atom and
  remaps later rows; bond deletion removes only bond-anchored records and bond
  facts for that endpoint pair;
- hide/recover does not alter the payload, whereas permanently clearing a
  hidden bond applies the bond-deletion rule; and
- force-field working-copy success commits current transport state atomically,
  while any failed commit restores the exact original payload reference.

These operations do not inspect coordinates, call CIP/stereo assignment, or
repair a descriptor. A record with one unmapped atom carrier is removed as a
whole; its ordered carrier sequence is never shortened or rewritten.

### 7.2 Level 2: complete chemical and spatial semantics

Level 2 validates representation, assigns descriptors from topology and/or
coordinates, and aligns stereo under an explicit atom mapping. Its public API
extends the same module:

```text
hotpot.cheminfo.stereo
|- enum StereoType
|- enum StereoIssueCode
|- class StereoValidationIssue
|- class StereoValidationReport
|  |- is_valid: bool
|  `- issues: tuple[StereoValidationIssue, ...]
|- class StereoAssignmentReport
|  |- payload: StereoPayload
|  `- unresolved: tuple[StereoValidationIssue, ...]
|- class StereoAlignmentReport
|  |- is_equivalent: bool
|  |- aligned_payload: StereoPayload
|  `- issues: tuple[StereoValidationIssue, ...]
|- validate_stereochemistry(mol, *, coordinates=None)
|    -> StereoValidationReport
|- assign_stereochemistry(mol, *, coordinates=None)
|    -> StereoAssignmentReport
`- align_stereochemistry(reference, target, atom_mapping)
     -> StereoAlignmentReport
```

The signatures to freeze are:

```python
def validate_stereochemistry(
    mol: Molecule,
    *,
    coordinates: NDArray[np.float64] | None = None,
) -> StereoValidationReport: ...

def assign_stereochemistry(
    mol: Molecule,
    *,
    coordinates: NDArray[np.float64] | None = None,
) -> StereoAssignmentReport: ...

def align_stereochemistry(
    reference: Molecule,
    target: Molecule,
    atom_mapping: Mapping[int, int],
) -> StereoAlignmentReport: ...
```

Level 2 owns these semantics rather than inheriting toolkit-specific enum
values:

- reference-order and permutation rules for tetrahedral centers;
- E/Z double-bond reference selection and equivalence;
- explicit/implicit hydrogen, isotope, radical, aromatic/Kekule handling;
- coordinate handedness, degeneracy tolerance, and coordinate/descriptor
  contradiction reporting;
- descriptor transformation under atom reordering and graph mappings;
- CIP labels as derived output, not the stored identity of a stereo element;
- cumulene/axial and atropisomeric elements; and
- tetrahedral, square-planar, trigonal-bipyramidal, square-pyramidal, and
  octahedral coordination configurations relevant to Hotpot complexes.

Implementation delivers this high-risk scope in ordered gates:

1. freeze the anchor/carrier/permutation algebra and an independent golden
   corpus;
2. implement manually specified tetrahedral and reference-relative alkene
   semantics;
3. implement deterministic parity transformation under atom mappings;
4. implement 3D assignment, degeneracy tolerance, and descriptor/coordinate
   contradiction reporting;
5. implement allene/cumulene, atropisomeric, and enhanced-group semantics;
6. implement substituent equivalence, stereogenicity, and CIP as derived
   output; and
7. implement square-planar, trigonal-bipyramidal, square-pyramidal,
   octahedral, and chelate-aware coordination semantics.

Each gate has its own tests and report; later gates cannot retroactively
change earlier stored descriptor meaning. An unsupported stereo type is
reported explicitly and never silently flattened to `UNSPECIFIED`.
RDKit/Open Babel mappings are adapters and test oracles to this Hotpot-owned
model. Wrapping RDKit `AssignStereochemistry()` is not accepted as a Hotpot
Level-2 implementation.

### 7.3 Relationship to `rdWrappers`

Level 1 is required before native RDKit graph construction so stereo data is
not dropped at the boundary. Level 2 is required before the implementation
report can claim chemically validated stereo preservation or inversion-free
coordinate generation. Until then, `rdWrappers` reports only exact payload
transport and embedding status.

Sanitization must not reorder atoms or silently rewrite Hotpot topology.
Native construction uses stable input rows, updates only the minimum RDKit
property cache required for embedding, and returns unsupported or rejected
molecular state as an explicit generation result.

## 8. Benchmark protocol

### 8.1 Fixed cohorts

The existing coordination benchmark supplies:

- all 187 original ligands; and
- the frozen 182-member CBond-positive coordination-complex cohort.

The benchmark reuses the existing canonical topology, donor indices, seeds,
Hotpot quality policy, and per-case artifact layout. It does not regenerate a
new favorable cohort for an ETKDG route.

Implementation audit found that the published 182-case manifest itself is not
present: only its recorded file SHA-256
`4416bb8558ef420fad3176a82845d02bd1e83d9b271347c97d65174f799a0377`
survives. Exhaustive recovery searched Git reachable/unreachable objects,
history, home/Trash, and the known local archives. The original generation
commands and identities were recovered, but the canonical file, reference
manifest, reports, and trajectories were not. The original byte digest cannot
be recreated because those artifacts contained volatile timestamps, paths,
runtimes, and lost validation/trajectory payloads.

A separate 182-case Eu smoke manifest found under `/tmp` may support
development but is not accepted as the formal cohort and is never reported as
such. Formal closure now requires either an external backup of the original
artifact/evidence or an explicit owner decision to create a normalized
semantic cohort v2 with a new identity. Stage 17 must not silently regenerate
a favorable replacement.

### 8.2 Required route matrix

| Route | Execution | Hotpot gate/perturb | Relaxation | Purpose |
|---|---|---|---|---|
| `R0` | Standalone ligand `rdWrappers` ETKDG | None | None | Prove the native Python/C++ component independently |
| `O0` | Current OBBuilder coordinates in current `hotpot.ff` | Current behavior | Current OB UFF | Frozen production baseline |
| `F1` | ETKDG replaces only the initial ligand coordinate provider | **Identical to `O0`** | **Identical to `O0`** | Isolate component substitution without workflow change |
| `C1` | Standalone stock whole-complex ETKDG | None | None | Test whether stock ETKDG can embed the complete graph |
| `C2` | Standalone coordination-aware ETKDG bounds | None | None | Test the supported complex extension |

`R0`, `C1`, and `C2` test candidate counts `1`, `8`, and `32` where supported.
`F1` supplies one deterministic initial conformer because activating a new
multi-candidate force-field loop would alter current control flow. Candidate
pruning settings, process count, CPU allocation, seeds, exact toolkit
versions, and wall-clock timing boundaries are recorded.

The mandatory force-field comparison is `O0` versus `F1` under identical
downstream stages and budgets. The standalone comparison reports generation
cost, success, and diversity without pretending that it measures complete
force-field throughput.

### 8.3 Execution versus offline observation

`R0`, `C1`, and `C2` contain no Hotpot geometry gate. Benchmark reporting may
evaluate a detached copy after generation to measure geometry. That post hoc
measurement is timed separately and cannot reject, repair, select, or perturb
the returned conformers.

This separation answers both questions:

- what native ETKDG returns as an independent component; and
- whether substituting only its initial coordinates into the unchanged
  force-field workflow changes final business outcomes or cost.

### 8.4 Per-stage evidence

Every applicable route reports at least:

- sampling status and returned conformer count;
- finite coordinate and exact atom-row preservation;
- topology, formal-charge, aromaticity, dative-direction, and Level-1 stereo
  payload parity;
- Level-2 stereo validity/alignment separately when that level is available;
- Open Babel force-field setup and minimization status;
- ligand sampling/minimization time separately from complex work;
- metal placement status and candidate count;
- coordination restoration attempts, forced bonds, and ring rejections;
- complete-complex entry/post-optimization piercing counts;
- terminal standard acceptance and failure names;
- total p50, p95, maximum, and aggregate wall time;
- deterministic same-seed result hashes for the pinned build; and
- for multi-conformer routes, returned/optimized unique-conformer evidence and
  the selected ordinal.

RMSD/diversity computation is benchmark evidence only in the first cut. It is
not automatically admitted into production candidate selection.

### 8.5 Integration gates

`rdWrappers` is retained as a standalone component after its own API, native,
and packaging tests pass; that decision does not wait for `F1`. Replacing only
the current force-field coordinate provider requires all of the following:

1. the final 182-complex route passes the existing standard acceptance for
   all 182 cases or the owner explicitly accepts named regressions;
2. no case changes atom order, topology, coordination direction, formal
   charge, or represented stereochemistry;
3. all sampling failures and downstream recoveries are stage-resolved rather
   than hidden by a broad fallback;
4. `F1` invokes the same downstream stages, settings, checks, and recovery
   policy as `O0`;
5. fixed-seed behavior is deterministic for the pinned build and execution
   configuration;
6. p50/p95 and aggregate resource use are reported against `O0`; and
7. the supported wheel/import matrix succeeds.

Regardless of `F1`, the existing `obWrappers` API and native component remain.
If `F1` is rejected, only its force-field adapter is omitted; the independently
useful `rdWrappers` component remains. `C1`/`C2` results are research evidence
only in this round. Removing ligand gates, perturbative retries, metal
placement, or any other force-field stage requires a later owner-approved
plan.

## 9. Implementation phases

### Phase 0 — Approved plan, log, and frozen baselines

- obtain owner approval for this document;
- create `implementation_log.md` and update it after every small implementation
  step;
- record repository/environment/toolkit identities;
- build the unchanged native sources in the dedicated worktree and run focused
  baselines;
- use the recovered provenance to request the published 182 artifact from an
  external backup, or obtain explicit owner approval for semantic cohort v2,
  before claiming a formal 187/182 baseline; keep smoke data separate;
- rerun the unchanged formal production benchmark when that artifact is
  available; and
- freeze per-stage baseline artifacts before refactoring the boundary.

### Phase 1 — Freeze API and exact dependency identity

- add class/function-level API tests for Sections 4 and 7 before filling in
  native behavior;
- pin Python metadata to `rdkit==2026.9.1`;
- update the supported Python matrix coherently to 3.10-3.14;
- lock the canonical RDKit 2026.09.1, Boost 1.86.0, Eigen, better-enums,
  Catch2, and RingDecomposerLib source archives plus the modern compiler,
  CMake/Ninja versions, SHA-256 values, licenses, configuration, and generated
  link map;
- run a minimal C++20/PIC `DistGeomHelpers` dependency-closure build probe
  before writing production bindings;
- add a build-time exact-version check and runtime version report; and
- document the owner-only upgrade procedure without adding multi-version code.

### Phase 2 — Common boundary, Level-1 stereo, and Open Babel parity

- introduce the common schema, C++17 `ArrayView`, and non-owning molecule view
  first without a production consumer;
- keep forwarding aliases/adapters at the existing `obWrappers` paths and
  preserve the public `obWrappers` API;
- migrate Open Babel structural validation and compatibility policy in a
  separate parity checkpoint;
- repair and separately test hidden-bond copy/assembly/removal and component
  atom-array ownership before attaching stereo lifecycle logic;
- revise/freeze Level-1 group lineage and dual source/current hydrogen DTOs;
- implement Level-1 storage and pure ordered remapping before toolkit/native
  packing;
- implement copying, component extraction/assembly, removal, hide/recover,
  working-copy transactions, serialization, packing, and exact round trips;
- separate static topology from conformer coordinate batches;
- add zero-copy common-view accessors to flat `ComplexSessionInput` without
  changing its aggregate layout;
- move Open Babel-specific validation to the Open Babel adapter; and
- prove result/report parity for all existing Open Babel build and force-field
  workflows before adding an RDKit consumer.

### Phase 3 — Standalone native RDKit C++ and Python components

- build the private C++20/PIC RDKit 2026.09.1 core with its exact transitive
  dependency closure;
- implement the C ABI SDK library and header-only C++17 facade independently
  of Python/pybind11;
- add a standalone C++ executable/CTest consumer;
- implement `_rd_native` only as a thin binding over the same core/C API;
- implement the public Python `rdWrappers` classes/functions from Section 4.2;
- construct ligand graphs directly from field-selective common views; and
- validate atom order, Level-1 stereo payload transfer, explicit hydrogens,
  import order, deterministic embedding, and non-mutation.

### Phase 4 — Standalone ETKDG validation

- run `R0` on all 187 ligands with counts `1`, `8`, and `32`;
- test both C++ and Python entry points against the same fixtures;
- report embedding success, deterministic hashes, conformer uniqueness,
  timings, and detached geometric observations;
- keep Hotpot gates, perturbations, Open Babel minimization, and force-field
  selection outside this route; and
- retain `rdWrappers` as a supported standalone component independent of the
  subsequent `F1` result.

### Phase 5 — Force-field component-substitution test

- implement the private backend-neutral sampler seam;
- connect the RDKit adapter only at the current initial coordinate-provider
  seam;
- select the first deterministic successful conformer without activating a
  new multi-conformer minimization/selection loop;
- prove by call trace and report comparison that every later ligand and
  complex stage is identical to `O0`;
- run `F1` across the frozen 182-complex cohort; and
- present the `O0`/`F1` outcome before changing the default coordinate
  provider. No existing geometry or optimization stage is removed in Phase 5.

### Phase 6 — Standalone coordination-aware ETKDG research

- implement stock complete-complex coordinate-only embedding as `C1`;
- add supported metal-donor/donor-pair bounds as `C2`;
- validate dative direction and metal/fragment transport;
- run both routes as standalone generators on the frozen cohort; and
- stop at evidence if supported RDKit extension points are insufficient.

No `C1`/`C2` path enters `hotpot.ff` in this stage.

### Phase 7 — Level-2 Hotpot stereochemistry

- execute the seven separately gated semantic increments in Section 7.2;
- freeze the carrier/permutation algebra and golden corpus before algorithms;
- implement representation validation, deterministic reference order,
  mapping-aware parity, coordinate assignment, and degeneracy reporting before
  advanced stereo families;
- add organic, axial/atropisomeric, enhanced-group, and coordination-stereo
  fixtures only at their declared gate;
- cross-check against pinned RDKit/Open Babel where each toolkit supports the
  case; and
- rerun `R0`/`F1` with chemical validity and alignment reports separated from
  Level-1 transport parity.

### Phase 8 — Compatibility, performance, and release validation

- rerun all force-field, native geometry, trajectory, working-copy, packaging,
  stereo, standalone C++, and public Python API tests;
- run the complete 187/182 matrix from clean processes;
- build/test supported Python 3.10-3.14 wheels with exactly
  `rdkit==2026.9.1`;
- test the C++ component without a Python interpreter;
- test the supported RDKit/Hotpot-first Python import order; retain the known
  OpenBabel-first third-party-wheel segfault only as an isolated diagnostic;
- audit installed headers/library, wheel native dependencies, licenses, and
  symbol visibility; and
- write the final implementation and validation report.

## 10. Test obligations

### 10.1 Common boundary

- exact dtype, shape, contiguity, missing-field, and empty-molecule-dimension
  behavior;
- borrowed-view lifetime and owning-session isolation;
- no full-schema packing when a consumer requests a subset;
- no topology repetition for a conformer batch;
- generic acceptance of dative bonds and adapter-specific rejection/handling;
- unchanged Open Babel mixed-invalid first-error order and text; and
- unchanged Open Babel results and diagnostics.

### 10.2 Standalone `rdWrappers`

- public Python class/function signatures and re-exports;
- installed C API and C++17 facade compile/link/run test without Python or
  pybind11;
- external consumers compiled with `_GLIBCXX_USE_CXX11_ABI=0` and `=1` against
  the same SDK library;
- `nm`/`readelf` proof that only approved `hotpot_rd_*` symbols are exported
  and `ldd` proof of no Conda/Python-RDKit dependency;
- equivalent C++ and Python results for fixed fixtures;
- exact `(K, N, 3)` coordinate shape and finite status;
- original atom-row order, including explicit hydrogens;
- no topology mutation or backend-added atoms;
- no mutation of input coordinates or `mol.conformers`;
- fixed-seed determinism under the recorded thread policy;
- disconnected component handling;
- macrocycle, aromatic, charged, and stereochemical cases;
- candidate-count behavior and failure reporting; and
- no Hotpot geometry or perturbation call inside the sampler.

### 10.3 Coordination-aware embedding

- directed coordination input;
- metal-donor bounds convention and unit tests;
- bounds-matrix symmetry/triangle-smoothing behavior;
- one- and multi-dentate ligands;
- multiple ligand components;
- unsupported metal/state reporting; and
- no RDKit or Open Babel force-field setup in standalone `C1`/`C2`.

### 10.4 Workflow and business behavior

- working-copy atomicity on success/failure;
- trajectory stage and selected-frame integrity;
- trace equality for every post-builder `O0`/`F1` stage;
- existing placement and restoration evidence unchanged;
- no duplicate ring scan at final acceptance;
- unchanged reserved `candidate_count` behavior;
- exact final topology and CBond direction; and
- complete 187/182 stage-resolved outcomes.

### 10.5 Stereochemistry

- Level-1 exact versioned-payload/DTO round trips through copy, components,
  assembly, removal, hide/recover, working-copy commit, Python/C++ packing,
  `obWrappers`, and `rdWrappers`;
- independent atom-array ownership and active/hidden-bond lifecycle integrity
  under copy, components, assembly, removal, clear, and recovery;
- enhanced-group scope isolation for independent molecules and scope retention
  when one source molecule is split and reassembled;
- exact source-hydrogen ledger retention, mechanical current-H updates during
  explicit-H materialization, and proof that native construction does not
  double-count source hydrogens;
- atomic rollback of Level-1 payload and current-H state on a failed
  force-field working-copy commit;
- explicit format-capability outcomes for SMILES/SDF/MOL2 rather than a false
  universal serialization guarantee;
- proof that Level 1 performs no validity, CIP, coordinate, or alignment work;
- tetrahedral permutation/parity fixtures and atom-reordering mappings;
- E/Z reference selection and equivalent/non-equivalent mappings;
- missing, explicit-H, isotope, radical, aromatic/Kekule, and degenerate cases;
- coordinate/descriptor agreement and contradiction reports;
- cumulene, axial, atropisomeric, and declared coordination geometries; and
- explicit unsupported-type reports with no silent stereo erasure.

## 11. Performance and resource policy

The benchmark separates:

- Hotpot-to-buffer packing;
- RDKit graph construction;
- ETKDG embedding;
- coordinate transfer;
- Open Babel ligand minimization;
- existing placement/restoration;
- complete-complex optimization; and
- diagnostic-only geometry measurement.

The standalone native path packs topology once per component and returns a
coordinate batch. It must not construct a Python RDKit molecule per conformer.
`F1` transfers only the selected coordinates into the existing force-field
stage; it does not add a new Open Babel batch-minimization loop.

Initial native ETKDG uses one internal thread per outer benchmark worker to
avoid nested oversubscription and nondeterministic scheduling. Alternative
threading is admitted only with separate resource and reproducibility evidence.

No fixed speedup is promised before Phase 1. The implementation report must
distinguish faster coordinate generation from additional cost caused by a
larger conformer count.

## 12. Packaging and release risks

The highest-risk implementation item is not the C++ adapter; it is reproducible
RDKit-native packaging across supported Python and wheel targets.

Required controls are:

- exact RDKit 2026.09.1, Boost, Eigen, better-enums, Catch2, and
  RingDecomposerLib upstream sources and digests;
- exact compiler/CMake/Ninja identities, frozen build options, generated link
  map, and a configure log proving no network download;
- license/notice inclusion;
- hidden RDKit/Boost/Eigen/other private C++ symbols with only versioned
  `hotpot_rd_*` C symbols exported;
- no dependence on a user's Conda RDKit ABI;
- no dependence on hashed libraries inside a Python RDKit wheel;
- auditwheel/native dependency inspection;
- import coexistence with Python RDKit; and
- source-build behavior documented separately from wheel installation.

**Because this plan intentionally introduces a public standalone C++
component, the matching public Hotpot headers and `hotpot_rdwrappers` library
must be installed as a versioned developer artifact and exercised from an
external consumer build.** Private RDKit headers are not installed. The Python
wheel statically absorbs the same private core into `_rd_native`; it is not
advertised as the C++ SDK. This avoids wheel RUNPATH/auditwheel renaming from
becoming the external C++ installation contract.

Wheel installation is offline. A source build either consumes the complete
locally cached, SHA-verified six-archive source closure or downloads those
exact archives once before configuration; it never uses `latest`, an
unverified system/Conda SDK, or an upstream configure-time fetch. A completely
offline source build requires the documented archive cache to be
pre-populated.

## 13. Explicit non-goals

This stage does not:

- claim that ETKDG is a metal-complex force field;
- use RDKit UFF/MMFF for authoritative complete-complex optimization;
- create a public backend/profile selector;
- change the overall ordering or policy of the current force-field workflow;
- add an intermediate Hotpot geometry gate after ETKDG;
- make RMSD/diversity a production selector without separate evidence;
- fork RDKit's embedding implementation when supported constraints fail;
- remove either the public `obWrappers` or `rdWrappers` component;
- activate `candidate_count` or add a multi-conformer force-field loop;
- support several RDKit versions for one Hotpot version or update RDKit
  automatically;
- replace Hotpot metal/coordination topology semantics;
- remove all Python RDKit use from drawing, ML, or conversion features; or
- claim experimental conformers are chemically definitive without reference
  structures.

## 14. Approved decisions and implementation refinements

The owner approved implementation on 2026-10-22. The following decisions,
including necessary consequences discovered by the implementation audit,
govern the work:

1. Archive this work as Stage 17 because Stage 16 is already occupied.
2. Pin this Hotpot version to RDKit 2026.09.1 / `rdkit==2026.9.1`, with no
   automatic upgrade or supported version range.
3. Expose and retain standalone native `rdWrappers` Python and C++ APIs, while
   also retaining the current public `obWrappers` component.
4. Treat the sampler as coordinate generation only, with no Hotpot gate,
   perturbation, or force-field optimization inside it.
5. Keep the existing public force-field functions and the complete current
   optimization flow unchanged; `F1` replaces only the initial coordinate
   provider.
6. Keep sampler replaceability behind a private composition seam and expose no
   force-field backend/profile selector.
7. Test stock whole-complex ETKDG and the supported custom-bounds extension as
   standalone research, not as a replacement for Hotpot metal placement in
   this round.
8. Generalize the native molecule transport so consumers request only needed
   fields and neither RDKit nor Open Babel policy contaminates the base schema.
9. Implement stereo in two explicit levels: exact uninterpreted transport,
   then Hotpot-owned chemical validation/assignment/alignment.
10. Require an owner-visible `O0`/`F1` decision before changing the default
    coordinate provider; no result in this stage authorizes deletion of an
    existing force-field stage or public wrapper component.
11. Support Python 3.10-3.14 for the Hotpot release pinned to RDKit 2026.09.1;
    do not maintain an independent RDKit Python 3.9 distribution.
12. Keep public consumers C++17-compatible through a C ABI/header-only facade,
    while isolating RDKit's private C++20 build.
13. Require either the externally recovered published 182 manifest or an
    owner-approved semantic cohort v2 before formal cohort claims; development
    smoke data remains explicitly labeled.
14. Give enhanced stereo groups a persistent lineage scope so independent
    component assembly cannot merge equal source group numbers accidentally.
15. Separate immutable source hydrogen encoding from current graph hydrogen
    state; only current state enters native RDKit construction.
16. Repair molecule copy/component/hidden-bond ownership as explicit Level-1
    prerequisites rather than compensating inside stereo adapters.
17. Lock every archive in the actual offline CMake dependency graph, including
    Eigen, better-enums, Catch2, and RingDecomposerLib, and reject any
    configure-time download.

Every implementation-driven revision is recorded in `implementation_log.md`
before dependent production work continues.
