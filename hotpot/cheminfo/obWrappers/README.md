# Native Open Babel Force-Field Wrapper

Chinese mirror: [README.zh.md](README.zh.md)

## 1. Purpose and scope

`hotpot.cheminfo.obWrappers` is Hotpot's direct native boundary for the Open
Babel operations used by the force-field subsystem. Its public functions take
a Hotpot `Molecule`; callers neither construct nor pass an Open Babel `OBMol`.

Python copies the molecule into typed, contiguous NumPy arrays. The pybind11
extension reads those buffers, constructs a temporary `OBMol` in C++, applies
registered backend workarounds, and runs `OBBuilder` or the complete
force-field epoch loop in C++. Only value results—coordinates, numerical
diagnostics, and immutable rule reports—return to Python.

The package currently provides:

- native 3D coordinate generation with `OBBuilder`;
- native steepest-descent and conjugate-gradient optimization;
- optional perturbation segments, increasing van der Waals cutoffs, stable
  stopping criteria, and bounded trajectory capture;
- a read-only registry of narrowly scoped Open Babel workaround rules; and
- auditable evidence for every applied rule.

It does not implement a replacement force field, decide whether a structure
is chemically acceptable, repair ring--bond piercing, restore coordination
bonds, or relocate a metal center. Those policies belong to
`hotpot.cheminfo.forcefields` and the geometry/chemistry layers above it.

## 2. Which API should application code use?

For normal Hotpot workflows, use the higher-level force-field package:

```python
from hotpot import read_mol
from hotpot.cheminfo import forcefields as ff


def main():
    mol = read_mol("CCO", "smi")
    report = ff.build_and_optimize(
        mol,
        forcefield="UFF",
        epochs=20,
        steps_per_epoch=25,
        seed=2026,
    )
    print(report.quality_report.passed)


if __name__ == "__main__":
    main()
```

That API owns hydrogen handling, organic/complex workflow selection,
ring--bond untangling, coordination-bond restoration, geometry validation,
trajectory persistence, seeded worker execution, warnings, and transactional
updates of the input molecule.

Use `obWrappers` directly only when you need a thin Open Babel operation or
need to inspect its native rule evidence:

```python
from hotpot import read_mol
from hotpot.cheminfo.obWrappers import build, optimize

mol = read_mol("CCO", "smi")
build_report = build(mol)
optimization = optimize(
    mol,
    "UFF",
    epochs=20,
    steps_per_epoch=25,
    retain_frames=True,
)

print(build_report.succeeded)
print(optimization.best_energy, optimization.energy_unit)
print(optimization.termination_reason)
```

Both calls update `mol.coordinates`. `optimize()` writes the selected
lowest-energy usable frame, while `terminal_coordinates` preserves the last
observed frame for diagnostics.

## 3. Public API summary

All supported public names are exported by `hotpot.cheminfo.obWrappers`.
The compiled `_ob_native` module and the buffer packer are private
implementation details.

| API | Purpose | Return value |
|---|---|---|
| `build(mol, *, stereo_warnings=None)` | Build 3D coordinates through native `OBBuilder` and apply pre-build rules | `BuildReport` |
| `optimize(mol, forcefield, *, ...)` | Run a native, epoch-based Open Babel optimization | `OptimizationReport` |
| `available_rules(stage=None)` | List compiled rules in deterministic execution order | `tuple[RuleDescriptor, ...]` |
| `inspect_rules(mol, stage, *, ...)` | Report which rules would apply without building or optimizing | `RuleExecutionReport` |

### 3.1 `build`

```python
build(
    mol: Molecule,
    *,
    stereo_warnings: bool | None = None,
) -> BuildReport
```

`mol` must be a Hotpot `Molecule`. `stereo_warnings=None` preserves the Open
Babel default; a Boolean is forwarded to the matching `OBBuilder.Build`
overload. Coordinates are committed to `mol` only when the builder reports
success.

```python
from hotpot import read_mol
from hotpot.cheminfo.obWrappers import build

mol = read_mol("OP(=O)(O)O", "smi")
report = build(mol)
print(report.succeeded)
print([item.descriptor.rule_id for item in report.rules.applications])
```

### 3.2 `optimize`

```python
optimize(
    mol: Molecule,
    forcefield: str,
    *,
    algorithm: str = "conjugate",
    epochs: int = 100,
    steps_per_epoch: int = 5,
    perturb_interval: int | None = None,
    perturbation_offsets: numpy.ndarray | None = None,
    retain_frames: bool = False,
    retain_epoch_history: bool = False,
    increasing_vdw: bool = False,
    vdw_cutoff_start: float = 1.0,
    vdw_cutoff_end: float = 10.0,
    energy_tolerance: float = 1.0e-6,
    stopping_window: int | None = None,
    maximum_energy_change_kj_mol: float = 1.0e-4,
    maximum_atom_displacement_angstrom: float = 1.0e-4,
    maximum_rms_gradient_kj_mol_angstrom: float = 1.0,
    maximum_gradient_kj_mol_angstrom: float = 5.0,
    singularity_threshold: float = 1.0e-6,
    repair_angle_radians: float = 1.0e-3,
) -> OptimizationReport
```

Important parameters:

| Parameter | Meaning |
|---|---|
| `forcefield` | Open Babel force-field plugin name, for example `"UFF"`, `"MMFF94"`, `"MMFF94s"`, or `"GAFF"` when available |
| `algorithm` | Exactly `"conjugate"` or `"steepest"` |
| `epochs`, `steps_per_epoch` | Upper optimization budget; each executed epoch submits one step block |
| `perturb_interval` | Start a new optimization segment after every specified number of epochs |
| `perturbation_offsets` | Contiguous offsets shaped `(K, N, 3)`, where `N` is the atom count and `K = (epochs - 1) // perturb_interval` |
| `retain_frames` | Return every executed epoch as an `OptimizationFrame` |
| `retain_epoch_history` | Retain the scalar energy history independently of coordinate frames |
| `increasing_vdw` | Rebuild the force-field segment with a linearly increasing van der Waals cutoff |
| `stopping_window` | Enable stable-window termination; `None` disables this extra stopping rule |
| `singularity_threshold`, `repair_angle_radians` | Numerical controls for the registered degenerate-torsion guard; they are not chemical acceptance criteria |

If perturbations are requested, the caller supplies the offsets explicitly.
The thin wrapper does not generate random perturbations. The higher-level
force-field API generates and records them as part of its workflow.

```python
import numpy as np

from hotpot import read_mol
from hotpot.cheminfo.obWrappers import build, optimize

mol = read_mol("CCCC", "smi")
build(mol)

epochs = 12
interval = 4
offsets = np.zeros(((epochs - 1) // interval, len(mol.atoms), 3))
report = optimize(
    mol,
    "UFF",
    algorithm="conjugate",
    epochs=epochs,
    steps_per_epoch=10,
    perturb_interval=interval,
    perturbation_offsets=offsets,
    retain_frames=True,
    retain_epoch_history=True,
    stopping_window=4,
)
print(report.epochs_completed, report.best_energy)
```

### 3.3 `available_rules`

```python
available_rules(
    stage: RuleStage | None = None,
) -> tuple[RuleDescriptor, ...]
```

The result is ordered by `(stage, priority, rule_id, version)`. Supplying a
stage filters the registry without changing that order.

```python
from hotpot.cheminfo.obWrappers import RuleStage, available_rules

for rule in available_rules(RuleStage.PRE_BUILD):
    print(rule.rule_id, rule.version, rule.priority)
```

### 3.4 `inspect_rules`

```python
inspect_rules(
    mol: Molecule,
    stage: RuleStage,
    *,
    singularity_threshold: float = 1.0e-6,
    repair_angle_radians: float = 1.0e-3,
) -> RuleExecutionReport
```

This is a read-only diagnostic operation. It constructs the same temporary
native representation and executes rule planning, but does not run
`OBBuilder`, set up a force field, or mutate `mol`.

```python
from hotpot import read_mol
from hotpot.cheminfo.obWrappers import RuleStage, inspect_rules

mol = read_mol("OP(=O)(O)O", "smi")
report = inspect_rules(mol, RuleStage.PRE_BUILD)
print(report.applied)
for application in report.applications:
    print(application.descriptor.rule_id, application.atom_indices)
```

## 4. Reports and value contracts

Public reports are frozen dataclasses. NumPy coordinate arrays remain arrays,
but report attributes and rule records cannot be reassigned.

| Type | Meaning |
|---|---|
| `BuildReport` | Builder success flag and pre-build rule evidence |
| `OptimizationReport` | Selected and terminal coordinates, frames, energy/gradient facts, budgets, termination facts, and all setup-rule evidence |
| `OptimizationFrame` | Facts recorded after one executed epoch |
| `RuleExecutionReport` | Ordered applications for one lifecycle stage; `.applied` is true when nonempty |
| `RuleApplication` | One rule's targets, metric, hybridization changes, and coordinate changes |
| `RuleDescriptor` | Stable rule ID, semantic version, stage, and priority |
| `HybridizationChange` | Atom index plus before/after hybridization values |
| `CoordinateChange` | Atom index plus before/after Cartesian coordinates |
| `RuleStage` | `PRE_BUILD` or `PRE_FORCEFIELD_SETUP` |
| `BondKindCode` | Stable integer codes used by the private typed-buffer boundary |
| `SingleOptimizationReport` | Contract used by Hotpot's internal short steepest-descent path |

All energies exposed by `OptimizationReport` are in `kJ/mol`. The original
Open Babel unit string is retained as `backend_energy_unit`. Key optimization
fields are:

- `coordinates`: the selected lowest-energy usable frame, also written to
  `mol.coordinates`;
- `terminal_coordinates`: the final observed frame, even when numerical
  failure ended the run;
- `frames`: epoch frames only when `retain_frames=True`;
- `selected_frame_index` and `best_epoch`: the selected executed epoch index;
- `final_energy` and `best_energy`: terminal and selected-frame energies;
- `termination_reason`: `converged`, `stability_reached`, `budget_exhausted`,
  or an explicit numerical-failure reason;
- `terminal_converged`: whether Open Babel reported convergence at the final
  observed frame; and
- `rules`: the combined pre-force-field-setup evidence from all native
  segments.

`selected_frame_index` identifies the executed epoch even when coordinate
frames were not retained. Do not index an empty `frames` tuple with it.

## 5. Native boundary and execution model

```text
Hotpot Molecule
    |
    | Python: copy into typed, C-contiguous NumPy arrays
    v
MoleculeData value buffers
    |
    | pybind11: validate and read buffers
    v
temporary C++ values -> temporary Open Babel OBMol
    |
    | registered rules + OBBuilder / force-field setup / epoch loop
    v
C++ result values
    |
    | Python: immutable reports and selected coordinates
    v
Hotpot Molecule
```

The schema contains:

| Buffer | dtype and shape |
|---|---|
| atomic numbers | `int32[N]` |
| formal charges | `int32[N]` |
| partial charges | `float64[N]` |
| coordinates | `float64[N, 3]` |
| atom aromatic flags | `uint8[N]` |
| bond atom rows | `int32[M, 2]` |
| bond orders | `float64[M]` |
| bond-kind codes | `uint8[M]` |
| bond aromatic flags | `uint8[M]` |
| optional unit cell | `float64[6]` |

This release deliberately has no `HpMol` or other C++ replacement for
`hotpot.Molecule`. `MoleculeData` is a transient transfer object, not a public
chemical object model. No SWIG object or `OBMol` pointer crosses the pybind11
boundary, and Python is not re-entered for each force-field epoch.

Open Babel uses process-global plugin and force-field state. Native operations
are therefore serialized by an internal recursive mutex within one process.
Process-level parallel workers remain the appropriate route for independent
molecules.

## 6. Built-in rules

### 6.1 Tetracoordinate P(V) builder guard

```text
rule_id  = tetracoordinate_pentavalent_phosphorus_build
version  = 1.0.0
stage    = PRE_BUILD
priority = 100
```

The rule narrowly matches neutral, tetracoordinate P(V) with one non-aromatic
P=O or P=S bond and three non-aromatic single bonds. It temporarily presents
the matching center to `OBBuilder` with hybridization 3 instead of 5, then
restores the native molecule state. The public report records the affected
atom and the temporary change. Formal charges, topology, and Hotpot bond
orders are not rewritten.

### 6.2 Degenerate non-linear torsion guard

```text
rule_id  = degenerate_nonlinear_torsion
version  = 1.0.0
stage    = PRE_FORCEFIELD_SETUP
priority = 100
```

For adjacent atoms $i-j-k$, the guard measures

$$
s(i,j,k)=
\frac{\lVert(\mathbf{x}_i-\mathbf{x}_j)\times
(\mathbf{x}_k-\mathbf{x}_j)\rVert}
{\lVert\mathbf{x}_i-\mathbf{x}_j\rVert
 \lVert\mathbf{x}_k-\mathbf{x}_j\rVert}
=|\sin\theta_{ijk}|.
$$

When a topology that participates in a proper torsion is exactly or nearly
collinear, $s\leq10^{-6}$ by default, the rule rotates a separable branch by
`1.0e-3` radians before force-field setup. The change avoids the known
non-finite UFF torsion-gradient input and is recorded explicitly. It does not
alter the UFF energy expression and is not a general geometry repair.

## 7. Extending the native wrapper

The framework isolates extension points so another force-field-specific
backend pathology can be added without introducing a Python/SWIG control loop:

1. implement a focused condition/action translation unit under `_native/`;
2. assign a stable `RuleDescriptor` and register a `RuleDefinition` with
   `RuleRegistrar`;
3. add the source to the `_ob_native` extension build;
4. add positive, exclusion, ordering, and boundary tests;
5. prove the real build or force-field workflow and relevant benchmark before
   enabling the rule by default.

Rules execute deterministically in `(stage, priority, rule_id, version)`
order. Later rules observe earlier changes to the private native snapshot.
Duplicate `(stage, rule_id)` registrations and more than 256 applications in
one plan are rejected. Python intentionally exposes registry inspection, not
runtime rule registration.

Keep condition/action rules narrow. They may prepare an evidenced Open Babel
input pathology and report their changes; they must not decide chemical
acceptance, select the global force-field workflow, suppress backend failure,
or become an unconditional fallback.

## 8. Reproducibility and limitations

- `build()` intentionally exposes no seed. Open Babel 3.2 keeps builder RNG
  state that cannot be reliably reset in the same process; repeated direct
  calls must not be assumed bitwise reproducible. Use the high-level
  force-field API with `seed=` when reproducible worker execution is required.
- A seed does not promise identical floating-point coordinates across Open
  Babel versions, compilers, CPUs, or platforms.
- `optimize()` selects the lowest-energy usable observed frame; convergence
  and chemical acceptance are separate facts. Use the high-level validation
  report before accepting a structure.
- Numerical failure preserves terminal coordinates and an explicit
  `termination_reason`; it is not silently converted to success.
- Hotpot dative bonds are rejected at this native boundary because Open Babel
  cannot represent that bond semantic losslessly. The complex workflow owns
  any temporary topology transformation required for coordination systems.
- Native extension compatibility is tied to the Open Babel ABI against which
  Hotpot was built. Install a matching wheel or rebuild Hotpot in the target
  environment.
- The wrapper addresses two evidenced backend pathologies only. It does not
  claim to make UFF generally accurate for coordination complexes.

## 9. Package layout

```text
hotpot/cheminfo/obWrappers/
├── __init__.py                 # supported public exports
├── builder.py                  # Hotpot-Molecule build facade
├── forcefield.py               # Hotpot-Molecule optimization facade
├── registry.py                 # read-only rule inspection
├── contracts.py                # immutable public reports
├── reports.py                  # native-to-public report conversion
├── packing.py                  # private typed NumPy buffer schema
├── native.py                   # lazy native loader and buffer handoff
├── settings.py                 # numerical rule defaults
├── _ob_native.pyi              # private extension typing surface
└── _native/
    ├── molecule_data.*         # transient C++ value buffers
    ├── openbabel_adapter.*     # value buffers <-> temporary OBMol
    ├── native_engine.*         # builder and force-field execution
    ├── rules.hpp               # native rule/report contracts
    ├── registry.*              # deterministic compiled registry
    ├── phosphorus_builder.cpp  # P(V) builder guard
    ├── degenerate_torsion.cpp  # torsion singularity guard
    └── native_bindings.cpp     # pybind11 module
```
