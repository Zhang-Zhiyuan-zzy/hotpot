# Hotpot composable xTB workflow: implementation plan

Status: **approved and implemented; stable-xTB official parity, real
coordination pipelines and the 187-structure Eu benchmark are complete; see
the reports for the explicit extended-actinide scientific boundary**

Branch: `feature/xtb-workflow`

Baseline: `116a255`

Official xTB source inspected:
`grimme-lab/xtb@68fe777a02418471b9ddfc7a9ea24b4021d7bca2`
(`bleed`, 2026-10-08). The locally available stable xTB 6.7.1 package was
also used to verify version-dependent behavior.

## 1. Executive decisions

1. The target sequence is:

   ```text
   CBond
     -> Hotpot build + UFF
     -> optional GFN-FF optimization
     -> GFN-xTB calculation/optimization
   ```

2. Every box is an independent node. Each node accepts a valid structure and
   can be invoked without running earlier nodes. The optional GFN-FF node is not
   hidden inside the GFN-xTB node.
3. CBond and the existing three-stage coordination-complex force-field kernel
   will not be changed. The xTB adapter consumes the final Hotpot `Molecule`
   committed by the existing force-field workflow.
4. The official xTB executable remains the numerical implementation. Hotpot
   will provide a thin process and stream adapter; it will not patch, fork, or
   recompile xTB in this stage.
5. Total-charge inference and spin-state inference become two independent
   chemistry services. A resolver composes their results but neither service
   depends on the xTB process wrapper.
6. Existing native formal-charge rules will be reused as the sole rule source.
   They will be separated into a pure inference operation and the existing
   mutating assignment operation; a second competing Lewis-charge
   implementation will not be added.
7. The default spin policy is explicitly named **lowest-spin parity**:
   zero unpaired electrons for an even electron count and one for an odd count.
   It is a documented assumption, not a claim that the physical ground state
   has been determined.
8. Explicit user charge and unpaired-electron values take precedence over
   inference. Ambiguous or unsupported inference fails explicitly; it never
   silently substitutes charge zero.
9. GFN0/1/2-xTB and GFN-FF have version-dependent element domains. The runner
   must reject an unsupported method before launching xTB:

   - GFN0/1/2-xTB end at Rn (`Z=86`) in both xTB 6.7.1 and the inspected
     `bleed` source;
   - the stable xTB 6.7.1 GFN-FF parameter file also ends at Rn;
   - official `bleed` contains the 2024 lanthanide reparameterization and
     actinide extension through Lr (`Z=103`).

   Therefore an Am (`Z=95`) structure must be rejected by all four methods on
   xTB 6.7.1. It may enter GFN-FF only when the resolved official installation
   exposes and passes validation for the extended parameter set; it can never
   continue into GFN0/1/2-xTB under the inspected official backend.
10. `hotpot/cheminfo/calculator.py` becomes the singular package
    `hotpot/cheminfo/calculator/`. The singular name follows the explicitly
    requested façade path and preserves the existing
    `hotpot.cheminfo.calculator` import boundary. Independent calculators live
    in separate modules or subpackages below it. The root-level
    `hotpot/calculator.py` façade is deleted rather than retained as a shim.
11. Exact bare shell commands such as `ff` and `xtb` will not be installed.
    `xtb` would collide with the official backend executable and can cause
    recursive self-resolution; `ff` is too generic for the global command
    namespace. Standalone nodes retain the unambiguous form `hotpot cbond`,
    `hotpot ff`, and `hotpot xtb`.
12. Hotpot gains a separate pipeline controller for concise composition and
    ordered artifacts. Its inline form is:

    ```text
    hotpot run --results-dir RESULTS -- \
      cbond ... :: ff ... :: xtb ... :: <registered-stage> ...
    ```

    The controller interprets stage names internally; it does not invoke a
    shell or shadow third-party executables.
13. xTB becomes the reference implementation for wrapping an external
    calculation program. Only execution primitives proven generic are placed
    in a private `hotpot/plugins/_harness/`; chemistry, method applicability,
    parsing, units, convergence and coordinate commit remain owned by
    `hotpot/plugins/xtb/`. No universal plugin base class is introduced from a
    single example.

## 2. Confirmed current state

### 2.1 Existing upstream nodes

| Node | Stable Python entry | Current behavior | Boundary for this work |
|---|---|---|---|
| CBond | `auto_build_cbond()` in `cheminfo/AImodels/cbond/apply.py` | Adds a metal and predicted coordination bonds; returns a Hotpot `Molecule` or detailed path result | Do not change model inference, site selection, thresholds, or graph mutation |
| Multiple CBond structures | `build_all_possible_cbond()` | Enumerates terminal coordination structures | Not folded into the default xTB workflow |
| General build + FF | `ff.build_and_optimize()` | Selects the organic or complex route | May be called by users before xTB; no kernel change |
| Explicit complex build + UFF | `ff.complexes_build()` | Builds ligand geometry, restores coordination bonds, optimizes and gates the complete complex | Authoritative upstream complex node |
| Existing force-field CLI | `hotpot ff` | Reads files, SMILES, or `-`; writes only molecular payload to stdout and diagnostics to stderr | Reused unchanged; its output is consumed downstream |

For deterministic use of the complete complex path, the documented shell
pipeline will request:

```bash
$ hotpot ff - --input-format smi --route complex --forcefield uff \
    --output-format sdf
```

It will not use `auto_optimize()` as an implicit xTB prerequisite. The current
`auto_optimize()` has its own FAST-first routing contract and is a separate
feature.

### 2.2 Existing formal-charge implementation

`hotpot/cheminfo/calculator.py` already provides:

- main-group Lewis/valence candidates;
- valence and total-charge-constrained models;
- separation of metal-ligand bonds before ligand analysis;
- an independent metal-charge resolver extension point;
- a default metal rule that preserves a nonzero formal charge and otherwise
  uses the element's current Hotpot default valence;
- explicit `preserve` behavior for authoritative imported charges.

This is the base implementation. The new work must not call
`Molecule.calc_mol_default_charge()`, whose older hydrogen-difference heuristic
can return zero for common charged species.

Current limitations that the new contract must expose are:

- `Molecule.charge` defaults to zero, so zero does not prove that a user
  explicitly selected a neutral state;
- `Molecule.copy()` does not preserve molecule-level charge or properties;
- `Molecule.components` does not assign component-level charge;
- a metal inserted by CBond currently has formal charge zero; the default metal
  oxidation-state rule must therefore be part of charge inference;
- transition-metal oxidation and spin state are not uniquely determined by a
  molecular graph;
- the current valence expression adds `sum_covalent_orders` and
  `implicit_hydrogens`; after `add_hydrogens()` the former already includes
  explicit H bonds while the latter can still hold the target H count, causing
  double counting and invalid formal charges;
- the current default spin convenience counts only explicitly materialized
  atoms, so it can give the wrong parity for an implicit-H molecular graph.

The explicit/implicit-H invariant must be corrected and regression-tested
before the charge rules are used at the post-UFF xTB boundary. This is a
prerequisite, not a later cleanup.

### 2.3 Legacy xTB prototype

The current `hotpot/plugins/xtb/core.py` is a prototype rather than a complete
workflow:

- it invokes an arbitrary executable by `subprocess` but does not record or
  verify the backend version;
- it changes the process-global working directory;
- it stores executable paths in a mutable `.cache.json` inside the package;
- it writes `.CHRG` and `.UHF`, but does not parse structured calculation
  results or commit optimized coordinates transactionally;
- its charge default uses `calc_mol_default_charge()`;
- its spin default is only electron parity;
- `set_mol_charge_unpairEs()` can ignore a supplied `unpair` value because it
  tests `self.unpair` instead of the argument;
- `set_opt()` uses `list.extend('--opt')`, producing characters rather than one
  option token;
- its tests depend on a locally cached executable and are not hermetic.

Once the new implementation takes ownership, this prototype, the empty xTB
writer plugin, `.cache.json`, and obsolete documentation/tests will be removed
rather than retained as a second compatibility path.

### 2.4 Current calculator module boundary

`hotpot/cheminfo/calculator.py` currently mixes four independent concerns:

| Concern | Current public surface | Required destination |
|---|---|---|
| Calculator marker | `Calculator` | `calculator/base.py` |
| Classical formal charge | `formal_charge()` plus charge model aliases | `calculator/formal_charges.py` initially; grow into a subpackage only when multiple files are needed |
| Legacy molecular-charge heuristic | `MolChargeCalculator` | `calculator/molecular_charge.py`, isolated from the new electronic-state rules |
| MCA inference adapter | `mca()` and lazy predictor cache | `calculator/mca_inference.py` |

The root-level `hotpot/calculator.py` currently re-exports four names, while
documentation and tests consume both that façade and
`hotpot.cheminfo.calculator`. The migration is atomic: remove the single file,
create the same-stem package, update all repository consumers, and never leave
`calculator.py` beside `calculator/`. The new package `__init__.py` is the only
public façade and defines an explicit `__all__`.

The `formal_charge()` assigning API remains mutating. Pure charge inference is
implemented beneath the same package and shared by that façade; moving files is
not permission to alter unrelated calculator behavior.

### 2.5 Current CLI and stream boundary

- `pyproject.toml` installs only the `hotpot` console script.
- `hotpot/__main__.py` imports several optional command modules while building
  the parser; this weakens node independence and must be replaced by lazy,
  selected-command registration.
- default single-result CBond stdout is a SMILES and can feed `hotpot ff`;
  `--bond-detail` and `--all-structures` emit human text and are not molecular
  streams.
- force-field stdout already contains molecular payload only and stderr is
  diagnostic, but callers must explicitly supply stdin as `-`.
- current SDF conversion does not round-trip arbitrary `Molecule.properties`.
  Electronic-state tags therefore require an explicitly tested stream codec;
  they cannot be assumed to survive the existing generic writer.
- separate shell processes cannot share a first command's `--results-dir`.
  Ordered cross-stage artifacts require a parent pipeline controller.

## 3. Official xTB I/O audit

### 3.1 Native capabilities

| Capability | Official CLI | Official C API | Consequence |
|---|---|---|---|
| Geometry from stdin | No supported `-` geometry input; the argument parser requires a real file | Receives typed arrays | CLI needs a temporary-file adapter |
| Molecular result on stdout | No; stdout is a human log | Results are available through getters | Hotpot must reserve stdout for its own molecular stream |
| Pure error-only stderr | No; warnings/errors can be printed on stdout and successful runs may write runtime notices to stderr | Structured environment errors | Success cannot be decided from empty/nonempty stderr |
| GFN-FF | `--gfnff` | `xtb_loadGFNFF()` | Both are official |
| GFN0/1/2-xTB | `--gfn 0/1/2` | Dedicated loader functions | Both are official |
| Geometry optimization | `--opt` | No optimizer in the published C API | Full official optimization must use the CLI |
| Machine-readable results | Written artifacts such as `xtbout.json`, `gfnff_lists.json`, `xtbopt.*`, `energy`, `charges`, and `wbo` | Getter functions | CLI adapter must validate and parse artifacts |

The official CLI syntax is file-oriented (`xtb [OPTIONS] FILE [OPTIONS]`). Its
argument parser checks whether a positional argument exists as a file. The
optimized structure is written to `xtbopt.<input-extension>` and the trajectory
to `xtbopt.log`; stdout is not a molecular record.

### 3.2 Chosen integration route

Hotpot will not recompile xTB for pipe support. The thin adapter will:

```text
Hotpot Molecule / input stream
        |
        v
resolve charge and unpaired electrons
        |
        v
create one private temporary work directory
        |
        v
write input.xyz + explicit .CHRG/.UHF
        |
        v
invoke the official absolute xtb executable path
        |
        v
validate exit code + convergence marker + required artifacts
        |
        v
parse coordinates/results and atom-order checks
        |
        v
commit coordinates to the Hotpot Molecule
        |
        +--> stdout: next-stage molecular payload only
        +--> stderr/native log: xTB diagnostics
```

Reasons not to fork or rebuild xTB now:

- a temporary XYZ file is negligible compared with GFN-FF/GFN-xTB runtime;
- patching stdin into the CLI would create an upstream maintenance and LGPL
  distribution burden without improving the numerical kernel;
- the official C API does not expose the official geometry optimizer, so an
  in-process replacement would require Hotpot to own a new optimizer loop;
- the file adapter preserves replaceable official binaries and enables direct
  parity tests against an unwrapped invocation.

An in-process C-API backend remains a possible later optimization for large
numbers of single-point/gradient calls. It is not an alternative backend in
this stage.

### 3.3 Element applicability is a backend capability

Element coverage is not inferred from a successful dry run or process return
code. xTB 6.7.1 accepts an Am input during `--define`; its GFN-FF can also
return code zero for an Am calculation while producing a degenerate zero or
non-finite result. These are not evidence of support.

The adapter therefore combines:

1. fixed GFN0/1/2 limits verified from the official parameter source;
2. the resolved GFN-FF parameter-set extent/fingerprint when it is inspectable;
3. a tested capability record for declared official builds; and
4. mandatory finite-energy, finite-gradient (when emitted), and finite-coordinate
   validation for every real result.

An unknown or unverifiable method/element combination is rejected
conservatively with an actionable applicability error. There is no automatic
method substitution.

The implementation tests will pin these upstream facts to source and runtime
evidence: CLI charge parsing in `src/prog/main.F90`, GFN-FF fragment-charge
handling in `src/gfnff/gfnff_setup.f90`, GFN0/1/2 `maxElem` declarations in
`src/xtb/gfn{0,1,2}.f90`, and the effective `.param_gfnff.xtb`. The stable
6.7.1 Am checks are retained as negative regressions: `--define` is
insufficient, GFN-xTB reports no basis, and GFN-FF can return non-finite values
with return code zero.

## 4. Target node contracts

| Node | Required input facts | Produced facts | Mutation/commit rule | Independent use |
|---|---|---|---|---|
| CBond | ligand graph, metal identity, model artifact | intended coordination topology | Existing behavior | Yes |
| Build + UFF | topology; optional initial coordinates | complete 3D geometry, FF report | Existing atomic commit | Yes |
| GFN-FF | 3D coordinates and total charge; component-charge evidence when available | optimized coordinates, energy, convergence and backend report | Coordinates commit only after a complete valid result | Yes |
| GFN-xTB | 3D coordinates, total charge and unpaired-electron count | optimized coordinates or single-point properties, energy and backend report | Coordinates commit only after a complete valid result | Yes |

The xTB nodes never add, remove, or reinterpret Hotpot bonds. They use xTB's
coordinates and numerical properties as backend evidence while retaining the
input Hotpot atom order and topology. Wiberg bond orders may be returned as
results but do not silently replace `BondKind` or covalent/coordination topology.

Execution success, xTB convergence, and optional Hotpot geometry acceptance are
three separate report fields. A terminal structure may be retained for
diagnosis without being labelled successful.

## 5. Charge and spin inference architecture

### 5.1 Package layout

```text
hotpot/cheminfo/calculator/
├── __init__.py                  # Sole public calculator façade and __all__
├── base.py                      # Calculator marker/base
├── formal_charges.py            # Pure inference + assigning formal_charge()
├── molecular_charge.py          # Isolated legacy MolChargeCalculator
├── mca_inference.py             # MCA adapter and lazy predictor cache
└── electronic_state/
    ├── __init__.py
    ├── contracts.py             # Immutable results and estimator Protocols
    ├── spin.py                  # Pure unpaired-electron inference
    └── resolver.py              # Charge/spin composition and overrides
```

Simple calculators remain modules; only electronic state warrants a nested
subpackage because it has contracts, independent inference policies and a
resolver. `electronic_state` imports the pure charge operation from
`formal_charges.py`; it does not contain a second Lewis-rule implementation.

This package is chemistry policy, not geometry. It does not invoke xTB and does
not perform coordinate optimization. Its `__init__.py` re-exports the approved
public entry points. No root-level `hotpot/calculator.py` remains.

### 5.2 Public contracts

Proposed public types and signatures:

```python
from typing import Optional, Protocol

class ChargeInferenceSource(str, Enum): ...
class SpinInferenceSource(str, Enum): ...

@dataclass(frozen=True)
class FragmentCharge:
    atom_indices: tuple[int, ...]
    charge: int
    source: ChargeInferenceSource

@dataclass(frozen=True)
class ChargeInferenceResult:
    atom_formal_charges: tuple[int, ...]
    fragments: tuple[FragmentCharge, ...]
    total_charge: int
    source: ChargeInferenceSource
    assumptions: tuple[str, ...]

@dataclass(frozen=True)
class SpinInferenceResult:
    unpaired_electrons: int
    multiplicity: int
    electron_count: int
    source: SpinInferenceSource
    assumptions: tuple[str, ...]

@dataclass(frozen=True)
class ElectronicState:
    charge: int
    unpaired_electrons: int
    multiplicity: int
    fragment_charges: tuple[int, ...]
    charge_source: ChargeInferenceSource
    spin_source: SpinInferenceSource
    assumptions: tuple[str, ...]

class ChargeEstimator(Protocol):
    def infer(self, mol: Molecule) -> ChargeInferenceResult: ...

class SpinEstimator(Protocol):
    def infer(self, mol: Molecule, charge: int) -> SpinInferenceResult: ...

def infer_charge(mol: Molecule, ...) -> ChargeInferenceResult: ...
def infer_lowest_spin(mol: Molecule, charge: int) -> SpinInferenceResult: ...
def resolve_electronic_state(
    mol: Molecule,
    *,
    charge: Optional[int] = None,
    unpaired_electrons: Optional[int] = None,
    charge_estimator: Optional[ChargeEstimator] = None,
    spin_estimator: Optional[SpinEstimator] = None,
) -> ElectronicState: ...
```

Concrete defaults:

- `ValenceFragmentChargeEstimator` reuses the present valence rules and metal
  resolver;
- `LowestSpinEstimator` implements only the explicitly documented parity rule;
- future oxidation-state rules or AI models implement the same narrow Protocol
  and can be supplied explicitly, without modifying the xTB runner.

No public annotation will use `Any`.

### 5.3 Default charge algorithm

1. Work from the Hotpot graph without modifying the caller's molecule.
2. Construct one representation-aware valence view. For a complete post-build
   structure, attached explicit H atoms are authoritative and stale stored
   implicit-H counts are not added. For a pre-materialization graph with no
   attached H, use its implicit-H count. A genuinely mixed/partial
   representation is reported as ambiguous until its H semantics can be
   normalized; it is never resolved by summing both counts blindly.
3. Verify that explicit-H and stored-H representations are chemically
   consistent. The same molecule before and after hydrogen materialization must
   produce identical charge evidence.
4. Separate metal-ligand coordination edges from covalent ligand topology using
   the existing bond semantics.
5. Order fragments deterministically by their smallest original atom index.
6. For each ligand fragment, apply the existing main-group valence/Lewis rules
   to obtain atom formal charges and sum them.
7. For each isolated metal, preserve a nonzero authoritative formal charge;
   otherwise call the existing metal charge resolver. For the current default
   element table this makes Eu and Am trivalent, while retaining an assumption
   record because oxidation state is not graph-unique.
8. Set total charge to the sum of all fragment charges.
9. Verify that atom-charge, fragment-charge, and total-charge sums agree.
10. Return immutable evidence. Do not mutate atom charges or `mol.charge`.

The existing public `formal_charge()` assignment function will consume the same
pure inference result and remain the explicit mutating operation. This removes
duplication while preserving its existing public behavior.

When a user supplies an explicit total charge, the resolver treats it as
authoritative. If inferred fragment charges cannot be reconciled with it, the
result reports the mismatch and requires either the constrained valence model
or explicit component/metal policy; it does not distribute the difference
arbitrarily.

### 5.4 Default spin algorithm

Before an xTB invocation, the serialized calculation structure must have a
complete explicit atom list, including hydrogen atoms, and finite 3D
coordinates. The xTB node does not silently build a missing geometry or add
atoms. For the actual serialized nuclear-charge sum `Z_total` and total
molecular charge `Q`:

```text
N_electrons = Z_total - Q
N_unpaired(lowest-spin parity) = N_electrons mod 2
multiplicity = N_unpaired + 1
```

Thus the default is singlet for an even electron count and doublet for an odd
count. This is appropriate only as the requested non-special low-spin default.
It cannot distinguish ligand-field states, ferro/antiferromagnetic coupling, or
metal ground-state multiplicities.

An explicit unpaired-electron count overrides the default and must have the same
parity as the total electron count. The API uses `unpaired_electrons`, not an
ambiguous `spin` integer. Multiplicity and electron count are derived and
recorded. Charge inference may operate on either an implicit- or explicit-H
graph; spin inference at execution time is always checked against the exact
explicit nuclei sent to xTB.

### 5.5 GFN-FF state handling

- Total charge is always passed explicitly.
- The default fragment-charge vector is retained in the inference report.
- For general xTB inputs such as XYZ/coord, the current official GFN-FF source
  accepts a second `.CHRG` line containing a whitespace-separated fragment
  charge vector. SDF/molfile and PDB follow different code paths. This is an
  implementation behavior rather than a stable documented CLI contract.
- GFN-FF derives fragments from its own adjacency graph and orders them by the
  first unvisited atom index. Hotpot enables the second line only for a tested
  xTB version after reproducing exactly the backend fragment count, membership,
  order and charge sum. Merely obtaining the same total is insufficient because
  upstream xTB does not fully validate vector length.
- If fragment identity/order cannot be proven, Hotpot passes only the documented
  total charge and marks fragment constraints as unapplied; it will not claim
  that component charges were enforced.
- GFN-FF does not use electronic spin in its energy expression. Its standalone
  node therefore does not require or pass `.UHF`. If a composed pipeline has
  already resolved spin metadata for a following GFN-xTB node, that metadata
  may be preserved in the Hotpot record but is marked unapplied by GFN-FF.

## 6. External-program harness and xTB reference plugin

### 6.1 Shared execution primitives

The current `hotpot/plugins/` tree is heterogeneous and is not retrofitted to
one base class. This stage adds only the external-process facts already needed
by xTB:

```text
hotpot/plugins/_harness/
├── __init__.py
├── contracts.py       # ProcessRequest and NativeProcessResult
├── executable.py      # Explicit/env/PATH resolution and generic version call
├── process.py         # shell=False process execution and byte/text capture
├── workspace.py       # Per-record isolated workspace lifecycle
└── provenance.py      # Backend identity, timing, log and artifact records
```

| Shared `_harness` owns | The xTB plugin owns |
|---|---|
| absolute executable resolution | xTB version/capability interpretation |
| argv/cwd/environment/timeout execution | method options and element applicability |
| isolated workspace creation/retention | XYZ, `.CHRG` and `.UHF` generation |
| raw return code/stdout/stderr/elapsed time | convergence and finite-result decisions |
| generic artifact paths, hashes and provenance | artifact meaning, units and parsing |
| no molecular mutation | atom mapping and transactional coordinate commit |

The generic process runner reports facts only. It cannot declare a chemical
calculation successful. This stage deliberately does not add a universal
`Plugin`, `Calculator`, or `prepare/run/collect` Protocol. A second modern
external backend is required before extracting a common lifecycle interface.

The intentionally narrow core contract is:

```python
@dataclass(frozen=True)
class ProcessRequest:
    argv: tuple[str, ...]
    cwd: Path
    env: Mapping[str, str]
    timeout_seconds: Optional[float] = None

@dataclass(frozen=True)
class NativeProcessResult:
    argv: tuple[str, ...]
    cwd: Path
    return_code: int
    stdout: str
    stderr: str
    elapsed_seconds: float

def run_process(request: ProcessRequest) -> NativeProcessResult: ...
```

### 6.2 xTB reference-plugin layout

```text
hotpot/plugins/xtb/
├── __init__.py          # Explicit public exports
├── contracts.py         # Enums, requests, results, reports and exceptions
├── backend.py           # xTB-specific version and parameter-set probe
├── capabilities.py      # Method/element and parameter-set applicability
├── adapter.py           # Molecule/state -> xTB files; result -> coordinates
├── runner.py            # xTB argv, artifacts and scientific success contract
├── workflow.py          # Independent GFN-FF and GFN-xTB public operations
├── stream.py            # stdin/stdout molecule records and metadata
├── stage.py             # Thin `hotpot run` molecular-stage adapter
├── cli.py               # Thin CLI over workflow.py
├── cli_doc.md            # Tested command examples
└── README.md             # Backend API, units, domains and limitations
```

Files removed after takeover:

```text
hotpot/plugins/xtb/core.py
hotpot/plugins/xtb/xtb_doc.md
hotpot/plugins/xtb/.cache.json
hotpot/cheminfo/_io/xtb.py   # currently an empty writer hook
```

The package README also serves as the implementation template for a future
`hotpot/plugins/xxxx/`: it identifies which five layers remain plugin-owned
(`contracts`, `adapter`, `runner/workflow`, `stream/cli`, and validation) and
which process primitives may be reused.

```text
hotpot/plugins/xxxx/
├── __init__.py
├── contracts.py       # Software-specific request/result and units
├── backend.py         # Identity, version and executable-specific probing
├── capabilities.py    # Methods, elements and feature domain
├── adapter.py         # Hotpot objects <-> native inputs/results
├── runner.py          # Native argv/artifacts and success interpretation
├── workflow.py        # Independent public operations
├── stream.py          # Molecular CLI records and reserved metadata
├── stage.py           # Optional controlled-pipeline adapter
├── cli.py
├── cli_doc.md
└── README.md          # Installation, API, limitations and validation evidence
```

A plugin may omit layers it does not need. This is a documented ownership
pattern, not an inheritance hierarchy.

No Gaussian, ORCA, remote API, or existing plugin is migrated in this stage.
There is no dynamic entry-point discovery yet.

### 6.3 Public backend types

```python
from typing import Optional, Union

class GFNXTBMethod(str, Enum):
    GFN0_XTB = "gfn0"
    GFN1_XTB = "gfn1"
    GFN2_XTB = "gfn2"

class XTBTask(str, Enum):
    SINGLEPOINT = "singlepoint"
    OPTIMIZE = "optimize"

@dataclass(frozen=True)
class XTBRequest: ...

@dataclass(frozen=True)
class XTBBackendInfo: ...

@dataclass(frozen=True)
class XTBRunReport: ...

class XTBError(RuntimeError): ...
class XTBExecutableError(XTBError): ...
class XTBInputError(XTBError): ...
class XTBExecutionError(XTBError): ...
class XTBResultError(XTBError): ...
```

The report will include at minimum:

- requested/effective method and task;
- resolved electronic state and inference provenance;
- executable absolute path and reported xTB version;
- exact argv, elapsed time, return code and convergence state;
- energy with an explicit unit field (raw Hartree; optional derived kJ/mol is
  separately named);
- atom count/order verification;
- paths or retained content for requested artifacts;
- native stdout/stderr, or paths to them when persistence is requested;
- whether optimized coordinates were committed;
- optional post-stage Hotpot geometry-quality report.

### 6.4 Executable resolution

Resolution precedence is deterministic:

1. explicit `executable=` argument;
2. `HOTPOT_XTB_EXECUTABLE`;
3. `shutil.which("xtb")` from `PATH`.

The resolved path is converted to an absolute path and probed once for version
and capabilities. Package directories are never modified. There is no hidden
download and no cache file under `site-packages`.

Hotpot records provenance but cannot prove that an arbitrary executable is an
official build. Documentation will recommend official `grimme-lab/xtb`
releases or a trusted conda-forge package.

### 6.5 Process lifecycle

- Build argv as a sequence and invoke without `shell=True`.
- Never call process-global `os.chdir()`; supply `cwd=` to `subprocess.run()`.
- Give every run a unique temporary directory.
- Always pass method-relevant electronic state explicitly through files and/or
  canonical CLI flags: charge for GFN-FF, and charge plus unpaired electrons for
  GFN-xTB. Never depend on stale current-directory files.
- Preserve `XTBPATH` and other user backend configuration, while setting thread
  environment only when explicitly requested.
- Decide execution success from exit code plus required parseable artifacts.
  Nonempty stderr alone is not failure.
- For optimization, require the xTB convergence marker and a complete final
  geometry. Preserve `NOT_CONVERGED`, logs and the terminal structure as failure
  evidence when available.
- Validate atom count and symbols before applying coordinates. Reject every
  non-finite energy, gradient, charge, or coordinate even when xTB returned
  code zero.
- Apply coordinates to the caller molecule only after all validations pass.
  A failed stage leaves the caller's coordinates unchanged.
- Retain the temporary directory only when explicitly requested.

### 6.6 Independent public operations

```python
def run_gfnff(
    mol: Molecule,
    *,
    task: XTBTask = XTBTask.OPTIMIZE,
    charge_state: Optional[ChargeInferenceResult] = None,
    charge: Optional[int] = None,
    executable: Optional[Union[str, Path]] = None,
    ...,
) -> XTBRunReport: ...

def run_gfn_xtb(
    mol: Molecule,
    *,
    method: GFNXTBMethod = GFNXTBMethod.GFN2_XTB,
    task: XTBTask = XTBTask.OPTIMIZE,
    state: Optional[ElectronicState] = None,
    charge: Optional[int] = None,
    unpaired_electrons: Optional[int] = None,
    executable: Optional[Union[str, Path]] = None,
    ...,
) -> XTBRunReport: ...
```

`run_gfnff()` and `run_gfn_xtb()` share the runner and adapters but neither
calls the other. Their public method domains do not overlap. Skipping GFN-FF
therefore changes only the user's composition, not an internal mode branch.

## 7. Standalone CLI and pipeline controller

### 7.1 Command

```text
hotpot xtb [3D-STRUCTURE-FILE/-] [options]
```

The input must describe a complete, explicit-atom, finite 3D structure. A bare
SMILES is intentionally not accepted by this node because silently invoking a
builder would violate node independence. A SMILES enters the composed workflow
through the existing `hotpot ff` build node.

Core options:

| Option | Meaning |
|---|---|
| `--method {gfnff,gfn0,gfn1,gfn2}` | Independent numerical method; default `gfn2` |
| `--task {singlepoint,optimize}` | Independent task; default `optimize` |
| `--charge INT` | Authoritative total-charge override |
| `--unpaired-electrons INT` | Authoritative xTB UHF/unpaired-electron override |
| `--charge-model {valence,valence-constrained,preserve}` | Named default inference model |
| `--input-format FORMAT` | Required when stdin cannot be identified safely |
| `--output-format FORMAT` | Molecular stdout/file format; default `sdf` |
| `-o/--output FILE` | Write molecular payload to a file; `-` means stdout |
| `--report FILE` | Write structured Hotpot/xTB JSON report |
| `--native-log FILE` | Persist captured native stdout and stderr |
| `--xtb-executable FILE` | Explicit official executable |
| `--threads INT` | Threads assigned to this xTB process |
| `--jobs INT` | Independent molecule processes; defaults to one |
| `--work-directory DIR` | Parent for isolated run directories |
| `--keep-work-directory` | Retain backend artifacts for inspection |
| `--post-check {off,basic,standard,strict}` | Optional post-xTB Hotpot geometry gate |
| `--doc` | Render tested Markdown examples |

The first implementation will expose only typed, tested xTB options. It will
not offer an arbitrary string forwarded to the shell or a generic catch-all
option list.

### 7.2 Stream contract

- stdin carries molecular records only;
- stdout carries molecular records only;
- Hotpot progress, warnings, and captured xTB log text go to stderr;
- `--report` and `--native-log` require separate files and cannot share stdout;
- one failed record yields a nonzero exit status and a structured report;
- record order is deterministic when `--jobs > 1`;
- stdin may appear only once.

SDF is the recommended shell-pipeline format because it retains coordinates,
connectivity and formal-charge annotations better than XYZ. The xTB process
adapter may still use a private XYZ file internally. xTB output coordinates are
applied to the original Hotpot graph, and Hotpot then serializes the result.

The current generic Open Babel conversion does not preserve arbitrary Hotpot
properties. `plugins/xtb/stream.py` will therefore own and test a narrow SDF
property codec for reserved total-charge, unpaired-electron, method, energy and
provenance fields. A following `hotpot xtb` node accepts them only after a
complete round trip and consistency check against the molecular record. This
does not silently promote every `Molecule.properties` value into SDF.

For formats that cannot carry these fields, users must repeat explicit state
options or accept newly reported inference. Under `hotpot run`, the controller
also stores the typed state in its manifest and does not rely on SDF alone.

CBond's human-only `--bond-detail` and current textual `--all-structures`
outputs are not valid upstream streams. The new CBond stage adapter calls the
existing Python result API, writes molecular records separately, and stores
details in its JSON report. Existing human CLI formatting remains unchanged.

### 7.3 Intended shell compositions

The canonical standalone form deliberately repeats `hotpot`. Bare `ff` and
`xtb` executables are not installed because of global-name and official-xTB
collisions.

Full pipeline:

```bash
$ hotpot cbond Eu 'LIGAND_SMILES' \
  | hotpot ff - --input-format smi --route complex --forcefield uff \
      --output-format sdf \
  | hotpot xtb - --input-format sdf --method gfnff --task optimize \
      --output-format sdf \
  | hotpot xtb - --input-format sdf --method gfn2 --task optimize \
      --output-format sdf \
  > final.sdf
```

Skip GFN-FF:

```bash
$ hotpot cbond Eu 'LIGAND_SMILES' \
  | hotpot ff - --input-format smi --route complex --forcefield uff \
      --output-format sdf \
  | hotpot xtb - --input-format sdf --method gfn2 --task optimize \
  > final.sdf
```

Standalone GFN-FF:

```bash
$ hotpot xtb input.sdf --method gfnff --task optimize -o gfnff.sdf \
    --report gfnff.json
```

Explicit electronic state:

```bash
$ hotpot xtb radical.sdf --method gfn2 --charge 0 \
    --unpaired-electrons 1 -o radical_opt.sdf
```

An Am complex is deliberately documented only through an explicitly validated
actinide-capable GFN-FF installation (for example the inspected official
`bleed`, not xTB 6.7.1):

```bash
$ hotpot xtb am_complex.sdf --method gfnff --charge 3 -o am_gfnff.sdf
```

Attempting `--method gfn2` on Am must fail before launch with an applicability
error; it must not fall back silently to GFN-FF.

### 7.4 `hotpot run` controller

The concise inline form is:

```bash
$ hotpot run --results-dir results/eu-001 -- \
    cbond Eu 'LIGAND_SMILES' \
    :: ff --route complex --rebuild --forcefield uff \
    :: xtb --method gfnff --task optimize \
    :: xtb --method gfn2 --task optimize
```

`::` is parsed only as an argv-stage separator. No command is evaluated as a
shell string. Removing the GFN-FF segment removes that node without changing
the others. A JSON workflow file will provide the equivalent non-inline form
for long or repeatedly executed pipelines.

```json
{
  "stages": [
    {"name": "cbond", "argv": ["Eu", "ligand.smi"]},
    {"name": "ff", "argv": ["--route", "complex", "--rebuild", "--forcefield", "uff"]},
    {"name": "xtb", "argv": ["--method", "gfnff", "--task", "optimize"]},
    {"name": "xtb", "argv": ["--method", "gfn2", "--task", "optimize"]}
  ]
}
```

```bash
$ hotpot run workflow.json --results-dir results/eu-001
```

The controller is separate from the external-process harness:

```text
hotpot/pipeline/
├── __init__.py
├── contracts.py       # MolecularStage, StageContext, StageResult, Artifact
├── registry.py        # Explicit, lazy built-in stage registry
├── runner.py          # Ordered execution and failure propagation
├── artifacts.py       # Atomic stage directories, hashes and manifest
├── cli.py             # `hotpot run` inline/JSON parsing
├── cli_doc.md         # Tested CLI examples rendered by `--doc`
└── README.md          # Controller API, payload and artifact contracts
```

The existing scientific modules expose thin adapters without changing their
kernels:

```text
hotpot/cheminfo/AImodels/cbond/stage.py
hotpot/cheminfo/forcefields/stage.py
hotpot/plugins/xtb/stage.py
```

The first controller version is explicitly a molecular-record pipeline. It is
not an arbitrary shell runner. A later non-molecular stage must declare a new
typed payload contract rather than pass an untyped object.

Every run has one ordered artifact tree:

```text
RESULTS/
├── manifest.json
├── input/
├── stages/
│   ├── 00-cbond/
│   │   ├── output.sdf
│   │   ├── report.json
│   │   ├── stderr.log
│   │   └── artifacts/
│   ├── 01-ff/
│   │   ├── output.sdf
│   │   ├── report.json
│   │   ├── stderr.log
│   │   └── trajectory/
│   ├── 02-xtb-gfnff/
│   │   ├── output.sdf
│   │   ├── report.json
│   │   ├── native.log
│   │   └── native/
│   └── 03-xtb-gfn2/
│       ├── output.sdf
│       ├── report.json
│       ├── native.log
│       └── native/
└── final.sdf
```

`manifest.json` records the complete argv, Hotpot version, stage order and
status, start/end time, input/output SHA-256 lineage, resolved electronic state
and provenance, third-party executable identity, and artifact paths. Each stage
writes to a temporary sibling and is atomically renamed only after its manifest
is complete. Failure stops downstream execution by default while retaining the
failed stage evidence. A future explicit policy may permit selected diagnostic
continuation; it is not an unconditional fallback.

In controlled mode every adapter receives a stage-local workspace under the
temporary stage directory. Third-party inputs, outputs, logs and trajectories
are created there or copied there before commit, so a successful run has no
untracked workflow artifact outside `RESULTS/`.

Standalone node CLIs and the controller call the same public operation/stage
adapter. The controller does not reimplement CBond, force-field, or xTB
scientific logic.

## 8. Completed-state file tree and source changes

### 8.1 Status legend and scope

- `[M]` (**modify**) is an existing file that will be edited in place.
- `[A]` (**add**) is a new target file that does not exist at the implementation
  baseline.
- `[D]` (**delete**) is an existing file that will be removed without a
  compatibility shim.
- `[G]` (**regenerate**) is a generator-owned subtree whose output contains a
  mixture of added, modified and deleted files.
- `[R]` (**runtime cleanup**) is an existing untracked runtime file that will be
  removed together with the source logic that creates it.

The tree below is the authoritative completed-state change boundary for this
stage. It lists every hand-maintained file that must be added, modified or
deleted; unrelated repository files are intentionally omitted. The `[G]`
subtree is regenerated as one unit rather than hand-edited file by file.
Planning files already created on the planning branch are marked `[M]`.
If implementation requires another production file, the plan must be revised
before that file is changed.

### 8.2 Complete in-scope tree

```text
.
├── [M] .gitignore
├── [M] README.md
├── [M] MANIFEST.in
├── .github/
│   └── workflows/
│       ├── [M] inference_compatibility.yml
│       └── [M] publish_pypi.yml
├── doc/
│   ├── [M] command.md
│   └── [G] html/**
│       # Regenerated output; remove legacy XtbCalculator references and
│       # describe the replacement API without manual per-file edits.
├── examples/
│   └── BayesianDesign/
│       └── [M] data_process.py
├── skills/
│   ├── [M] usage.claude.md
│   └── [M] usage.codex.md
├── hotpot/
│   ├── [M] __main__.py
│   ├── [D] calculator.py
│   ├── cheminfo/
│   │   ├── [M] core.py
│   │   ├── [D] calculator.py
│   │   ├── calculator/
│   │   │   ├── [A] __init__.py
│   │   │   ├── [A] base.py
│   │   │   ├── [A] formal_charges.py
│   │   │   ├── [A] molecular_charge.py
│   │   │   ├── [A] mca_inference.py
│   │   │   └── electronic_state/
│   │   │       ├── [A] __init__.py
│   │   │       ├── [A] contracts.py
│   │   │       ├── [A] spin.py
│   │   │       └── [A] resolver.py
│   │   ├── _io/
│   │   │   └── [D] xtb.py
│   │   ├── AImodels/
│   │   │   ├── cbond/
│   │   │   │   └── [A] stage.py
│   │   │   └── mca/
│   │   │       └── [M] README.md
│   │   └── forcefields/
│   │       └── [A] stage.py
│   ├── pipeline/
│   │   ├── [A] __init__.py
│   │   ├── [A] contracts.py
│   │   ├── [A] registry.py
│   │   ├── [A] runner.py
│   │   ├── [A] artifacts.py
│   │   ├── [A] cli.py
│   │   ├── [A] cli_doc.md
│   │   └── [A] README.md
│   └── plugins/
│       ├── _harness/
│       │   ├── [A] __init__.py
│       │   ├── [A] contracts.py
│       │   ├── [A] executable.py
│       │   ├── [A] process.py
│       │   ├── [A] workspace.py
│       │   └── [A] provenance.py
│       └── xtb/
│           ├── [M] __init__.py
│           ├── [R] .cache.json
│           ├── [D] core.py
│           ├── [D] xtb_doc.md
│           ├── [A] contracts.py
│           ├── [A] backend.py
│           ├── [A] capabilities.py
│           ├── [A] adapter.py
│           ├── [A] runner.py
│           ├── [A] workflow.py
│           ├── [A] stream.py
│           ├── [A] stage.py
│           ├── [A] cli.py
│           ├── [A] cli_doc.md
│           └── [A] README.md
├── tests/
│   ├── [M] run_coverage.sh
│   ├── [M] run_inference_compatibility.sh
│   ├── cbond/
│   │   └── [A] test_stage.py
│   ├── readme/
│   │   └── [M] test_readme_examples.py
│   ├── test_main/
│   │   └── [A] test_command_loading.py
│   ├── test_cheminfo/
│   │   ├── [M] test_calculator.py
│   │   ├── [M] test_charge_calculators.py
│   │   ├── [M] test_import_safety.py
│   │   ├── [M] test_mca_calculator.py
│   │   ├── calculator/
│   │   │   ├── [A] test_package_contract.py
│   │   │   ├── [A] test_charge_inference.py
│   │   │   ├── [A] test_hydrogen_representation.py
│   │   │   ├── [A] test_spin_inference.py
│   │   │   └── [A] test_electronic_state_resolver.py
│   │   └── forcefields/
│   │       └── [A] test_stage.py
│   ├── test_plugin/
│   │   ├── [D] test_xtb.py
│   │   ├── test_harness/
│   │   │   ├── [A] test_executable.py
│   │   │   ├── [A] test_process.py
│   │   │   ├── [A] test_workspace.py
│   │   │   └── [A] test_provenance.py
│   │   └── test_xtb/
│   │       ├── [A] conftest.py
│   │       ├── fixtures/
│   │       │   └── [A] fake_xtb.py
│   │       ├── [A] test_backend.py
│   │       ├── [A] test_capabilities.py
│   │       ├── [A] test_adapter.py
│   │       ├── [A] test_runner.py
│   │       ├── [A] test_workflow.py
│   │       ├── [A] test_stream.py
│   │       ├── [A] test_documentation.py
│   │       └── [A] test_real_backend.py
│   ├── test_cli/
│   │   ├── [A] test_xtb_cli.py
│   │   └── [A] test_run_cli.py
│   ├── test_pipeline/
│   │   ├── [A] test_contracts.py
│   │   ├── [A] test_registry.py
│   │   ├── [A] test_runner.py
│   │   ├── [A] test_artifacts.py
│   │   └── [A] test_builtin_stages.py
│   └── benchmarks/
│       └── coordination_complexes/
│           ├── [M] README.md
│           ├── [M] cli.py
│           ├── [A] xtb_workflow.py
│           └── [A] test_xtb_workflow.py
└── plan/
    ├── [M] README.md
    └── 12_xtb_workflow/
        ├── [M] README.md
        ├── [M] xtb_workflow_implementation.md
        ├── [M] compatibility_audit.md
        ├── [A] xtb_workflow_implementation_report.md
        └── [A] xtb_workflow_validation_report.md
```

`README.2026.md` is deliberately absent: it is a historical snapshot, not an
active API guide, and will not be rewritten to conceal the old interface that
it documented. `pyproject.toml` also remains unchanged: existing `hotpot*`
package discovery already includes the new packages and xTB remains an external
executable rather than a new mandatory Python dependency. Runtime Markdown is
included through `MANIFEST.in`, and wheel/sdist contents are enforced by the
modified publishing workflow.

The names `formal_charges.py` and `mca_inference.py` are intentional. A module
named `formal_charge.py` or `mca.py` would compete with the same-named function
exported by `calculator/__init__.py` through Python package attributes. The
implementation names avoid that collision while preserving the public
functions `formal_charge()` and `mca()`.

`hotpot/plugins/xtb/.cache.json` currently exists only as ignored runtime state,
not as a Git-tracked source file. `[R]` means that the migration removes the
local file and the code that recreates it; `[M] .gitignore` removes its now-dead
ignore rule.

### 8.3 Change summary

| Location | Planned change | Kernel impact |
|---|---|---|
| `hotpot/cheminfo/calculator.py` -> `hotpot/cheminfo/calculator/**` | Atomically split independent calculators; retain one explicit package façade | Structural refactor; existing calculator behavior retained |
| `hotpot/cheminfo/calculator/electronic_state/**` | Add pure charge/spin inference contracts and composition | New chemistry service |
| `hotpot/calculator.py` | Delete root façade and migrate all repository consumers | Deliberate public-path removal; no shim |
| `hotpot/plugins/_harness/**` | Add narrow executable/process/workspace/provenance primitives | New non-chemical infrastructure |
| `hotpot/plugins/xtb/**` | Replace prototype with typed adapter, runner, workflows, CLI and documentation | New downstream backend |
| `hotpot/pipeline/**` | Add typed molecular-stage controller and ordered result manifests | New orchestration; no scientific kernel |
| `hotpot/cheminfo/AImodels/cbond/stage.py` | Adapt existing CBond result API to machine records/reports | Downstream adapter only |
| `hotpot/cheminfo/forcefields/stage.py` | Adapt existing FF public operations and trajectory artifacts | Downstream adapter only |
| `hotpot/__main__.py` | Register `hotpot xtb` and `hotpot run`; make command loading lazy | CLI composition only |
| `hotpot/cheminfo/_io/xtb.py` | Remove empty writer hook after replacement | Dead-code cleanup |
| `.gitignore` and `hotpot/plugins/xtb/.cache.json` | Remove the obsolete mutable package-cache mechanism and its ignore rule | Dead-state cleanup |
| `tests/test_cheminfo/calculator/**` | Calculator split plus charge/spin unit and regression tests | Test only |
| `tests/test_pipeline/**` | Stage parsing, manifests, failure propagation and result-tree tests | Test only |
| `tests/test_plugin/test_harness/**`, `tests/test_plugin/test_xtb/**` | Generic process tests plus fake executable, real backend and API tests | Test only |
| `tests/test_cli/**`, `tests/test_main/test_command_loading.py` | stdin/stdout/stderr, shell pipeline, controlled pipeline and lazy command-loading tests | Test only |
| `tests/benchmarks/coordination_complexes/**` | Optional xTB/GFN-FF benchmark adapter and report | Benchmark only |
| `README.md`, `doc/command.md`, skills, examples, generated API documentation | Publish active imports, CLI usage and the replacement xTB API | Documentation/consumer migration only |
| `MANIFEST.in`, CI workflows and test launchers | Include runtime docs, exercise new suites and inspect built distributions | Packaging/CI only |

No planned production changes are permitted in:

```text
hotpot/cheminfo/AImodels/cbond/apply.py
hotpot/cheminfo/forcefields/_native/**
hotpot/cheminfo/obWrappers/_native/**
hotpot/cheminfo/forcefields/workflows.py   # except no change is currently needed
hotpot/cheminfo/obconvert.py               # xTB owns its narrow SDF metadata codec
hotpot/cheminfo/geometry/**
hotpot/cheminfo/graph/**
```

If implementation discovers that one of these files must change, work pauses
and the plan is revised before that change.

## 9. Atomic implementation sequence

Each phase is one independently revertible commit unless tests reveal a smaller
necessary split. Every commit is pushed to `feature/xtb-workflow` after its
focused tests pass.

1. `docs(plan): define composable xtb workflow`
   - retain planning commit `4d4fbbb` and commit this reviewed revision plus the
     separate compatibility audit.
2. `test(calculator): lock calculator behavior before split`
   - cover the four current façade names, formal-charge mutation, legacy charge
     behavior, MCA lazy loading and exact import consumers.
3. `refactor(calculator): split cheminfo calculator package`
   - atomically replace `cheminfo/calculator.py` with the package layout;
   - delete `hotpot/calculator.py` and migrate code, docs, skills and tests;
   - preserve lazy MCA loading and explicitly verify wheel contents.
4. `test(state): define charge and spin inference contracts`
   - add neutral, ionic, radical, metal, explicit/implicit-H and ambiguity
     fixtures before changing charge rules.
5. `refactor(state): expose pure fragment charge inference`
   - correct H representation handling and extract/reuse the existing
     formal-charge calculation;
   - preserve the assigning `formal_charge()` behavior and prove inference does
     not mutate the source molecule.
6. `feat(state): add explicit lowest-spin inference`
   - add independent spin estimator and combined resolver;
   - connect existing default spin conveniences to the single formula only when
     their public contract is preserved.
7. `test(harness): define external process facts`
   - specify path resolution, argv/cwd/env, timeout, log capture, workspace and
     provenance behavior without xTB semantics.
8. `feat(harness): add external process primitives`
   - implement the narrow private `_harness` and its isolated tests.
9. `test(xtb): define backend and artifact contracts`
   - add a deterministic fake xTB executable and success/failure fixtures.
10. `feat(xtb): add backend probe and typed runner`
    - xTB version/parameter capability, argv construction, process invocation,
      artifacts and explicit failure objects.
11. `feat(xtb): add molecule and result adapters`
    - XYZ/state files, fragment checks, atom-order checks, artifact parsing,
      finite-value validation and transactional coordinate commit.
12. `feat(xtb): expose independent gfnff and gfn-xtb nodes`
    - public Python operations and method-domain preflight.
13. `feat(cli): add composable hotpot xtb command`
    - stdin/stdout/stderr contract, dedicated SDF state codec, reports and
      tested documentation.
14. `test(pipeline): define controller and artifact contracts`
    - inline `::` parsing, JSON workflows, stage order, atomic directories,
      hashes, manifests and failure propagation.
15. `feat(pipeline): add controller and built-in stage adapters`
    - add `hotpot run`, lazy stage registration, ordered result directories,
      and thin CBond/FF/xTB adapters without kernel changes.
16. `test(xtb): validate official backend and both pipeline forms`
    - direct official CLI parity, standalone shell composition, controlled
      pipeline composition and optional GFN-FF path.
17. `refactor(xtb): remove superseded prototype`
    - remove `core.py`, `.cache.json`, empty writer hook and obsolete tests/docs;
    - complete consumer search, packaging and installed-wheel smoke checks.
18. `docs(xtb): publish template, api limits and validation report`
    - document the external-software wrapper template, supported
      versions/elements, inference assumptions, runtime and benchmark evidence.
19. `test(xtb): extend official state and optimization parity`
    - add GFN2 coordinate, ionic and radical parity checks.
20. `test(xtb): validate official coordination pipelines`
    - run real CBond/FF/GFN2 pipelines with and without optional GFN-FF.
21. `test(xtb): add coordination refinement benchmark`
    - execute the four-route 187-structure Eu benchmark with retained failures.
22. `ci(xtb): include pipeline tests in Python matrix`
    - add the calculator, harness, xTB and pipeline suites to the maintained
      CPython compatibility runner.
23. `docs(xtb): require rebuild in controlled workflows`
    - make the CBond-to-force-field coordinate contract explicit.
24. `ci(xtb): exercise refinement benchmark contracts`
    - include the benchmark module in coverage and compatibility CI.
25. `docs(xtb): record official and corpus validation`
    - record the official parity and four-route Eu benchmark evidence, then
      validate CPython 3.9-3.14 and all six ABI wheels from this snapshot.
26. `test(xtb): validate README europium workflow`
    - execute the exact Eu/acetate/GFN-FF/GFN2 optimization example.
27. `docs(xtb): close workflow validation`
    - reconcile final coverage, CPython 3.9-3.14 runtime and six-wheel evidence;
    - remove stale pending language and qualify the GFN-FF timing conclusion.

## 10. Validation plan

### 10.1 Calculator package migration

- pre-split and post-split public calculator results are identical on the same
  fixtures;
- `hotpot.cheminfo.calculator` exposes only its declared `__all__` and retains
  lazy MCA model loading;
- all repository imports, documentation examples and core error messages use
  the new canonical façade;
- `import hotpot.calculator` fails after installation, proving the deleted root
  façade was not accidentally retained in the wheel;
- no same-stem `cheminfo/calculator.py` remains beside the package;
- the installed wheel passes calculator, MCA and charge tests on Python
  3.9-3.14.

### 10.2 Charge inference

Minimum fixtures:

| Case | Expected charge behavior |
|---|---|
| `CCO` | neutral valence result |
| `[NH4+]` | `+1`, not the historical false zero |
| `CC(=O)[O-]` | `-1` |
| `C[N+](=O)[O-]` | net zero with localized atom charges |
| `[BH4-]` | `-1` |
| `[Zn](Cl)Cl` | metal and ligands treated separately; net zero |
| CBond-added Eu/Am plus neutral ligand | default trivalent metal assumption recorded |
| Am plus anionic ligand | fragment charges sum to expected complex charge |
| explicit total-charge override | override retained; mismatch is explicit |
| unsupported nonmetal/ambiguous metal | inference error or explicit resolver requirement, never zero fallback |

Assertions cover original atom order, no source mutation, deterministic fragment
order, atom/fragment/total sum equality, and preserved public `formal_charge()`
behavior.

### 10.3 Spin inference

- neutral even-electron molecule -> `0`, singlet;
- odd-electron radical -> `1`, doublet;
- charged even/odd systems use `N_electrons = sum(Z)-Q`;
- explicit high-spin value survives unchanged;
- parity-inconsistent explicit input is rejected;
- Eu/Am and transition-metal defaults are marked assumed lowest spin rather
  than inferred physical ground states;
- custom and future AI estimators satisfy the same Protocol tests.

### 10.4 Hermetic harness and runner tests

The generic `_harness` is first tested without xTB for executable precedence,
absolute paths, cwd isolation, environment and timeout handling, concurrent
workspaces, exact stdout/stderr capture, elapsed time and artifact hashes. These
tests assert process facts only.

A fake executable will reproduce official file names and controlled outcomes:

- successful single point;
- successful optimization;
- nonzero exit;
- stderr text during success;
- zero exit with missing result artifact;
- zero exit with `NaN`/`Inf` energy, gradient or coordinates;
- malformed/truncated optimized geometry;
- atom-count or atom-symbol reorder;
- explicit non-convergence marker;
- paths containing spaces;
- simultaneous runs with isolated work directories;
- optional retained artifacts and deterministic cleanup.

These tests run in the normal Python 3.9-3.14 CI matrix without requiring xTB.

### 10.5 Real official xTB tests

Against a recorded official binary version:

1. Compare Hotpot wrapper results with direct official CLI calls using identical
   geometry, method, charge, UHF and options.
2. Cover GFN-FF and GFN2-xTB, neutral, ionic and radical examples.
3. Compare final energy within parser precision and final coordinates within a
   declared RMSD/maximum-displacement tolerance.
4. Verify the GFN-FF two-line fragment-charge input against systems with
   reordered atoms and multiple distinctly charged fragments before enabling
   it as an enforced production feature.
5. Demonstrate that every method rejects Am on xTB 6.7.1, that GFN0/1/2 reject
   it on all supported builds, and that GFN-FF accepts it only with a validated
   extended official parameter set. Do not infer applicability from input
   serialization, `--define`, or return code alone.
6. Record executable path, xTB version, platform, thread count and commands.

Real-backend tests are marked integration tests and skip with a clear reason
when xTB is unavailable; fake-runner tests remain mandatory.

### 10.6 Standalone CLI and shell pipeline

- stdout contains only parseable molecular records;
- all native logs and warnings are on stderr or in the requested log file;
- redirection and `set -o pipefail` work;
- SDF state tags round-trip between two `hotpot xtb` nodes;
- `cbond -> ff -> gfnff -> gfn2` succeeds for an element within the complete
  method domain, such as Eu;
- `cbond -> ff -> gfn2` succeeds when GFN-FF is omitted;
- each of `ff`, `gfnff`, and `gfn2` accepts a standalone valid structure;
- `--charge` and `--unpaired-electrons` override defaults in both Python and CLI;
- a quality-failed terminal structure is distinguishable from a successful one.

### 10.7 Controlled pipeline and results directory

- inline `::` and JSON definitions produce the same normalized stage plan;
- stage tokens are passed as argv arrays and never interpreted by a shell;
- CBond detail/all-structure evidence goes to reports while molecular output
  remains parseable;
- standalone and controlled execution use the same stage operations;
- every stage directory contains output, status, elapsed time and SHA-256
  lineage consistent with `manifest.json`;
- a terminated or failed stage cannot leave a completed-looking directory;
- failure retains evidence and prevents downstream execution by default;
- concurrent runs using the same parent directory receive distinct run roots;
- no bare `ff` or `xtb` console script is installed, and xTB backend resolution
  cannot resolve to Hotpot itself;
- lazy CLI registration allows `hotpot ff` to start without loading CBond or
  xTB optional dependencies.

### 10.8 Coordination benchmark

The existing 187-ligand corpus was executed in opt-in stages:

1. run charge/spin inference and xTB input preparation on every structure for
   which CBond + UFF produced a structure;
2. run GFN-FF over the supported structures and report launch, convergence,
   parse, and geometry-gate counts separately;
3. run GFN2-xTB only for structures whose elements are inside its parameter
   domain;
4. compare direct UFF -> GFN2 with UFF -> GFN-FF -> GFN2;
5. report median, p90, p95, maximum and total time per stage;
6. retain failures with charge/spin provenance, native logs and terminal
   structures;
7. do not merge unsupported-element, execution-failure, non-convergence and
   post-geometry-quality failure into one success-rate denominator.

The completed run used Eu, official xTB 6.7.1 and 64 requested cores. Direct
GFN2 passed 182/187 ligands and 151/182 available Eu complexes. Optional
GFN-FF pre-refinement followed by GFN2 passed 182/187 ligands and 132/182
complexes. The lower complex-chain reliability is retained as measured
evidence, not hidden by a combined denominator or fallback. Its lower
all-record median is affected by frequent early termination and is not evidence
that successful paths are faster. Exact timing and failure classes are
recorded in `xtb_workflow_validation_report.md`.

## 11. Acceptance criteria

The production architecture, official optimization/ionic/radical cases, real
controlled coordination pipelines and 187-structure Eu benchmark are
complete. CPython 3.9-3.14 runtime tests and all six ABI-wheel checks also pass.
Extended-GFN-FF fragment-charge/actinide validation is the remaining scientific
boundary. Exact completed evidence and applicability limits are recorded in
`xtb_workflow_validation_report.md`.

Implementation is complete only when:

- calculator logic is separated under `hotpot.cheminfo.calculator`, the root
  `hotpot.calculator` module is absent, and all in-repository consumers use the
  canonical façade;
- CBond and force-field kernel diffs are empty;
- generic process facts are implemented once in `_harness` while all xTB
  scientific decisions remain in the xTB plugin;
- GFN-FF and GFN-xTB can each run independently from Python and CLI;
- the optional GFN-FF node can be inserted or removed without internal mode
  changes;
- the wrapper uses an official external executable and records version/path;
- stdin/stdout/stderr shell composition is covered by an actual subprocess
  test;
- charge inference correctly handles the listed common ions and fragment sums;
- default low-spin inference is explicit in reports and user-overridable;
- unsupported xTB element domains fail before launch without fallback;
- source molecule coordinates change only after a valid successful result;
- units, energy, convergence, provenance and failure evidence are explicit;
- `hotpot run` executes the same node operations, creates the specified ordered
  artifact tree, and records verified SHA-256 lineage;
- no bare `ff` or `xtb` executable is installed;
- fake-runner tests pass across Python 3.9-3.14;
- real official-backend parity tests pass on the declared supported xTB version;
- old xTB code and mutable package cache have been removed;
- package build, wheel smoke test, CLI documentation examples and
  `git diff --check` pass.

## 12. Explicit non-goals

- changing CBond predictions, thresholds or model artifacts;
- changing UFF/Open Babel/native three-stage complex construction;
- silently discovering the physically correct spin state of a metal complex;
- adding a second electronic-structure backend as an xTB fallback;
- supporting GFN-xTB for elements absent from official parameter files;
- implementing a Hotpot geometry optimizer around the xTB C API;
- forking or distributing a patched xTB binary inside the Hotpot wheel;
- refactoring existing Gaussian, ORCA, ML, plotting, or database plugins into
  the new external-process primitives;
- defining a universal calculation-plugin base class or dynamic third-party
  plugin discovery from the xTB example alone;
- accepting arbitrary shell strings in `hotpot run`;
- inferring protonation/tautomer states not encoded by the input structure;
- silently changing Hotpot bond topology from xTB bond orders.

## 13. Approval checklist

Implementation starts only after review accepts all of the following:

- use the official external xTB executable through an isolated temporary-file
  adapter; do not fork or recompile xTB;
- replace `cheminfo/calculator.py` with the same-name package, split its
  independent calculators, and intentionally delete the root-level
  `hotpot/calculator.py` façade;
- keep CBond and the existing UFF/complex force-field kernels unchanged;
- expose GFN-FF and GFN-xTB as independent Python and CLI nodes;
- require complete explicit-atom 3D input at the xTB node and reject bare
  SMILES there;
- reuse one corrected, pure formal-charge rule source and keep charge and spin
  estimators independently replaceable;
- name the default spin assumption `lowest-spin parity` and never present it as
  a physical ground-state determination;
- enable GFN-FF fragment-charge constraints only after exact backend fragment
  identity/order validation;
- treat method/element coverage as a probed backend capability, including the
  xTB 6.7.1 versus extended-GFN-FF Am boundary;
- reserve stdout for molecular payload and route native diagnostics to stderr
  or an explicit log file;
- keep standalone commands as `hotpot <node>` and use `hotpot run` for concise
  composition and one controlled results directory; do not install bare command
  aliases;
- make xTB the documented reference plugin while sharing only proven generic
  external-process primitives;
- remove the superseded legacy xTB prototype after the new path passes parity
  and integration tests, rather than keeping compatibility branches.

