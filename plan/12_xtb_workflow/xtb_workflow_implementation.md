# Hotpot composable xTB workflow: implementation plan

Status: **review draft; do not implement before approval**

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
hotpot/cheminfo/electronic_state/
├── __init__.py
├── contracts.py      # Enums, immutable result records and Protocols
├── charge.py         # Pure fragment/formal/total-charge inference
├── spin.py           # Pure unpaired-electron inference
└── resolver.py       # Explicit overrides + composition only
```

The package is chemistry policy, not geometry. It does not invoke xTB and does
not perform coordinate optimization.

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

## 6. xTB backend package

### 6.1 Proposed layout

```text
hotpot/plugins/xtb/
├── __init__.py          # Explicit public exports
├── contracts.py         # Enums, requests, results, reports and exceptions
├── executable.py        # Executable resolution and version probe
├── capabilities.py      # Method/element and parameter-set applicability
├── adapter.py           # Molecule/state -> xTB files; result -> coordinates
├── runner.py            # Isolated subprocess lifecycle
├── workflow.py          # Independent GFN-FF and GFN-xTB public operations
├── stream.py            # stdin/stdout molecule records and metadata
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

### 6.2 Public backend types

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

### 6.3 Executable resolution

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

### 6.4 Process lifecycle

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

### 6.5 Independent public operations

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

## 7. CLI design

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

The xTB wrapper will add reserved SDF data fields for the resolved total charge,
unpaired-electron count, method and energy. A following `hotpot xtb` node treats
those fields as explicit provenance rather than recomputing them. For formats
that cannot carry these fields, users must repeat explicit state options or
accept a newly reported inference.

### 7.3 Intended shell compositions

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

## 8. Planned source changes

| Location | Planned change | Kernel impact |
|---|---|---|
| `hotpot/cheminfo/electronic_state/**` | Add pure charge/spin inference contracts and composition | New chemistry service |
| `hotpot/cheminfo/calculator.py` | Move existing pure charge machinery behind the new inference service; keep `formal_charge()` as the assigning façade | Behavior-preserving refactor plus evidence return path |
| `hotpot/calculator.py` | Export intentionally public inference operations if approved | Public façade only |
| `hotpot/plugins/xtb/**` | Replace prototype with typed adapter, runner, workflows, CLI and documentation | New downstream backend |
| `hotpot/__main__.py` | Register `hotpot xtb` | CLI composition only |
| `hotpot/cheminfo/_io/xtb.py` | Remove empty writer hook after replacement | Dead-code cleanup |
| `tests/test_cheminfo/electronic_state/**` | Charge/spin unit and regression tests | Test only |
| `tests/test_plugin/test_xtb/**` | Fake executable, real backend and API tests | Test only |
| `tests/test_cli/test_xtb_cli.py` | stdin/stdout/stderr and shell-pipeline tests | Test only |
| `tests/benchmarks/coordination_complexes/**` | Optional xTB/GFN-FF benchmark adapter and report | Benchmark only |
| `pyproject.toml`, package manifests | Include docs; no mandatory xTB Python dependency | Packaging only |

No planned production changes are permitted in:

```text
hotpot/cheminfo/AImodels/cbond/apply.py
hotpot/cheminfo/forcefields/_native/**
hotpot/cheminfo/obWrappers/_native/**
hotpot/cheminfo/forcefields/workflows.py   # except no change is currently needed
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
   - commit this reviewed plan and archive index entry.
2. `test(state): define charge and spin inference contracts`
   - add regression fixtures for neutral molecules, ions, radicals, metal
     complexes and ambiguous states before changing production code.
3. `refactor(state): expose pure fragment charge inference`
   - extract/reuse the existing formal-charge calculation;
   - preserve `formal_charge()` assignment behavior;
   - verify source molecules are not mutated by inference.
4. `feat(state): add explicit lowest-spin inference`
   - add independent spin estimator and combined resolver;
   - connect existing default spin conveniences to the single formula where
     doing so preserves their contract.
5. `test(xtb): define executable and process contracts`
   - add a deterministic fake xTB executable fixture and failure cases.
6. `feat(xtb): add typed executable runner`
   - executable discovery, version probe, unique workspaces, process capture and
     explicit failure objects.
7. `feat(xtb): add molecule and result adapters`
   - XYZ/state files, atom-order checks, artifact parsing, transactional
     coordinate commit and unit-labelled results.
8. `feat(xtb): expose independent gfnff and gfn-xtb nodes`
   - public Python operations and method-domain preflight.
9. `feat(cli): add composable hotpot xtb command`
   - stdin/stdout/stderr contract, SDF state tags, reports and documentation.
10. `test(xtb): validate official backend parity and pipeline`
    - real official xTB tests, direct CLI parity, optional GFN-FF path and full
      shell composition.
11. `refactor(xtb): remove superseded prototype`
    - remove `core.py`, `.cache.json`, empty writer hook and obsolete tests/docs;
    - full consumer search and packaging checks.
12. `docs(xtb): publish api limits and validation report`
    - record supported versions/elements, inference assumptions, measured
      runtime and benchmark evidence.

## 10. Validation plan

### 10.1 Charge inference

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

### 10.2 Spin inference

- neutral even-electron molecule -> `0`, singlet;
- odd-electron radical -> `1`, doublet;
- charged even/odd systems use `N_electrons = sum(Z)-Q`;
- explicit high-spin value survives unchanged;
- parity-inconsistent explicit input is rejected;
- Eu/Am and transition-metal defaults are marked assumed lowest spin rather
  than inferred physical ground states;
- custom and future AI estimators satisfy the same Protocol tests.

### 10.3 Hermetic runner tests

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

### 10.4 Real official xTB tests

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

### 10.5 CLI and full pipeline

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

### 10.6 Coordination benchmark

The existing 187-ligand corpus will be used in opt-in stages:

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

## 11. Acceptance criteria

Implementation is complete only when:

- CBond and force-field kernel diffs are empty;
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
- inferring protonation/tautomer states not encoded by the input structure;
- silently changing Hotpot bond topology from xTB bond orders.

## 13. Approval checklist

Implementation starts only after review accepts all of the following:

- use the official external xTB executable through an isolated temporary-file
  adapter; do not fork or recompile xTB;
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
- remove the superseded legacy xTB prototype after the new path passes parity
  and integration tests, rather than keeping compatibility branches.

