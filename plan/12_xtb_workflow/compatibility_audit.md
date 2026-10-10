# Compatibility audit for the calculator, xTB and pipeline refactor

Status: **read-only pre-implementation audit**

This report is intentionally separate from the implementation plan. It records
interfaces and behavior that will change, remain stable, or require migration.

## 1. Executive classification

| Area | Compatibility level | Decision |
|---|---|---|
| `hotpot.cheminfo.calculator` file -> package | Source-compatible for approved re-exports | Keep the singular qualified name and place the public façade in `calculator/__init__.py` |
| root `hotpot.calculator` | Deliberately breaking | Delete it; update all repository consumers; do not add a shim |
| calculator implementation module paths | Breaking for private imports and serialized Python objects | Publish a mechanical migration table; do not preserve private paths |
| existing `hotpot cbond` / `hotpot ff` | Preserved | Add stage adapters without changing scientific kernels or existing human modes |
| new `hotpot xtb` | Replaces an incomplete plugin prototype | Remove the old API only after new parity and consumer checks |
| new `hotpot run` | Additive | Keep it separate from standalone node CLIs |
| bare `ff` / `xtb` executables | Rejected | Do not occupy global names or shadow the official xTB binary |
| SDF electronic-state metadata | New, versioned stream behavior | Implement and test a dedicated reserved-field codec |
| external-process `_harness` | New private infrastructure | Do not force existing plugins onto it |

## 2. Calculator package migration

### 2.1 Naming resolution

The request mentions both plural `calculators` and the exact path
`hotpot/cheminfo/calculator/__init__.py`. This plan selects the singular package
path because it follows the explicit path and preserves the existing import
namespace:

```text
hotpot/cheminfo/calculator.py
    -> hotpot/cheminfo/calculator/__init__.py
```

The file and directory must never coexist in one revision. The migration commit
atomically deletes the file and creates the package.

### 2.2 Current consumers that must move

Root imports from `hotpot.calculator` currently appear in:

- `README.md` and `README.2026.md`;
- `skills/usage.claude.md`;
- README example tests and charge/MCA calculator tests;
- `hotpot/cheminfo/AImodels/mca/README.md`.

`hotpot/cheminfo/core.py` also emits two error messages that instruct users to
call `hotpot.calculator.mca`. All are changed to the canonical
`hotpot.cheminfo.calculator` façade in the same commit that removes the root
module.

Current direct consumers of `hotpot.cheminfo.calculator` are limited to tests.
One test imports private `_get_mca_predictor`; after the split it must import the
private helper from `hotpot.cheminfo.calculator.mca`, not from the public
façade.

### 2.3 Public migration table

| Removed path | Canonical path |
|---|---|
| `hotpot.calculator.Calculator` | `hotpot.cheminfo.calculator.Calculator` |
| `hotpot.calculator.MolChargeCalculator` | `hotpot.cheminfo.calculator.MolChargeCalculator` |
| `hotpot.calculator.formal_charge` | `hotpot.cheminfo.calculator.formal_charge` |
| `hotpot.calculator.mca` | `hotpot.cheminfo.calculator.mca` |

The façade will define `__all__`. Names accidentally importable from the old
module because it lacked `__all__` are not treated as public API.

### 2.4 Serialization and reflection break

Moving `Calculator`, `MolChargeCalculator` and functions into implementation
modules changes their `__module__` values. Old pickle, joblib or cloudpickle
payloads that record implementation paths may not deserialize. The project has
no evidence that such payloads are a supported interchange format, so no
module-alias compatibility code is planned. The release notes must state this
break.

### 2.5 Behavioral boundary

- `formal_charge()` remains an assigning, mutating façade.
- the new pure charge inference result is additive;
- correcting explicit/implicit-H double counting can intentionally change
  previously incorrect charge results;
- `MolChargeCalculator` remains isolated as legacy behavior and is not used by
  the xTB workflow;
- MCA remains lazily imported so calculator import does not load the model
  runtime.

The pre-split regression suite distinguishes intentional H-rule corrections
from accidental changes caused by file movement.

### 2.6 Packaging boundary

Current setuptools discovery includes `hotpot*`, so the new package should be
discovered automatically. Wheel tests must still prove that:

- `hotpot/cheminfo/calculator.py` and `hotpot/calculator.py` are absent;
- `hotpot/cheminfo/calculator/**` is present;
- no stale module from an in-place installation masks the package;
- imports pass on CPython 3.9-3.14.

## 3. CLI compatibility

### 3.1 Why the exact bare pipeline is rejected

This form is technically possible only by installing additional global console
scripts:

```bash
hotpot cbond ... | ff ... | xtb ...
```

It is unsafe:

- `xtb` is already the official backend executable name;
- Hotpot backend discovery could resolve the Hotpot wrapper itself and recurse;
- installing or uninstalling either package could replace the other command;
- `ff` is too generic and collision-prone;
- global options on the first shell process cannot control later independent
  processes.

No bare aliases are installed. Existing standalone syntax remains valid:

```bash
hotpot cbond ... | hotpot ff ... | hotpot xtb ...
```

Concise controlled syntax is additive through `hotpot run`, where `cbond`,
`ff`, and `xtb` are internal stage identifiers rather than PATH executables.

### 3.2 Existing command behavior

| Existing behavior | Treatment |
|---|---|
| `hotpot cbond` default single-result SMILES | Preserved |
| CBond `--bond-detail` human text on stdout | Preserved for standalone human mode; controller consumes the Python result and writes details to `report.json` |
| CBond `--all-structures` human ranking text | Preserved for standalone human mode; controller writes a molecular multi-record output plus report |
| `hotpot ff` default MOL2 output | Preserved |
| `hotpot ff -` explicit stdin marker | Preserved; controller supplies records through its adapter |
| force-field quality failure returns diagnostic evidence | Preserved and recorded as a failed stage; no unconditional continuation |

`hotpot run` is new and does not reinterpret an existing command. Its stage
registry is lazy so running FF does not require importing CBond or xTB optional
dependencies.

### 3.3 Result-directory behavior

An ordinary Unix pipeline has no parent Hotpot process that can collect every
node's artifacts. Therefore `--results-dir` belongs to `hotpot run`, not to a
global option on the first independent command. Standalone nodes retain their
own `--output`, `--report`, trajectory and native-log options.

The controller creates a new run manifest and stage directories. It never
silently redirects a standalone node's existing output path.

## 4. Molecular stream compatibility

Current Open Babel conversion copies atoms, bonds, total charge and crystal
cell information, but not arbitrary `Molecule.properties`. The earlier design
assumption that custom SDF state tags already round-trip is false.

The xTB stream adapter will define a small, namespaced and versioned SDF field
set. It will parse only those fields, verify them against the structure and
leave unrelated properties untouched. General Hotpot property serialization is
outside this stage.

Consequences:

- a shell pipeline using a format without metadata must repeat charge/spin
  overrides or accept fresh inference;
- `hotpot run` stores typed electronic state independently in `manifest.json`;
- reserved fields are additive to SDF but may be visible to external tools;
- round-trip tests must cover multiple records, charged complexes, coordination
  bonds, atom order and tools that discard unknown SDF fields.

The new xTB node also requires complete explicit-atom finite 3D input. Inputs
accepted by the old prototype despite missing a valid xTB geometry can now be
rejected earlier. This is a correctness change, not a compatibility fallback.

## 5. Legacy xTB plugin breakpoints

The repository still contains consumers of `XtbCalculator` and
`xtb_batch_run`:

- `examples/BayesianDesign/data_process.py`;
- `tests/test_plugin/test_xtb.py`;
- generated HTML API files and `plugins/xtb/xtb_doc.md`.

The new typed operations do not preserve those signatures. Before deleting the
prototype, the example is migrated or explicitly retired, tests are replaced,
and generated documentation is regenerated or removed. No wrapper that routes
the old broad API through the new runner is planned.

Executable resolution changes from mutable package `.cache.json` and legacy
paths to explicit argument, environment variable, then PATH. Existing local
installations that relied only on `.cache.json` must set
`HOTPOT_XTB_EXECUTABLE` or put the official executable on PATH.

## 6. External-program harness compatibility

`hotpot/plugins` currently contains unrelated ML, plotting, database and
calculation modules. The new private `_harness` is not a new mandatory plugin
framework. Existing plugins are not moved and do not inherit a new base class.

The shared process result is a new typed record, not an emulation of
`subprocess.CompletedProcess` or the legacy xTB return values. Scientific
success remains plugin-specific. A common plugin Protocol and dynamic entry
points are deferred until a second modern external calculation backend provides
real comparison evidence.

## 7. Compatibility gates before removal

The breaking removals occur only after these checks pass:

1. full consumer searches for old calculator and xTB paths;
2. README, skills, examples and error-message migration;
3. pre/post calculator behavioral comparison;
4. fake and real xTB parity tests;
5. standalone CLI and controlled-pipeline round trips;
6. source and wheel installation tests on Python 3.9-3.14;
7. inspection of wheel contents for stale modules/cache files;
8. release notes containing the import, serialization and executable-resolution
   migration tables.

These gates verify the intended clean break. They do not authorize hidden
compatibility branches.
