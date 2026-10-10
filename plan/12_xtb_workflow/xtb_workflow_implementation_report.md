# Composable xTB workflow implementation report

## 1. Result

Branch: `feature/xtb-workflow`

Baseline: `116a255`

Implemented and validated range: `4d4fbbb..8313d55`

The approved calculator split, electronic-state services, external-process
harness, official xTB plugin, standalone `hotpot xtb` command, controlled
`hotpot run` pipeline, built-in CBond/FF/xTB stage adapters, and legacy xTB
prototype removal are complete. Subsequent validation phases add official
charged/open-shell and optimization parity, two real coordination pipelines,
and a reproducible four-route 187-structure benchmark.

The implementation does not change the CBond prediction kernel or the native
three-stage coordination-complex force-field kernel. The force-field CLI was
structurally separated so the CLI and pipeline adapter share one route
selection operation.

## 2. Delivered architecture

| Layer | Responsibility | Main files | Boundary |
|---|---|---|---|
| Calculator package | Canonical calculator façade; pure formal-charge inference; independent charge/spin resolution; MCA and legacy charge adapters | `hotpot/cheminfo/calculator/**` | Chemistry policy only; no xTB execution |
| Generic harness | Executable resolution, argv/cwd/environment/timeout, raw process facts, temporary workspace, hashes, provenance | `hotpot/plugins/_harness/**` | No method, element, charge, convergence, or molecule semantics |
| xTB plugin | Backend probe, method applicability, molecule/XYZ conversion, native runner, result parsing, coordinate commit, SDF state stream, CLI/stage adapters | `hotpot/plugins/xtb/**` | Official executable remains the numerical backend |
| Controlled pipeline | Typed payloads/stages, lazy registry, preflight, atomic persistence, SHA-256 lineage, failure propagation | `hotpot/pipeline/**` | No shell execution and no scientific fallback policy |
| Built-in stages | Adapt existing CBond and FF operations and the new xTB operation to one stage contract | `hotpot/cheminfo/AImodels/cbond/stage.py`, `hotpot/cheminfo/forcefields/stage.py`, `hotpot/plugins/xtb/stage.py` | Thin adapters; existing scientific kernels remain authoritative |

The three computation nodes remain independent:

```text
CBond -> Hotpot build + UFF -> optional GFN-FF -> GFN-xTB
```

Each downstream node accepts a valid structure without requiring the preceding
node. GFN-FF is not an internal mode of GFN-xTB.

## 3. Public surfaces

### 3.1 Python

- `hotpot.cheminfo.calculator` is the sole calculator façade.
- `run_gfnff()` runs an independent GFN-FF operation.
- `run_gfn_xtb()` runs an independent GFN0/1/2-xTB operation.
- `run_pipeline()` executes ordered `StageSpec` objects.
- `register_stage()` exposes an explicit lazy stage registry.

The xTB workflow commits coordinates only after a successful, converged result
has finite parsed values and the atom element sequence agrees with the input.
A single-point request never commits coordinates.

### 3.2 CLI

- `hotpot xtb`: standalone file/stdin/stdout molecular-stream adapter;
- `hotpot run`: controlled inline or strict-JSON workflow with one result tree;
- existing `hotpot cbond` and `hotpot ff` behavior remains available.

No bare `ff` or `xtb` console script is installed. Backend resolution is:

```text
explicit executable argument
    > HOTPOT_XTB_EXECUTABLE
    > PATH
```

### 3.3 Artifacts and streams

The standalone xTB command reserves stdout for SDF molecular records. Native
diagnostics are kept separate. Its narrow `HOTPOT_XTB_*` metadata carries
state, method, energy, and provenance between xTB nodes.

The controller writes a new result root containing input evidence, atomic stage
directories, `manifest.json`, and `final.sdf` only after full success. Every
declared artifact has a relative path, byte size, and SHA-256 digest.

## 4. Scientific contract

- xTB inputs must contain all atoms explicitly and finite Cartesian 3D
  coordinates.
- Explicit charge and unpaired-electron values are authoritative.
- Default spin inference is the named lowest-spin parity assumption, not a
  physical ground-state prediction.
- GFN-FF receives charge but no spin option; GFN0/1/2-xTB receive charge and
  unpaired-electron count.
- Stable xTB 6.7.1 is conservatively limited to `Z <= 86` for the bundled
  GFN-FF and GFN0/1/2 parameter sets. Am is rejected before launch.
- No method is substituted after an applicability or execution failure.
- xTB results do not rewrite Hotpot bond kinds or coordination topology.

## 5. Migration and deliberate breaks

| Removed surface | Replacement or decision |
|---|---|
| root `hotpot.calculator` | Use `hotpot.cheminfo.calculator`; no shim |
| calculator implementation file | Replaced atomically by the same-name package |
| `XtbCalculator`, `xtb_batch_run` | Use typed `run_gfnff`, `run_gfn_xtb`, CLI, or stages |
| package-local `.cache.json` executable state | Use explicit argument, environment, or PATH |
| empty `_io/xtb.py` writer | Use the plugin's strict SDF stream codec |
| old BayesianDesign xTB export/filter helpers | Retired with their obsolete artifact layout |

Private module and serialized Python-object paths are not compatibility
contracts. No conditional compatibility branch was added for them.

## 6. Commit ledger

| Phase | Commit | Purpose |
|---:|---|---|
| 1 | `4d4fbbb`, `82ff25e`, `c4fcdf8` | Approve and refine the architecture, compatibility audit, and target tree |
| 2 | `27c8203` | Lock calculator behavior before movement |
| 3 | `f13b782` | Split the canonical calculator package and remove the root façade |
| 4–6 | `a286e48`, `d80b69f`, `e3be153` | Define state contracts, pure charge inference, and lowest-spin resolution |
| 7–8 | `a214128`, `db9fc49` | Define and implement generic external-process facts |
| 9–13 | `439122a`, `bda6eef`, `796c597`, `b4a9761`, `ea93af4` | Build xTB contracts, backend, adapters, independent nodes, stream, and CLI |
| 14–15 | `8ea5083`, `aa7d34c`, `03724ff`, `d58c1e3`, `ab04158` | Specify and implement the controlled pipeline and built-in stages |
| 16 | `e17c2af` | Validate official backend parity and both composition forms |
| 17 | `5328071` | Remove the superseded prototype and verify distributions |
| 18 | `64ec730` | Publish API limits and the initial validation report |
| 19 | `b791f5e`, `652a3c7` | Add official optimization, ionic and radical parity with Python 3.9-compatible tests |
| 20 | `9fbf984` | Validate two real CBond/FF/official-xTB coordination pipelines |
| 21 | `837c804` | Add and execute the four-route 187-structure coordination benchmark |
| 22 | `8313d55` | Include CLI, pipeline, harness and xTB tests in the Python matrix |

## 7. Deferred work

The following were not required to keep the delivered API honest and remain
separate validation or future-design work:

- a universal plugin base class or dynamic third-party entry-point discovery;
- an in-process xTB C API backend;
- official extended-GFN-FF actinide support;
- physical oxidation-state or spin-state prediction;
- pipeline resume or parallel stage execution;
- complete runtime and wheel validation on every CPython 3.9-3.14
  interpreter.

The xTB package is documented as a reference responsibility layout, not as a
premature mandatory superclass for unrelated software.

The installed stable xTB 6.7.1 backend has received direct energy,
optimization-coordinate, ionic and radical parity checks, plus real
coordination-pipeline validation. The 187-structure benchmark found that
direct GFN2 passed 151/182 Eu complexes, whereas optional GFN-FF pre-refinement
followed by GFN2 passed 132/182. This evidence supports composition and failure
reporting; it does not justify claiming that GFN-FF improves complex
robustness.

Stable xTB 6.7.1 remains explicitly unable to process Am. Extended-GFN-FF
fragment-charge or actinide behavior is outside the validated implementation
until a corresponding official backend is built, identified and tested.
