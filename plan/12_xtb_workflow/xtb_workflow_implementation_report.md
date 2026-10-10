# Composable xTB workflow implementation report

## 1. Result

Branch: `feature/xtb-workflow`

Baseline: `116a255`

Implementation and validation record: Phase 1 started at commit `4d4fbbb` and
Phase 28 closed at commit `5c4dc8f` (38 commits after baseline; use
`git log 4d4fbbb^..5c4dc8f` for the inclusive range).

The approved calculator split, electronic-state services, external-process
harness, official xTB plugin, standalone `hotpot xtb` command, controlled
`hotpot run` pipeline, built-in CBond/FF/xTB stage adapters, and legacy xTB
prototype removal are complete. Subsequent validation phases add official
charged/open-shell and optimization parity, three real coordination pipelines,
a reproducible four-route 187-structure benchmark, the CPython 3.9-3.14
runtime matrix, and six clean ABI-wheel checks.

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

### 2.1 Workflow-relevant final implementation layout

```text
hotpot/
├── __main__.py                         # top-level command dispatch
├── cheminfo/
│   ├── calculator/
│   │   ├── __init__.py                 # sole calculator façade
│   │   ├── base.py
│   │   ├── formal_charges.py
│   │   ├── mca_inference.py
│   │   ├── molecular_charge.py
│   │   └── electronic_state/
│   │       ├── contracts.py
│   │       ├── resolver.py
│   │       └── spin.py
│   ├── AImodels/cbond/stage.py         # CBond pipeline adapter
│   └── forcefields/stage.py            # FF pipeline adapter
├── plugins/
│   ├── _harness/                       # backend-neutral process facts
│   │   ├── contracts.py
│   │   ├── executable.py
│   │   ├── process.py
│   │   ├── provenance.py
│   │   └── workspace.py
│   └── xtb/                            # all xTB scientific semantics
│       ├── adapter.py
│       ├── backend.py
│       ├── capabilities.py
│       ├── cli.py
│       ├── contracts.py
│       ├── runner.py
│       ├── stage.py
│       ├── stream.py
│       └── workflow.py
└── pipeline/
    ├── artifacts.py
    ├── cli.py
    ├── contracts.py
    ├── registry.py
    └── runner.py

tests/
├── test_cheminfo/calculator/           # state and migration contracts
├── test_plugin/test_harness/           # generic process contracts
├── test_plugin/test_xtb/               # fake and official xTB evidence
├── test_pipeline/                      # controller/artifact contracts
├── test_cli/                           # xtb and run CLI contracts
└── benchmarks/coordination_complexes/  # 187-structure refinement benchmark
```

The previous root `hotpot/calculator.py`, legacy xTB calculator, package-local
executable cache, empty xTB writer and obsolete Bayesian xTB consumers are not
parallel compatibility paths; they were removed after consumer and package
content checks.

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

## 6. Complete Phase 1–28 implementation record

The distinction between a test fence, an implementation commit, CI inclusion,
and a completed external run is preserved below. Adding a test to a matrix is
not reported as proof that the matrix ran, and adding a benchmark program is
not reported as the corpus result.

| Phase | Commit(s) | Implementation result | Verification at that phase |
|---:|---|---|---|
| 1 | `4d4fbbb`, `82ff25e`, `c4fcdf8` | Defined the official-executable architecture, compatibility decisions, responsibility boundaries and final target tree. | Document review and clean-diff checks only; no numerical claim. |
| 2 | `27c8203` | Locked the calculator façade, formal-charge mutation, legacy charge and lazy MCA behavior before movement. | Pre-migration calculator regression suites passed. |
| 3 | `f13b782` | Replaced `cheminfo/calculator.py` with the calculator package, removed root `hotpot/calculator.py`, and migrated imports, docs and tests. | Calculator, charge, MCA and README consumers passed with the new canonical import. |
| 4 | `a286e48` | Added charge, spin, electronic-state, hydrogen-representation and package contracts before their implementation. | Strict expected-failure TDD fence; not recorded as a completed feature. |
| 5 | `d80b69f` | Added pure fragment-charge inference, typed charge contracts, deterministic fragment ordering and corrected H handling while retaining the mutating façade. | Charge, hydrogen and legacy façade tests passed; pure inference left the source molecule unchanged. |
| 6 | `e3be153` | Added named lowest-spin-parity inference, combined state resolution, explicit overrides and provenance. | Spin, resolver and package-contract suites passed. |
| 7 | `a214128` | Specified executable resolution, process, workspace and provenance contracts without xTB semantics. | Strict expected-failure harness fence. |
| 8 | `db9fc49` | Implemented `_harness` primitives for argv/cwd/env/timeout/streams, isolated workspaces, hashes and provenance. | Path precedence, process facts, timeout, workspace and hashing tests passed. |
| 9 | `439122a` | Added deterministic fake-xTB fixtures and contracts for capabilities, requests, results, artifacts and failures. | Conditional/expected-failure implementation fence; no official numerical claim. |
| 10 | `bda6eef` | Implemented version probing, method/element capabilities, typed native requests/results and runner error evidence. | xTB 6.7.1 domain, Am preflight rejection, argv, artifacts and failure paths passed. |
| 11 | `796c597` | Added molecule/XYZ conversion, result parsing, finite checks, atom-sequence checks and transactional coordinate commit. | Adapter tests and the official-integration scaffold passed; invalid results did not mutate inputs. |
| 12 | `b4a9761` | Exposed independent `run_gfnff()` and `run_gfn_xtb()` operations with explicit electronic-state consumption. | Independent nodes, override conflicts, preflight and workspace lifecycle passed. |
| 13 | `ea93af4` | Added `hotpot xtb`, strict SDF state streaming, stdin/stdout/stderr separation, ordered parallel records and packaged CLI documentation. | CLI, stream, resource and top-level command-loading tests passed. |
| 14 | `8ea5083` | Specified pipeline controller, payload, artifact, hash, manifest and failure-propagation contracts. | Strict expected-failure pipeline fence. |
| 15 | `aa7d34c`, `03724ff`, `d58c1e3`, `ab04158` | Implemented typed payloads and lazy registry, atomic artifacts and runner, CBond/FF/xTB adapters, then `hotpot run`. FF route selection was shared without changing its scientific kernel. | Registry, lineage, failure stop, stage adapters, strict JSON and inline `::` CLI suites passed. |
| 16 | `e17c2af` | Added official neutral GFN2/GFN-FF single-point parity and both composition forms. | Two official energy comparisons passed; shell and controlled composition used the deterministic fake backend. |
| 17 | `5328071` | Removed the superseded xTB calculator, cache, empty writer, obsolete consumers, tests and generated references. | Consumer search, package-content absence and external wheel smoke checks passed. |
| 18 | `64ec730` | Published xTB/pipeline API guides, applicability limits, wrapper template and initial implementation/validation reports. | 1686 passed, 2 opt-in skips, 4 warnings and 49 subtests; two official parity cases; CPython 3.11 wheel and sdist checks. |
| 19 | `b791f5e`, `652a3c7` | Added official GFN2 optimized-coordinate, chloride-ion and hydrogen-radical parity, then made the tests Python 3.9 compatible. | Official direct-parity coverage increased from two to five cases. |
| 20 | `9fbf984` | Added real Zn/NCCO CBond → FF → GFN2 and CBond → FF → GFN-FF → GFN2 pipelines. | Both real pipelines passed; official suite increased from five to seven. |
| 21 | `837c804` | Added the resumable, hashed, failure-preserving four-route 187-structure benchmark implementation and contracts. | Benchmark unit/contracts passed; the full corpus result was not claimed until Phase 25. |
| 22 | `8313d55` | Added calculator, harness, xTB, CLI and pipeline suites to the CPython 3.9–3.14 compatibility runner. | CI gate inclusion only; full six-runtime completion was recorded in Phase 27. |
| 23 | `b54b59b` | Made `ff --rebuild` explicit after CBond because retained 2D coordinates are not a completed complex build. | README, pipeline documentation, CLI normalization and benchmark guidance agreed. |
| 24 | `5751e00` | Added refinement-benchmark contracts to maintained coverage and compatibility CI. | Contract tests passed in both maintained runners. |
| 25 | `d6a948f` | Recorded five direct parity cases, two real pipelines and the full 187-structure Eu benchmark; froze the production snapshot used for wheel validation. | Four-route aggregate results and failures were checked against the retained manifest and artifacts. |
| 26 | `8e015d3` | Added the exact README Eu/acetate four-node workflow as a real official integration case. | Eu–O topology, charge `+3`, zero unpaired electrons, both coordinate commits and all quality gates passed; official suite became eight. |
| 27 | `b4827ba` | Closed the reports, removed stale matrix-pending language and corrected the GFN-FF timing interpretation. | Final coverage, CPython 3.9–3.14 runtimes and six ABI wheels were reconciled with their retained logs. |
| 28 | `565c639`, `bb9a737`, `1706de4`, `9477d3f`, `5c4dc8f` | Published the root README xTB evidence and path-free corpus summary, brought every README test into maintained coverage, made model materialization explicit, added JUnit failure annotations, and defined the observed 0.25 kJ/mol MCA display tolerance without weakening atom/site checks. | README tests passed locally; hosted coverage run `38050145537` and all 13 jobs of compatibility run `38050145565` passed. The two corrective CI iterations are retained in the validation report rather than hidden. |

Across the Phase 1–28 snapshot, 111 tracked files changed relative to baseline.
The scientific kernels for CBond prediction and the native three-stage complex
force-field workflow were not changed by this work.

## 7. Final delivered contracts

1. Calculator state policy has one canonical package and one pure charge rule
   source.
2. External execution facts live in `_harness`; xTB scientific decisions stay
   in the xTB plugin.
3. GFN-FF and GFN-xTB are independently callable from Python and CLI.
4. `hotpot run` is a typed controller, not a shell evaluator; failures stop
   downstream stages and persist evidence atomically.
5. Coordinates are committed only after a finite, converged, atom-consistent
   optimization result; single points never mutate coordinates.
6. Stable xTB applicability is checked before launch, without method fallback.
7. Every persisted stage artifact carries size and SHA-256 lineage.

## 8. Scientific boundary and optional future directions

The only unclosed scientific boundary in this stage is real extended-GFN-FF
fragment-charge/actinide behavior. Am is rejected before launch by stable xTB
6.7.1, and no extended backend has yet been validated.

The following are optional future architecture or model directions. They are
not incomplete acceptance requirements for the delivered workflow:

- a universal plugin base class or dynamic third-party entry-point discovery;
- an in-process xTB C API backend;
- physical oxidation-state or spin-state prediction;
- pipeline resume or parallel stage execution.

The xTB package is documented as a reference responsibility layout, not as a
premature mandatory superclass for unrelated software.

The installed stable xTB 6.7.1 backend has received direct energy,
optimization-coordinate, ionic and radical parity checks, plus three real
coordination-pipeline validations. The 187-structure benchmark found that
direct GFN2 passed 151/182 Eu complexes, whereas optional GFN-FF pre-refinement
followed by GFN2 passed 132/182. This evidence supports composition and failure
reporting. The chain's lower all-record median is confounded by frequent early
failure and does not justify claiming that GFN-FF improves successful-path
speed or complex robustness.

Stable xTB 6.7.1 remains explicitly unable to process Am. Extended-GFN-FF
fragment-charge or actinide behavior is outside the validated implementation
until a corresponding official backend is built, identified and tested.

The final delivery coverage suite, now including all README executable and
evidence tests, passed `1715` tests with 8 official opt-in skips, 4 warnings and
49 subtests in 281.75 s at 36% aggregate coverage. Runtime tests passed on
CPython 3.9-3.14, and cp39/cp310/cp311/cp312/cp313/cp314 wheels each passed
clean isolated installation and native/resource smoke checks.
