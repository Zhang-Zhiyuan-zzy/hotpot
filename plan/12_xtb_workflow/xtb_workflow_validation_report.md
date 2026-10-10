# Composable xTB workflow validation report

## 1. Scope and environment

This report separates hermetic fake-backend tests, direct parity against the
official executable, official end-to-end pipelines, the 187-structure
coordination benchmark, and distribution checks. Evidence from one category is
not used as a substitute for another.

| Item | Value |
|---|---|
| Main validation Python | CPython 3.11.16 |
| Official backend | xTB 6.7.1, revision `edcfbbe` |
| Official executable | Resolved environment executable; absolute path retained in integration artifacts |
| Final coverage suite | `1715 passed, 8 skipped, 4 warnings, 49 subtests passed` |
| Coverage-suite wall time | 281.75 s |
| Aggregate measured coverage | 36% |
| Official opt-in integration suite | `8 passed in 5.78 s` |
| README executable/evidence suite | `18 passed` |
| Hosted coverage and Codecov gate | Run `38050145537`: pass |
| Hosted compatibility gate | Run `38050145565`: 13/13 jobs pass |

All eight coverage-suite skips are the official opt-in tests when their
integration environment variables are absent. With the official xTB and CBond
integration resources enabled, all five direct-parity tests and all three
controlled end-to-end tests passed.

The hosted workflows do not install the official xTB executable, so the eight
official tests are intentionally opt-in there. Their local official-backend
result is reported separately from the hermetic CI result.

## 2. Automated evidence

### 2.1 Hermetic backend and composition

The fake executable tests cover:

- successful single-point and optimization artifacts;
- native nonzero exits, non-convergence, missing and malformed artifacts;
- non-finite energy, gradient, charge, and coordinate rejection;
- atom-count and element-order mismatch without coordinate commit;
- paths with spaces, timeout, workspace isolation and optional retention;
- stdout molecular purity and native-log separation;
- multi-record ordering with parallel jobs;
- GFN-FF optimization piped by the operating system into GFN2 single-point;
- optional GFN-FF inside a controlled pipeline, manifest lineage, artifacts,
  electronic-state propagation, and final SDF creation;
- quality and execution failure persistence without downstream execution.

These tests are deterministic contract tests. They do not claim official
numerical validation.

### 2.2 Official direct parity

Five tests compare the wrapper with direct calls to the same official xTB
6.7.1 executable using identical geometry, electronic state, method and
single-thread environment:

| Case | Comparison | Tolerance | Result |
|---|---|---:|---|
| Neutral methanol, GFN2 single point | Energy from `xtbout.json` | `1e-12 Eh` absolute | Pass |
| Neutral methanol, GFN-FF single point | Energy from `gfnff_lists.json` | `1e-12 Eh` absolute | Pass |
| Neutral methanol, GFN2 optimization | Energy and optimized coordinates | `1e-12 Eh`; `1e-8 Å` coordinate absolute | Pass |
| Chloride anion, GFN2 single point | Charge `-1`, zero unpaired electrons and energy | `1e-12 Eh` absolute | Pass |
| Hydrogen radical, GFN2 single point | Charge `0`, one unpaired electron and energy | `1e-12 Eh` absolute | Pass |

The tests also verify backend identity and atom order. Single-point operations
do not commit coordinates; the successful optimization does.

### 2.3 Official controlled coordination pipelines

Three real pipelines use the production CBond model, Hotpot force-field stage
and official xTB executable:

```text
Zn + NCCO -> CBond -> FF --rebuild -> GFN2 single point
Zn + NCCO -> CBond -> FF --rebuild -> GFN-FF optimize -> GFN2 single point
Eu + O=C(O)C -> CBond -> FF --rebuild -> GFN-FF optimize -> GFN2 optimize
```

All three passed. The tests verify the complete stage manifest and artifacts,
standard post-stage geometry gates, finite energies and final 3D coordinates.
The Zn workflows retain Zn-N/Zn-O coordination, charge `+2` and zero unpaired
electrons. The exact README Eu workflow retains Eu-O coordination, charge
`+3`, zero unpaired electrons, coordinate commits from both optimization
stages, and successful FF/GFN-FF/GFN2 geometry gates. `--rebuild` is
intentional: CBond defines coordination topology, but its retained input
coordinates must not be mistaken for a completed 3D complex build.

### 2.4 Reproduction commands

The maintained hermetic and README suites are launched by one repository
entry point:

```bash
./tests/run_coverage.sh
```

That runner includes `tests/readme`; a README push therefore validates its
executable examples and the machine-readable evidence behind its benchmark
tables rather than merely triggering an unrelated workflow.

The source/runtime matrix is launched with:

```bash
./tests/run_inference_compatibility.sh
```

The official integration suite requires an independently installed xTB and
the real CBond model:

```bash
HOTPOT_XTB_INTEGRATION=1 HOTPOT_CBOND_INTEGRATION=1 \
  python -m pytest -q \
    tests/test_plugin/test_xtb/test_official_integration.py \
    tests/test_plugin/test_xtb/test_official_pipeline_integration.py
```

The complete corpus command and its source/cohort identity requirements are
maintained in `tests/benchmarks/coordination_complexes/README.md`. The
benchmark is opt-in and always requires `--official`; it is not silently
launched by pytest.

## 3. The 187-structure coordination benchmark

### 3.1 Execution contract

The benchmark used the 187 extractant structures with Eu, official xTB 6.7.1,
16 case workers and four xTB threads per worker (64 requested cores). It
compared four routes:

1. ligand -> GFN2 optimization;
2. ligand -> GFN-FF optimization -> GFN2 optimization;
3. Eu-ligand complex -> GFN2 optimization;
4. Eu-ligand complex -> GFN-FF optimization -> GFN2 optimization.

All routes retain electronic-state provenance, native logs, last finite/final
structures and geometry-gate reports. The complete run took 683.41 s wall time.
The benchmark manifest SHA-256 is
`389a939c193ea9c4067ebb8a482be113f36e194ad619666b54e3500472090865`.
The retained aggregate `summary.json` SHA-256 is
`c2e3312baef2fd930dfc7d635e62e16be9a0b2673ac6b9bbf9dc1d5b5b801945`.
The ligand denominator is 187. CBond produced 182 Eu-ligand complex inputs, so
the complex denominator is 182; these denominators are not merged. The full
upstream Hotpot run passed the standard geometry gate for 182/187 ligand
records and 181/182 complex records. Terminal structures that failed that
upstream gate were retained as explicit inputs rather than silently removed.

The public, path-free aggregate evidence is stored in
`assets/readme/xtb_coordination_benchmark.json`. A README regression test checks
its internal count/rate/resource consistency, checks route statistics against
both the root README and this report, and checks the retained manifest and
summary digests against this report.

### 3.2 Results

| Target and route | Passed | Quality failure | Execution failure | Pass rate | Median | p90 | p95 | Maximum | Aggregate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Ligand, direct GFN2 | 182/187 | 3 | 2 | 97.326% | 5.637 s | 26.678 s | 35.095 s | 93.906 s | 2129.786 s |
| Ligand, GFN-FF -> GFN2 | 182/187 | 3 | 2 | 97.326% | 3.936 s | 17.346 s | 24.001 s | 83.590 s | 1313.523 s |
| Eu complex, direct GFN2 | 151/182 | 9 | 22 | 82.967% | 7.561 s | 48.039 s | 60.378 s | 361.824 s | 3374.847 s |
| Eu complex, GFN-FF -> GFN2 | 132/182 | 6 | 44 | 72.527% | 6.647 s | 44.835 s | 60.719 s | 181.407 s | 2963.767 s |

GFN-FF itself was inexpensive: median 0.227 s for ligands and 0.258 s for
complexes. However, it did not improve Eu-complex robustness. For the complex
chain, 33 executions stopped in GFN-FF (31 topology/factorization failures that
led to non-finite values and two native `SIGABRT`/double-free failures); 11
additional structures reached GFN2 and failed SCC convergence. All 22 direct
GFN2 complex execution failures were SCC non-convergence, with one also
reporting non-finite numerical behavior.

The ligand failures include three post-optimization quality failures and two
execution failures. Source inspection identified an exact H-H overlap in one
input and an extreme-gradient case. These records remain failures; the
benchmark does not silently repair or remove them.

The measured conclusion is limited but clear: for this Eu corpus and xTB 6.7.1,
direct GFN2 is the more reliable complex route. The lower all-record median of
the GFN-FF chain is confounded by its higher failure rate and early termination;
it is not evidence that successful complex paths are faster. GFN-FF introduces
a substantial native failure class and must remain optional rather than a
claimed speed or robustness step.

## 4. Packaging and compatibility evidence

- The complete CPython 3.9-3.14 runtime matrix passed. CPython 3.9-3.13 each
  passed the 1400-test core suite, 255 SMARTS cases, and the final xTB delta of
  `41 passed, 3 skipped`.
- CPython 3.14 completed the final clean runner with
  `1411 passed, 12 skipped, 3 xfailed`, plus 255 SMARTS cases, and exit code 0.
- CPython 3.9 used Open Babel 3.1.0; CPython 3.10-3.14 used Open Babel 3.2.1.
- Six ABI-specific wheels passed clean isolated validation: cp39, cp310, cp311,
  cp312, cp313 and cp314. For every wheel, ordinary `pip install`, `pip check`,
  native graph loading, xTB imports/resources and `hotpot xtb --doc` passed.
- The six-wheel snapshot was commit `d6a948f`; later changes through `5c4dc8f`
  affect only documentation, tests and CI, not package ABI or production code.
- The source distribution content contract also passed. New calculator,
  harness, xTB and pipeline modules were present; the removed calculator
  facade, old xTB prototype, mutable cache and empty writer were absent.
- The published README chemistry examples are mapped to executable tests, and
  its force-field, gallery and xTB benchmark tables are checked against tracked
  artifacts.

The maintained-suite warnings are known force-field quality warnings and an
existing `search/logic.py` syntax warning. None was attributed to xTB.

### 4.1 GitHub workflow gates

Every push runs `.github/workflows/test_push.yml`, which executes the maintained
coverage runner and uploads its XML report to Codecov. Changes to README, xTB,
pipeline, calculator, force-field, model or related test paths also run
`.github/workflows/inference_compatibility.yml` with:

- one package/sdist content and installed-wheel smoke job;
- CPython 3.9, 3.10, 3.11, 3.12, 3.13 and 3.14 runtime jobs;
- native-wheel build/install/smoke jobs for all six CPython ABIs.

Official numerical xTB tests remain an explicit local/integration gate because
the hosted runner has no official xTB backend. Hermetic xTB, CLI, pipeline and
benchmark-contract tests run in hosted CI on every supported Python version.

### 4.2 Final hosted-CI closure

Adding `tests/readme` to maintained coverage exposed a single cross-platform
assertion: one non-site carbon printed as 324.00 kJ/mol in the recorded CPU
output and 324.25 kJ/mol on the hosted ONNX Runtime. Atom order, every other
value, the nucleophilic N value and all site flags were unchanged. The test now
keeps atom/table/site checks exact and permits only the observed 0.25 kJ/mol
display boundary for MCA values. This is not a scientific acceptance
tolerance for model training or benchmark scores.

The coverage workflow also now caches and explicitly verifies both inference
model bundles before running tests with local-only model resolution. JUnit
failures are emitted as workflow annotations, so a public failure identifies
the exact test instead of exposing only pytest's exit code. An initial attempt
at this change placed `runner.temp` in job-level `env`; GitHub rejected that
context before creating a job. Commit `1706de4` moved it to the executable
steps, and the invalid intermediate configuration is not part of the final
workflow contract.

The final commit under test was `5c4dc8f`. Hosted
[`Tests and coverage`](https://github.com/Zhang-Zhiyuan-zzy/hotpot/actions/runs/38050145537)
passed the maintained suite and Codecov upload. Hosted
[`Inference compatibility`](https://github.com/Zhang-Zhiyuan-zzy/hotpot/actions/runs/38050145565)
passed package smoke, all six CPython 3.9-3.14 runtime jobs and all six native
wheel jobs.

## 5. Explicit scientific boundaries

The following acceptance items are closed by this report: official energy
parity, official optimization-coordinate parity, ionic and radical states,
real CBond/FF/xTB end-to-end execution and the complete four-route
187-structure Eu benchmark.

Two boundaries remain and must not be inferred as supported:

1. Stable xTB 6.7.1 has no valid Am path for the bundled GFN-FF or GFN0/1/2
   parameter sets. Hotpot rejects Am before launch. An inspected extended
   upstream source tree is not a validated installed backend.
2. Real extended-GFN-FF fragment-charge and actinide behavior has not been
   validated. It requires an identified, built and versioned official backend
   with those capabilities. Stable-xTB Eu results cannot establish it.
