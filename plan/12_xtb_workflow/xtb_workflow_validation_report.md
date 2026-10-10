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
| Maintained suite before the final benchmark additions | `1686 passed, 2 skipped, 4 warnings, 49 subtests passed` |
| Maintained-suite wall time | 264.33 s |
| Aggregate measured coverage | 36% |
| Final focused xTB/pipeline/CLI suite | `182 passed, 7 skipped` |
| Official opt-in integration suite | `7 passed` |

The seven focused skips are the opt-in official-backend tests when their
integration environment variables are absent. With the official xTB and CBond
integration resources enabled, all five direct-parity tests and both controlled
end-to-end tests passed.

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

Two real pipelines use the production CBond model, Hotpot force-field stage and
official xTB executable:

```text
Zn + NCCO -> CBond -> FF --rebuild -> GFN2 single point
Zn + NCCO -> CBond -> FF --rebuild -> GFN-FF optimize -> GFN2 single point
```

Both passed. The tests verify the complete stage manifest and artifacts,
inferred charge `+2` and zero unpaired electrons, standard post-stage geometry
gates, retained Zn-N/Zn-O coordination topology, finite energies and final 3D
coordinates. `--rebuild` is intentional: CBond defines coordination topology,
but its retained input coordinates must not be mistaken for a completed 3D
complex build.

A separate Eu/corpus-case-1 manual run of the full four-node workflow also
completed successfully. The observed stage times were 0.172 s for CBond,
1.634 s for the Hotpot FF build, 0.098 s for GFN-FF and 0.166 s for GFN2. Its
GFN-FF and GFN2 energies were `-1.945034 Eh` and `-55.680347 Eh`, respectively.
This is an operability check, not a population-level timing estimate.

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
The ligand denominator is 187. CBond produced 182 Eu-ligand complex inputs, so
the complex denominator is 182; these denominators are not merged. The full
upstream Hotpot run passed the standard geometry gate for 182/187 ligand
records and 181/182 complex records. Terminal structures that failed that
upstream gate were retained as explicit inputs rather than silently removed.

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
direct GFN2 is the more reliable complex route. GFN-FF pre-refinement reduces
median time among the complete population but introduces a substantial native
failure class. It must remain optional rather than a claimed robustness step.

## 4. Packaging and compatibility evidence

- An isolated CPython 3.11 wheel was built:
  `hotpot_zzy-0.5.4.0-cp311-cp311-linux_x86_64.whl`.
- The wheel installed outside the checkout and passed import/resource/CLI
  smoke checks.
- The source distribution was built and its content contract passed.
- xTB/pipeline API READMEs were present in the wheel and source distribution,
  alongside their runtime CLI guides.
- New calculator, harness, xTB and pipeline modules were present; removed
  calculator facade, old xTB prototype, mutable cache and empty writer were
  absent.
- Active calculator/xTB/pipeline sources compile with CPython 3.9 through 3.14.
  The complete six-interpreter runtime matrix remains a distinct gate until
  its run finishes successfully.

The maintained-suite warnings are known force-field quality warnings and an
existing `search/logic.py` syntax warning. None was attributed to xTB.

## 5. Explicit boundaries and remaining work

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

The complete CPython 3.9-3.14 runtime/wheel matrix is also an engineering gate
separate from the scientific xTB acceptance evidence above.
