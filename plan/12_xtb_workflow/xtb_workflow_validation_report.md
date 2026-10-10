# Composable xTB workflow validation report

## 1. Scope and environment

This report distinguishes hermetic fake-backend tests, direct official-backend
parity, manual runtime smoke checks, and distribution validation. Passing one
category is not presented as evidence for another.

| Item | Value |
|---|---|
| Main validation Python | CPython 3.11.16 |
| Official backend | xTB 6.7.1, revision `edcfbbe` |
| Maintained suite | `1686 passed, 2 skipped, 4 warnings, 49 subtests passed` |
| Maintained-suite wall time | 264.33 s |
| Aggregate measured coverage | 36% |
| Final focused xTB/pipeline/CLI/docs suite | `176 passed, 2 skipped` |

The two focused skips are the opt-in official-backend tests when the integration
environment variables are absent. With official integration enabled, both
parity tests passed.

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

The actual OS-pipe and controlled-composition tests use the fake numerical
backend so they remain deterministic and do not claim official numerical
validation.

### 2.2 Official direct parity

Two neutral-methanol single-point calculations compared the Hotpot wrapper with
direct calls to the same official executable and identical geometry, state,
method, and environment:

| Method | Comparison | Tolerance | Result |
|---|---|---:|---|
| GFN2-xTB | Wrapper energy vs direct `xtbout.json` | `1e-12 Eh` absolute | Pass |
| GFN-FF | Wrapper energy vs direct `gfnff_lists.json` | `1e-12 Eh` absolute | Pass |

Both tests also verify backend identity, parsed atom order, and that a
single-point calculation does not commit coordinates.

### 2.3 Manual official runtime smoke

These single-system checks demonstrate operability, not statistically valid
performance:

| Workflow | Energy evidence | Wall time | Native time |
|---|---|---:|---:|
| Water GFN2 single-point | `-5.065772968305 Eh` | 1.17 s | 0.0182 s |
| Water GFN-FF optimization | `-0.327655824559189 Eh` | part of 1.43 s chain | about 0.0099 s |
| Resulting water GFN2 single-point | `-5.070330330443 Eh` | part of 1.43 s chain | about 0.0172 s |

The chained SDF retained total charge 0. GFN-FF received no spin option; the
following GFN2 node resolved zero unpaired electrons. For this tiny input,
process and Python startup dominate the native numerical time.

## 3. Packaging and compatibility evidence

- An isolated CPython 3.11 wheel was built:
  `hotpot_zzy-0.5.4.0-cp311-cp311-linux_x86_64.whl`.
- The wheel installed outside the checkout and passed import/resource/CLI
  smoke checks.
- The source distribution was built and its content contract passed.
- The Phase 18 xTB/pipeline API READMEs were present in both the wheel and
  source distribution, alongside their runtime CLI guides.
- New calculator, harness, xTB, and pipeline modules were present.
- Removed calculator façade, old xTB prototype, mutable cache, empty writer,
  and old docs were absent.
- Active calculator/xTB/pipeline sources compiled with CPython 3.9 through
  3.14. This is a syntax/import-compatibility check, not a claim that the full
  runtime suite was repeated on all six interpreters in this phase.

The four maintained-suite warnings are known force-field quality warnings and
an existing `search/logic.py` syntax warning. No warning was attributed to the
xTB integration.

## 4. Validation boundaries still open

The following original-plan acceptance evidence has not been produced and must
not be inferred from the completed tests:

1. official xTB optimization-coordinate parity using declared RMSD and maximum
   displacement tolerances;
2. official ionic and radical calculations;
3. a complete official-backend shell and controlled `cbond -> ff -> xtb`
   calculation rather than fake-backend composition;
4. the 187-structure CBond/UFF/GFN-FF/GFN-xTB benchmark;
5. real extended-GFN-FF fragment-charge validation and actinide coverage;
6. the complete runtime/wheel test suite on every CPython 3.9–3.14 interpreter.

The delivered core is therefore validated for its typed contracts, failure
semantics, state/stream composition, official neutral single-point energy
parity, and packaging. It has not yet received full scientific acceptance for
all methods, electronic states, optimizations, supported platforms, or the
coordination benchmark.
