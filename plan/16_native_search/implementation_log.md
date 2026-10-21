# Stage 16 implementation log

## Recording policy

This log is updated during implementation rather than reconstructed at the
end. Every step records the intended result, work performed, verification,
unexpected findings, plan adjustments, and the corresponding Git checkpoint.

Only the single approved RI-style search implementation is permitted. This
stage does not retain VF2, NetworkX `GraphMatcher`, an algorithm selector, a
profile, or a runtime fallback.

## Baseline

| Item | Result |
| --- | --- |
| Date | 2026-10-21 |
| Working branch | `perf/native-search` |
| Implementation starting commit | `4bc7eaf` |
| Planning baseline | `2cf0893` |
| Python | 3.11.16, GCC 14.3.0 runtime build |
| Host platform | Linux 5.15.0-139-generic x86_64, glibc 2.31 |
| Native compiler | GCC 9.4.0, C++17 capable |
| NetworkX | 3.6.1 |
| NumPy | 2.4.6 |
| RDKit | 2026.09.1 |
| Focused search/SMARTS baseline | 306 passed, 49 subtests passed in 2.23 s |
| MCA baseline | 183 passed in 13.37 s |
| Quick benchmark wall/RSS | 17.77 s; 212,960 KiB maximum RSS |
| Seven-wildcard stress | 5,040 mappings; 1,104.369 ms median; 5,094,700 traced bytes |

The focused baseline command was:

```text
python -m pytest -q tests/smarts_conformance \
  tests/test_cheminfo/test_search.py \
  tests/test_cheminfo/test_search_mapping.py \
  tests/test_cheminfo/test_smarts.py
```

The MCA baseline command was:

```text
python -m pytest -q tests/mca
```

The quick benchmark artifact was written outside the repository to
`/tmp/plan16-search-quick-baseline.json`; GNU `time -v` evidence was written to
`/tmp/plan16-search-quick-baseline.rusage.txt`.

The active runtime environment does not directly provide the `pybind11`
Python package and has setuptools 83, above the declared build cap. Native
build evidence will therefore use the repository's isolated `uv` command with
setuptools 77-81 and pybind11 3.x rather than modifying product dependencies or
claiming this shell as a valid wheel-build environment.

## Step record

### Step 0 - Freeze implementation authorization

Status: complete

Planned:

- retain one deterministic RI-style implementation;
- preserve the reviewed public API and substantive search semantics; and
- begin implementation only after explicit owner authorization.

Performed:

- recorded the owner's decision to retain RI-style only;
- rejected a second VF2 implementation and public algorithm selection; and
- retained the mandatory native extension with no NetworkX execution fallback.

Verification:

- planning commits `5d39bac` and `4bc7eaf` contain the reviewed normative
  design, implementation choice, API tree, gates, and approval checklist.

Adjustment: none.

### Step 1 - Establish executable baselines and freeze contracts

Status: complete

Planned:

- run focused search/SMARTS and MCA baselines;
- capture the existing quick performance baseline and resource use;
- freeze untested public and degenerate-query behavior before production
  changes; and
- record native build-environment constraints.

Performed:

- completed the focused and MCA baselines recorded above;
- completed the quick generic-search benchmark;
- audited public signatures, one-shot `Hits` providers, bounded iteration,
  sparse query keys, empty queries, duplicate nodes/edges, self-loops,
  mutation, callable behavior, packaging, and downstream consumers; and
- identified that the current benchmark omits isolated `has_match`, bounded,
  recursive, mixed-callable, native-allocation, and MCA P1 measurements.

- added 65 frozen contract cases covering the complete reviewed signature and
  alias manifest, descriptors, enums/errors, empty queries, sparse duplicate
  query keys, duplicate-edge last-wins matching with original result edges,
  programmatic self-loops, every bounded-iteration boundary and invalid limit,
  and one-shot target-to-query `Hits` providers;
- confirmed that these fixtures pass against the unchanged NetworkX execution
  path; and
- recorded SHA-256 `333468f4640f7cbfe8ee1b3d2602b078789f1b820e38d2fec15f9191af8f9749`
  for the quick benchmark JSON and
  `36ab4218f4af0a2cd9a7bd8dcc3badff604600dc2ee3583c587b162477449be5`
  for its GNU-time evidence.

Verification:

```text
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_cheminfo/test_search_native_contract.py
```

Result: 65 passed in 1.01 s.

Unexpected findings:

- the local shell is suitable for Python regression tests but not an
  authoritative isolated native build;
- the maintained build compiles all native extensions and therefore requires
  Open Babel headers even for search-only changes; and
- no tracked MCA P1-only benchmark reproduces the prior external site-search
  timing, so final claims must distinguish the committed generic benchmark,
  maintained MCA regressions, and any separately frozen MCA performance run.

Adjustment:

- add focused contract fixtures before changing execution;
- use isolated native builds and test installed wheels outside the checkout;
  and
- do not invent an MCA performance claim where the repository lacks a frozen
  reproducible input.

Git checkpoint: recorded in the Stage 0 contract commit immediately following
this log entry.

### Step 2 - Introduce typed parser predicate IR

Status: pending

### Step 3 - Implement and prove the deterministic native topology kernel

Status: pending

### Step 4 - Add native predicates and leaf-only Python callbacks

Status: pending

### Step 5 - Complete recursive and Hotpot SMARTS semantics

Status: pending

### Step 6 - Perform the atomic public cutover

Status: pending

### Step 7 - Add generic target reuse and verify MCA integration

Status: pending

### Step 8 - Close packaging, performance, and final validation

Status: pending
