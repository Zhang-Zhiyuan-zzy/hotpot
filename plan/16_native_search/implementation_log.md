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

Git checkpoint: `a4547f7` (`test(search): freeze native migration contracts`).

### Step 2 - Introduce typed parser predicate IR

Status: complete

Planned:

- replace only parser-owned Python closures with immutable typed predicates;
- preserve direct callability, exact representations, Boolean order,
  aromatic-state inference, recursive cache behavior, and existing bare
  set/private bond constraints; and
- keep arbitrary programmatic callables opaque and untouched.

Performed:

- added private typed atom and bond operation/field nodes with frozen identity
  semantics;
- represented every active atom primitive, Boolean operation, recursive leaf,
  multi-bond alternative, and implicit single-or-aromatic bond without
  changing parser output shapes outside parser-owned callables;
- retained left-to-right short circuiting, recursive molecule-state cache
  invalidation, exact `repr`, and callable behavior; and
- added focused tests covering every typed operation family, aromatic hints,
  hydrogen ambiguity, direct `parse_bracket_atom()` use, immutable identity,
  opaque-callable separation, mutation visibility, and recursive atom/bond
  invalidation.

Verification:

```text
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/smarts_conformance \
  tests/test_cheminfo/test_search.py \
  tests/test_cheminfo/test_search_mapping.py \
  tests/test_cheminfo/test_smarts.py \
  tests/test_cheminfo/test_search_native_contract.py
```

Result: 398 passed, 49 subtests passed in 2.36 s.

Unexpected finding: the active parser has a deliberately mixed output model:
bare atom and explicit single-bond constraints are sets/private guards, while
compound atom and multi/implicit bond expressions are parser-owned callables.
Uniformly converting all constraints would have changed the current public
query representation, so the implementation keeps that boundary.

Adjustment: native query compilation must recognize typed IR positively by
its private class, then separately compile existing set/guard constraints and
arbitrary opaque callables. It must never classify every `Callable` as typed.

Git checkpoint: `f97ef7c` (`refactor(search): add typed SMARTS predicate IR`).

### Step 3 - Implement and prove the deterministic native topology kernel

Status: complete

Planned:

- add the mandatory in-tree C++17 extension with no graph-library dependency;
- implement the reviewed deterministic, component-aware RI-style traversal;
- prove the topology kernel independently through synthetic compatibility
  matrices and an independent injection oracle; and
- retain the Python facade as the sole bounded-iteration owner.

Performed:

- added compact sorted-key graph storage, canonical undirected edges, CSR
  adjacency, degree data, self-loop lookup, and edge lookup;
- implemented the fixed component/root/frontier/parent query order, ascending
  target candidate order, injectivity, non-induced edge checks, disconnected
  queries, self-loops, raw automorphism enumeration, conservative domains,
  and the approved safe forward checks;
- added a lazy native cursor whose lifetime owns an immutable problem snapshot
  and which releases the GIL while advancing;
- exposed only a private synthetic compatibility-table boundary for this
  proof stage, plus a private adapter from query-aligned native tuples to the
  legacy target-to-query mapping-provider protocol; and
- registered the extension sources, type stub, and source-distribution files.

Verification:

```text
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/test_cheminfo/test_search_native_kernel.py \
  tests/test_cheminfo/test_search_native_contract.py \
  tests/test_cheminfo/test_search_mapping.py
```

Result: 83 passed in 0.99 s, including 14 native-kernel tests. The kernel
suite exhaustively compared all simple query graphs through three vertices
against all simple target graphs through four vertices using an independent
permutation oracle. It also covered empty and disconnected queries,
self-loops, non-induced matching, every automorphism, sparse keys, input and
cursor lifetimes, synthetic node/edge rejection, and every facade bound.

The isolated `uv` build compiled and linked the four search translation units
into `_search_native.cpython-311-x86_64-linux-gnu.so` without warnings or
errors. The repository-wide forced build was stopped only after the search
extension had linked and setup had moved on to recompiling unrelated existing
extensions. `ldd` showed only the platform C/C++ runtime libraries and no
graph or chemistry toolkit dependency.

Unexpected finding: the repository setup builds every native extension even
when only the search extension changed. A focused search build is therefore
useful during development, while the final packaging gate must still exercise
the complete clean wheel build.

Adjustment: keep the synthetic compatibility boundary private and temporary;
the next step replaces it with the production predicate program without
switching public `Searcher` prematurely.

Git checkpoint: `b1aa3cd` (`feat(search): add deterministic native RI kernel`).

### Step 4 - Add native predicates and leaf-only Python callbacks

Status: complete

Planned:

- compile normalized programmatic queries and typed SMARTS IR into a private
  immutable native schema;
- pack only the target facts required by the compiled query;
- evaluate typed atom, bond, topology, ring, and semantics predicates in C++;
  and
- preserve arbitrary named constraints and callables as lazy Python leaves
  inside the same RI traversal.

Performed:

- added a private query compiler that reproduces simple-graph normalization,
  sparse duplicate-atom keys, last-duplicate-edge matching, self-loops,
  insertion-ordered constraints, private typed-IR recognition, and recursive
  programs;
- added query-selective target packing for native scalar fields, full-graph and
  ligand-skeleton degree/connectivity/valence, relevant-ring counts and sizes,
  total hydrogen, bond kind, and authoritative bond aromaticity;
- introduced one production `NativeProblem` over the existing RI cursor rather
  than a second topology executor;
- implemented conservative native domains and tri-state Boolean pruning that
  never invokes an opaque predicate or recursion speculatively;
- retained the exact global non-callable attribute preflight, then evaluated
  constraints in insertion order with bond guards first;
- reacquired the GIL only for lazy `hasattr`/`getattr`, Python membership, or
  opaque-callable leaves, passed the complete `Atom`/`Bond`, ignored the
  callable keyword name, and propagated the original exception; and
- retained owning Python references through the cursor lifetime while native
  scalar-only traversal remains GIL-free.

Boundary findings and fixes:

- Python membership is implemented with the general containment protocol, not
  a set-only API, because the public mutable `kwargs` mapping can be replaced
  with another container;
- signed-int64 overflow is classified as a host-evaluated value instead of
  failing native compilation;
- unused derived fields are not evaluated, preventing a wildcard query over a
  dummy atomic-number-zero atom from touching its undefined periodic group;
  and
- complete relevant-ring families are materialized at most once per required
  target view rather than once per atom.

### Step 5 - Complete recursive and Hotpot SMARTS semantics

Status: complete

Performed:

- compiled recursive SMARTS into nested native query programs rather than an
  opaque callable;
- added query-key-zero anchored existence search using the same RI cursor with
  an anchored root plan and shared injectivity across disconnected nested
  components;
- memoized recursive existence only by nested program and target position
  within one immutable target snapshot; and
- covered full-graph/ligand-skeleton values, Relevant Cycle semantics,
  coordination bonds, explicit SMARTS bond guards, Boolean order, and
  recursion inherited semantics in native-versus-legacy mapping-set tests.

Verification:

```text
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  tests/smarts_conformance \
  tests/test_cheminfo/test_search.py \
  tests/test_cheminfo/test_search_mapping.py \
  tests/test_cheminfo/test_smarts.py \
  tests/test_cheminfo/test_search_native_contract.py \
  tests/test_cheminfo/test_search_native_kernel.py \
  tests/test_cheminfo/test_search_native_compiler.py \
  tests/test_cheminfo/test_search_native_production_kernel.py \
  tests/test_cheminfo/test_search_native_predicate_parity.py
```

Result: 499 passed, 49 subtests passed in 2.61 s. The new production-focused
subset contains 128 tests, and the native-versus-NetworkX parity module alone
contains 66 cases. It compares complete mapping sets while separately requiring
repeatable native order; it does not freeze legacy order.

The final incremental isolated build linked the current production source and
all focused tests imported that rebuilt extension. Linkage still contains only
the platform C/C++ runtime libraries.

Adjustment: keep public `Searcher` on its historical executor until this full
predicate/SMARTS checkpoint is committed. The next step performs the three
public method changes atomically and removes production `GraphMatcher` use.

Git checkpoint: recorded in the native-predicate commit immediately following
this log entry.

### Step 6 - Perform the atomic public cutover

Status: pending

### Step 7 - Add generic target reuse and verify MCA integration

Status: pending

### Step 8 - Close packaging, performance, and final validation

Status: pending
