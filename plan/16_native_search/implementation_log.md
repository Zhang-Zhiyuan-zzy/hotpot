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

Git checkpoint: `c704bd6` (`feat(search): execute predicates in native RI
kernel`).

### Step 6 - Perform the atomic public cutover

Status: complete

Planned:

- switch all three public `Searcher` methods together;
- retain the Python facade as the only limit/truncation and result owner;
- remove every production `GraphMatcher` invocation from generic and recursive
  search; and
- run the frozen API plus direct and downstream consumer matrix.

Performed:

- routed `has_match()`, `iter_mappings()`, and `search()` through one freshly
  compiled `NativeProblem` and the existing private mapping-source adapter;
- retained `_MappingIterator`, `Hits`, `Hit`, direct one-shot `Hits` provider
  construction, grouping, mapped/induced bonds, read-only mappings, mutable
  `get_hit`, and the facade-owned `N + 1` probe unchanged;
- left public `Substructure.construct_graph()` and its NetworkX compatibility
  graph untouched while removing NetworkX isomorphism imports from production
  search execution; and
- changed direct `_RecursiveEvaluator.__call__` compatibility behavior to
  collect anchor results from the native cursor, while compiled search uses
  the anchored nested native program directly.

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
  tests/test_cheminfo/test_search_native_predicate_parity.py \
  tests/test_cheminfo/test_search_native_cutover.py \
  tests/mca \
  tests/test_cheminfo/test_bond_metadata.py \
  tests/test_cheminfo/test_ligand_rings.py \
  tests/test_cheminfo/test_relevant_ring_integration.py \
  tests/test_cheminfo/test_mol_assemble.py \
  tests/test_cheminfo/test_mca_calculator.py \
  tests/test_ds/test_stage13_query.py \
  tests/test_cheminfo/test_import_safety.py \
  tests/test_main/test_command_loading.py
```

Result: 739 passed, 49 subtests passed in 32.28 s.

Five explicit cutover tests sabotage `NetworkX GraphMatcher` and exercise all
three public methods plus direct recursive-predicate evaluation. They also
prove first-result termination, mutation visibility between calls, the native
`Hits.graph_matcher` adapter, and read-only mappings. Source inspection finds
no `GraphMatcher` or isomorphism import under `hotpot.cheminfo.search`.

Unexpected finding: none. The retained NetworkX dependency is now limited to
the public compatibility graph and unrelated graph facilities, as planned.

Git checkpoint: `414ccd3` (`feat(search): cut over public search to native RI`).

### Step 7 - Add generic target reuse and verify MCA integration

Status: complete

Planned:

- prepare one target snapshot for an ordered group of generic compiled queries;
- reuse that snapshot across outer and recursive native problems;
- let MCA reuse the generic operation without moving rule names, precedence,
  site eligibility, symmetry selection, or prediction assembly into C++; and
- prove exact site identity while meeting the one-worker P1 performance gate.

Performed:

- added a private frozen prepared-query record containing compiled query IR,
  Python-object requirements, and derived-target requirements;
- added ordered multi-query execution that unions those requirements, packs the
  Hotpot target once, constructs one native `NativeTarget`, and lazily creates
  each `NativeProblem` against the shared target;
- moved target graph construction and parsed atom/bond records into that
  immutable `NativeTarget`; recursive queries receive the same shared object;
- removed the old dict-target `NativeProblem` overload so there remains one
  native construction path;
- cached the 24 invariant MCA query programs at module import and projected
  each rule's anchor directly from raw native mappings, while leaving rule
  order, first-rule assignment, metal eligibility, atom-index ordering, exact
  labeled-graph symmetry representatives, and `is_Nuc_site` ownership in the
  existing Python/MCA layers; and
- added a reproducible 24-molecule one-worker benchmark runner and structural
  tests for one target pack, one native target, ordered duplicate queries,
  requirement union, no recompile, direct anchor projection, and empty groups.

Unexpected finding and correction:

- pybind11 converts Python `None` to an empty `std::shared_ptr<NativeTarget>` by
  default. A direct isolated probe of `NativeProblem(query, None)` therefore
  exited with signal 11 before target dereference. The sole binding now rejects
  `None` at its typed boundary with `TypeError`; no fallback constructor or
  runtime backend was added.

Verification:

```text
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
  --import-mode=importlib \
  tests/test_cheminfo/test_search_native_compiler.py \
  tests/test_cheminfo/test_search_native_production_kernel.py \
  tests/test_cheminfo/test_search_native_predicate_parity.py \
  tests/test_cheminfo/test_search_native_cutover.py \
  tests/test_cheminfo/test_search_native_prepared.py \
  tests/mca/test_site_detection.py \
  tests/mca/test_site_detection_coordination.py \
  tests/mca/test_pipeline_workers.py \
  tests/mca/test_cpu_backend.py \
  tests/mca/test_gpu_routing.py \
  tests/test_cheminfo/test_mca_calculator.py
```

Result: 159 passed in 11.69 s. The complete MCA plus calculator follow-up was
186 passed in 19.42 s.

The frozen 24-molecule P1 corpus retained the exact ordered site/type payload
(`jq -cS .sites` SHA-256
`0dfc9fb2cb048d53dd1aa6cc05535d9147155bb4433775930608e56b63f9b866`).
Its median fell from 0.234229852 s at the post-cutover/pre-reuse baseline to
0.033436576 s with shared native targets, a 7.005x speedup. The final run used
three warmups and nine measured rounds, produced 717.78 molecules/s, and had
204,996 KiB process peak RSS. Evidence files:

- `/tmp/plan16-mca-site-shared-target.json`, SHA-256
  `1815563a186f3744e3443e6e602e974f79d2fa7c63bc08110175c8592c2516f2`;
- `/tmp/plan16-mca-site-shared-target.rusage.txt`, SHA-256
  `801c2587a952e4f95495154494a9892b4d85b95eb8fc59544217c67e56da2eca`.

No persistent molecule cache was introduced: every public search or MCA
molecule call still takes a fresh target snapshot, so mutations are visible on
the next call. The native extension contains no MCA name, rule, site type,
eligibility policy, or result type.

Git checkpoint: `fd21cc7` (`perf(search): share prepared targets across
queries`). Production-only synthetic compatibility bindings were then removed
in `f152b63` (`refactor(search): remove synthetic test executor`); exhaustive
kernel coverage continues through the independent Python brute-force oracle
and the production `NativeTarget`/`NativeProblem` path.

### Step 8 - Close packaging, performance, and final validation

Status: in progress

Planned:

- add one reproducible generic performance/resource gate and run it unchanged
  against the historical executor and the native candidate;
- close short-query regressions without adding a cache, fallback, selector, or
  second topology executor;
- validate sanitizers, packaging, supported Python versions, downstream
  consumers, and MCA CPU/GPU scientific parity; and
- update public documentation and write the final implementation report.

Performed so far:

- added the committed gate runner in `ab18152`; it covers public existence,
  bounded `0`/`1`/`64`/unbounded search, 5,040 complete embeddings, recursive
  SMARTS, mixed Python-callable leaves, repeated calls, phase timings,
  `tracemalloc`, current/max RSS, substantive unordered mapping digests, and
  ordered-repeat determinism;
- ran the historical `2cf0893` implementation and the first native candidate
  on the same host, Python 3.11.16, CPU 0 affinity, and
  `PYTHONHASHSEED=0`. The first candidate passed semantic parity, achieved a
  2.208x ten-workload geometric mean and a 7.787x complete-enumeration
  speedup, but failed the non-regression gate for existence hit, existence
  miss, and bounded zero;
- traced those failures to fixed Python-to-C++ snapshot/record construction,
  not RI traversal. Added a bool-only native existence probe, reused it for
  the facade-owned zero-limit truncation probe, removed the single-query
  multi-query detour, packed only fields and semantic records required by the
  compiled query, and stopped duplicating native scalar fields into a second
  C++ Python-object map;
- retained a fresh target and query snapshot on every public invocation, and
  added focused tests proving query/target mutation visibility, zero-limit
  truncation, no mapping materialization in existence probes, exact sparse
  requirements, and production predicate parity; and
- removed Python 3.10-only `dataclass(slots=True)` use after the clean Python
  3.9 matrix exposed it. Commit `92859b0` records that compatibility fix.

Performance requalification:

| Workload | NetworkX baseline median | Final native median | Speedup |
| --- | ---: | ---: | ---: |
| existence hit | 0.196333 ms | 0.133673 ms | 1.469x |
| existence miss | 0.169952 ms | 0.163490 ms | 1.040x |
| bounded zero | 0.197355 ms | 0.144014 ms | 1.370x |
| bounded one | 0.238252 ms | 0.164793 ms | 1.446x |
| bounded 64 | 2.872069 ms | 0.493045 ms | 5.825x |
| complete 5,040 mappings | 215.118008 ms | 27.440589 ms | 7.839x |
| recursive complete | 1.381150 ms | 0.451948 ms | 3.056x |
| repeated bounded 64 | 2.863868 ms | 0.495280 ms | 5.782x |

The final native ten-workload geometric mean is 2.747x; the five required
categories weighted equally produce 3.018x. No native-only representative is
more than 5% slower than baseline. Mixed-callable complete enumeration is
1.409x faster; mixed-callable existence is 15.4% slower and is reported
separately as required. It remains the same RI traversal with only individual
callable leaves entering Python.

Both reports pass every semantic check and share substantive mapping-set
SHA-256
`3453a51f80ebf2e998d453dadb75e3bac65f88fb5292d3db84adb300b6489ce8`.
The historical JSON/rusage SHA-256 values are
`5835f0e331d399fb8e96a6fce8055691ba0af27563147afcc868e0ba4bbd5383`
and `69f4d09bbff24c1d8a10c4acfa89a510ae4fb48c8056f505751e5db15bf75a0f`;
the final candidate values are
`6b556e8df046f8aca16ccc59fdf5b80d79ba894e487624278aa41fc98103870d`
and `3312bc4cdf831034e698d533611f4439f896a29062f3e04c2332e0d1eda3a037`.
Fifty repeated composite cycles retained 4,308 traced bytes versus 12,263 at
baseline, showed zero current/max RSS growth, and were 3.007x faster.

Validation already completed:

- focused post-correction native/compiler/predicate/cutover/prepared/contract
  suite: 171 passed in 4.62 s;
- broad native search, SMARTS/conformance, MCA, molecule assembly and
  conversion suite: 736 passed plus 49 subtests in 24.84 s before the final
  sparse-record correction; the same broad suite will be repeated at the
  final checkpoint;
- exact-production UBSAN build: 182 passed in 4.70 s with no diagnostic;
- exact-production ASAN build: a fixed-count RI/predicate/recursive/callable/
  lifetime harness passed with no diagnostic. Full ASAN pytest cannot be
  claimed because ASAN plus this pybind11/toolchain combination stalls while
  translating the exhausted cursor's C++ `py::stop_iteration`; normal and
  UBSAN builds exhaust the same cursor correctly;
- clean CPython 3.11 wheel and sdist, isolated install, native import and real
  `[C]-[O]` public search passed. Wheel linkage contains only standard runtime
  libraries and no third-party graph matcher;
- ai-14 CPU routing at 255 sites retained exact MCA/site/joint hashes and did
  not start GPU resources; the 256-site four-A800 path retained exact
  MCA/site/joint hashes. The joint digest covers `is_Nuc_site` and site type.

Unexpected findings and adjustments:

- the first strict performance run was intentionally treated as a gate
  failure, not averaged away. Sparse requirement-complete records fixed the
  overhead while retaining one implementation and per-call mutation
  visibility;
- the first Python 3.9 matrix build compiled the extension but failed import
  on `dataclass(slots=True)`, which is unavailable before Python 3.10. The
  private predicate records do not require slots, so the unsupported option
  was removed rather than adding a compatibility branch; and
- ASAN's exhausted-cursor exception-translation limitation is recorded as a
  sanitizer coverage deviation. UBSAN, ordinary pytest exhaustion tests,
  weak-reference lifetime tests, repeated resource measurements, and the
  bounded ASAN harness remain independent evidence, but are not mislabeled as
  a complete leak-sanitizer pass.

Remaining before closure: repeat the broad suite on the final sparse-record
implementation, finish CPython 3.9--3.14 clean builds/tests, rerun sanitizer
checks on the final source, finish the 4,096-molecule ai-14 workload, complete
source/docs/package audits, and record the final Git checkpoints.
