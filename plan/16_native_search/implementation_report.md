# Stage 16 native-search implementation report

- Status: implemented and validated
- Branch: `perf/native-search`
- Planning baseline: `2cf0893`
- Production implementation head: `60288d2`
- Validation-harness head: `e8f72eb` (test-only)
- Date: 2026-10-21

## 1. Outcome

Hotpot substructure search now has one production topology executor: an
in-tree C++17 deterministic RI-style subgraph-monomorphism kernel exposed
through pybind11. `Searcher.has_match()`, `iter_mappings()`, and `search()` all
use it. There is no VF2 implementation, algorithm selector, backend/profile
option, NetworkX execution fallback, retained legacy matcher, or MCA-specific
search engine.

The implementation preserves the approved substantive contract:

- complete embedding sets and every supported Hotpot query/SMARTS predicate;
- deterministic enumeration for identical inputs and deterministic
  predicates, without making historical NetworkX order a contract;
- bounded `max_matches` counts, prefix and truncation behavior;
- opaque Python callable predicates as individual GIL-protected leaves inside
  the native DFS, never as a whole-search fallback;
- full-graph and ligand-skeleton semantics, recursive anchoring, rings,
  coordination behavior, automorphisms, self-loops and disconnected queries;
- all existing public classes, signatures, aliases and result protocols; and
- MCA rule order, precedence, site type, eligibility, symmetry representative,
  atom ordering and `is_Nuc_site` ownership in the existing MCA/Python layer.

The native cutover is also a performance improvement. On the same pinned CPU,
the final generic-search five-category geometric mean is 3.047x, the complete
5,040-embedding workload is 7.842x faster, and no native-only representative
is more than 5% slower than the historical executor. Final one-worker MCA site
detection is 6.001x faster than the post-cutover/pre-reuse baseline with the
exact ordered site/type payload unchanged.

## 2. Implemented architecture

The production path is:

```text
mutable Substructure
  -> fresh typed query IR snapshot
mutable Molecule
  -> fresh requirement-complete target snapshot
  -> shared NativeTarget
  -> NativeProblem (query plan + domains + predicate VM)
  -> deterministic RI MappingCursor
  -> existing MappingProxy / Hit / Hits facade
```

The query compiler separates typed native sets/expressions from opaque host
sets and callables. The target packer emits only fields and semantic records
actually required by the compiled query. Native scalar values are stored once
in C++, while general Python values and original Atom/Bond objects remain
owned only when a host predicate needs them.

`NativeProblem.has_match()` uses the same `MappingCursor` but returns only a
boolean. This avoids constructing a mapping that the caller would discard.
The same probe implements the observable `max_matches=0` truncation check;
zero remains a real existence probe rather than an unconditional empty fast
return.

Recursive SMARTS builds nested `NativeProblem` objects against the same
`NativeTarget`. Anchoring remains query-key 0, recursive result memoization is
per target snapshot, and nested queries stay in the same RI executor.

Generic ordered multi-query execution unions field/object requirements and
packs one target. MCA prepares its 24 invariant query programs once and runs
them against one target per molecule. No target is cached across public calls:
query and molecule mutation between calls remains visible.

MCA's separate post-match symmetry-representative step remains MCA-owned and
continues to use labeled NetworkX `is_isomorphic` checks. It is not part of
generic substructure search and was intentionally outside this nativeization;
site ordering and representative semantics therefore remain unchanged.

## 3. Public API result

No approved public signature changed. The existing API tree remains centered
on `Query`, `QueryAtom`, `QueryBond`, `Substructure`, `Searcher`, `Hits`,
`Hit`, `SmartsSemantics`, `SmartsSyntaxError`, and
`UnsupportedSmartsError`, with the same facade identities through
`hotpot.cheminfo.search`, `hotpot.cheminfo`, and `hotpot`.

Notable observable details retained by tests include:

- read-only query-to-target mapping views;
- facade-owned `N + 1` bounded probing and authoritative `truncated` state;
- target-to-query direction of the private matcher-like adapter retained in
  `Hits.graph_matcher`;
- one-shot external `Hits` providers and lazy mutable `Hits.hits` caching;
- mapped versus induced bonds, grouped automorphisms, empty queries and sparse
  query keys;
- exact exception propagation from opaque callables; and
- query/target mutation visibility on the next method call.

The native extension's implementation-private surface contains
`MappingCursor`, `NativeTarget`, and `NativeProblem`. `NativeProblem` provides
`query_keys`, `cursor()`, and the bool-only `has_match()`. The development-only
synthetic compatibility executor was removed before closure.

## 4. Implementation sequence

| Commit | Result |
| --- | --- |
| `5d39bac` | Create the reviewed native-search plan |
| `4bc7eaf` | Freeze RI selection, API tree, signatures and validation gates |
| `a4547f7` | Add executable legacy contracts and baselines |
| `f97ef7c` | Introduce immutable typed SMARTS predicate IR |
| `b1aa3cd` | Add deterministic RI graph kernel and brute-force tests |
| `c704bd6` | Execute native predicates, recursive programs and callable leaves |
| `414ccd3` | Cut all three public search methods over atomically |
| `fd21cc7` | Share prepared targets across generic queries and MCA rules |
| `f152b63` | Remove the synthetic production/test interface |
| `ab18152` | Add the reproducible generic performance/resource gate |
| `92859b0` | Restore Python 3.9 compatibility for private predicate records |
| `60288d2` | Remove short-query marshalling overhead and pass strict performance gates |
| `c7144cf` | Canonicalize signature-contract rendering across supported Python versions |
| `e8f72eb` | Canonicalize resolved and unresolved forward annotations |

The detailed planned/performed/verified/adjusted record for every step is in
[implementation_log.md](implementation_log.md).

## 5. Main file changes

| Area | Result |
| --- | --- |
| `search/_predicate_ir.py`, `query_ir.py` | Typed atom/bond operations, recursive programs and exact target requirements |
| `search/native.py` | Fresh sparse target packing, prepared-query groups and native result adapter |
| `search/_native/graph_data.*` | Deterministic normalized graph/CSR representation |
| `search/_native/matcher.*` | Single RI plan, domains, DFS, forward checks and cursor |
| `search/_native/production.*` | Predicate VM, Python leaf trampoline, recursion, target/problem ownership |
| `search/_native/bindings.cpp`, `_search_native.pyi` | pybind11 cursor/target/problem bindings and bool existence probe |
| `search/search.py` | Atomic facade cutover while retaining result protocols |
| `AImodels/mca/site_detection.py` | Prepared invariant rules and one-target rule execution |
| `tests/test_cheminfo/test_search_native_*` | Contracts, oracle kernel, compiler, production, parity, cutover and reuse tests |
| CI, coverage, README and SMARTS/MCA docs | Native build/smoke coverage and current execution documentation |
| `plan/16_native_search/` | Normative plan, live log, reproducible runners and this report |

The reviewed private layout proposed separate query-program, predicate and
Python-callback translation units. During implementation those closely coupled
records were consolidated into `production.cpp/.hpp`. This is an internal file
layout simplification: the typed IR, native predicate VM and GIL callback
boundary remain distinct responsibilities, and no semantic or public-API
change results from the consolidation.

## 6. Correctness evidence

### 6.1 Final local suites

At exact implementation head `60288d2`:

- native contract/kernel/compiler/production/predicate/cutover/prepared suite:
  186 passed under UBSAN in 4.59 s;
- final targeted downstream closure suite covering search,
  SMARTS/conformance, MCA, molecule assembly and molecule conversion: 740
  passed plus 49 subtests in 24.67 s;
- final focused sparse/existence correction suite: 171 passed in 4.62 s;
- `git diff --check`, Python compilation and shell syntax checks passed.

At validation-harness head `e8f72eb`, whose production source is unchanged
from `60288d2`, the maintained coverage entry point passed in one complete
isolated-worktree run: 2,437 passed, 12 skipped and 49 subtests passed in
295.80 s. Repository-wide branch coverage was 40%; the native-search Python
execution modules were 94--100% covered. Temporary writable
`HOTPOT_MODEL_HOME` and `HF_HOME` directories were used so CBond model and Xet
cache writes did not target the sandboxed user cache. The run exited zero.

The required full dataset-consumer gate also passed locally: 437 passed and
one optional CSD/SCO95 local-evidence test skipped in 182.38 s. Its direct
native-search consumer creates `SmartsSearcher("O")`, executes
`Searcher.has_match()`, and passed independently in 2.58 s. The focused
dataset query file is retained in the maintained coverage target list; the
full governance-heavy dataset suite remains recorded closure evidence.

### 6.2 Semantic and determinism coverage

Tests cover:

- independent exhaustive small labeled graphs, complete mapping sets and
  deterministic order;
- empty, disconnected, self-loop, sparse-key and duplicate normalization
  cases;
- symmetric chains/cycles, automorphisms and grouped atom-set results;
- all typed atom/bond operation families, programmatic native/host sets and
  missing attributes;
- full-graph/ligand coordination and relevant-ring behavior;
- anchored recursive queries including disconnected nested components;
- pure callable, mixed callable/native, exception, short-circuit and lifetime
  behavior;
- every bounded boundary, including zero, exact-bound, over-bound and empty
  results;
- thread sharing and cross-process/hash-seed sequence stability; and
- sabotage tests proving that production search never constructs NetworkX
  `GraphMatcher`.

The historical and native heavy performance reports both passed every
semantic assertion and produced the same substantive mapping-set SHA-256:

```text
3453a51f80ebf2e998d453dadb75e3bac65f88fb5292d3db84adb300b6489ce8
```

## 7. Generic search performance

Method: identical committed runner, same host, CPython 3.11.16, CPU affinity
0, `PYTHONHASHSEED=0`, heavy profile. The baseline is `2cf0893`; the final
candidate is clean detached commit `60288d2`.

| Workload | Baseline median | Native median | Speedup |
| --- | ---: | ---: | ---: |
| `has_match` hit | 0.196333 ms | 0.131299 ms | 1.495x |
| `has_match` miss | 0.169952 ms | 0.163019 ms | 1.043x |
| bounded 0 | 0.197355 ms | 0.143101 ms | 1.379x |
| bounded 1 | 0.238252 ms | 0.161817 ms | 1.472x |
| bounded 64 | 2.872069 ms | 0.485992 ms | 5.910x |
| complete 5,040 mappings | 215.118008 ms | 27.430439 ms | 7.842x |
| complete public hits | 217.353249 ms | 29.281501 ms | 7.423x |
| recursive complete | 1.381150 ms | 0.449192 ms | 3.075x |
| recursive `has_match` | 0.532820 ms | 0.386694 ms | 1.378x |
| repeated bounded 64 | 2.863868 ms | 0.489629 ms | 5.849x |

The ten native-only workloads have a 2.774x geometric mean. Equal-weight
category speedups are existence 1.249x, bounded 2.289x, complete 7.630x,
recursive 2.058x and repeated 5.849x; their geometric mean is 3.047x. All
native-only items pass the plan's no-more-than-5% regression gate.

Mixed-callable complete enumeration is 1.413x faster. Mixed-callable
`has_match` is 11.8% slower and is deliberately reported separately, as the
plan requires. Its topology traversal remains native and every callable test
passes; the result does not conceal whole-search Python fallback.

Final phase medians were 0.041579 ms query-IR compilation, 0.048332 ms target
packing, 0.032762 ms `NativeTarget`, 0.032692 ms `NativeProblem`, 8.632086 ms
raw cursor, 18.896409 ms direction adapter, 27.152573 ms public mappings and
29.223731 ms grouped public hits.

Fifty repeated composite cycles improved from 26.256108 to 8.814914 ms
(2.978x). Both implementations showed zero current/max RSS growth. Retained
traced growth fell from 12,263 to 4,237 bytes and traced peak fell from 364,851
to 309,313 bytes. Process peak RSS was 214,612 KiB baseline and 217,044 KiB
candidate; the 1.13% difference is bounded process footprint, not per-call
growth.

Evidence SHA-256:

| Artifact | SHA-256 |
| --- | --- |
| committed runner | `c449ffb04e9e239948fa6caa1d03428652a3f476d1bb1b9ee600906e45c57df6` |
| baseline JSON | `5835f0e331d399fb8e96a6fce8055691ba0af27563147afcc868e0ba4bbd5383` |
| baseline rusage | `69f4d09bbff24c1d8a10c4acfa89a510ae4fb48c8056f505751e5db15bf75a0f` |
| final `60288d2` JSON | `e9e5bce38f3ded612a2f473b0e28753a5792a1d0e09f27d38b8a906678b6358e` |
| final `60288d2` rusage | `29137a4a56797843139e485545b3cfed477ae7fc43bc1cc52a6038b8a6b171b0` |

## 8. MCA performance and scientific parity

The final clean-commit 24-molecule one-worker site workload has median
0.039032 s, 614.87 molecules/s and 204,284 KiB maximum RSS. The corresponding
post-cutover/pre-reuse baseline was 0.234230 s, so the final speedup is 6.001x,
above the required 1.5x.

The ordered site/type payload SHA-256 is unchanged:

```text
0dfc9fb2cb048d53dd1aa6cc05535d9147155bb4433775930608e56b63f9b866
```

Final evidence JSON/rusage SHA-256 values are
`9cb87111b86ee5968340dc9286182b466d02a419db5e0360128207d1758fd6c8`
and `d15f06558140cd377618f5b7d9228a75f8846f21ec603df4e0c953429f5362f9`.

Exact source commit `60288d2fd15e84d5adfb8dc88a0c44e69549ec67` was then
compiled and run on ai-14 from `~/mca_test/hotpot-native-60288d2`. The search
extension SHA-256 was
`49ee25158e1b60db89eef64b20f05970a3e300495fd3fe22f01e5462eaf10602`.
The final remote run covered both routing sides of the 256-site threshold and
the 4,096-molecule cross-capacity workload:

- 255 sites: GPU-preferred routing remained on CPU, started no GPU resource,
  matched explicit CPU, restored affinity and retained MCA/site/joint digests;
- 256 sites: four A800 consumers on free GPUs 2--5 ran with 64 CPUs at
  29.64 molecules/s and retained the same three digests as the historical
  implementation; and
- 4,096 molecules: the workload crossed the 8,192-atom admission capacity in
  13 cycles, ran at 47.55 molecules/s and retained the exact expected MCA,
  site and joint digests.

The exact remote digests were:

| Scale | Route | MCA SHA-256 | Site SHA-256 | Joint SHA-256 |
| ---: | --- | --- | --- | --- |
| 255 | automatic CPU; equal to explicit CPU | `98c18f01a0584a6558d958622015c63de83e512e0bca276e7f750fde10de1363` | `3a7e755ce3f0dcf891ec304f6201a7a034a82ddcc56ab2d33f5facbb3b69bdd2` | `87d0417b1b19258f125b763499ad0accc4615772fd87412c763e4213293d7336` |
| 256 | four A800 GPUs and 64 CPUs | `71a50dcf8e33bef6b50a385769251be6c15d84c745fb924b1a97e6abfdd10ec9` | `33519e36432b580e6c3e79500c4e906279c3f74536fe44232d3bab266c621813` | `4bd05408483946e518058f7f7d096ff6eb309bc1e182847ffa6ee7abdfd459a1` |
| 4,096 | four A800 GPUs, 64 CPUs, 13 admission cycles | `ad00da4e45c7e4462f7d5b9eab1ed6918989e18dec8a6a4a1299659084b81544` | `a7339ccdce12c9f53ac13ddebe650868d82332ae748f27202eeee063232482e2` | `b81031d437b6766bf96282dac21410f1b0e55fda2117f5c856ff88b67c18d7eb` |

The joint digest includes atom MCA, site MCA, site index/type and
`is_Nuc_site`; it is not a shape-only comparison.

Remote evidence is preserved under
`~/mca_test/results/native_search_60288d2`. JSON SHA-256 values are
`b914a7c3f869cb1ea3de9a447ad660ccd22ef5248b5b1305d47dfa2a65a9fe38`
(255), `c4b1d1ba2830d45ba6bb42f796d029aa19143a52cb23f59906477dabf1ab58e6`
(256) and
`20aaf4b9fdf2bdce40a9d5d34fa83af77905cc0850ac8bff137cde58b95eeb3f`
(4,096).

The first final-source GPU invocation omitted the NVIDIA wheel library paths
and failed before inference because ONNX Runtime could not load
`libcudnn.so.9`. Its log was preserved, the library paths were supplied
explicitly, and both GPU validations then passed. No package or environment
was modified to obtain the result.

## 9. Sanitizers and lifetime diagnostics

The final source was compiled separately with ASAN and UBSAN.

- UBSAN (`-fsanitize=undefined -fno-sanitize-recover=all`): 186 focused tests
  passed in 4.59 s with no diagnostic, including iterator exhaustion,
  callbacks, recursion, mutation and lifetime tests.
- ASAN (`-fsanitize=address`): a fixed-count native harness covering RI
  traversal, positive/negative bool probes, typed predicates, recursion,
  opaque callables, self-loops, shared targets and object destruction passed
  with no diagnostic.

Full ASAN pytest is not claimed. With this GCC 9/libasan/pybind11 toolchain,
the extension stalls after `MappingCursor::next_mapping()` correctly returns
empty while pybind11 translates the deliberately exhausted iterator into C++
`py::stop_iteration`. Direct debugging confirmed that traversal reaches and
returns from the exhausted branch; removing GIL release does not change the
stall. Normal builds on supported Pythons and the complete UBSAN suite execute
the same exhaustion path correctly.

LeakSanitizer/Valgrind was not available as an independent full-suite gate.
The report therefore relies on the bounded ASAN harness, UBSAN suite,
weak-reference ownership tests, repeated-call zero-growth measurements and
clean shutdown tests, and records this as residual diagnostic coverage rather
than calling it a complete leak-sanitizer pass.

## 10. Packaging and source audit

At exact commit `60288d2`, a clean CPython 3.11 wheel and sdist were built,
installed outside the checkout, and exercised through real public search.

| Artifact | Size | SHA-256 |
| --- | ---: | --- |
| wheel | 3,059,282 bytes | `8cd7bfa1580031eecaa0514da24a227cfb0d1f6d20501ba2c74e1a1eba786773` |
| sdist | 1,290,419 bytes | `fc3a0aef02d5386fa69edee74ef7735c920e571bfe8d484d9070ca2179583045` |

The wheel contains the extension, stub and all native sources. The installed
wheel passed real `[C]-[O]`, positive/negative `has_match`, and matching/missing
`max_matches=0` truncation cases. The sdist contains all eight required native
`.cpp/.hpp` files and the stub.

`ldd`, `readelf`, symbol and source scans found only standard runtime
libraries: no Boost, VF2, NetworkX executor, igraph, LEMON, nauty or other
third-party graph matcher. Production search source contains no
`GraphMatcher`, backend selector, compatibility executor or synthetic problem.
NetworkX remains used for the public `Substructure.construct_graph()`
compatibility projection, MCA-owned post-match automorphic-site
deduplication via `nx.is_isomorphic`, and unrelated Hotpot graph surfaces. It
does not participate in substructure traversal or predicate evaluation.

The locally built wheel is ordinary `linux_x86_64`; manylinux repair remains a
CI/publish workflow responsibility. The workflow now imports the native
extension and executes real public search after wheel installation, and its
sdist check explicitly requires every search native source.

## 11. Supported-Python matrix

Clean wheels were built from exact production commit
`60288d2fd15e84d5adfb8dc88a0c44e69549ec67` for every supported interpreter.
Each wheel was then installed from `/tmp`, outside the checkout, and imported
from isolated `site-packages`. The ABI-tagged extension, positive and negative
`has_match`, and hit/miss `max_matches=0` truncation behavior passed on every
version.

| Python | Native/SMARTS validation | Isolated wheel | Wheel SHA-256 |
| --- | --- | --- | --- |
| 3.9.25 | final native + SMARTS: 468 passed | Pass (`cp39`) | `a910f98d83ae2b59b7e582bae6fe7ce2af4bd700dea228f763ee5aa4d0e05b9f` |
| 3.10.20 | final native + SMARTS: 468 passed | Pass (`cp310`) | `b5cd2e60f9424b333d4697317620215f2b10b5d6a378bf90a36f46a71faf0d13` |
| 3.11.16 | inference-compatibility: 2,141 passed, 13 skipped, 3 xfailed, 49 subtests; SMARTS: 282 passed | Pass (`cp311`) | `f5bdc7fc7d6f52e5c6a2deecfe9a2662cb9a11e3664bf0959cc238be8f65ddb7` |
| 3.12.13 | inference-compatibility: 2,141 passed, 13 skipped, 3 xfailed, 49 subtests; SMARTS: 282 passed | Pass (`cp312`) | `423619b917ab2fa8f1ff884889e64c3ea9f0dd60393d71c4139f2d6a0638ffa5` |
| 3.13.15 | inference-compatibility: 2,141 passed, 13 skipped, 3 xfailed, 49 subtests; SMARTS: 282 passed | Pass (`cp313`) | `f69fef7b2c00959809f603f0c97827569f7af7c24be322f671fe2faf45289ecd` |
| 3.14.7 | final native + SMARTS: 468 passed; see rendering note below | Pass (`cp314`) | `fac04b871571819bc193498ffa982ac8cc7e2a5a8027fab6d5bb6f3da06d1d60` |

The first 3.9 attempt used the pre-fix private predicate records and exposed
unsupported `dataclass(slots=True)` use; `92859b0` removed that option. The
first 3.14 broad run had 2,134 passes and only seven failures: Python 3.14
renders existing `typing.Union` and forward annotations differently in
`inspect.signature()`. Commits `c7144cf` and `e8f72eb` canonicalized the
contract test structurally. The corrected contract produced 49/49 passes on
both 3.11 and 3.14, and the final 3.14 native/SMARTS suite produced 468/468
passes. Neither compatibility correction changes production search code.

The matrix proves Linux x86-64 build, isolated installation and native-search
behavior for all six CPython ABIs. It does not claim auditwheel/manylinux
repair or cross-distribution binary compatibility.

## 12. Deviations and residual risks

1. The reviewed private source layout was consolidated into
   `production.cpp/.hpp`, as explained in Section 5; behavior and boundaries
   are unchanged.
2. Full ASAN iterator-exhaustion pytest and a separate leak sanitizer are not
   available on the local toolchain. This limitation is explicit in Section 9
   and is not represented as a pass.
3. Mixed-callable `has_match` is 11.8% slower than baseline. Mixed-callable
   complete search is faster, semantics pass, and callable workloads were
   explicitly excluded from hiding native-only geometric performance. Future
   optimization may reduce GIL/host-predicate fixed cost without changing the
   current boundary.
4. Linux x86-64 is the validated native platform in this implementation
   cycle. Deterministic order is covered across threads/process hash seeds and
   the supported local Python matrix, but other operating systems require
   their normal CI qualification.

## 13. Acceptance summary

| Gate | Result |
| --- | --- |
| One generic RI executor for all public search methods | Pass |
| Public signatures, aliases and result protocols | Pass |
| Complete mapping-set and SMARTS semantics parity | Pass |
| Deterministic sequence and bounded behavior | Pass |
| Predicate-only Python callable execution | Pass |
| Full/ligand, ring, coordination, recursion and mutation | Pass |
| Downstream search/SMARTS/MCA/molecule/dataset consumers | Pass; full dataset gate: 437 passed, one optional local-evidence skip |
| Maintained coverage entry point | Pass: 2,437 passed, 12 skipped, 49 subtests; 40% repository-wide branch coverage |
| MCA site/type/order/`is_Nuc_site` and prediction parity | Pass locally and on exact final source at ai-14 (255 CPU route, 256 GPU route and 4,096 cross-capacity) |
| CPython 3.9--3.14 native build/test matrix | Pass; Linux x86-64 boundary noted |
| Native memory/lifetime diagnostics | Pass with explicit ASAN exhaustion/leak-sanitizer limitation |
| Generic and MCA performance gates | Pass |
| No third-party or legacy production substructure matcher | Pass |

Old behavior remains available only through Git history. The delivered source
contains one implementation and one interface, consistent with the approved
Hotpot design philosophy.
