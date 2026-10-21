# Native Hotpot search execution implementation plan

Status: proposed; implementation is not authorized.

Planning branch: `perf/native-search`

Planning baseline: `2cf0893`

## 1. Objective and owner decisions

Replace the NetworkX `GraphMatcher` execution path behind
`hotpot.cheminfo.search` with one native subgraph-monomorphism executor while
preserving Hotpot's public search model and substantive graph-search semantics.
This is a generic search-layer change. It is not an MCA-only matcher and does
not introduce a second chemical-search dialect.

The owner has made these decisions:

1. NetworkX's historical mapping enumeration order is not a permanent public
   contract. Complete search results must remain semantically identical, and
   the new executor must produce one deterministic mapping sequence for
   repeated identical work.
2. Arbitrary Python callables remain supported. A callable is a Python
   predicate leaf inside the native traversal; its presence must never route
   the complete search through NetworkX or another Python graph matcher.
3. Planning precedes implementation. No native search code is authorized by
   this document until the plan receives a separate approval.

The implementation must retain one public interface and one topology-search
engine. It must not expose backend selection, named profiles, compatibility
flags, or an MCA-specific execution path.

## 2. Confirmed current-state audit

### 2.1 Public and observable surfaces

The current public search model includes:

- `Query`, including set-valued attribute constraints and arbitrary callable
  constraints;
- `QueryAtom` and `QueryBond`, including programmatic construction;
- `Substructure.from_smarts()`, `from_mol()`, `from_smiles()`, mutation through
  `add_atom()`/`add_bond()`, and the public `construct_graph()` method;
- `Searcher.has_match()`, `iter_mappings()`, and `search()`;
- complete query-index-to-target-index mappings;
- raw query automorphisms;
- `max_matches` and the one-extra-result `truncated` contract;
- `Hits` grouping by target atom set without discarding raw mappings;
- `Hit` mapped atoms, mapped bonds, and induced bonds; and
- `SmartsSemantics.FULL_GRAPH` and
  `SmartsSemantics.LIGAND_SKELETON`.

`Hits` is itself exported, and repository tests construct it with a
matcher-like object. That constructor/materialization protocol remains
available. Native `Searcher` may add a private constructor from mappings, but
must not remove the existing public behavior.

### 2.2 Current execution path

`Searcher` currently constructs a NetworkX `GraphMatcher` and calls
`subgraph_monomorphisms_iter()`. Recursive SMARTS creates additional
`GraphMatcher` instances. The matcher invokes Python node and edge predicates,
and `Hits` materializes, groups, and sorts mappings in Python.

MCA performs a second NetworkX workload after SMARTS matching: it builds a
labeled graph, refines node colors, and runs rooted exact isomorphism checks to
remove automorphic site candidates. That MCA-specific representative policy is
not the definition of generic `Searcher` and must not be embedded into the
generic matcher.

The current target graph remains the complete `Molecule.atom_bond_graph` even
under `LIGAND_SKELETON`. That named semantics changes topology-derived
predicates such as degree, connectivity, valence, and ring membership; it does
not delete metal-ligand edges from the graph traversed by `GraphMatcher`.
Native execution must preserve this distinction.

### 2.3 Current consumers

The audit scope includes at least:

- `Molecule.search_substructure()` and direct `Searcher` callers;
- SMARTS conformance, mapping, enumeration, recursion, coordination, ring, and
  mutation behavior;
- molecule assembly and fragment workflows;
- dataset `SmartsSearcher`;
- MCA ordered nucleophilic-site rules and symmetry filtering; and
- programmatic queries using custom atom/bond attributes or Python callables.

MCA is therefore a high-value performance and scientific-parity workload, but
cannot define the complete search contract.

### 2.4 Existing evidence and required rebaseline

The repository already contains a strict SMARTS conformance matrix,
coordination fixtures, regression cases, deterministic fuzzing, mapping tests,
and performance benchmarks. The existing report records 400/400 deterministic
fuzz cases and identifies raw automorphism enumeration as a retained worst
case: seven disconnected wildcards produce 5,040 mappings with a recorded
median of 886.56 ms and 5.42 MB peak traced Python memory on that host.

The existing MCA reference workload records 121.624 seconds for single-worker
site detection and 17.052 seconds with nine site workers for 256 molecules.
These measurements motivate the work but are not cutover baselines. Stage 0
must rerun the current branch with fixed inputs, environment metadata, and
machine identity before implementation changes execution.

## 3. Normative semantic contract

### 3.1 Graph matching

The native executor must implement an undirected, injective subgraph
monomorphism:

- every query vertex maps to one distinct target vertex;
- every query edge maps to a compatible target edge;
- additional target edges among mapped vertices are allowed;
- disconnected query components remain supported;
- the target topology is the full Hotpot molecular graph; and
- atom and bond acceptance is decided by the query predicates, not by a
  replacement toolkit's perception model.

For built-in/typed predicates, and for pure deterministic opaque predicates,
the complete unbounded set of query-index-to-target-index mappings must equal
the current Hotpot result for every supported query and stable target. Query
automorphisms remain distinct raw mappings. A stateful or random callable does
not define one stable mathematical match set and is outside this equivalence
claim, while remaining accepted as described in Section 5.

### 3.2 Query predicates

The following behavior remains normative:

- an empty constraint accepts a compatible target object;
- a set-valued constraint performs current attribute-membership semantics;
- a missing target attribute rejects the candidate;
- `QueryAtom.map_number` remains metadata and never becomes a matching
  condition;
- SMARTS Boolean expression order and short-circuit behavior are retained;
- recursive SMARTS is anchored at its current query atom and inherits the
  selected semantics profile;
- relevant-ring and coordination semantics remain Hotpot-defined;
- `_smarts_exclude_aromatic` and `_smarts_bond_kind` retain their current
  behavior; and
- unsupported or malformed SMARTS remains explicitly rejected rather than
  being delegated to RDKit or Open Babel.

The current generic callable behavior is exact and unusual: when a constraint
value is callable, its keyword/attribute name is not resolved and no
`hasattr()` check is performed. The callable receives the complete target
`Atom` or `Bond` object. Thus `QueryAtom(any_name=lambda atom: ...)` is valid
and must remain valid. Only non-callable set constraints interpret their key
as a target attribute name.

The standard SMARTS parser currently represents internal expressions as
Python-callable objects. Before native matching, those parser-owned expressions
must acquire an immutable typed Boolean IR. Detecting `Callable` alone is not a
valid implementation because it would incorrectly keep normal SMARTS in
Python.

### 3.3 Mapping and result objects

The native executor emits mappings as compact target-index tuples in query
index order. The Python facade reconstructs the existing read-only mapping
views and result objects.

The following remain unchanged:

- mapping direction is query index to target index;
- complete search retains every raw automorphism;
- `Hits` groups mappings by the mapped target atom set;
- mappings within a `Hit` are not discarded;
- `mapped_bonds()` follows query-edge mappings;
- `induced_bonds` describes every target edge inside the mapped atom set;
- result mapping views remain read-only; and
- `has_match()` terminates after the first accepted embedding.

Native-produced `Hits` must also retain the complete current object behavior:

- `sub`, `mol`, `get_hit`, `max_matches`, `truncated`, and `graph_matcher`
  remain present;
- callers may set `get_hit=False` after construction, changing iteration and
  indexing to expose target atom sets as today;
- length, truth value, membership, indexing, `_get_nodes_set()`, and lazy
  `.hits` materialization retain their current meaning; and
- the `graph_matcher` attribute becomes a private native mapping-source
  adapter with a repeatable `subgraph_monomorphisms_iter()` protocol rather
  than a NetworkX `GraphMatcher` instance.

The adapter preserves the behavior consumed by `Hits` and current repository
callers; it is not a second search engine. Depending on the concrete NetworkX
class through `Hits.graph_matcher` was never a Hotpot search abstraction and is
explicitly replaced by this adapter.

`max_matches` continues to count raw embeddings. `truncated` remains true only
when an additional raw embedding exists beyond the requested bound. Because
legacy enumeration order is released, a bounded search may return a different
valid N-prefix than NetworkX. Its count, validity, determinism, and truncation
meaning must remain unchanged.

### 3.4 Mutation, reuse, and errors

Reusing a `Searcher` after query or target mutation must observe the new state
on the next call. A compiled-query cache is allowed only behind a complete
fingerprint covering topology, constraint values, typed expression IR,
semantics profile, and opaque predicate identity.

Each search method snapshots native query and target values at method entry.
Mutating a query, target, or callable-observed object while its search is in
progress is unsupported; the next method call must observe the mutation. This
rule is required to make native iteration and cross-process determinism
well-defined and is called out for owner approval in Section 15.

Initially, each invocation repacks the target graph. Target caching is not
permitted until Hotpot exposes a mutation generation that proves the cached
view is current.

Current public error categories remain stable: malformed/unsupported SMARTS,
invalid `max_matches`, incompatible objects, and missing query payloads must
not change. When an opaque callable is actually evaluated and raises, the
original exception and traceback propagate rather than becoming a non-match.
The new traversal order does not promise that a stateful callable is evaluated
at the same point, or at all in a bounded/existence search, as under NetworkX.

## 4. Deterministic native enumeration

The new sequence is implementation-defined but deterministic. It is not
required to reproduce NetworkX ordering and is not promised to remain the same
after a separately reviewed future algorithm change.

The initial native executor must use an explicit total order:

1. query components are ordered by their smallest original query index;
2. each component chooses its initial query vertex from a static selectivity
   tuple ending with the original query index as the final tie-break;
3. subsequent frontier selection uses a documented total tuple containing
   mapped-neighbor count, query degree, predicate selectivity, and query index;
4. target candidates are visited in ascending stable target atom index;
5. target adjacency is stored and visited in ascending stable target index;
6. complete mappings are emitted as target-index tuples ordered by original
   query index; and
7. no unordered-container iteration or parallel enumeration may affect the
   emitted sequence.

The exact selectivity tuple and field weights must be frozen in a versioned
native constant and documented before Stage 2 is merged. They are performance
policy, not chemical semantics.

For built-in predicates, identical query/target content and the same native
implementation version must produce the same mapping byte sequence across
repeated calls, processes, worker counts, and supported platforms. This
guarantee assumes the input graph indices and predicate values are identical.

For an opaque callable, determinism additionally requires that the callable is
pure and deterministic for the lifetime of one search. Callable invocation
count, invocation order, and side effects are not compatibility contracts once
legacy NetworkX enumeration order has been released. Every callable result
actually returned to the matcher remains authoritative, and every exception
actually raised by an invoked callable propagates unchanged.

## 5. Python-callable hybrid contract

The matcher retains one native topology traversal for every query, including a
query containing only opaque callables.

At a standard typed predicate leaf, C++ evaluates the predicate without the
GIL. At an opaque leaf, the executor:

1. retains a safe owning reference to the original Python predicate and exact
   target `Atom` or `Bond` object;
2. reacquires the GIL;
3. invokes only that predicate leaf;
4. converts its return value according to current truth-value semantics;
5. propagates any Python exception with its traceback; and
6. releases the GIL before continuing native candidate search/backtracking.

The opaque slot stores the original constraint value, not an attribute lookup.
Its keyword name is ignored exactly as it is by current `Query.match()`, and
the complete target `Atom`/`Bond` object is passed to the callable.

Opaque user callables are evaluated lazily. The first implementation must not
eagerly build a truth table and must not memoize an opaque callable result,
because either change can alter invocation count, exception timing, or
state-sensitive behavior. Native recursive SMARTS and typed internal
expressions may use per-search memoization because they are part of the fixed
Hotpot expression contract.

Programmatic constraints on unknown/custom Python attributes use the same
host-evaluated predicate-leaf mechanism when their values cannot be represented
in the native scalar domain. This is not a topology fallback: graph traversal,
injectivity, frontier selection, pruning, and mapping enumeration remain in
C++.

No execution path may instantiate a NetworkX `GraphMatcher` because a callable
or custom attribute is present.

## 6. Target architecture and data boundaries

### 6.1 One public layer, one executor

```text
existing public Query/Substructure/Searcher API
                    |
                    +--> Python SMARTS parser
                    |       -> immutable typed predicate/query IR
                    |
                    +--> target graph packer
                            -> indexed typed arrays
                            -> Python object handles only for opaque leaves
                    |
                    v
          _search_native native executor
                    |
                    +--> deterministic subgraph monomorphism
                    +--> native typed predicates
                    +--> recursive-program memoization
                    +--> leaf-only Python callback trampoline
                    |
                    v
          compact query-index-ordered mappings
                    |
                    v
              existing Hit/Hits facade
```

### 6.2 Query program

`compile_query(Substructure)` produces an immutable private `QueryProgram`
containing:

- query vertex and edge topology;
- stable original query indices;
- typed atom and bond predicate bytecode;
- ordered Boolean operations with explicit short-circuit structure;
- recursively compiled query programs;
- semantics-profile identifiers;
- map-number metadata; and
- slots holding opaque Python predicates where required.

The IR is a private implementation format with an explicit schema version. It
is not a new public query language.

### 6.3 Target graph

`pack_target(Molecule)` produces an indexed private `TargetGraph` containing
at least:

- atomic number, formal charge, aromatic flag, implicit hydrogen count,
  isotope, and stable atom index;
- bond endpoints, order, `BondKind`, aromatic flag, direction where consumed,
  and stable bond index;
- sorted adjacency offsets and neighbor indices;
- full-graph and ligand-skeleton derived values required by typed predicates;
- authoritative Hotpot relevant-ring membership; and
- Python `Atom`/`Bond` handles only when an opaque leaf may need them.

Packing copies values once per search invocation. C++ must not depend on
`Molecule`, `Atom`, or `Bond` Python object layout for native-only predicates.

### 6.4 Recursive queries

Recursive SMARTS compiles to nested native `QueryProgram` objects. Anchored
existence is memoized by `(recursive program, target atom index)` within one
search invocation. It must not invoke NetworkX or be represented as an opaque
Python callable.

### 6.5 Generic multi-query reuse

After the single-query executor is correct, add a private generic prepared
target/multi-query operation. It packs one target graph once and evaluates an
ordered sequence of compiled queries. MCA may use that operation for its 24
rules, but the operation belongs to `hotpot.search` and contains no MCA site
names, precedence, metal eligibility, or output assembly.

## 7. Public API, source, and packaging impact

Public signatures and result classes remain unchanged. No `backend`,
`device`, `profile`, or `native=True` option is added.

Proposed private source layout:

```text
hotpot/cheminfo/search/
    search.py                 # existing public objects/facade
    smarts.py                 # existing parser, emits typed expression IR
    query_ir.py               # private immutable query program schema
    native.py                 # extension loading, packing, result adaptation
    _search_native.pyi        # private type surface
    _native/
        bindings.cpp
        graph_data.hpp/.cpp
        query_program.hpp/.cpp
        matcher.hpp/.cpp
        predicates.hpp/.cpp
        python_predicate.hpp/.cpp
```

`Substructure.construct_graph()` remains available for public compatibility,
but `Searcher` stops using it for execution. `Hits` retains its current
matcher-consuming constructor and mutable `get_hit` behavior. Native
`Searcher` supplies a private native mapping-source adapter as
`Hits.graph_matcher`; this adapter can create the deterministic native mapping
iterator repeatedly. Neither retained surface is a production search
fallback.

NetworkX may remain a Hotpot dependency for unrelated/public graph surfaces.
The cutover claim is specifically that production `hotpot.search` execution
does not construct or invoke a NetworkX `GraphMatcher`.

The extension uses C++17 and pybind11 and must build in the supported CPython
3.9 through 3.14 matrix. Import failure is explicit; the package must not
silently select the old matcher.

## 8. Implementation stages and atomic commit sequence

Implementation begins only after plan approval. Each stage is one reviewable
commit or a short atomic commit group with its stated tests green.

### Stage 0 - Freeze contracts, oracle fixtures, and baselines

Tasks:

1. inventory every public query/search/result behavior and repository
   consumer;
2. run and record the complete current search, SMARTS conformance, and consumer
   suites;
3. generate immutable NetworkX-era oracle fixtures containing target/query
   inputs, complete raw mapping sets, grouped hits, mapped bonds, bounded
   results, truncation, and expected errors;
4. add an independent brute-force oracle for small labeled graphs;
5. record current generic-search and MCA P1 performance, RSS, and result
   digests; and
6. record Python, NetworkX, Open Babel, RDKit, compiler, CPU, and operating
   system metadata.

Exit criteria:

- the semantic inventory covers every item in Section 3;
- mapping-set oracle fixtures use built-in/typed or pure deterministic
  predicates and are deterministic and reviewable;
- stateful callable fixtures verify only the explicitly retained invocation
  result and exception rules, not NetworkX call count/order;
- legacy mapping order is recorded only for diagnosis, not adopted as the new
  permanent contract; and
- no production execution code has changed.

### Stage 1 - Introduce typed internal query IR

Tasks:

1. replace parser-owned lambda expressions with immutable typed Boolean nodes;
2. retain their Python `__call__` behavior for existing direct predicate use;
3. encode atom/bond fields, topology predicates, Boolean short-circuiting,
   semantics profiles, and recursive programs;
4. distinguish typed internal expressions from opaque user callables; and
5. fingerprint every mutable query input needed for safe compilation reuse.

Exit criteria:

- the current NetworkX executor still produces all Stage 0 oracle results;
- standard SMARTS contains no opaque predicate solely because of parser
  implementation details; and
- direct programmatic callable behavior remains green.

### Stage 2 - Implement deterministic native graph kernel

Tasks:

1. add typed target/query topology bindings;
2. implement injective subgraph monomorphism, disconnected queries, and raw
   automorphism enumeration;
3. freeze deterministic selection/candidate-order constants;
4. implement `has_match`, unbounded iteration, `max_matches`, and one-extra
   truncation detection; and
5. compare direct extension output with the independent brute-force oracle.

Exit criteria:

- exhaustive small labeled graphs have identical complete embedding sets;
- repeated and cross-process mapping sequences are byte-identical;
- bounded searches are valid, deterministic, and report truncation correctly;
  and
- production `Searcher` is not yet switched.

### Stage 3 - Implement predicates and Python leaf callbacks

Tasks:

1. implement every supported native atom/bond/topology predicate;
2. add safe Python-object ownership and a GIL callback trampoline;
3. preserve callable keyword-name independence, lazy evaluation, short-circuit
   behavior, and actual exception propagation;
4. support callable-only, native-only, and mixed queries in the same native
   DFS; and
5. add lifetime, reference-count, reentrancy, and callback-exception tests.

Exit criteria:

- all programmatic `QueryAtom`/`QueryBond` tests pass against the extension;
- instrumented callable tests prove that no NetworkX matcher is constructed;
- only opaque predicate leaves enter Python; and
- sanitizer/refcount checks show no leak or use-after-free.

### Stage 4 - Implement complete Hotpot SMARTS semantics

Tasks:

1. compile and execute recursive SMARTS natively;
2. implement full-graph and ligand-skeleton derived predicates without
   changing traversal topology;
3. consume authoritative Hotpot relevant-ring and coordination semantics;
4. preserve query mutation and recursive-cache invalidation behavior; and
5. pass every strict, corpus, regression, metamorphic, coordination, and fuzz
   test.

Exit criteria:

- complete unbounded mapping sets match every applicable stable-predicate
  Stage 0 oracle;
- supported/unsupported syntax boundaries and error categories match;
- no recursive expression uses the opaque-callable branch; and
- all existing search conformance tests pass.

### Stage 5 - Atomic public cutover

Tasks:

1. route `Searcher.has_match()`, `iter_mappings()`, and `search()` through the
   native executor in one commit;
2. adapt native tuples to current read-only mappings and `Hit`/`Hits` objects;
3. remove the production NetworkX `GraphMatcher` execution path;
4. preserve the native mapping-source `graph_matcher` adapter, mutable
   `get_hit`, bounded state, lazy `.hits`, and collection protocols;
5. keep public `construct_graph()` and direct `Hits` construction behavior;
   and
6. run every direct and downstream consumer suite.

Exit criteria:

- public signatures and schemas are unchanged;
- all semantic, consumer, and determinism gates pass;
- source inspection and sabotage tests prove there is no runtime GraphMatcher
  fallback; and
- the worktree contains one current search executor.

### Stage 6 - Generic reuse and MCA integration

Tasks:

1. add private prepared-target and ordered multi-query execution;
2. make MCA reuse one packed target for its compiled rule sequence;
3. keep MCA precedence, site type, metal eligibility, symmetry representative,
   and `is_Nuc_site` logic outside the generic native matcher; and
4. verify exact MCA atom/site/joint digests on CPU and GPU paths.

Exit criteria:

- no MCA-specific rule or result type exists in `_search_native`;
- target packing occurs once per MCA molecule/rule group;
- site indices, types, order, MCA values, and `is_Nuc_site` are unchanged; and
- GPU resource scheduling and threshold routing remain unchanged.

### Stage 7 - Packaging, performance, documentation, and closure

Tasks:

1. build clean wheels for every supported Python version;
2. run native sanitizers and leak checks;
3. run frozen generic and MCA performance workloads;
4. update search/SMARTS documentation and the implementation log/report;
5. remove temporary executable oracle code; and
6. retain only immutable fixtures, independent brute-force tests, and Git
   history as reference evidence.

Exit criteria:

- the full validation matrix and performance gates pass;
- source and wheel contents contain no legacy production backend;
- clean isolated installs import and execute the extension; and
- final documentation reports deviations, measurements, commits, and residual
  risks.

## 9. Validation matrix

Correctness gates precede performance claims.

| Area | Required evidence |
| --- | --- |
| Graph kernel | Exhaustive small labeled graphs against an independent brute-force oracle |
| Complete parity | Exact unbounded raw embedding sets for built-in/typed and pure deterministic callable predicates |
| Enumeration | Repeated, cross-process, worker-count, and supported-platform sequence stability |
| Bounded search | Every limit boundary, raw-embedding counts, valid deterministic prefix, and one-extra truncation |
| Automorphisms | Symmetric chains, cycles, disconnected queries, duplicate target atom-set grouping |
| Results | Read-only mappings, mapped/induced bonds, mutable `get_hit`, lazy `.hits`, collection behavior, native `graph_matcher` adapter, legacy `Hits` construction |
| SMARTS | Strict feature matrix, full corpus, regressions, invalid syntax, supported/unsupported boundary |
| Semantics profiles | Full graph, ligand skeleton, metals, coordination bonds, relevant rings |
| Recursion | Nested recursion, anchoring, cache reuse, atom/bond mutation invalidation |
| Programmatic queries | Set constraints, empty sets, missing attributes, map metadata, mutation between calls, unsupported mutation during an active search |
| Callables | Keyword-name independence, complete Atom/Bond argument, callable-only and mixed queries, actual exception propagation, object lifetime |
| Concurrency | Shared read-only Searcher, threads, spawn workers, repeated calls |
| Consumers | Molecule search, mol assembly/fragments, dataset SMARTS, MCA CPU/GPU |
| MCA scientific parity | Exact sites, site types/order, atom/site MCA, joint digest, `is_Nuc_site` |
| Packaging | CPython 3.9-3.14 clean build, wheel content, isolated import and smoke test |

The core repository commands will include at least:

```text
python -m pytest -q tests/smarts_conformance
python -m pytest -q tests/test_cheminfo/test_search.py
python -m pytest -q tests/test_cheminfo/test_search_mapping.py
python -m pytest -q tests/test_cheminfo/test_smarts.py
python -m pytest -q tests/mca
python -m pytest -q tests/test_ds
```

Additional focused consumer commands must be recorded during Stage 0 rather
than inferred after implementation.

## 10. Performance and resource gates

Stage 0 freezes the exact benchmark inputs and current measurements. Cutover
requires all of the following on the same host and affinity:

1. native-only built-in query execution achieves at least a 2.0x geometric
   mean speedup across existence, bounded, complete, recursive, and repeated
   searches;
2. no representative native-only benchmark is slower than baseline by more
   than 5% outside the recorded noise interval;
3. the 5,040-mapping disconnected-query stress case is at least 3.0x faster or
   uses at most one third of the baseline peak traced allocation, while
   returning the identical complete embedding set;
4. one-worker MCA site detection is at least 1.5x faster with exact site
   parity;
5. mixed-callable searches remain bounded and correct; they are reported
   separately and are not allowed to hide a whole-search Python fallback;
6. target packing, native execution, callback time, and result materialization
   are reported separately and together end to end;
7. repeated searches do not grow native or Python memory; and
8. the extension does not add persistent per-target caches or unbounded result
   buffers.

If correctness passes but a performance gate fails, the native executor does
not cut over. Changing a threshold requires an explicit plan amendment and
owner review; it must not be relaxed inside the final report.

## 11. Cutover, history, and rollback

The final source tree contains one production matcher. It does not contain a
NetworkX backend selector, automatic fallback, environment toggle, compatibility
module, or retained MCA-specific matcher.

The historical implementation may be executed from its Git commit when an
investigation needs it. During development, Stage 0 may use the baseline commit
to generate oracle evidence. Final tests retain data fixtures and an
independent small-graph oracle, not a copied executable legacy engine.

Failure to import the required extension is explicit. It is not converted into
a slower Python search.

Python predicate-leaf execution is part of the current matcher contract, not a
fallback backend: only the opaque leaf reacquires the GIL and the native graph
traversal remains active.

## 12. Risks and mitigations

| Risk | Required mitigation |
| --- | --- |
| Standard SMARTS accidentally remains Python | Typed parser-owned IR and a gate asserting zero opaque leaves for standard corpus queries |
| Native and current complete match sets diverge | Frozen oracle fixtures plus independent exhaustive small-graph testing for stable predicates |
| New sequence varies by platform | Explicit total order, sorted adjacency/candidates, no unordered iteration, cross-platform digests |
| `max_matches` hides a semantic mismatch | Validate every returned mapping and compare unbounded sets separately; test all limit boundaries |
| Callable causes complete fallback | Sabotage NetworkX construction and instrument native traversal/callback counts |
| Eager callback evaluation changes errors | Lazy opaque leaves, no initial opaque-result memoization, and unchanged propagation for exceptions actually raised |
| Recursive SMARTS re-enters Python | Recursive native programs and anchored per-search memoization |
| Ligand semantics deletes topology edges | Pack full topology; apply ligand view only in the relevant predicate leaves |
| Ring behavior is redefined | Consume Hotpot relevant-ring facts or port that exact primitive with its own parity proof |
| Query/target mutation sees stale state | Complete query fingerprint; repack target on every invocation initially |
| Native object outlives Python object | Owning references, GIL-safe destruction, refcount and sanitizer tests |
| Extension ABI/build failure | Existing pybind11 build conventions and the full CPython clean-wheel matrix |
| Native work only benefits MCA | Generic direct benchmarks and every non-MCA consumer are cutover gates |

## 13. Acceptance criteria

The implementation is complete only when all of the following are true:

1. one generic native executor backs every `Searcher` method;
2. public search signatures, result schemas, and `Hits` collection/state
   behavior are unchanged, with `graph_matcher` represented by the approved
   native adapter;
3. complete unbounded embedding sets for built-in/typed and pure deterministic
   callable predicates match the frozen Hotpot semantics;
4. the native sequence is deterministic under Section 4;
5. bounded searches preserve validity, counts, and truncation semantics;
6. every supported standard SMARTS predicate executes through typed native IR;
7. opaque Python callables execute only at predicate leaves inside native DFS;
8. no callable, recursion, or custom attribute causes whole-search fallback;
9. full-graph, ligand-skeleton, coordination, ring, mutation, and cache behavior
   passes the complete conformance matrix;
10. all programmatic-query and downstream-consumer tests pass;
11. MCA exact site/type/order/`is_Nuc_site` and prediction digests are unchanged;
12. the supported Python wheel matrix passes clean builds and isolated imports;
13. native memory, object lifetime, threading, and exception behavior pass their
    diagnostics;
14. every Section 10 performance gate passes; and
15. production code contains no legacy matcher or runtime backend selection.

## 14. Explicit non-goals

This stage does not:

- create an MCA-specific search engine;
- replace Hotpot semantics with RDKit or Open Babel SMARTS;
- make old NetworkX enumeration order permanent;
- promise deterministic output from a stateful or random user callable;
- promise callable invocation order/count or callable side effects;
- nativeize all of `Molecule`, chemistry perception, or every NetworkX use in
  Hotpot;
- change MCA site rules, precedence, symmetry representative policy, or model;
- add a public batch/multi-query API without separate approval; or
- retain a serial/parallel, Python/native, or old/new compatibility profile.

## 15. Approval checklist

Owner-confirmed decisions already incorporated:

- [x] complete substantive search semantics remain unchanged;
- [x] legacy NetworkX mapping order is not a permanent contract;
- [x] the native implementation has deterministic repeatable enumeration;
- [x] Python callables remain supported as predicate-only Python execution;
- [x] callable use never degrades the full matcher to Python; and
- [x] the project targets generic `hotpot.search`, not only MCA.

Derived consequences requiring review with this plan:

- [ ] bounded `max_matches=N` may return a different valid N-prefix than the
  NetworkX implementation because the approved order changes;
- [ ] callable result/exception semantics remain supported, but invocation
  order, invocation count, and side effects are not contracts;
- [ ] each search uses a method-entry query/target snapshot; mutation during an
  active search is unsupported and mutation between calls remains visible;
- [ ] `Hits.graph_matcher` remains present as a native mapping-source adapter,
  not as a NetworkX `GraphMatcher` instance;
- [ ] deterministic built-in-query sequences are required across supported
  platforms for one native implementation version;
- [ ] the native extension is mandatory at runtime and has no NetworkX
  fallback; and
- [ ] MCA symmetry representative selection remains MCA-owned rather than
  becoming generic `Searcher` behavior.

Approval of this document authorizes planning closure only. Native search code
must begin only after an explicit implementation instruction.
