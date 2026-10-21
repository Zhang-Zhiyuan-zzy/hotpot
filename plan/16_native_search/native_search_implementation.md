# Native Hotpot search execution implementation plan

Status: approved for implementation on 2026-10-21.

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
3. Planning precedes implementation. The owner approved this plan and selected
   the single RI-style implementation on 2026-10-21.

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
- non-callable constructor values become mutable sets, with an empty set acting
  as a wildcard;
- a missing target attribute rejects the candidate;
- constraints are ANDed and evaluated in `Query.kwargs` insertion order with
  current short-circuit behavior whenever their leaf is reached;
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

Each native cursor owns an immutable sorted `query_keys` vector and emits
compact target-index tuples aligned with that vector. The Python facade always
reconstructs a mapping with `dict(zip(query_keys, target_values))`, never by
enumerating tuple positions. This preserves sparse normalized keys such as
`{0, 2}` before producing the existing read-only query-key-to-target-index
mapping views and result objects.

The following remain unchanged:

- public mapping direction is normalized query-node key to target index;
- complete search retains every raw automorphism;
- no symmetry breaking or target-atom-set uniquification occurs in the native
  enumerator, and every complete injective query-index mapping is emitted once;
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
  adapter rather than a NetworkX `GraphMatcher` instance.

The public `Hits(...)` constructor accepts a possibly one-shot provider and
calls `subgraph_monomorphisms_iter()` exactly once while eagerly materializing
mapping groups. That provider yields the legacy NetworkX direction,
target/molecule index to query index; `Hits` reverses it into the public
query-to-target direction. Repeated iteration applies to the materialized
`Hits`, not to the provider. A Searcher-created native adapter may be
repeatable, but repeatability is not a constructor requirement. The adapter
preserves the behavior consumed by `Hits`; it is not a second search engine.
Depending on the concrete NetworkX class through `Hits.graph_matcher` was never
a Hotpot search abstraction and is explicitly replaced by this adapter.

`max_matches` continues to count raw embeddings. `truncated` remains true only
when an additional raw embedding exists beyond the requested bound. Because
legacy enumeration order is released, a bounded search may return a different
valid N-prefix than NetworkX. Its count, validity, determinism, and truncation
meaning must remain unchanged.

The exact bounded cases remain:

- `max_matches=0` exposes no mapping, probes the first embedding, and sets
  `truncated=True` only when that embedding exists;
- fewer than or exactly `N` embeddings require exhaustion and leave
  `truncated=False`;
- more than `N` embeddings consume result `N + 1` internally, expose only the
  first `N`, and set `truncated=True`; and
- the probe embedding is never exposed through iteration or `Hits`.

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

1. normalize the programmatic query to the current simple-graph semantics in
   Section 6.3;
2. discover connected components by scanning normalized query-node keys and
   sorted adjacency in ascending integer order, then order components by their
   smallest query-node key;
3. choose each nonempty component root by full-topology query degree
   descending, then query-node key ascending;
4. process one complete component before the next and repeatedly select only
   an unchosen vertex adjacent to the already ordered prefix, using the exact
   key: number of already ordered query neighbors descending, full-topology
   query degree descending, then query-node key ascending;
5. assign each non-root vertex's search parent to its already ordered neighbor
   with the greatest query-order position;
6. visit target candidates, candidate-domain bitsets, and target adjacency in
   ascending stable target atom index;
7. check edges to earlier query neighbors, including multiple required checks,
   in ascending normalized query-neighbor and edge-key order;
8. emit complete mappings aligned with ascending normalized query-node keys;
   and
9. never allow unordered-container iteration, parallel enumeration, or pointer
   values to affect the emitted sequence or predicate-check order.

The query order is prepared once from the normalized query before target DFS;
"already ordered" refers to that compilation prefix, not runtime mapping
success. Full-topology degree here is graph degree in the topology traversed by
the matcher, with a self-loop counted twice. It is not SMARTS `D`,
ligand-skeleton degree, valence, ring membership, or a predicate selectivity
score. Degree pruning uses only the safe lower bound `query_degree <=
target_degree`, because additional target edges are allowed.

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

### 6.2 Selected implementation: in-tree C++17 RI-style kernel

The proposed production matcher is a Hotpot-owned C++17 kernel in
`_search_native`. It uses the C++ standard library for storage and pybind11
only for the extension boundary and opaque-predicate trampoline. It does not
delegate topology enumeration to Boost.Graph, igraph, RDKit, Open Babel,
NetworkX, or another graph-matching library.

The kernel is a deterministic, component-aware, RI-style domain-filtered
backtracking monomorphism search. "RI-style" describes the ordered-query and
parent-neighbor candidate scheme; this is not a claim that Hotpot embeds or
forks the RI reference implementation. Its initial algorithm is fixed as
follows:

1. pack the target as compact sorted adjacency offsets, neighbor indices, and
   edge indices;
2. build conservative native unary candidate bitsets from safe full-topology
   degree lower bounds and typed atom predicates; an opaque callable, custom
   Python-attribute leaf, or not-yet-resolved Boolean branch must remain in the
   domain rather than being evaluated speculatively;
3. compile the total query order specified in Section 4 and assign every
   non-component-root query vertex an earlier ordered query neighbor as its
   search parent;
4. let a component root scan its candidate domain in ascending target index,
   and let every other query vertex scan the sorted neighbors of its mapped
   parent intersected with its domain and the unused-target set;
5. for each candidate, check in order: unused target; the complete ordered atom
   expression with its current short-circuit semantics; every normalized query
   self-loop and edge to an earlier mapped query vertex in the Section 4 order,
   including typed and opaque bond predicates; then the forward checks;
6. apply only these initial forward checks: (a) for the newly bound `(q, t)`,
   the number of distinct unmapped neighbors of `q` must not exceed the number
   of unused neighbors of `t` in the full traversal topology; and (b) every
   remaining query vertex must have a nonempty intersection of its conservative
   native domain, unused target vertices, and the target adjacency required by
   each already mapped query neighbor, including only native bond filters that
   can safely reject a pair. A not-yet-started component root uses its global
   unused domain. Forward checking must not invoke an opaque callable or custom
   Python-attribute leaf;
7. recurse and stream each complete target-index tuple in ascending normalized
   query-node-key order; and
8. expose a lazy raw-embedding cursor that pauses after each yielded mapping
   and performs no independent public `max_matches` truncation.

Disconnected components remain inside the same DFS and share one injectivity
state. They are not searched independently and combined by a Python Cartesian
product. Standard typed predicates execute in C++; an opaque callable is
invoked lazily at the candidate/predicate leaf defined in Section 5. Native
recursive SMARTS invokes this same kernel with an anchored initial binding.
No symmetry breaking is applied: distinct query-index mappings, including
automorphisms later grouped into one `Hit`, are each emitted exactly once.
Hall-set, label-count, or stronger pruning may be added only after a separate
brute-force parity proof demonstrates that it cannot remove an embedding.

There is exactly one bounded-iteration owner. The existing Python facade
limiter retains `max_matches`, `yielded`, the `N + 1` probe, and `truncated` for
both `Searcher.iter_mappings()` and `Hits`. It pulls at most `N + 1` raw
embeddings from the native cursor and never exposes the probe. The native
cursor must not hide or pre-consume its own probe result. `has_match()` pulls
exactly one raw embedding and stops. This avoids nested limiters producing an
incorrect `truncated` value while still stopping native DFS as soon as the
facade stops requesting results.

This choice is deliberate rather than an unqualified preference for custom
code:

| Candidate | Decision and reason |
| --- | --- |
| Boost.Graph `vf2_subgraph_mono` | Strongest generic alternative and adaptable through its visitor/comparator facilities, but Boost is not a current build dependency and does not contractually provide Hotpot's cross-version order. Hotpot would still own disconnected ordering, typed/recursive state, callback/GIL behavior, and the exact streaming-prefix/adapter contract. |
| igraph or another evaluated packaged VF2/VF3 engine | Its callbacks and early-stop facilities can be adapted, but the evaluated API does not jointly guarantee Hotpot's deterministic order, typed predicate VM, anchored recursion, Python-leaf behavior, and mapping-source protocol; it also adds an ABI/runtime dependency. |
| RDKit or Open Babel | Rejected because their matchers couple traversal to their own molecular query, aromaticity, and SMARTS semantics and do not implement arbitrary Hotpot attribute/callable predicates. |
| Glasgow-style constraint solver | Not selected because integrating its solver model adds substantial scope while Hotpot would still own predicate callbacks, deterministic all-mapping order, and streaming result adaptation. |
| In-tree deterministic RI-style kernel | Selected because it keeps the small topology kernel, predicate VM, deterministic order, recursion, early stop, and callable boundary under one testable contract with no new graph-library dependency. |

Authoritative Hotpot ring, coordination, atom, and bond facts may be packed as
inputs. Reusing those facts is not delegating graph matching. A later proposal
to replace the in-tree kernel with a third-party matcher requires a reviewed
plan amendment and complete semantic, deterministic-order, packaging, and
performance requalification.

The custom-kernel correctness risk is controlled by the independent
brute-force injection oracle, exhaustive small-graph comparison, frozen
NetworkX-era fixtures, native sanitizers, and exact cross-process/platform
sequence digests required below. No pruning rule may enter production solely
because it improves a benchmark.

### 6.3 Query program

`compile_query(Substructure)` first reproduces the simple undirected query
projection currently built by `Substructure.construct_graph()`, without using
that NetworkX object as the production executor input:

- a node key is the current `QueryAtom.idx`, which is derived by
  `sub.query_atoms.index(atom)`; adding the same `QueryAtom` object more than
  once therefore collapses to its first key and may leave sparse integer keys;
- query bonds are inserted in list order into one undirected edge per endpoint
  pair, including self-loops; repeated bonds on the same unordered pair
  overwrite the matching predicate so the last `QueryBond` wins;
- the original `query_bonds` list remains unchanged for result construction,
  so `Hit.mapped_bonds()` continues to include one result entry per original
  list item even when matching used one collapsed edge;
- an empty normalized query makes `has_match()` true, makes
  `iter_mappings()` yield one read-only empty mapping, and makes `search()`
  yield one `Hit` whose `atom_indices` is an empty `frozenset` and whose
  mapping is empty; and
- a normalized query self-loop requires a compatible target self-loop, with
  the current predicate and error behavior.

Malformed direct list/end-point mutations retain their current failure rather
than being silently repaired. Stage 0 freezes these degenerate programmatic
cases explicitly. Changing or rejecting them later requires separate API
approval; nativeization must not reinterpret the lists as a multigraph.

The normalized form is compiled into an immutable private `QueryProgram`
containing:

- query vertex and edge topology;
- stable normalized query-node keys and sorted key-to-program-position maps;
- typed atom and bond predicate bytecode;
- ordered Boolean operations with explicit short-circuit structure;
- recursively compiled query programs;
- semantics-profile identifiers;
- map-number metadata; and
- slots holding opaque Python predicates where required.

The IR is a private implementation format with an explicit schema version. It
is not a new public query language.

### 6.4 Target graph

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
The packed topology must reproduce the current undirected simple
`Molecule.atom_bond_graph` node/edge meaning, including any representable
self-loop and stable Hotpot atom/bond indices, without importing RDKit or Open
Babel graph perception into matching.

### 6.5 Recursive queries

Recursive SMARTS compiles to nested native `QueryProgram` objects. Anchored
execution prebinds normalized nested-query node `0` to the candidate target
atom, validates node `0`'s complete atom predicate and self-loop, and marks the
target atom used. The component containing node `0` is ordered from the prefix
`{0}` using the remaining Section 4 frontier rule; every edge back to node `0`
is checked normally. Other disconnected components then scan their global
domains under the same injectivity state. Anchored existence is memoized by
`(recursive program identity, target atom index)` only within the same
immutable `TargetGraph` search snapshot. It must not invoke NetworkX or be
represented as an opaque Python callable.

### 6.6 Generic multi-query reuse

After the single-query executor is correct, add a private generic prepared
target/multi-query operation. It packs one target graph once and evaluates an
ordered sequence of compiled queries. MCA may use that operation for its 24
rules, but the operation belongs to `hotpot.cheminfo.search` and contains no
MCA site names, precedence, metal eligibility, or output assembly.

## 7. Frozen public API, source, and packaging impact

No `backend`, `device`, `profile`, `native=True`, or other search-mode option
is added. The freeze below covers parameter names and order,
positional-versus-keyword-only status, defaults, return protocols, exported
aliases, enum/error identities, and documented mutable object state. It is not
limited to retaining class names.

### 7.1 Public API tree and signatures

The canonical definitions live under `hotpot.cheminfo.search`. The same
classes, enum, and exception objects are re-exported through
`hotpot.cheminfo` and the top-level `hotpot` facade. Native implementation
work must not introduce replacement wrapper class identities at those import
paths.

The formal facade export set is exactly `Query`, `QueryAtom`, `QueryBond`,
`Substructure`, `Searcher`, `Hits`, `Hit`, `SmartsSemantics`,
`SmartsSyntaxError`, and `UnsupportedSmartsError`. Names that happen to leak
through a package lacking `__all__` are not promoted into this contract. This
is an intended export classification: `hotpot.cheminfo.search` does not
currently define its own `__all__`, while `search.py.__all__` contains the
seven query/search/result classes.

```text
hotpot.cheminfo.search                 [also re-exported by hotpot.cheminfo and hotpot]
├── Query(**kwargs: Union[Container, Any])
│   ├── match(obj)
│   ├── __repr__()
│   ├── label                          [abstract-decorated getter; base remains instantiable]
│   └── kwargs                         [public mutable mapping]
├── QueryAtom(sub=None, map_number=None, **attrs)
│   ├── from_atom(atom, include_attrs=None, exclude_attrs=None) [classmethod]
│   ├── map_number                     [read-only property]
│   ├── idx                            [read-only property]
│   ├── label                          [read-only property]
│   └── sub                            [public state]
├── QueryBond(atom1, atom2, **attrs)
│   ├── match(obj)
│   ├── a1idx / a2idx / sub / label   [read-only properties]
│   └── atom1 / atom2                  [public state]
├── Substructure()
│   ├── from_mol(mol, addition_atom_attr=None, addition_bond_attr=None) [classmethod]
│   ├── from_smarts(smarts, *, semantics=SmartsSemantics.FULL_GRAPH) [classmethod]
│   ├── from_smiles(smiles, addition_atom_attr=None, addition_bond_attr=None) [classmethod]
│   ├── add_atom(atom_query)
│   ├── add_bond(atom1, atom2, **bond_attrs)
│   ├── construct_graph()
│   ├── __repr__()
│   └── query_atoms / query_bonds / query_graph
├── Searcher(substructure)
│   ├── has_match(mol)
│   ├── iter_mappings(mol, max_matches=None)
│   ├── search(mol, max_matches=None)
│   └── substructure                   [public state]
├── Hits(sub, mol, graph_matcher, get_hit=True, max_matches=None)
│   ├── hits                           [getter-only; lazy cached mutable list]
│   ├── __iter__() / __len__() / __bool__()
│   ├── __getitem__(item) / __contains__(item)
│   └── sub / mol / graph_matcher / get_hit / max_matches / truncated
├── Hit(mol, sub, atom_indices, mappings=())
│   ├── mapped_bonds(mapping_index=0)
│   ├── mapped_atom_indices(query_idx)
│   ├── mapped_atoms(query_idx)
│   ├── mappings                       [getter-only tuple of read-only mappings]
│   ├── induced_bonds                  [getter-only tuple]
│   ├── atom_indices                   [frozenset]
│   └── mol / sub / atoms / bonds      [atoms and bonds are mutable lists]
├── SmartsSemantics(str, Enum)
│   ├── FULL_GRAPH = "full_graph"
│   ├── LIGAND_SKELETON = "ligand_skeleton"
│   ├── rings_for(atom) / bonds_for(atom)
│   └── degree(atom) / connectivity(atom) / valence(atom)
├── SmartsSyntaxError(ValueError)
└── UnsupportedSmartsError(NotImplementedError)

hotpot.cheminfo.search.smarts
├── substructure_from_smarts(smarts, *, semantics=SmartsSemantics.FULL_GRAPH)
├── parse_bracket_atom(expr_text, *, semantics=SmartsSemantics.FULL_GRAPH)
├── tokenize(smarts)
└── TokenType(str, Enum)              [ATOM="ATOM", BRACKET="BRACKET",
                                      BOND="BOND", BRANCH_L="BRANCH_L",
                                      BRANCH_R="BRANCH_R", RING="RING",
                                      DOT="DOT"]

hotpot.Molecule / hotpot.cheminfo.Molecule
└── search_substructure(smarts, *, semantics="full_graph")
```

The exact frozen callable signatures are:

```python
Query(**kwargs: Union[Container, Any])
Query.match(self, obj)
Query.__repr__(self)
Query.label.fget(self)

QueryAtom(sub: "Substructure" = None, map_number: int = None, **attrs)
QueryAtom.from_atom(atom: Atom, include_attrs: Sequence[str] = None,
                    exclude_attrs: Sequence[Literal["atomic_number", "is_aromatic"]] = None)
QueryAtom.map_number.fget(self)
QueryAtom.label.fget(self)
QueryAtom.idx.fget(self) -> int

QueryBond(atom1: QueryAtom, atom2: QueryAtom, **attrs)
QueryBond.match(self, obj)
QueryBond.label.fget(self)
QueryBond.a1idx.fget(self) -> int
QueryBond.a2idx.fget(self) -> int
QueryBond.sub.fget(self)

Substructure()
Substructure.from_mol(
    mol: Molecule,
    addition_atom_attr: dict[int, dict[str, set]] = None,
    addition_bond_attr: dict[Union[int, tuple[int, int]], dict[str, set]] = None,
) -> "Substructure"
Substructure.from_smarts(smarts: str, *,
                         semantics: SmartsSemantics = SmartsSemantics.FULL_GRAPH) -> "Substructure"
Substructure.from_smiles(
    smiles: str,
    addition_atom_attr: dict[int, dict[str, set]] = None,
    addition_bond_attr: dict[Union[int, tuple[int, int]], dict[str, set]] = None,
)
Substructure.add_atom(self, atom_query: Union[Atom, QueryAtom])
Substructure.add_bond(self, atom1: Union[int, QueryAtom],
                      atom2: Union[int, QueryAtom], **bond_attrs)
Substructure.construct_graph(self)
Substructure.__repr__(self)

Searcher(substructure: "Substructure")
Searcher.has_match(self, mol: Molecule) -> bool
Searcher.iter_mappings(self, mol: Molecule,
                       max_matches: int = None) -> Iterator[MappingProxyType]
Searcher.search(self, mol: Molecule, max_matches: int = None) -> "Hits"

Hits(sub, mol, graph_matcher, get_hit: bool = True,
     max_matches: int = None)
Hits.__iter__(self)
Hits.__getitem__(self, item: int) -> Union[frozenset[int], "Hit"]
Hits.__len__(self)
Hits.__bool__(self)
Hits.__contains__(self, item: Union[Iterable[int], "Hit"])
Hits.hits.fget(self) -> list["Hit"]
Hit(mol, sub, atom_indices, mappings=())
Hit.mapped_bonds(self, mapping_index: int = 0)
Hit.mapped_atom_indices(self, query_idx: int) -> tuple[int, ...]
Hit.mapped_atoms(self, query_idx: int) -> tuple[Atom, ...]
Hit.induced_bonds.fget(self)
Hit.mappings.fget(self)

SmartsSemantics.rings_for(self, atom)
SmartsSemantics.bonds_for(self, atom)
SmartsSemantics.degree(self, atom) -> int
SmartsSemantics.connectivity(self, atom) -> int
SmartsSemantics.valence(self, atom) -> float

substructure_from_smarts(smarts: str, *,
                         semantics: SmartsSemantics = SmartsSemantics.FULL_GRAPH) -> Substructure
parse_bracket_atom(expr_text: str, *,
                   semantics: SmartsSemantics = SmartsSemantics.FULL_GRAPH) -> Dict[str, object]
tokenize(smarts: str) -> List[Tuple[TokenType, str]]
Molecule.search_substructure(self, smarts: str, *, semantics="full_graph")
```

The existing difference between SMARTS defaults is intentional compatibility:
parser helpers and `Substructure.from_smarts()` expose the enum default, while
`Molecule.search_substructure()` exposes the string default. Both forms still
resolve to `FULL_GRAPH`; the native work must not normalize the public
signatures as incidental cleanup.

`QueryAtom.from_atom()` and the three `Substructure.from_*()` factories remain
`classmethod` descriptors and construct through `cls`; the method kind is part
of the freeze even though a bound `inspect.signature()` omits `cls`.
`Query.label` remains abstract-decorated, but `Query` does not use `ABCMeta`
and is currently instantiable; native dispatch must not introduce abstract-base
enforcement.

`parse_bracket_atom()` retains its usable `{"predicate": callable_expression}`
return shape and predicate behavior; the private concrete expression class is
not frozen. `tokenize()` retains its historical ordered list of
`(TokenType, text)` pairs and the exact `TokenType` values shown in the tree.
Other non-underscore names exposed incidentally by `smarts.py` lacking an
`__all__` -- including `BondOrder`, `AromaticFlag`, token constants, imported
query classes, typing names, and module imports -- are not frozen as public
native-search API. Adding a restrictive `smarts.__all__` is also out of scope
because it would itself change wildcard-import behavior.

`Searcher.iter_mappings()` continues to return an iterator of read-only
query-index-to-target-index mappings. Its concrete private class name is not
frozen, but it retains `__iter__()`, `__next__()`, and the observable public
state `max_matches`, `yielded`, and `truncated`. `truncated` becomes
authoritative after the bound/exhaustion check under the current protocol.

`Hits` retains its matcher-like constructor protocol: `graph_matcher` accepts
an object whose `subgraph_monomorphisms_iter()` may be one-shot and is invoked
exactly once per `Hits` construction. Its yielded dictionaries use
target-index-to-query-index direction. Searcher-created instances make the one
explicit concrete-type substitution in this plan: `graph_matcher` becomes a
native mapping-source adapter instead of a NetworkX `GraphMatcher`. The
attribute, one-shot-compatible constructor protocol, repeated iteration of the
materialized `Hits`, mutable `get_hit`, lazy cached mutable `hits` list,
grouping, and collection behavior remain frozen.

`Substructure.construct_graph()` continues to return the cached undirected
NetworkX `Graph` compatibility projection even though production `Searcher`
no longer consumes it. Its nodes use the current normalized integer keys and
hold the original `QueryAtom` under `"qa"`; each simple query edge holds the
last inserted `QueryBond` under `"qb"`. Repeated calls return the same graph
object whenever the existing cache-validity conditions hold; detected or
degenerate structural states retain their current rebuild behavior.

`Hit(..., mappings=())` retains its exact signature, but an empty `mappings`
argument is not promoted into a valid construction promise: current
construction calls `mapped_bonds()` and raises `IndexError`. Likewise,
`Substructure.from_mol(..., addition_bond_attr=caller_dict)` currently mutates
that dictionary when converting tuple endpoint keys to bond indices. This is
an observable but incidental factory quirk outside the native executor; this
stage leaves it unchanged, and any cleanup requires separate approval.

The following are implementation-private and are not added to the public
contract: `_MappingIterator` class identity; `Searcher._graph_matcher()`,
`_node_match()`, and `_edge_match()`; `resolve_smarts_semantics()`; parser
`_Predicate` classes; typed query IR classes; `_smarts.py`; and
`smarts_parser/`. Existing private compatibility such as
`Hits._get_nodes_set()` is regression-tested where currently consumed but is
not promoted into a new public API.

Stage 0 records an executable signature/export manifest using
`inspect.signature`, alias identity checks across all three facades, enum
value checks, exception-base checks, and result-protocol fixtures. It checks
presence and identity of the ten intended facade exports and the seven names
in `search.py.__all__`; it does not assert that the package defines an
`__all__` or that `dir()` contains only ten names. Any required signature
change discovered during implementation stops the stage and requires a plan
amendment rather than being hidden in the native cutover.

### 7.2 Proposed private source layout

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

NetworkX may remain a Hotpot dependency for `construct_graph()` and unrelated
public graph surfaces. The cutover claim is specifically that production
`hotpot.cheminfo.search` execution does not construct or invoke a NetworkX
`GraphMatcher`.

### 7.3 Packaging boundary

The in-tree kernel becomes a new `Pybind11Extension` declaration in
`setup.py`. It uses C++17 and the already required pybind11 build dependency;
it adds no linked or runtime graph-search library. It must build in the
supported CPython 3.9 through 3.14 matrix. Import failure is explicit, and the
package must not silently select the old matcher.

## 8. Implementation stages and atomic commit sequence

Implementation begins only after plan approval. Each stage is one reviewable
commit or a short atomic commit group with its stated tests green.

### Stage 0 - Freeze contracts, oracle fixtures, and baselines

Tasks:

1. inventory every public query/search/result behavior and repository
   consumer;
2. freeze the Section 7.1 signature/export manifest, enum/error identities,
   mapping-iterator protocol, and matcher-provider compatibility fixtures;
3. run and record the complete current search, SMARTS conformance, and consumer
   suites;
4. generate immutable NetworkX-era oracle fixtures containing target/query
   inputs, complete raw mapping sets, grouped hits, mapped bonds, bounded
   results, truncation, expected errors, empty queries, self-loops, duplicate
   query bonds, and duplicate `QueryAtom` object insertion;
5. add an independent brute-force oracle for small labeled graphs;
6. record current generic-search and MCA P1 performance, RSS, and result
   digests; and
7. record Python, NetworkX, Open Babel, RDKit, compiler, CPU, and operating
   system metadata.

Exit criteria:

- the semantic inventory covers every item in Section 3;
- every Section 7.1 signature, default, keyword-only marker, export alias, and
  return protocol is captured by a failing-on-change test;
- a single-use `Hits` provider fixture proves exactly one provider call and
  target-to-query input direction;
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

1. add the in-tree C++17/pybind11 extension and compact target/query topology
   bindings without adding a third-party graph-matcher dependency;
2. make the topology kernel accept precomputed synthetic node/edge
   compatibility tables so Stage 2 can be proved independently of the Hotpot
   predicate VM introduced in Stage 3;
3. implement the Section 6.2 RI-style candidate domains, ordered-parent DFS,
   injective subgraph monomorphism, disconnected queries, safe forward checks,
   and raw automorphism enumeration;
4. freeze deterministic selection/candidate-order constants and prohibit
   unordered-container iteration from affecting emitted order;
5. implement the lazy raw native cursor and one-result `has_match` pull without
   a second native `max_matches` owner;
6. prove the existing facade limiter's `0`, fewer-than-`N`, exactly-`N`, and
   greater-than-`N` behavior over the native cursor; and
7. compare direct extension output with the independent brute-force oracle.

Exit criteria:

- exhaustive small labeled graphs have identical complete embedding sets;
- empty, disconnected, self-loop, and normalized duplicate-node/edge query
  cases match their Stage 0 fixtures, including exact sparse query keys rebuilt
  with `zip(query_keys, target_values)`;
- repeated and cross-process mapping sequences are byte-identical;
- bounded searches are valid, deterministic, and report truncation correctly;
- extension linkage inspection shows no external graph-matcher library; and
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
- the complete predicate VM replaces Stage 2 synthetic compatibility tables
  without changing any kernel mapping sequence for equivalent tables;
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
- no recursive expression uses the opaque-callable branch;
- anchored node-0, disconnected nested components, self-loops, and per-target
  snapshot memoization pass focused fixtures; and
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
6. run the frozen signature/export/return-protocol manifest; and
7. verify facade-owned `N + 1` probing through both `iter_mappings()` and
   `search()`/`Hits`, with no hidden native probe; and
8. run every direct and downstream consumer suite.

Exit criteria:

- every Section 7.1 public signature, alias identity, enum/error identity,
  schema, and return protocol is unchanged except for the approved concrete
  `Hits.graph_matcher` adapter type;
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
2. inspect wheels and extension linkage to prove that no new graph-search
   library or undeclared native dependency is present;
3. run native sanitizers and leak checks;
4. run frozen generic and MCA performance workloads;
5. update search/SMARTS documentation and the implementation log/report;
6. remove temporary executable oracle code; and
7. retain only immutable fixtures, independent brute-force tests, and Git
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
| Public API | Exact signatures/defaults/keyword-only markers, three-facade alias identities, SMARTS helper/token protocol, enum values, error bases, and mapping/result protocols |
| Graph kernel | Exhaustive small labeled graphs against an independent brute-force oracle, including empty, disconnected, self-loop, duplicate-node, and duplicate-edge normalization cases |
| Complete parity | Exact unbounded raw embedding sets for built-in/typed and pure deterministic callable predicates |
| Enumeration | Repeated, cross-process, worker-count, and supported-platform sequence stability |
| Bounded search | Facade-owned `N + 1`, every `0`/`<N`/`=N`/`>N` boundary through iterator and Hits paths, raw counts, deterministic prefix, and no double limiter |
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
| Project-owned matcher contains a faulty pruning rule | Begin with only the Section 6.2 safe checks; require exhaustive brute-force equivalence before adding every stronger pruning rule |
| A third-party matcher becomes an undeclared semantic dependency | Linkage/source inspection and a plan amendment before any topology executor other than the approved in-tree kernel |
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
2. the complete Section 7.1 API tree, signatures, aliases, defaults, result
   schemas, iterator protocol, and `Hits` collection/state behavior are
   unchanged, with `graph_matcher` represented by the approved native adapter;
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
15. the shipped executor is the approved Section 6.2 in-tree kernel, source and
    wheel linkage contain no third-party graph matcher, and production code
    contains no legacy matcher or runtime backend selection.

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
- [x] the project targets generic `hotpot.cheminfo.search`, not only MCA; and
- [x] the plan freezes the existing API at class/function/method signature
  level rather than redesigning it during nativeization.

Derived consequences requiring review with this plan:

- [x] the production topology executor is the Section 6.2 in-tree C++17
  deterministic RI-style kernel, with pybind11 as its binding layer and no
  third-party graph-matcher dependency;
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

The owner explicitly authorized implementation after approving the single
RI-style topology executor. Execution evidence belongs in the implementation
log and final report rather than changing this normative contract silently.
