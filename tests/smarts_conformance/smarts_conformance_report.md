# SMARTS parser and substructure-search conformance report

Date: 2026-09-15

Audited revision: `406c41f705d4e10099f8504ff52f5a27d33e9798`

Active implementation: `hotpot/cheminfo/search/smarts.py` and
`hotpot/cheminfo/search/search.py`

Production code changed by this audit: **no**

## Executive result

The new test suite is usable as an offline regression gate, but the active
implementation is not yet fully conformant with the frozen Hotpot SMARTS
contract.

- Supported-behaviour suite: **184 passed, 19 deselected**. This includes an
  exact known-mismatch gate that executes all 1,332 corpus cases.
- Existing SMARTS baseline: **72 passed, 49 subtests passed**.
- Existing baseline + MCA rules + supported conformance: **288 passed,
  19 deselected, 49 subtests passed** on Python 3.14; **288 passed,
  19 deselected** on Python 3.9.
- Strict conformance suite: **184 passed, 19 failed**. The failing tests are
  deliberately neither skipped nor xfailed.
- Deterministic corpus: **1,332 cases**; 1,278 passed and 54 mismatched the
  intended contract.
- The parser coverage target is met: `smarts.py` has **95.53% statement** and
  **91.10% branch** coverage. The search/result module has **94.92% statement**
  and **80.43% branch** coverage.

The highest scientific-correctness risks are explicit single/double SMARTS
bonds falsely matching aromatic bonds, 25 element symbols not behaving like
their atomic-number equivalents, and malformed queries being accepted as
valid query graphs. The highest resource risk is eager enumeration of all
query automorphisms, which grows factorially for disconnected symmetric
queries.

## Frozen contract and preprocessing

The adopted dialect is the project-documented Daylight-like subset plus the
Hotpot coordination-chemistry extensions `M`, `Ln`, `An`, `NPn[-m]`, and
`NGn[-m]`. The full normative decisions are recorded in
`SMARTS_CONFORMANCE.md`.

Target SMILES are read by `hotpot.read_mol(text, "smi")`. Open Babel performs
SMILES parsing and supplies atomic number, formal charge, aromaticity, bond
order, and implicit-hydrogen information. Matching itself uses Hotpot objects
and NetworkX; it does not use RDKit objects. Ring membership and ring sizes use
`networkx.cycle_basis`, so ambiguous polycyclic `R<n>`/`r<n>` cases are audit
cases rather than enforced core truth until a ring-set contract is chosen.

Malformed syntax must raise `ValueError`; recognized but unsupported syntax
must raise `NotImplementedError`; target-preparation failures must stay
distinct. Unknown exceptions are allowed to propagate from the thin adapter
and count as crashes. Atom mappings are metadata. Embeddings preserve query
atom order, while target atom sets are represented separately.

## Delivered test structure

All delivered files are new test/audit material under
`tests/smarts_conformance/`:

```text
tests/smarts_conformance/
├── README.md
├── SMARTS_CONFORMANCE.md
├── smarts_conformance_report.md
├── feature_matrix.yaml
├── adapter.py
├── audit_corpus.py
├── corpus.py
├── corpus_validator.py
├── corpus/{manifest.json,known_mismatches.json}
├── test_adapter.py
├── test_corpus_validator.py
├── test_differential_normalize.py
├── test_enumeration.py
├── test_extensions_exhaustive.py
├── test_full_corpus_contract.py
├── test_fuzz_generators.py
├── test_metamorphic.py
├── test_parse_and_query_graph.py
├── test_real_rules.py
├── test_regressions.py
├── test_search_integration.py
├── test_search_objects.py
├── test_semantics.py
├── differential/{README.md,cases.py,normalize.py,runner.py}
├── fuzz/{README.md,generators.py,run_deterministic.py,runner.py}
└── benchmarks/{README.md,benchmark_smarts.py,runner.py}
```

The corpus is generated deterministically from reviewed truth tables. Every
case carries an ID, classification, feature tags, dialect, and provenance.
No external corpus was imported, and reference-engine output is evidence only;
it is never used to rewrite the golden expectations.

## Commands and observed results

Run these commands from the repository root. Replace the explicit interpreter
with the equivalent project environment where appropriate.

```bash
# Existing focused baseline
/tmp/hotpot-py314/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_cheminfo/test_search.py \
  tests/test_cheminfo/test_search_mapping.py \
  tests/test_cheminfo/test_smarts.py \
  tests/test_smart_parser.py
# 72 passed, 49 subtests passed

# Smoke
/tmp/hotpot-py314/bin/python -m pytest -q -p no:cacheprovider \
  -m smarts_smoke tests/smarts_conformance
# 38 passed, 165 deselected

# Supported core contract
/tmp/hotpot-py314/bin/python -m pytest -q -p no:cacheprovider \
  -m 'smarts_core and not smarts_known_failure' tests/smarts_conformance
# 184 passed, 19 deselected

# Strict core contract: currently expected to return non-zero
/tmp/hotpot-py314/bin/python -m pytest -q -p no:cacheprovider \
  -m smarts_core tests/smarts_conformance
# 184 passed, 19 failed

# Corpus schema/count gate and exhaustive audit
/tmp/hotpot-py314/bin/python -m tests.smarts_conformance.corpus_validator
/tmp/hotpot-py314/bin/python -m tests.smarts_conformance.audit_corpus \
  --output /tmp/hotpot-smarts-corpus-audit.json
# schema gate passes; strict audit returns 1 because 54 cases disagree

# Coverage
COVERAGE_FILE=/tmp/hotpot-smarts.coverage \
  /tmp/hotpot-py314/bin/python -m coverage run --branch \
  --source=hotpot/cheminfo/search -m pytest -q -p no:cacheprovider \
  -m 'smarts_core and not smarts_known_failure' tests/smarts_conformance
COVERAGE_FILE=/tmp/hotpot-smarts.coverage \
  /tmp/hotpot-py314/bin/python -m coverage report -m \
  hotpot/cheminfo/search/smarts.py hotpot/cheminfo/search/search.py

# Heavy evidence collectors
/tmp/hotpot-py314/bin/python \
  -m tests.smarts_conformance.differential.runner \
  --output /tmp/hotpot-smarts-differential.json
timeout 60s /tmp/hotpot-py314/bin/python \
  -m tests.smarts_conformance.fuzz.run_deterministic \
  --seed 20260915 --output /tmp/hotpot-smarts-fuzz.json
/tmp/hotpot-py314/bin/python \
  -m tests.smarts_conformance.benchmarks.benchmark_smarts \
  --output /tmp/hotpot-smarts-benchmark.json
```

The differential runner defaults to audit mode and records findings while
returning zero. Use its `--strict` option when differences must fail a job.
The corpus audit and fuzz runner return non-zero for the confirmed findings;
this is intentional and prevents audit output from masquerading as a pass.

## Functional coverage

| Layer | Evidence |
|---|---:|
| Valid parsing | 454 cases / 454 unique SMARTS / 13 generator families |
| Definitely invalid parsing | 208 cases / 208 unique SMARTS / 9 generator families |
| SMARTS-target semantics | 670 pairs / 306 unique SMARTS / 11 broad families; 327 positive and 343 near-negative |
| Total deterministic corpus | 1,332 cases |
| Exhaustive element identity | atomic numbers 1-118, symbol and `[#Z]` forms |
| Hotpot extensions | `M`, `!M`, all `Ln`/`An`, periods 1-7, groups 1-18 |
| Production MCA rules | all 24 rules, each with positive and methane control; 48 checks |
| Differential | 277 cases, including a 118-element equivalence matrix |
| Fixed-seed fuzz | 400 cases: 100 valid, 100 invalid, 100 robustness, 100 structured stress |
| Enumeration | query-order mappings, target sets, symmetry and automorphisms |
| Search | object API, single search, batch-vs-linear baseline, invalid target separation |
| Metamorphic | OR/AND commutation, double negation, equivalent spelling, and true target-atom renumbering compared through stable atom IDs |

The corpus exceeds the requested minima of 300 valid, 150 invalid, and 300
semantic pairs. Counts are not presented as 1,332 independent grammar forms:
472 semantic pairs form the deliberate exhaustive 118-element
symbol/atomic-number matrix, and 64 verify atom-map metadata over 32 map
labels. Core atom identities, atom properties (`D`, `X`, `v`, `H`,
`R`, `r`), Boolean precedence, bond types, branches, rings, disconnected
queries, recursive SMARTS, atom maps, aromaticity, result enumeration, and
project extensions all have direct tests. Unsupported chirality, isotopes,
directional bonds, ring bonds, and lowercase `h`/`x` are tested for rejection
classification rather than invented semantics.

## Code coverage

Coverage was measured from the supported deterministic core, excluding the
tests intentionally red for known production defects.

| Module | Statements | Statement coverage | Branches | Branch coverage |
|---|---:|---:|---:|---:|
| `hotpot/cheminfo/search/smarts.py` | 559 | 534/559 = 95.53% | 292 | 266/292 = 91.10% |
| `hotpot/cheminfo/search/search.py` | 197 | 187/197 = 94.92% | 46 | 37/46 = 80.43% |
| Combined | 756 | 721/756 = 95.37% | 338 | 303/338 = 89.64% |

The parser target of 90% statements and 85% branches is met. Important
remaining branches include rare malformed-expression paths, unsupported bond
alternatives, both-end ring-bond declarations, and a few Searcher construction
and object-conversion paths. Mutation testing was not run because no mutation
tool is installed in the isolated environments; this remains a test-quality
risk even though the strict regressions demonstrate sensitivity to the known
faults.

## Reference engines

The differential run used:

- Hotpot at Git revision `406c41f`, NetworkX 3.6.1, target aromaticity from
  Open Babel through Hotpot;
- RDKit 2026.03.6, sanitized targets, `useChirality=False`, raw matches with
  `uniquify=False`, and `maxMatches=1,000,000`;
- Open Babel runtime 3.1.0 (`openbabel-wheel` 3.1.1.23), using
  `OBSmartsPattern.GetMapList` and `GetUMapList`.

Of 277 cases, 218 were unanimous, 34 were classified as Hotpot project
mismatches, 17 as reference/dialect disagreements, and 8 as Hotpot-only
extensions. Across the 118-element `[Symbol]` versus `[#Z]` matrix, Hotpot had
25 inequivalences, Open Babel 15, and RDKit 4. These counts are not interpreted
as an engine-quality ranking: unsupported superheavy elements and different
aromatic/ring conventions are kept separate from manually reviewed core
expectations.

Raw mappings, engine-unique mappings, and unordered target atom sets are now
reported as three distinct quantities. Exact duplicate raw mappings are not
discarded. Embedding tuples use engine-local target indices and are not
compared directly across independent SMILES readers. Cross-engine decisions
use acceptance, existence, raw-embedding counts, and normalized target-set
counts; native engine-unique counts remain diagnostic because each backend
defines uniquification differently. This prevents apparent atom-index
agreement from being mistaken for a verified common atom identity.

## Confirmed implementation issues

Each issue below is reproducible offline on both supported test environments
unless stated otherwise. `Options: default` means Hotpot's current public
search API exposes no alternate matching option for the tested behavior.

### SMARTS-PARSE-BOUNDARY-001 — empty queries/components are accepted

- Classification: parser/query-graph bug
- SMARTS: `""`, whitespace, `.`, `.C`, `C.`, `C..O`
- Target SMILES: `CC`
- Options: default
- Expected: reject every query with `ValueError`
- Actual: all are accepted; empty/whitespace/`.` searches produce one empty
  `Hit`
- Failure phase: tokenization/query compilation
- Evidence: strict regression, invalid corpus, RDKit/Open Babel differential
  for non-empty malformed-dot cases
- Reproducibility: deterministic, every run
- Suspected module: `smarts.py:202-272`; no non-empty-query and component-state
  validation
- Optimization: validate parser state at every dot and at end-of-input; reject
  empty query graphs before invoking NetworkX

### SMARTS-PARSE-GRAPH-002 — invalid graph topology is compiled

- Classification: query-graph compiler bug
- SMARTS: `=C`, `C11`, `C1C1`, `C-(C)`
- Target SMILES: not required
- Options: default
- Expected: reject leading bond, self-loop, duplicate edge, and branch bond
  without a following branch atom
- Actual: all four compile; `C11` yields one atom plus a self-bond and `C1C1`
  yields two parallel/duplicate query edges
- Failure phase: query compilation
- Evidence: strict regression and invalid corpus
- Reproducibility: deterministic, every run
- Suspected module: `smarts.py:202-272` and `smarts.py:893-914`
- Optimization: use an explicit parser state machine and reject self/duplicate
  edges before `Substructure.add_bond`

### SMARTS-RECURSIVE-001 — empty recursive SMARTS is accepted

- Classification: query-compiler bug
- SMARTS: `[$()]`, `[C;$()]`
- Target SMILES: not required
- Options: default
- Expected: `ValueError`
- Actual: both compile as a one-atom query containing an empty recursive query
- Failure phase: atom-expression/query compilation
- Evidence: strict regression and invalid corpus
- Reproducibility: deterministic, every run
- Suspected module: `smarts.py:484-488`
- Optimization: require a non-empty recursive body before constructing
  `_RecursivePredicate`

### SMARTS-DIAGNOSTIC-001 — malformed logic leaks `IndexError`

- Classification: robustness/diagnostic bug
- SMARTS: `[!]`, `[!!]`, `[C!]`, `[C&!]`
- Target SMILES: not required
- Options: default
- Expected: controlled `ValueError` with query-compilation phase
- Actual: `IndexError: string index out of range`
- Failure phase: atom-expression compilation
- Evidence: four strict parameterized regressions
- Reproducibility: deterministic, every run
- Suspected module: `smarts.py:494`
- Optimization: validate operand availability in the expression parser and
  introduce a small structured parse-error type carrying position and phase

### SMARTS-ATOM-WILDCARD-001 — documented bare `a` and `A` fail

- Classification: parser/documented-core mismatch
- SMARTS: `a`, `A`
- Target SMILES: benzene and ethane
- Options: default
- Expected: any aromatic atom / any aliphatic atom
- Actual: `ValueError: Unknown atom symbol in SMARTS`
- Failure phase: query compilation
- Evidence: strict regression; RDKit and Open Babel both accept the forms
- Reproducibility: deterministic, every run
- Suspected module: bare-atom construction around `smarts.py:790-815`
- Optimization: special-case bare `a`/`A` consistently with bracket `[a]` and
  `[A]`, or explicitly remove the promise from the dialect documentation

### SMARTS-ATOM-ELEMENT-001 — element symbols collide with primitives

- Classification: parser/atom-semantics bug
- SMARTS: `[Symbol]` versus `[#Z]` for all elements 1-118
- Target SMILES: isolated bracket element of the same atomic number
- Options: default
- Expected: the two spellings are equivalent
- Actual: 25 Hotpot inequivalences: He, Al, Ar, As, Rb, Ru, Rh, Ag, Xe, Dy,
  Ho, Hf, Re, Au, Hg, At, Rn, Ra, Ac, Am, Rf, Db, Hs, Ds, and Rg. Eight are
  rejected (`He`, `Al`, `Ag`, `Xe`, `Re`, `Hg`, `Am`, `Rg`); the remainder
  compile to an unsatisfiable or incorrect conjunction.
- Failure phase: atom-expression query compilation/semantics
- Evidence: strict exhaustive regression and three-engine 118-element matrix
- Reproducibility: deterministic, every run
- Suspected module: `_parse_atom_primitive`; single-letter `H`, `A`, `D`, `X`,
  and `R` primitives are considered before longest valid element symbols
- Optimization: tokenize bracket atom identities with longest-valid-element
  precedence, then parse property primitives in their syntactic context

### SMARTS-ATOM-H-001 — lowercase `h`/`x` rejection is inconsistent

- Classification: unsupported-feature classification bug
- SMARTS: `[h]`, `[h1]`, `[x]`, `[x2]`
- Target SMILES: not required
- Options: default
- Expected: all recognized as unsupported and rejected with
  `NotImplementedError`
- Actual: `[h]` is accepted as an unintended aromatic element-like predicate,
  `[h1]` is misclassified as an isotope, and `[x]`/`[x2]` raise `ValueError`
- Failure phase: atom-expression query compilation
- Evidence: strict regression
- Reproducibility: deterministic, every run
- Suspected module: `_parse_atom_primitive`
- Optimization: reserve `h` and `x` lexically; return a consistent unsupported
  classification until their semantics are implemented

### SMARTS-BOND-AROM-001 — explicit numeric bonds match aromatic bonds

- Classification: matching-semantics bug; high scientific priority
- SMARTS: `*-*`, `*=*`, `[#6]-[#6]`, `[#6]=[#6]`, `c-c`, `c=c`
- Target SMILES: `c1ccccc1`
- Options: default
- Expected: no explicit single or double query matches an aromatic edge
- Actual: all six queries match benzene
- Failure phase: edge matching
- Evidence: strict regression and RDKit/Open Babel differential
- Reproducibility: deterministic, every run
- Suspected module: `smarts.py:818-857`; numeric bond order is tested without
  excluding `bond.is_aromatic`
- Optimization: make `-`, `=`, and `#` predicates require the requested order
  and `not bond.is_aromatic`, including comma-separated alternatives

### SMARTS-RING-r0-001 — `r0` does not mean acyclic

- Classification: atom matching-semantics bug
- SMARTS: `[C;r0]`
- Target SMILES: `C`
- Options: default NetworkX cycle-basis ring model
- Expected: match the acyclic carbon
- Actual: no match
- Failure phase: atom matching
- Evidence: strict regression and reviewed semantic case
- Reproducibility: deterministic, every run
- Suspected module: numeric `r` predicate in `smarts.py`
- Optimization: implement `r0` as absence from every perceived ring. Do not
  change polycyclic `R<n>`/`r<n>` until the ring-set contract is decided.

### SMARTS-EXT-RANGE-001 — malformed extension ranges are accepted

- Classification: Hotpot-extension parser bug
- SMARTS: `[NP]`, `[NP0]`, `[NP8]`, `[NG0]`, `[NG19]`
- Target SMILES: not required
- Options: Hotpot extension dialect
- Expected: reject incomplete and out-of-domain period/group expressions
- Actual: all compile; `[NP]` is interpreted as elemental N AND P, while the
  numeric cases produce invalid/unsatisfiable predicates
- Failure phase: atom-expression query compilation
- Evidence: strict regression and boundary corpus
- Reproducibility: deterministic, every run
- Suspected module: extension recognition/range parsing in `smarts.py`
- Optimization: establish token boundaries and enforce period 1-7 and group
  1-18. Decide and document whether reversed ranges such as `[NP5-3]` are
  invalid before changing their current silent normalization.

### SMARTS-HIT-BONDS-001 — `Hit.bonds` returns induced target edges

- Classification: result-enumeration/API bug
- SMARTS/target SMILES: `C.C`/`CC`; `CCC`/`C1CC1`
- Options: default, grouped target-atom-set hits
- Expected: 0 and 2 query-mapped bonds respectively
- Actual: 1 and 3 bonds; all edges induced by the matched atom set are returned
- Failure phase: result materialization
- Evidence: two strict regressions
- Reproducibility: deterministic, every run
- Suspected module: `search.py:555-556`
- Optimization: derive `Hit.bonds` from each query-edge mapping; expose a
  separately named `induced_bonds` property if that behavior is also useful

### SMARTS-PERF-EAGER-001 — eager mapping enumeration is factorial

- Classification: algorithmic complexity / denial-of-service risk
- SMARTS: `*.*.*.*`, increasing through seven disconnected wildcards
- Target SMILES: equal-sized disconnected carbon atoms
- Options: default; all automorphisms retained
- Expected: existence checks and bounded searches should terminate without
  materializing all mappings
- Actual: `Hits.__init__` exhausts the GraphMatcher iterator. Raw mappings grow
  24, 120, 720, 5,040 for sizes 4-7. Median time grows from 2.66 ms to
  14.48 ms, 96.36 ms, and 753.33 ms; traced peak memory reaches 5.43 MB at 7.
- Failure phase: enumeration/result materialization
- Evidence: deterministic benchmark
- Reproducibility: deterministic counts; timings are machine-dependent
- Suspected module: `search.py:475` and `search.py:489-505`
- Optimization: add an existence fast path, lazy iteration, `max_matches`, and
  an explicit `truncated` flag. Cache compiled query graphs where lifecycle
  permits. Never silently truncate.

### SMARTS-QUERY-ATOM-001 — `QueryAtom.from_atom` is unusable

- Classification: search-object construction bug
- SMARTS: not applicable
- Target SMILES: `C`
- Options: default
- Expected: construct a query atom matching the input Hotpot atom
- Actual: `TypeError: 'int' object is not iterable`
- Failure phase: query-object construction
- Evidence: strict regression
- Reproducibility: deterministic, every run
- Suspected module: `search.py:197-198`; scalar attributes are passed to
  `set(...)`, and a default `None` `include_attrs` is iterated
- Optimization: wrap scalar values as singleton sets and iterate
  `include_attrs or ()`

### SMARTS-DOC-001 — documentation does not match active behavior

- Classification: documentation defect
- SMARTS/target/options: not applicable
- Expected: documentation describes the active parser and result objects
- Actual: `hotpot/cheminfo/smarts.md` gives incorrect Boolean-precedence and
  `parse_bracket_atom` return details; `Hit.atom_indices` is documented as an
  ordered list but is a `frozenset`; aromatic-bond prose does not accurately
  express the intended constraint
- Failure phase: documentation review
- Evidence: source/documentation comparison
- Reproducibility: static
- Suspected module: `hotpot/cheminfo/smarts.md` and `search.py` docstrings
- Optimization: update documentation after the target contract is approved,
  and generate small API examples as doctests where practical

### SMARTS-MAINT-001 — four parser implementations remain in the tree

- Classification: maintenance/divergence risk
- SMARTS/target/options: not applicable
- Expected: one authoritative implementation, or explicit versioned backends
- Actual: the active `search/smarts.py` coexists with
  `search/_smarts.py`, `cheminfo/parse_smarts.py`, and
  `search/smarts_parser/`; their contracts and quality differ
- Failure phase: repository discovery
- Evidence: import and source-tree audit
- Reproducibility: static
- Suspected module: parser package organization
- Optimization: audit all imports, mark legacy implementations as such, then
  archive/remove them in a separate compatibility change

## Corpus, differential, fuzz, and performance findings

The 1,332-case corpus recorded 8 failures among 454 intended-valid parses,
13 failures among 208 intended-invalid parses, and 33 failures among 670
semantic pairs. The strict test intentionally aggregates each corpus category
into one failing test so all cases are evaluated and the JSON artifact contains
every mismatch.

The fixed-seed fuzz run accepted all 100 grammar-generated valid evaluations
(60 unique query strings). Each invalid case was then made by one named local
corruption of that iteration's valid seed: 88 unique invalid strings were
produced, 63 were rejected and 37 were incorrectly accepted. The accepted
mutations concentrate in leading/trailing/duplicate components and leading
bonds. All 100 arbitrary robustness inputs were unique: 99 were controlled
rejections and 1 was accepted. All 100 structured-stress evaluations were
accepted (28 unique queries): 23 deep branches, 27 nested recursive queries,
28 mixed Boolean expressions, and 22 multiple-ring queries. Mutation records
retain their base SMARTS, mutation name, zero-based edit position, and payload.
The four streams use fixed child seeds derived from the reported seed, so a
generator change in one stream does not shift the other streams.
No crash, timeout, or non-deterministic result occurred in this fixed seed.
Separate curated regressions still prove that malformed Boolean expressions
can leak `IndexError`, so the fuzz result must not be read as “no crash is
possible.”

Repeated benchmark outputs were deterministic, threaded and linear batch
search returned identical results, and concurrent query compilation agreed
with serial compilation. Threads were slower than the linear adapter
at the tested small sizes, so parallelism should not be advertised as a speed
optimization without a workload-specific benchmark. The only native batch
facility tested here is a test adapter; Hotpot currently has no fingerprint
index, pagination, streaming, `maxMatches`, or native batch/index API to test.
The benchmark now also covers deep branches, multiple rings, nested recursion,
wide combined AND/OR expressions, final-atom near-match failure, and concurrent
query compilation. Its checksums are stable SHA-256 digests.

## Disputed and unsupported areas

The following should not be “fixed” until the project chooses a contract:

- `[H,Cl]` and related elemental-H versus H-count context;
- `R<n>`/`r<n>` on fused, bridged, spiro, and cage systems, because NetworkX
  cycle basis and symmetrized SSSR are not equivalent;
- redundant nested branches, both-end ring-bond declarations, and ring labels
  longer than `%nn`;
- internal whitespace acceptance;
- Open Babel aromaticity changes around metal coordination.

Currently explicit unsupported features are isotopes, atom chirality,
directional bonds, ring-bond predicates, general bond negation, reaction
SMARTS, component-level grouping, native indexing, streaming/pagination,
`maxMatches`, and a chirality search option. They must remain visibly
unsupported rather than being interpreted through a fallback.

## Recommended repair order

1. Fix explicit-bond/aromatic matching and element-token collisions first;
   both can silently produce chemically wrong matches.
2. Add parser state validation for empty components, leading/dangling bonds,
   self/duplicate edges, empty recursion, and missing Boolean operands.
3. Correct `r0`, extension bounds, `Hit.bonds`, and `QueryAtom.from_atom`.
4. Add lazy/bounded enumeration with explicit truncation metadata before
   exposing SMARTS search to untrusted or large disconnected queries.
5. Resolve the disputed dialect/ring decisions, update public documentation,
   and only then consolidate legacy parsers.

Each repair should make its matching `smarts_known_failure` regression pass;
the marker must then be removed from that repaired test rather than weakening
the expected value. Production changes should be small and independently
committed so semantic changes can be bisected.

## Adding regressions and reviewing golden data

For a new defect, minimize the SMARTS/SMILES pair, classify the failure phase,
assign a stable issue/case ID, add the intended assertion to
`test_regressions.py`, attach provenance and an exit condition, update
`feature_matrix.yaml`, then run smoke, strict core, corpus validation, and the
relevant differential case.

Golden expectations must be reviewed as a source diff. No runner in this suite
rewrites golden data from current Hotpot, RDKit, or Open Babel output. A
reference-engine disagreement is supporting evidence, not sufficient reason to
change the Hotpot contract.

## Remaining limitations

- Mutation testing was not executed.
- No sanitizer/native-memory instrumentation was available.
- The repository-wide legacy test collection was attempted but could not run
  in the lightweight SMARTS environment because `numba`, `torch_geometric`,
  `scikit-learn`, and `requests` are absent; the XTB plugin also tries to write
  its package-local cache. The directly relevant existing SMARTS/Search/MCA
  tests were run and passed.
- The smoke/core/heavy profiles are defined and runnable but are not yet wired
  into the repository's hosted CI configuration.
- No large external licensed SMARTS corpus was imported; instead, exhaustive
  periodic-table/extension matrices and deterministic generated cases were
  used with explicit provenance.
- Native indexed search cannot be assessed because it does not exist in the
  current API.
- Benchmark timings are diagnostic snapshots, not cross-machine performance
  thresholds.

These limitations do not invalidate the confirmed defects above, but they are
the next areas to expand after production behavior is corrected.
