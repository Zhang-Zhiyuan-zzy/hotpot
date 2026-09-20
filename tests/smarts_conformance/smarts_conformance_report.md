# SMARTS parser and substructure-search conformance report

Original audit date: 2026-09-15

Relevant-Cycle supplement: 2026-09-20

Supplement validation revision: `4d847f4`

Audited production revision: `7b262a9663d930d86186f5473fab60af2d70931a`

Active implementation: `hotpot/cheminfo/search/{smarts.py,_smarts_syntax.py,
semantics.py,errors.py,search.py}`

## Executive result

The scoped SMARTS regression gate is green after the parser, result-object,
bond-metadata, and coordination-semantics changes.

- Strict `smarts_core`: **252 passed** on Python 3.9 and **252 passed** on
  Python 3.14.
- Relevant-Cycle supplement: **255 passed** on both Python 3.9 and 3.14,
  including three marked cubane tests for `R/r`, edge-order invariance, and
  RDKit parity.
- Smoke profile: **38 passed, 214 deselected** on Python 3.14.
- Deterministic corpus: **1,332/1,332 passed**; zero mismatches.
- Focused legacy Search/SMARTS, MCA site detection, bond metadata, and ligand
  ring tests: **136 passed, 49 subtests passed** on Python 3.14.
- The inference compatibility matrix passed on every Python version from 3.9
  through 3.14: **190 focused tests** plus **252 strict SMARTS tests** per
  interpreter (Python 3.9 reports the same subtests as ordinary tests).
- Strict-suite coverage of the active search modules: **903/945 statements**
  and **362/402 branches**, reported as **94% total** by Coverage.py.

The former failing-conformance and corpus-mismatch figures described the
pre-repair implementation and are no longer current. No repository-wide test
result is claimed here.

## Frozen contract and preprocessing

The dialect is the documented Daylight-like subset plus Hotpot's `M`, `Ln`,
`An`, `NPn[-m]`, and `NGn[-m]` extensions. Malformed syntax raises
`SmartsSyntaxError` (`ValueError`); recognized unsupported syntax raises
`UnsupportedSmartsError` (`NotImplementedError`).

Targets are Hotpot molecular graphs. Open Babel performs input perception, but
matching uses Hotpot objects and NetworkX, not RDKit. `BondKind` records
`SINGLE`, `DOUBLE`, `TRIPLE`, `AROMATIC`, `ZERO`, `DATIVE`, or `UNKNOWN`
separately from numeric order.

Two public semantics profiles select the view used by topology-sensitive atom
primitives:

| Profile | `D` / `X` | `v` | `R` / `r` |
|:--|:--|:--|:--|
| `FULL_GRAPH` (default) | All graph edges; `X` adds implicit H | Numeric bond-order sum plus implicit H | `Molecule.rings` |
| `LIGAND_SKELETON` | Non-metals omit metal--ligand edges; metals retain full coordination | Ligand view; only covalent/aromatic `BondKind` values contribute | `Molecule.ligand_rings` |

The ligand profile never mutates the molecular graph. Recursive SMARTS inherits
the selected profile. Explicit `-` and implicit aliphatic single bonds require
`BondKind.SINGLE`; `DATIVE`, `ZERO`, and `UNKNOWN` match `~` but not `-`.

Profile selection does not rerun hydrogen perception: `X` and `v` retain the
reader-provided `Atom.implicit_hydrogens`. In the frozen fixtures Open Babel
3.1 assigns a coordinated amine donor zero implicit H from MOL2 but one from
SDF despite identical numeric bond topology. The corresponding ligand-profile
values are therefore `X3/v3` and `X4/v4`.

Open Babel 3.1 maps MOL2 `du`, `un`, and `nc` to the same numeric order-zero
representation and does not preserve the lexical source token. Hotpot therefore
records all three as `UNKNOWN`. The fixture contract deliberately refuses to
guess whether such an edge was zero-order or dative.

MCA rules compile with `LIGAND_SKELETON`. Reliable MCA site selection then
applies a separate applicability-domain rule: metal centres and atoms directly
bound to a metal are excluded. This does not erase per-atom MCA predictions.

## Delivered test structure

The suite remains under `tests/smarts_conformance/` and contains:

- parser, atom-expression, graph-syntax, and result-object contracts;
- exhaustive element and Hotpot-extension matrices;
- a deterministic 1,332-case parse/semantic corpus;
- direct Hotpot coordination graphs for D/X/v and bond-kind truth tables;
- MOL2/SDF perception fixtures and repository CIF cases;
- full-graph versus ligand-skeleton ring and recursive-profile tests;
- differential, fuzz, and benchmark evidence collectors.

The coordination fixture manifest is
`fixtures/coordination/manifest.yaml`. Golden data is source-controlled and is
never silently regenerated from Hotpot or a reference engine.

## Commands and observed results

The original commands were run from the repository root on 2026-09-15.

```bash
/tmp/hotpot-py314/bin/python -m pytest -q -p no:cacheprovider \
  -m smarts_core tests/smarts_conformance
# 252 passed in 2.36s

/tmp/hotpot-py39/bin/python -m pytest -q -p no:cacheprovider \
  -m smarts_core tests/smarts_conformance
# 252 passed in 2.64s

/tmp/hotpot-py314/bin/python -m pytest -q -p no:cacheprovider \
  -m smarts_smoke tests/smarts_conformance
# 38 passed, 214 deselected in 1.35s

/tmp/hotpot-py314/bin/python -m pytest -q -p no:cacheprovider \
  tests/smarts_conformance/test_corpus_validator.py \
  tests/smarts_conformance/test_full_corpus_contract.py
# 12 passed in 1.18s

/tmp/hotpot-py314/bin/python \
  -m tests.smarts_conformance.audit_corpus \
  --output /tmp/hotpot-smarts-corpus-audit-current.json
# 1332 passed; 0 failed
```

The focused compatibility command was:

```bash
/tmp/hotpot-py314/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_cheminfo/test_search.py \
  tests/test_cheminfo/test_search_mapping.py \
  tests/test_cheminfo/test_smarts.py \
  tests/test_smart_parser.py \
  tests/mca/test_site_detection.py \
  tests/mca/test_site_detection_coordination.py \
  tests/test_cheminfo/test_bond_metadata.py \
  tests/test_cheminfo/test_ligand_rings.py
# 136 passed, 49 subtests passed in 1.87s
```

The packaged compatibility runner was also executed without exclusions:

```bash
UV_CACHE_DIR=/tmp/hotpot-audit-uv-cache \
  bash tests/run_inference_compatibility.sh 3.9 3.10 3.11 3.12 3.13 3.14
# Every interpreter: 190 focused tests + 252 strict SMARTS tests passed
```

These timings are local diagnostic observations, not performance thresholds.

The Relevant-Cycle supplement was run from the repository root on 2026-09-20:

```bash
UV_CACHE_DIR=/tmp/hotpot-ring-audit-uv-cache \
  bash tests/run_inference_compatibility.sh 3.9 3.14
# Each interpreter: 710 passed, 4 skipped, 3 xfailed in the focused matrix.
# Each interpreter: 255 passed in the strict SMARTS gate.
```

## Corpus evidence

The deterministic corpus was audited independently of the aggregate pytest
assertion:

| Corpus | Cases | Failures |
|:--|--:|--:|
| Valid parsing | 454 | 0 |
| Definitely invalid parsing | 208 | 0 |
| SMARTS-target semantics | 670 | 0 |
| **Total** | **1,332** | **0** |

`corpus/known_mismatches.json` has an empty `issues` object. Any new mismatch,
removed expectation, or changed classification must be reviewed as a source
diff.

## Code coverage

Coverage was measured with branch tracking over the strict Python 3.14 suite:

| Module | Statements covered | Branches covered | Reported coverage |
|:--|--:|--:|--:|
| `_smarts_syntax.py` | 109 / 114 | 47 / 52 | 94% |
| `errors.py` | 2 / 2 | 0 / 0 | 100% |
| `search.py` | 274 / 284 | 59 / 68 | 95% |
| `semantics.py` | 26 / 26 | 6 / 6 | 100% |
| `smarts.py` | 492 / 519 | 250 / 276 | 93% |
| **Total** | **903 / 945** | **362 / 402** | **94%** |

Coverage is supporting evidence only; semantic matrices and regression cases
remain the acceptance criteria.

## Reference engines

The optional differential runner was repeated on the repaired implementation:
**252/277** cases were unanimous, **17** were classified as reference/dialect
disagreements, and **8** were Hotpot-only extensions. The run used RDKit
2026.03.6 and Open Babel 3.1.0. These are audit results rather than normative
goldens; Hotpot-only coordination extensions must not be interpreted through
engines with different syntax.

## Resolved implementation issues

The strict regressions now verify the following repairs:

- empty queries/components, malformed branches, leading/dangling bonds,
  self-loops, duplicate edges, and empty recursion are rejected;
- malformed Boolean logic raises a controlled syntax error;
- bare `a`/`A`, element-symbol precedence through atomic number 118, and
  `r0` behave according to the contract;
- lowercase `h`/`x` are consistently classified as unsupported;
- explicit numeric bonds no longer match aromatic bonds;
- period/group extension bounds are validated;
- `QueryAtom.from_atom`, mapped `Hit.bonds`, `induced_bonds`, existence search,
  bounded iteration, and explicit truncation behave as documented;
- full-graph and ligand-skeleton coordination semantics are selected explicitly
  and recursive SMARTS preserves the selected profile.

The historical `smarts_known_failure` marker is still attached to some repaired
regression tests as provenance; those tests pass in the strict gate and are not
xfails or exclusions.

## Fuzz and performance evidence

The fixed-seed (`20260915`) fuzz run evaluated **400/400** cases with zero
failures: 100 valid, 100 invalid mutations, 100 robustness strings, and 100
structured-stress inputs. The quick benchmark remained deterministic, and its
threaded and serial checksums agreed. The principal retained worst case is raw
automorphism enumeration: seven disconnected wildcards produce 5,040 mappings
in a median **886.56 ms** with **5.42 MB** peak traced Python memory on this
machine. `has_match` and bounded enumeration let callers avoid that work when
they only need existence or a limit; they do not provide a fingerprint index or
make arbitrary graph-isomorphism workloads constant-time.

## Disputed and unsupported areas

The following boundaries remain explicit:

- `R<n>`/`r<n>` in fused, bridged, spiro, and cage systems follow Hotpot's
  Relevant Cycle family; this is deterministic and symmetry-preserving but is
  not claimed to be identical to every toolkit's SSSR or smallest-ring policy;
- Open Babel controls input aromaticity and implicit-hydrogen perception;
- MOL2 `du`/`un`/`nc` cannot be distinguished after Open Babel 3.1 parsing;
- `LIGAND_SKELETON` excludes every topological metal--nonmetal edge on the
  non-metal side, including potentially covalent organometallic bonds;
- isotope, atom chirality, directional/ring-bond predicates, general bond
  negation, reaction SMARTS, component grouping, and native indexing remain
  unsupported.

These cases must stay explicit rather than being accepted through fallback
logic. A future organometallic policy should be a new named profile, not an
`ignore_metals` Boolean added to the current profiles.

## Adding regressions and reviewing golden data

For a new defect, minimize the SMARTS/target pair, classify the failure phase,
assign a stable case ID, add the intended assertion, update
`feature_matrix.yaml`, and run smoke, strict core, corpus validation, and the
relevant differential or coordination fixture.

Reference-engine disagreement is evidence, not sufficient reason to change the
Hotpot contract. Golden changes require review and must never be generated as
an unconditional fallback from current output.

## Remaining limitations

- This report does not claim a repository-wide pytest pass.
- Mutation testing and native-memory sanitizers were not run.
- No large external licensed SMARTS corpus is included.
- Performance measurements are local and not CI thresholds.
