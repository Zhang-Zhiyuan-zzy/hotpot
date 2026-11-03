# SMARTS parsing and search

Hotpot parses its supported SMARTS subset directly and executes it through one
deterministic C++17 RI subgraph-monomorphism engine. Use the ordinary `hotpot`
exports; canonical query composition lives in `cheminfo.operations.search`,
pure grammar/comparisons in `cheminfo.rules.chemistry.substructure`, the existing
profile/typed-expression contracts in `cheminfo.structure.substructure`, and
prepared/native transport in `cheminfo.adapters`. There is no backend selector
or whole-search Python fallback.

## Syntax and chemistry

- Atoms: element symbols, aromatic lowercase symbols, `*`, `a` and `A`.
- Bonds: `-`, `=`, `#`, `:`, `~`, and comma-separated alternatives such as `=,:`.
  Explicit single/double/triple kinds exclude aromatic bonds.
- Logic precedence: `!` > adjacency/`&` > `,` > `;`. Thus `[C,N;D1]` means
  `(C OR N) AND D1`. Semicolon is the low-precedence AND.
- Charge: `+`, `++`, `+2`, `-`, `--`, `-2`.
- `D`: explicit degree; `X`: degree plus implicit H; `v`: the selected view's
  numeric bond-order summary plus implicit H, not valence-electron count.
- `[H]` names hydrogen; `H` after an atom identity or `H<n>` constrains total
  explicit plus implicit H. Atom-map labels such as `:1` are metadata.
- `R`/`r`: membership in Relevant Cycles; `R<n>` counts them, `r<n>` matches
  a cycle size, and `r0` requires no selected cycle.
- Anchored recursion such as `$([N]-C=O)` uses nested query atom0 as its anchor.
- Hotpot extensions: `[M]`, `[!M]`, `[Ln]`, `[An]`, `[NP3-5]`, `[NG3-8]`.
  NP ranges are1–7 and NG ranges1–18.
- Directional `/` and `\`, chirality `@`/`@@`, and SMARTS isotope predicates
  remain explicitly unsupported. This concerns query matching, not the separate
  source-isotope/stereo transport contract.

`SmartsSemantics.FULL_GRAPH` is the default. `LIGAND_SKELETON` changes D/X/v/ring
descriptors while retaining the full searched topology. For nonmetals it omits
metal–nonmetal incident bonds from those descriptors; a metal retains its
incident bonds. Full-graph v retains Hotpot's integer all-bond-order sum; ligand
v sums only single/double/triple/aromatic kinds. Both use the current source
implicit H. Profiles do not reinterpret a source H convention.

An explicit bond symbol takes precedence. Without one, two explicitly aromatic
atoms require an aromatic bond; other atom pairs accept single-or-aromatic.
Aromatic matching uses the named Hotpot representation and Relevant Cycles.

## Public calls

`Substructure.from_smarts(smarts, *, semantics=SmartsSemantics.FULL_GRAPH)`
builds the ordinary mutable QueryAtom/QueryBond graph.
`Searcher.has_match(mol)` probes the first embedding; `iter_mappings(mol,
max_matches=None)` streams read-only query-index to molecule-index mappings.
The limit counts raw embeddings, including automorphisms; zero still probes
for truncation. `search()` groups Hits deterministically. Mapped bonds and
induced bonds remain distinct.

`operations.search.smarts.substructure_from_smarts` has the same parser entry
contract. Its `parse_bracket_atom(expr_text, *, semantics=...)` returns exactly
`{"predicate": <typed callable expression>}`; maps are excluded from that dict.
`tokenize(smarts)` retains `(TokenType,text)` tuples, including DOT; the lower
`tokenize_with_spans` retains half-open source spans. Syntax failures raise
SmartsSyntaxError(ValueError); unsupported syntax raises
UnsupportedSmartsError(NotImplementedError).

```python
import hotpot as hp

molecule = hp.read_mol("CC(=O)N", "smi")
query = hp.Substructure.from_smarts("[#6;$([#6](=[#8])[#7])]")
searcher = hp.Searcher(query)
assert searcher.has_match(molecule)
assert len(tuple(searcher.iter_mappings(molecule))) == 1
```

Each public search snapshots current Query/Molecule facts. Subsequent mutations
are seen on the next call. Direct recursive predicates also use fresh tasks;
there is no persistent Molecule-keyed result cache. Immutable prepared queries
and one target may be reused within an MCA detection task. Native recursive
cache entries belong only to that problem/target lifetime.

Typed chemistry executes natively. Opaque Python callables receive the original
whole Atom/Bond only when reached; host sets/containers and attribute preflight
also retain their Python behavior. Missing noncallable attributes preflight
before callbacks, then leaves short-circuit in insertion order. Compilation
never runs user callbacks. Traversal/backtracking stay native; host reentry
acquires the GIL, and reached exceptions propagate.
