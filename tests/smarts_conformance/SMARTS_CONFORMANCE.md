# Hotpot SMARTS conformance contract

This document separates the intended contract from observations of the current
implementation. A currently observed defect is never promoted to the contract
merely to make a test pass.

## Scope and entry points

The active implementation is
`hotpot.cheminfo.search.smarts.substructure_from_smarts`, reached publicly as
`hotpot.Substructure.from_smarts`. Matching is performed by
`hotpot.cheminfo.search.Searcher.search` on a Hotpot `Molecule`, and
`Molecule.search_substructure` is the convenience entry point. Tests and
coverage in this suite deliberately exclude the older, unreferenced parser
implementations in `hotpot/cheminfo/parse_smarts.py`,
`hotpot/cheminfo/search/_smarts.py`, and
`hotpot/cheminfo/search/smarts_parser/`.

The target dialect has three profiles:

- `hotpot_core`: a documented Daylight-like subset implemented on Hotpot's
  NetworkX molecular graph.
- `hotpot_extension`: the coordination-chemistry tokens `M`, `Ln`, `An`,
  `NPn[-m]`, and `NGn[-m]`.
- `compatibility_audit`: valid constructs or semantics implemented by other
  engines but not promised by Hotpot. These do not silently become core.

Reaction SMARTS, component-level grouping, query serialization, indexing,
fingerprint prefilters, `maxMatches`, and a chirality option are outside the
current API.

## Target preparation

Targets are read through `hotpot.read_mol(text, "smi")`. This path uses Open
Babel to parse SMILES and populate atomic number, formal charge, aromaticity,
bond order, and implicit-hydrogen fields. Hotpot then matches its own Atom and
Bond objects; RDKit objects are not used by the production matcher.

Ring membership and ring sizes are derived from `networkx.cycle_basis` through
`Molecule.rings`. Thus `R<n>` and `r<n>` in fused, bridged, spiro, and cage
systems are explicitly ring-model-sensitive. Only unambiguous acyclic and
single-ring cases are core assertions. Polycyclic differences are classified
as `ring_model_disagreement` until a ring-set contract is selected.

Target preprocessing failure is distinct from query rejection and is exposed
as `OSError` by the current public reader.

## Core atoms and atom expressions

The target core supports element identities, `[#n]`, `*`, `[a]`, `[A]`, formal
charge, atom maps, `D`, `X`, `v`, `H`, `R`, and `r`.

- `D<n>` is the number of explicit graph neighbours.
- `X<n>` is explicit graph neighbours plus `Atom.implicit_hydrogens`.
- `v<n>` is the sum of graph bond orders plus implicit hydrogens.
- `H<n>` in an atom-identity context is explicit graph hydrogens plus implicit
  hydrogens. A standalone `[H]` is elemental hydrogen.
- `R` means membership in any Hotpot cycle-basis ring; `R<n>` counts those
  rings. `R0` means no such ring.
- `r` means membership in any Hotpot cycle-basis ring; `r<n>` means membership
  in at least one basis ring of size `n`. The desired core meaning of `r0` is
  acyclic, although the current implementation does not satisfy it.
- Atom-map numbers are metadata. They preserve query labels and never constrain
  a match.

Logical precedence is `!` then high-precedence/implicit AND (`&` and adjacent
primitives), then OR (`,`), then low-precedence AND (`;`). Repeated negation is
allowed. Recursive `$()` expressions are anchored at query atom zero of the
recursive query. An empty recursive query is invalid.

Lowercase `h` and ring-connectivity `x` are currently unsupported, not invalid
syntax. Isotopes and atom chirality `@`/`@@` are also unsupported. Rejection of
an unsupported feature must remain distinguishable from malformed syntax.

## Bonds and graph syntax

Core bond tokens are implicit bond, `-`, `=`, `#`, `:`, `~`, and the currently
implemented comma-separated bond alternatives. Directional `/` and `\\` bonds,
ring-bond `@`/`!@`, and general bond negation are unsupported.

The intended implicit bond contract is:

- between two explicitly aromatic atoms, require an aromatic bond;
- otherwise accept a single or aromatic bond, matching the parser's documented
  Daylight-like default.

Explicit `-`, `=`, and `#` require a non-aromatic bond of the requested order.
An aromatic bond's internal Kekule order must not make it satisfy an explicit
single or double query.

Branches, one-digit and `%nn` ring closures, ring-label reuse after closure,
and dot-separated disconnected queries are core. A query and every dot
component must contain at least one atom. Leading, trailing, or consecutive
dots, leading/dangling bonds, empty branches, self-loop ring closures, duplicate
query edges, and unclosed branches/rings are invalid.

A disconnected query is matched as one disconnected query graph against a
single Hotpot `Molecule`; no component-level grouping syntax is implemented.

## Hotpot extensions

- `M`: atoms for which Hotpot's `Atom.is_metal` is true.
- `Ln`: atomic numbers 57 through 71.
- `An`: atomic numbers 89 through 103.
- `NPn[-m]`: inclusive period number or period range, bounded to 1 through 7.
- `NGn[-m]`: inclusive group number or group range, bounded to 1 through 18.

These strings must not be fed directly to RDKit as an oracle: several are
valid RDKit expressions with unrelated meanings. Extension truth is checked
against Hotpot element metadata and frozen atomic-number sets.

## Enumeration and result identity

An embedding is a tuple of target atom indices in query-atom order. Sorting is
allowed only across the outer collection; indices inside an embedding are not
sorted. A target atom set is a separate, order-free `frozenset`.

`Searcher.search` currently groups all raw query automorphisms that have the
same target atom set into one `Hit`. `Hit.mappings` retains the read-only
query-index-to-target-index mappings. The intended meaning of `Hit.bonds` is the
target bonds corresponding to query edges; the current induced-subgraph
behavior is tracked as a contract defect.

The API currently enumerates eagerly and has no truncation flag or result
limit. The adapter reports these capabilities as unsupported rather than
inventing values. A test-only batch adapter is a deterministic linear scan and
is not represented as a native index.

## Error contract

- malformed query text: `ValueError` during lexing or query compilation;
- recognized but unsupported feature: `NotImplementedError`;
- invalid target input: `OSError` during target preparation;
- unknown exceptions: propagated by the conformance adapter and treated as
  crashes, never converted to a successful result.

Diagnostics are compared by stable phase and exception type. Exact prose is
recorded for evidence but is not a compatibility guarantee.
When a current parser diagnostic explicitly contains a position, the test
adapter reports it as a zero-based character offset; otherwise
`error_position` is `None` rather than an invented location.

## Known deviations under test

The conformance audit currently covers, without fixing, empty queries and empty
components, leading bonds, self/duplicate ring edges, empty recursion,
uncontrolled `IndexError`, bare `a`/`A`, standard element-token prefix
collisions, lowercase `h`, explicit bonds matching aromatic bonds, `r0`, induced
`Hit.bonds`, and eager factorial enumeration. Polycyclic `R<n>/r<n>` cases and
ambiguous hydrogen syntax are reported separately as dialect/model questions.
