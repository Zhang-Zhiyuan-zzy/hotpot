# Hotpot SMARTS conformance contract

This document defines the supported behavior of the active NetworkX-backed
SMARTS implementation. Observations from RDKit or Open Babel are evidence, not
automatic changes to this contract.

## Scope and entry points

The active compiler is
`hotpot.cheminfo.search.smarts.substructure_from_smarts`, exposed through
`hotpot.Substructure.from_smarts`. Matching remains the responsibility of
`Searcher`, `Hits`, and `Hit`; `Molecule.search_substructure` is the convenience
entry point. The legacy parser modules are outside this conformance target.

The supported dialect consists of:

- `hotpot_core`: the documented Daylight-like subset;
- `hotpot_extension`: `M`, `Ln`, `An`, `NPn[-m]`, and `NGn[-m]`;
- two named target-semantics profiles, `FULL_GRAPH` and
  `LIGAND_SKELETON`;
- `compatibility_audit`: reference-engine behavior not promised by Hotpot.

Reaction SMARTS, component grouping, query serialization, fingerprint
prefilters, chirality matching, and a native molecular index are outside the
current API. Bounded raw-embedding enumeration is available through
`Searcher.iter_mappings(..., max_matches=n)` and
`Searcher.search(..., max_matches=n)`; truncation is explicit.

## Target preparation and bond metadata

SMILES and structure files are read through Hotpot's normal readers. Open
Babel supplies atom identity, charge, aromaticity, numeric bond order, and
implicit hydrogen perception; the matcher then operates only on Hotpot
`Molecule`, `Atom`, and `Bond` objects with NetworkX.

`Bond.bond_kind` preserves semantic categories separately from numeric order:
`SINGLE`, `DOUBLE`, `TRIPLE`, `AROMATIC`, `ZERO`, `DATIVE`, and `UNKNOWN`.
`bond_direction`, `bond_source`, and source metadata are retained where the
source backend exposes them. No matcher may silently reinterpret `UNKNOWN` as
single or dative.

Open Babel 3.1 loses information for MOL2 `du`, `un`, and `nc`: all three are
reported as numeric order zero with indistinguishable flags. Hotpot therefore
records them as `BondKind.UNKNOWN`, not `ZERO` or `DATIVE`. A MOL2 bond token
`1` is perceived as `SINGLE`. This is an input-representation limitation and is
covered by fixed fixtures; it is not repaired by guessing chemistry.

## Named semantics profiles

The keyword-only `semantics` argument is accepted by
`Substructure.from_smarts`, `substructure_from_smarts`,
`parse_bracket_atom`, and `Molecule.search_substructure`. It accepts
`SmartsSemantics` or its exact string value.

### `FULL_GRAPH`

`FULL_GRAPH` is the default and preserves prior callers:

- `D<n>` is the number of explicit graph neighbours;
- `X<n>` is that degree plus `Atom.implicit_hydrogens`;
- `v<n>` is the numeric bond-order sum plus implicit hydrogens;
- `R`/`R<n>` and `r`/`r<n>` use `Molecule.rings`, whose current ring model is
  `networkx.cycle_basis`.

Metal--ligand edges therefore affect donor degree/connectivity and may create
full-graph chelate rings.

### `LIGAND_SKELETON`

`LIGAND_SKELETON` is a non-mutating descriptor view. It does not delete edges
from the molecule and does not change the graph traversed by `Searcher`:

- for a non-metal atom, `D`, `X`, and `v` exclude bonds for which
  `Bond.is_metal_ligand_bond` is true;
- for a metal atom, `D` and `X` retain all incident graph edges, so
  coordination queries such as `[M;X6]` remain expressible;
- ligand-profile `v` counts only `SINGLE`, `DOUBLE`, `TRIPLE`, and `AROMATIC`
  kinds; `DATIVE`, `ZERO`, and `UNKNOWN` contribute zero;
- `R`/`r` use `Molecule.ligand_rings`, which obtains a cycle basis after
  filtering metal--ligand edges on a copied graph.

The profile does not rerun hydrogen perception. Both `X` and `v` use the
existing `Atom.implicit_hydrogens` supplied by the reader. In the frozen
fixtures Open Babel 3.1 gives a coordinated amine donor zero implicit H from
MOL2 and one from SDF despite identical numeric bond topology; after removal of
the metal edge the ligand-profile values are respectively `X3/v3` and
`X4/v4`.

`Bond.is_metal_ligand_bond` is a topological metal--nonmetal predicate, not a
claim that every such bond is dative. Consequently this profile intentionally
describes a ligand skeleton; organometallic covalent semantics may require a
future separately named profile rather than a Boolean switch.

Recursive `$()` expressions inherit the parent profile. The recursive cache
signature includes atom state, graph connectivity, numeric order,
`BondKind`, and aromaticity so metadata changes cannot reuse a stale match.

## Core atoms and atom expressions

The core supports element symbols, `[#n]`, `*`, bare/bracket `a` and `A`,
formal charge, atom maps, `D`, `X`, `v`, `H`, `R`, `r`, Boolean logic, and
anchored recursion.

- `H<n>` in an identity context is explicit plus implicit hydrogen count; a
  standalone `[H]` denotes elemental hydrogen.
- `R` means membership in any selected-view cycle-basis ring and `R<n>` counts
  those rings.
- `r` means ring membership, `r<n>` means membership in a selected-view basis
  ring of size `n`, and `r0` means acyclic in that view.
- atom maps are query metadata and never constrain a match.

Logical precedence is `!`, high-precedence/implicit AND (`&` and adjacency),
OR (`,`), then low-precedence AND (`;`). Empty recursion is invalid.
Lowercase `h`, ring-connectivity `x`, isotopes, and atom chirality are
recognized as unsupported and raise `NotImplementedError`.

## Bonds and graph syntax

Core bond tokens are implicit bond, `-`, `=`, `#`, `:`, `~`, and implemented
comma-separated alternatives. Directional `/` and `\\`, ring-bond `@`/`!@`,
and general bond negation are unsupported.

- `-`, `=`, and `#` match exactly `BondKind.SINGLE`, `DOUBLE`, and `TRIPLE`.
- An implicit aliphatic single uses the same `SINGLE` test; it cannot bypass
  the `BondKind` contract.
- `:` matches aromatic bonds and `~` matches every existing edge, including
  `DATIVE`, `ZERO`, and `UNKNOWN`.
- Between two explicitly aromatic atoms, an omitted bond requires aromaticity;
  otherwise the documented implicit default accepts semantic single or
  aromatic.

Branches, one-digit and `%nn` ring closures, ring-label reuse, and disconnected
dot components are supported. Empty components, leading/dangling bonds, empty
branches, self-loops, duplicate query edges, and unclosed branches/rings are
invalid.

## Hotpot extensions

- `M`: `Atom.is_metal` is true.
- `Ln`: atomic numbers 57--71.
- `An`: atomic numbers 89--103.
- `NPn[-m]`: inclusive period or period range, bounded to 1--7.
- `NGn[-m]`: inclusive group or group range, bounded to 1--18.

These extensions must not be sent directly to another toolkit as an oracle:
some strings have different meanings there.

## Enumeration and result identity

An embedding maps query indices to target indices. `Searcher.search` groups
query automorphisms with the same target atom set into one `Hit`, while
`Hit.mappings` retains all read-only mappings. `Hit.bonds` contains only target
bonds corresponding to query edges; `Hit.induced_bonds` separately exposes all
target edges induced by the hit atoms.

`Searcher.has_match` is the existence fast path. `iter_mappings` streams raw
embeddings, and `max_matches` bounds raw embeddings rather than unique grouped
hits. `Hits.truncated` reports whether more mappings existed; truncation is
never silent.

## MCA applicability boundary

MCA nucleophilic-site rules compile with `LIGAND_SKELETON`, allowing organic
motifs to be interpreted independently of coordination-induced degree and ring
changes. Site eligibility is a separate model-domain policy: metal centres and
atoms directly bound to a metal are excluded from `Molecule.mca_sites`.
Per-atom MCA inference may still populate `Atom.mca`; that does not certify the
atom as a reliable site.

## Error contract

- malformed query text: `SmartsSyntaxError`, a `ValueError` subclass;
- recognized unsupported syntax: `UnsupportedSmartsError`, a
  `NotImplementedError` subclass;
- invalid target input: target-reader error, currently normally `OSError`;
- unknown exceptions: propagated, never converted to a successful match.

## Current conformance status

At revision `7b262a9`, the strict `smarts_core` suite passes on Python 3.9 and
3.14: **252 passed** on each interpreter. The deterministic corpus contains
**1,332/1,332 passing cases** and `corpus/known_mismatches.json` is empty.
These numbers describe the scoped SMARTS suite, not all repository tests.

Polycyclic cycle-basis behavior, Open Babel aromaticity/hydrogen perception,
MOL2 coordination-token loss, unsupported SMARTS features, and the deliberate
ligand-skeleton treatment of organometallic bonds remain documented boundaries,
not hidden fallbacks.
