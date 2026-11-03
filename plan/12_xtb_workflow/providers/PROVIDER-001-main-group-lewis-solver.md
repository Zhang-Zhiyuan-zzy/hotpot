# PROVIDER-001: supported main-group Lewis solver

Type: independently callable scientific inference provider. It is not an
empirical ranking rule and is not implicitly run by `STRICT`. The coordination
screening profile may call it conditionally for components inside its reviewed
main-group applicability domain.

## Theory and basis

For a fixed atom/H inventory and a supported ordinary main-group domain, bond
orders, nonbonding/radical electrons and atom formal charges can be solved
jointly under electron and charge constraints. Resonance-equivalent
localizations are one scientific class. The method is not universal for
organometallic, multicentre or unknown-protonation systems.

## Implementation

- Consume an immutable covalent-component view derived from
  `MolecularFacts.semantic_bonds`, a complete H inventory and, when known, a
  hard-valid component or system $Q$ constraint.
- The current `fixed-bond-main-group-v2` method requires a known site-total-H
  slot for every non-H atom, including known zero. A supplied explicit-H neighbor
  alone does not resolve an unknown site-total-H slot. Explicit H atoms are
  counted particles and do not require their own implicit-H assignment; known
  positive implicit H retains the existing nonduplication convention.
- After the existing unsupported-element and fixed-bond checks, an unresolved
  non-H site-total-H slot produces an empty `NOT_EVALUATED/OUT_OF_DOMAIN` result
  with `PROVIDER-001.INCOMPLETE_HYDROGEN_INVENTORY` and the affected atom indices.
  This enforces the complete-H applicability prerequisite without changing
  molecular facts, net-charge evidence, charge/electron accounting, coordination
  exclusion, enumeration or budgets.
- For a coordination complex, solve only the coordination-separated main-group
  ligand/counterion component. Do not treat a metal--ligand coordination edge
  as an ordinary covalent valence contribution and do not mutate the source
  graph.
- Generate joint Lewis allocations rather than guessing atom charges
  independently.
- Enforce electron conservation and $\sum_i q_i=Q$ for complete candidates;
  quotient resonance/symmetry classes and use deterministic limits.
- When $Q$ is absent, propose a net-charge value only inside a separately
  validated closed-shell main-group component domain; return component-charge
  candidates for the joint composer rather than deciding the intact complex
  charge. Otherwise return candidates or abstain.
- Resource-limited search is labelled truncated, never globally unique.
- Return `CandidateDomainScope` and `MaterializationStatus` with every
  charge/Lewis result. Closed-shell is a declared solver-domain constraint,
  not a preference that one
  common charge is chemically true. Within that domain, return all non-
  equivalent satisfying Q/Lewis classes; optional rules and default policies
  rank/select them later.

## Acceptance

- Ordinary neutral/ionic main-group and carboxylate/nitro/azide resonance
  fixtures are correct modulo equivalence.
- Atom order and equivalent Kekule forms preserve results.
- Equivalent source encodings that resolve to the same semantic coordination
  boundary preserve ligand-component results.
- Radical, hypervalent, electron-deficient and out-of-domain cases abstain or
  retain explicit alternatives.
- Search limits preserve honest `TRUNCATED/RESOURCE_LIMIT` status.
- A singleton result is reported scientifically unique only when the searched
  candidate domain is `EXHAUSTIVE`.

Reference algorithmic basis: Welsh and Allison, *J. Cheminform.* 2019, 11,
18, DOI `10.1186/s13321-019-0340-0`, plus versioned Hotpot domain templates.
