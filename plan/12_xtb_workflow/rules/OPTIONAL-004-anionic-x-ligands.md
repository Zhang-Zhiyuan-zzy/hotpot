# OPTIONAL-004: ordinary anionic X-type ligand prior

Type: optional empirical rule. Planned screening default: **enabled**.

| Metadata | Value |
|---|---|
| Rule version | `draft-1` |
| Phase | `COORDINATION_COUNTING` |
| Requires rules | none |
| Consumes | fixed protonation/coordination-boundary facts, compatible Lewis or motif evidence, ionic-counting request |
| Produces | ionic `CoordinationFragmentAssignment` candidates |
| Required providers/data | versioned X-ligand/protonation registry |
| Review state | `DRAFT` |
| Owner approval | pending individual review |
| Primary basis | ionic counting plus Green CBC, DOI `10.1016/0022-328X(95)00508-N` |
| Applicability artifact | X-ligand/protonation registry; digest not yet frozen |
| Candidate domain scope | `DOMAIN_BOUNDED` |

## Theory and basis

With fixed protonation and an ionic counting convention, halide, hydride,
hydroxide, alkoxide, carboxylate and cyanide commonly contribute a $-1$
bookkeeping charge.

## Implementation

- Match curated whole-ligand motifs, not a donor atom in isolation.
- Operate on ligand components obtained from the immutable semantic
  coordination-boundary view; do not delete or reinterpret bonds in the source
  `Molecule`.
- Require explicit H/protonation and compatible formal-charge/resonance facts.
- Action proposes an ionic ligand bookkeeping candidate and source evidence
  that the joint composer may combine with metal OS and counterion candidates.
- Ambident, bridging, radical or non-innocent cases retain alternatives or
  abstain.

## Acceptance

- Representative halide, hydride, hydroxide, alkoxide, carboxylate and cyanide
  fixtures propose $-1$.
- Neutral acid/alcohol forms do not match their deprotonated rule.
- Carboxylate resonance/Kekule forms give one equivalent result.
- System charge is changed only through a separate hard-valid composition.
- Equivalent metal-bond encodings that resolve to the same semantic
  coordination boundary yield the same ligand assignment.
