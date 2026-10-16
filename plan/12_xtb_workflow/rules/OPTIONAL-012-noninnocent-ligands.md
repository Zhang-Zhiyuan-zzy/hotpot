# OPTIONAL-012: non-innocent ligand alternatives

Type: optional empirical rule. Planned screening default: **disabled**.

| Metadata | Value |
|---|---|
| Rule version | `draft-1` |
| Phase | `REDOX_CANDIDATE` |
| Requires rules | none |
| Consumes | molecular facts, net/Lewis and coordination-counting evidence |
| Produces | correlated ligand-redox, local OS and radical/spin candidate factors |
| Required providers/data | versioned non-innocent motif/protonation registry |
| Review state | `DRAFT` |
| Owner approval | pending motif-by-motif review |
| Primary basis | Luca--Crabtree, DOI `10.1039/C2CS35228A`, plus primary motif sources |
| Applicability artifact | redox/protonation motif registry; digest not yet frozen |
| Candidate domain scope | `DOMAIN_BOUNDED` |

## Theory and basis

NO, O2, dithiolene, quinone and related ligands can support multiple ligand-
centered redox/radical descriptions for one connectivity. A metal-only
oxidation-state assignment may therefore be non-unique.

## Implementation

- Use reviewed whole-motif templates including protonation, bond semantics,
  charge and radical alternatives.
- Action emits one `CandidateProposal` containing correlated ligand-redox,
  local-OS and radical/spin factors plus proposal-guarded compatibility
  constraints. The joint composer performs bounded path expansion and
  conservation checks; this rule does not independently select one path or
  overwrite $Q$.
- Equivalent resonance localizations collapse to one scientific path.
- Keep disabled until motif-specific expert tests and source snapshots exist.

## Acceptance

- Curated NO/O2/dithiolene/quinone examples retain expected alternatives.
- Innocent lookalikes and wrong protonation do not match.
- Each alternative conserves charge and records radical evidence.
- Representation-equivalent inputs yield equivalent candidate sets.
