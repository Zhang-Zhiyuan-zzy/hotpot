# OPTIONAL-007: ordinary Th(IV) prior

Type: optional empirical rule. Planned screening default: **enabled**.

| Metadata | Value |
|---|---|
| Rule version | `draft-1` |
| Phase | `OXIDATION_STATE_CANDIDATE` |
| Requires rules | none; consumes any compatible counting source |
| Consumes | metal identity/environment, semantic coordination boundary, local ligand/counting candidates, explicit/source OS candidates |
| Produces | Th(IV) OS proposal/rank evidence |
| Required providers/data | versioned Th domain/exception registry |
| Review state | `DRAFT` |
| Owner approval | pending individual review |
| Primary basis | source-tagged molecular thorium oxidation-state review set |
| Applicability artifact | Th(II/III/IV) domain/exception registry; digest not yet frozen |
| Candidate domain scope | `DOMAIN_BOUNDED` |

## Theory and basis

Th(IV) is the dominant state in much molecular thorium coordination chemistry,
but lower-valent thorium chemistry exists and must remain representable.

## Implementation

- Apply to ordinary non-redox Th coordination profiles after ligand/counting
  candidates exist.
- Propose/rank local `+4` without requiring or deriving a preselected $Q$, and
  without deleting lower-valent candidates.
- Store the `+4` preference as deferred rank evidence; apply it only after
  charge--OS composition and coupled hard validation.
- Let the joint composer combine the proposal with complete ligand and
  counterion assignments under one named convention.
- Explicit/source OS and recognized low-valent/redox motifs outrank or disable
  the prior.

## Acceptance

- Ordinary curated Th(IV) complexes rank `+4`.
- Explicit low-valent examples retain their state.
- The rule abstains when ligand accounting cannot form a legal assignment.
- Backend element support remains a separate decision.
