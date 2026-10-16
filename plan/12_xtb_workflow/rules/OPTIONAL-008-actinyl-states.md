# OPTIONAL-008: actinyl oxidation-state candidates

Type: optional empirical rule. Planned screening default: **disabled**.

| Metadata | Value |
|---|---|
| Rule version | `draft-1` |
| Phase | `OXIDATION_STATE_CANDIDATE` |
| Requires rules | none |
| Consumes | semantic actinyl facts, net-charge candidates, counting and geometry evidence |
| Produces | element-specific actinyl OS candidates and ranks |
| Required providers/data | versioned actinyl topology/redox/protonation registry |
| Review state | `DRAFT` |
| Owner approval | pending specialist review |
| Primary basis | element-specific actinyl spectroscopy/structure review set |
| Applicability artifact | actinyl topology/redox/protonation registry; digest not yet frozen |
| Candidate domain scope | `DOMAIN_BOUNDED` |

## Theory and basis

A recognized near-linear O=An=O actinyl unit provides strong but element- and
environment-dependent evidence for high actinide oxidation states. Geometry
alone is not sufficient, and multiple redox/protonation states are possible.

## Implementation

- Require an explicit semantic actinyl motif plus element-specific,
  source-tagged candidate data; geometry is corroborating evidence only.
- Propose a candidate set and ranking, never a universal unique state.
- Handle oxo protonation, bent structures, reduced actinyl and ligand-centered
  redox as explicit alternatives/out-of-domain paths.
- Keep the rule disabled until specialist fixtures and sources are approved.

## Acceptance

- Curated U/Np/Pu actinyl cases reproduce reviewed candidate sets.
- Linear O-An-O geometry without compatible topology does not match.
- Reduced/protonated counterexamples retain alternatives.
- Every proposed full assignment obeys HARD-004/HARD-005.
