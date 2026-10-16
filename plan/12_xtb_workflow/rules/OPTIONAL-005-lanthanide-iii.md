# OPTIONAL-005: ordinary lanthanide(III) prior

Type: optional empirical rule. Planned screening default: **enabled**.

| Metadata | Value |
|---|---|
| Rule version | `draft-1` |
| Phase | `OXIDATION_STATE_CANDIDATE` |
| Requires rules | none; consumes any compatible counting source |
| Consumes | metal identity/environment, semantic coordination boundary, local ligand/counting candidates, explicit/source OS candidates |
| Produces | Ln(III) OS proposal/rank evidence |
| Required providers/data | versioned Ln domain/exception registry |
| Review state | `DRAFT` |
| Owner approval | pending individual review |
| Primary basis | source-tagged lanthanide coordination/oxidation-state review set |
| Applicability artifact | Ln(II/III/IV) domain and exception registry; digest not yet frozen |
| Candidate domain scope | `DOMAIN_BOUNDED` |

## Theory and basis

The trivalent state dominates much molecular lanthanide coordination
chemistry, but stable Eu(II), Yb(II), Sm(II), Ce(IV), Tb(IV) and other
exceptions prevent it from being a hard or universal oxidation-state default.

## Implementation

- Apply only to recognized lanthanide centers in a non-redox coordination
  profile with sufficient local ligand/counting context.
- Propose/rank local `+3`; retain source-supported `+2/+4` and explicit
  alternatives before total system charge is selected.
- Store the `+3` preference as deferred rank evidence; it is applied only after
  charge--OS composition and coupled hard validation.
- Never derive system $Q$ from the metal prior alone.
- Let the joint composer combine the local OS proposal with complete ionic
  ligand and counterion assignments; only scientifically complete ionic-
  accounting paths may derive $Q$ under the declared convention.
- Redox-active ligands, element-specific exception motifs or contradictory
  source facts force alternatives or abstention.

## Acceptance

- Ordinary Ln donor complexes rank `+3` without deleting alternatives.
- Eu/Yb/Sm(II), Ce/Tb(IV) and explicit conflicting fixtures are not forced to
  `+3`.
- All complete assignments obey HARD-004.
- The same local OS proposal can participate in more than one charge path
  without being copied or independently selected.
- Result is labelled empirical, not authoritative.
