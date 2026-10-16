# OPTIONAL-010: transition-metal ligand-field spin ranking

Type: optional empirical rule. Planned screening default: **disabled**.

| Metadata | Value |
|---|---|
| Rule version | `draft-1` |
| Phase | `ELECTRONIC_STATE_RANKING` |
| Requires rules | OPTIONAL-009 compatible major version |
| Consumes | local `two_s` candidates, metal/OS, geometry and ligand-field class |
| Produces | electronic-state rank adjustments |
| Required providers/data | calibrated ligand-field domain classifier/reference set |
| Review state | `DRAFT` |
| Owner approval | pending specialist review |
| Primary basis | held-out metal/OS/geometry/ligand-field reference set required |
| Applicability artifact | calibrated domain classifier; not yet available |
| Candidate domain scope | not applicable; ranking-only rule |

## Theory and basis

For many $d$-block configurations, ligand-field strength and coordination
geometry influence high-/low-spin ordering. No universal ligand-name or
geometry rule covers all oxidation states, distortions and covalency.

## Implementation

- Operate only in a calibrated `(element, OS, d-count, geometry, ligand class)`
  domain.
- Rank candidates from OPTIONAL-009; never create parity-illegal states.
- Record geometry classifier, ligand-field evidence, training/reference domain
  and counterexamples.
- Abstain outside calibration, for non-innocent ligands or uncertain geometry.

## Acceptance

- Curated high-/low-spin fixtures are held out by metal/ligand class.
- Confidence and coverage are reported separately.
- Distorted/out-of-domain structures abstain.
- Disabling the rule preserves the candidate set and removes only its ranking.
