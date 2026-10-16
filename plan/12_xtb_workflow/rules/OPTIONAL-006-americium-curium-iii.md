# OPTIONAL-006: ordinary Am(III)/Cm(III) prior

Type: optional empirical rule. Planned screening default: **enabled**.

| Metadata | Value |
|---|---|
| Rule version | `draft-1` |
| Phase | `OXIDATION_STATE_CANDIDATE` |
| Requires rules | none; consumes any compatible counting source |
| Consumes | metal identity/environment, semantic coordination boundary, local ligand/counting candidates, explicit/source OS candidates |
| Produces | Am/Cm(III) OS proposal/rank evidence |
| Required providers/data | versioned Am/Cm redox-domain registry |
| Review state | `DRAFT` |
| Owner approval | pending individual review |
| Primary basis | source-tagged Am/Cm aqueous and donor-coordination review set |
| Applicability artifact | Am/Cm oxidation/redox domain registry; digest not yet frozen |
| Candidate domain scope | `DOMAIN_BOUNDED` |

## Theory and basis

Am(III) and Cm(III) are common in ordinary aqueous/donor coordination
chemistry, while other oxidation states exist under specialized redox
conditions.

## Implementation

- Apply only to recognized Am/Cm donor-complex profiles with fixed boundary,
  sufficient local ligand/counting evidence and no explicit actinyl/redox
  motif.
- Propose/rank local `+3`; retain every hard-valid source or redox alternative
  before total system charge is selected.
- Store the `+3` preference as deferred rank evidence; apply it only after
  charge--OS composition and coupled hard validation.
- Never derive system $Q$ from this prior alone; the joint composer requires a
  complete compatible ligand/counterion assignment.
- Do not claim backend support: element/method capability is checked later.
- Attach domain, source and redox caveats.

## Acceptance

- Ordinary curated Am/Cm donor complexes rank `+3`.
- Explicit non-III or redox-active cases are not overwritten.
- Stable xTB backend limitations remain `BACKEND_UNSUPPORTED`, not OS conflict.
- Full assignments satisfy charge conservation.
