# OPTIONAL-002: common disconnected monoatomic ions

Type: optional empirical rule. Planned screening default: **enabled**.

| Metadata | Value |
|---|---|
| Rule version | `draft-1` |
| Phase | `CHARGE_EVIDENCE` |
| Requires rules | none |
| Consumes | `MolecularFacts` and original-component identities |
| Produces | original-component charge candidates and evidence |
| Required providers/data | versioned monoatomic-ion registry |
| Review state | `DRAFT` |
| Owner approval | pending individual review |
| Primary basis | curated common-ion records with source per element/charge |
| Applicability artifact | monoatomic-ion registry; source release and digest not yet frozen |
| Candidate domain scope | `DOMAIN_BOUNDED` |

## Theory and basis

Many explicit one-atom salt components have well-established dominant ionic
charges. This is useful only when the computational boundary and disconnected
component identity are explicit.

## Implementation

- Match one-atom original components, not atoms temporarily disconnected by
  force-field or coordination operations.
- Read candidate charges from a versioned source registry. Preserve explicit
  isotope/formal-charge/source evidence at higher authority.
- Propose only the matched original-component charge. The joint composer may
  derive system $Q$ only after every required component assignment is complete;
  this rule never promotes one ion's charge to system net charge.
- Do not infer oxidation state for a bonded metal center.
- Ambiguous/multivalent entries produce candidates or abstain.

## Acceptance

- Common alkali, alkaline-earth and halide counterions match source-backed
  values.
- A coordinated Na/Cl atom is not classified as a disconnected counterion.
- Multivalent transition-metal atoms do not receive an arbitrary unique value.
- Component and molecular sums are hard-validated.
