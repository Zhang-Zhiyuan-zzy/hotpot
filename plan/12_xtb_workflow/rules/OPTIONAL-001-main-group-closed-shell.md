# OPTIONAL-001: supported main-group closed-shell prior

Type: optional empirical rule. Planned screening default: **conditionally
enabled for supported main-group molecules/components after individual
validation**.

| Metadata | Value |
|---|---|
| Rule version | `draft-1` |
| Phase | `ELECTRONIC_STATE_CANDIDATE` |
| Requires rules | none |
| Consumes | `MolecularFacts`, one hard-valid partial charge path, complete supported `LewisFormalChargeResult` |
| Produces | `SpinContributionCandidate(two_s=0)` with `MOLECULAR` or `COMPONENT` scope |
| Required providers/data | PROVIDER-001 result or authoritative equivalent Lewis result |
| Review state | `DRAFT` |
| Owner approval | pending individual review |
| Primary basis | complete supported Lewis allocation; Welsh--Allison algorithmic reference |
| Applicability artifact | main-group template/counterexample registry; digest not yet frozen |
| Candidate domain scope | `DOMAIN_BOUNDED` |

## Theory and basis

A complete, even-electron Lewis allocation with no radical electrons in a
validated ordinary main-group domain supports a singlet execution prior. Even
electron parity alone does not prove a closed shell.

## Implementation

- Applicability requires one hard-valid partial path carrying legal component
  charge, a complete supported Lewis allocation, no radical evidence, and no
  transition/f-block atom inside the evaluated component, non-innocent,
  multicentre or unsupported hypervalent domain. The path need not be the final
  selected state.
- For a standalone main-group molecule, action proposes
  `SpinContributionCandidate(scope=MOLECULAR, two_s=0)`. For a semantically
  coordination-separated ligand/counterion component, it uses
  `scope=COMPONENT`; the electronic-state composer, not this rule, couples the
  contribution with metal/local-spin candidates and emits complete
  `ElectronicStateCandidate` objects.
- It abstains rather than treating every even-electron system as closed shell.
- Domain templates and counterexamples are versioned data.

## Acceptance

- Water and ordinary saturated/aromatic closed-shell fixtures propose
  `two_s=0`.
- O2, carbenes and biradicals do not match merely by parity. A metal complex is
  never assigned total `two_s=0` from ligand evidence, although an independently
  validated ligand component may receive component `two_s=0`.
- Atom ordering and equivalent Kekule forms do not change applicability.
- Proposal passes HARD-006 and never overwrites explicit spin.
