# OPTIONAL-011: f-block Hund/free-ion execution prior

Type: optional empirical rule. Planned screening default: **conditionally
enabled inside the reviewed applicability registry**.

| Metadata | Value |
|---|---|
| Rule version | `draft-1` |
| Phase | `ELECTRONIC_STATE_RANKING` |
| Requires rules | OPTIONAL-009 compatible major version |
| Consumes | f-block local `two_s` candidates on a hard-valid charge/OS path |
| Produces | execution-prior rank adjustments |
| Required providers/data | versioned f-block free-ion/SOC applicability registry |
| Review state | `DRAFT` |
| Owner approval | pending f-block specialist review |
| Primary basis | NIST ionic/free-ion terms plus domain-specific validation |
| Applicability artifact | f-block execution-prior/SOC caveat registry; digest not yet frozen |
| Candidate domain scope | not applicable; ranking-only rule |

## Theory and basis

For many localized $4f^n/5f^n$ ions, a maximum-spin/free-ion-derived state is
a useful execution starting point. Strong spin-orbit coupling and crystal-
field effects mean spin-only $2S$ and any projected UHF are not a complete
physical $J$-state model.

## Implementation

- Consume reviewed ionic-configuration candidates from OPTIONAL-009.
- Rank the maximum-spin candidate only in a versioned element/OS/domain table.
- Label it `EXECUTION_PRIOR`, retain all lower-spin/coupling alternatives and
  record the spin-orbit limitation.
- Explicit/source spin evidence outranks this rule.
- Outside the reviewed localized-ion domain, abstain so DEFAULT-004 remains the
  final legal execution fallback.

## Acceptance

- In its reviewed ordinary localized-ion domain, Eu(III) ranks `two_s=6` while
  retaining `0,2,4` as scientific alternatives.
- The historical Eu recovery is treated as execution evidence, not ground-
  state proof.
- Unsupported/strongly covalent or ambiguous-OS systems abstain.
- Every result passes HARD-006.
