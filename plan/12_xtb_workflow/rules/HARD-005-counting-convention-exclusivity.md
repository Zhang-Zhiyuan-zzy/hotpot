# HARD-005: coordination-counting convention exclusivity

Type: hard, non-toggleable.

## Theory and basis

Ionic counting, donor-return cleavage and Green CBC assign boundary electrons
under different conventions. Applying two conventions to the same metal-
ligand electron pair double counts it.

## Implementation

- Represent the convention with an enum and immutable result identity.
- A state path selects at most one treatment for each coordination boundary.
- Ionic counting may expose bookkeeping ligand charges; donor-return records
  electron ownership; CBC records L/X/Z and donated electrons.
- Keep alternative conventions as separate candidates. Do not merge their
  fields into one generic fragment charge.

## Acceptance

- The same complex may have distinct convention-labelled results without
  changing intact-system $Q$.
- One coordination edge cannot contribute twice to an OS equation.
- CBC and donor-return outputs fail type checks when used as ionic charges.
- Every result serializes and restores its convention.
