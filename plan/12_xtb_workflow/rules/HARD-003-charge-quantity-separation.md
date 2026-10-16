# HARD-003: charge-quantity separation

Type: hard, non-toggleable.

## Theory and basis

Net charge, Lewis formal charge, original-component charge, oxidation state,
population/partial charge and coordination electron counting answer different
questions. Numeric equality in one example does not make them interchangeable.

## Implementation

- Use distinct result types and field names for every quantity.
- Do not round partial charges into formal or net charges.
- Do not turn ionic, donor-return or CBC ligand bookkeeping into intact-system
  component charges.
- Do not use a metal oxidation state as its atom formal charge unless an
  explicit, validated mapping states that convention.
- Every conversion operation names its convention and provenance.

## Acceptance

- `NH3->BF3` retains neutral donor-return fragments and the distinct intact-
  graph formal-charge description.
- CBC L/X/Z output cannot populate an ionic ligand-charge field.
- Backend partial charges cannot overwrite source formal charges or $Q$.
- Type-level tests prevent accidental cross-assignment.
