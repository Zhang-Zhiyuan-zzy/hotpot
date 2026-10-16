# DEFAULT-001: `STRICT` policy

Type: named default decision policy.
Runtime registry: `StatePolicyRegistry`.

## Theory and basis

The low-level library default should preserve uncertainty and avoid applying
domain priors that a caller did not request.

## Implementation

- Always run every hard rule.
- Preserve and validate attributable explicit/source values.
- Enable no optional empirical or AI rule.
- Never select neutral charge, a common oxidation state or a high/low spin
  merely to continue.
- Return candidates, `AMBIGUOUS` or `UNRESOLVED` when evidence is insufficient.
- For hard-valid `STRICT_ANALYSIS`, return decision action `NOT_REQUESTED`;
  analysis does not pretend that an execution choice failed. Hard validation
  runs first, so illegal/conflicting authoritative input still returns
  `INADMISSIBLE`/`CONFLICT`.
- Decide `NO_AUTOMATIC_STATE` when a backend needs a unique value that no
  attributable input selected; never pause for an interactive choice.

## Acceptance

- Empty `RuleSet` remains empty.
- `STRICT_ANALYSIS` returns `NOT_REQUESTED` with no selected identity.
- Identical explicit inputs produce identical results across entry points.
- A metal complex without sufficient evidence is not silently assigned $Q$,
  OS or a physical spin.
- An unresolved batch record terminates with `NO_AUTOMATIC_STATE` and does not
  prevent later records from being resolved.
- Hard-invalid input cannot be rescued by this policy.
