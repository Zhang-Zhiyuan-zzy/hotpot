# RECOVERY-001: legal-state retry ordering

Type: optional pipeline recovery policy. Default: **disabled**.

## Theory and basis

Failure to converge in one legal electronic state may justify trying another
independently legal candidate. Convergence does not rank physical states.

## Implementation

- The recovery policy requests a bounded, lazily expanded sequence of remaining
  hard-admissible execution paths from the factorized candidate space; it does
  not require all alternatives to be materialized in advance.
- `RecoveryBudget` limits backend attempts independently from the scientific
  `CandidateBudget`; requesting a new path still uses the recorded candidate-
  search frontier and certification rules.
- The pipeline controller decides whether to retry; scientific inference rules
  never control this transition.
- Preserve one workspace/result per attempt; never overwrite the first failure.
- Do not create new charge/OS/spin values from error text.

## Acceptance

- Illegal candidates never enter the retry list.
- Retry is opt-in, bounded and independently attributable.
- Success of a later attempt does not relabel earlier candidates physically
  wrong or the successful state a ground state.
- Disabling the recovery policy performs no extra backend work.
