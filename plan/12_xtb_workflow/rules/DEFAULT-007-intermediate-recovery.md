# DEFAULT-007: finite intermediate warning and recovery

Type: pipeline continuation decision policy.
Runtime registry: `PipelineRecoveryPolicyRegistry`.

## Theory and basis

A low-level optimizer can produce a finite but chemically poor intermediate
that a later method may repair. Intermediate quality is evidence, not always a
terminal product decision.

## Implementation

- Preserve `stage_input`, finite `stage_final`, `last_finite` and
  `best_recoverable`; full trajectory remains opt-in.
- A finite quality failure produces a warning and deterministic recovery-frame
  selection unless the route declares a hard intermediate gate.
- Non-finite/singular frames roll back and do not propagate.
- Final acceptance remains strict and independent.
- Do not compare raw energies across different methods when selecting a frame.

## Acceptance

- Finite warning cases can continue and retain the warning.
- Non-finite frames select an earlier finite frame or stop the route.
- Final failure is not hidden by intermediate continuation.
- Recovery selection and artifact hashes are reproducible.
