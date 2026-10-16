# HARD-002: net, atom and component charge conservation

Type: hard, non-toggleable.

## Theory and basis

For one exact computational boundary, net charge $Q$ is an integer. If a result
claims complete atom formal charges $q_i$ and complete original-component
charges $Q_c$, then

$$Q=\sum_i q_i=\sum_c Q_c.$$

A graph alone need not determine $Q$.

## Implementation

- Validate explicit/source $Q$ independently of Lewis inference.
- Permit a joint composer to create a derived $Q$ only when a declared hard
  conservation equation uniquely determines it from complete, identity-bound
  parent assignments. Record the equation, convention and parent candidate
  identities; a derived $Q$ is not new empirical evidence.
- Enforce each sum only when the corresponding assignment is declared
  complete; incomplete assignments remain partial evidence.
- Distinguish an explicit neutral value from an uninitialized zero.
- Return `CONFLICT` for contradictory complete facts and `UNRESOLVED` for
  missing facts. Never choose neutrality merely to continue.

## Acceptance

- Positive, negative and explicit-zero cases pass.
- Conflicting atom or component sums return `CONFLICT` and do not launch a
  backend.
- Partial atom evidence is not falsely treated as a complete sum.
- A graph-indeterminate system can return candidates or `UNRESOLVED`.
- A derived charge without complete parents or a unique conservation solution
  is rejected.
