# HARD-009: state invalidation after identity change

Type: hard, non-toggleable.

## Theory and basis

A resolved state applies only to the chemical boundary and semantic graph from
which it was obtained. Coordinates alone normally do not change that identity;
protonation, composition, redox or accepted topology changes do.

## Implementation

- Recompute the chemical identity after every operation capable of changing
  atoms, hydrogens, components, bond semantics or accepted reaction topology.
- Reuse a state only when its identity still matches.
- Coordinate-only optimization preserves the state binding after atom-map
  verification.
- A topology proposal from a backend does not invalidate or replace the graph
  until an explicit topology-reconciliation operation accepts it.

## Acceptance

- Coordinate updates retain valid state packets.
- Protonation, atom deletion/addition and accepted bond changes invalidate
  them.
- Stale states cannot cross CLI, stream or pipeline boundaries unnoticed.
- Round trips either preserve the identity or report information loss.
