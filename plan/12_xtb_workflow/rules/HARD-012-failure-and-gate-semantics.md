# HARD-012: failure and geometry-gate semantics

Type: hard, non-toggleable control contract.

## Theory and basis

Scientific conflict, backend incompatibility, runtime failure and geometry
quality are different facts. A poor intermediate geometry may be repairable by
a later method; an illegal electronic state is not made legal by convergence.

## Implementation

- Maintain separate scientific, projection, executability, runtime and
  geometry statuses.
- Preflight rejects non-finite/singular input, invalid atom mapping and
  unsupported backend states.
- Intermediate finite quality failures are preserved as warning/recovery facts
  unless the selected route explicitly requires a hard gate.
- Final acceptance evaluates the delivered product and retains failure
  artifacts/last finite frames.
- Unknown programming errors propagate; they are not converted to a scientific
  status.

## Acceptance

- No backend call occurs after scientific/projection conflict.
- Finite intermediate warning can continue to a configured later stage.
- NaN/Inf coordinates never propagate unchanged.
- Final failure retains artifacts without being reported as success.
- Aggregate reports keep every failure class separate.
