# HARD-008: backend-specific state projection

Type: hard, non-toggleable.

## Theory and basis

A scientific state and a backend input representation are related but not
identical. xTB UHF, Gaussian multiplicity and an ECP effective-electron count
belong to different execution contexts.

## Implementation

- Keep backend name/version, method, basis/parameter set, ECP and effective-
  electron definition out of `MolecularFactsIdentity` and inside
  `BackendStateProjection`.
- Project only a hard-admissible selected execution candidate.
- GFN-FF consumes $Q$ but not UHF; GFN1/2-xTB consumes $Q$ and UHF.
- Return `BACKEND_UNSUPPORTED` or `PROJECTION_CONFLICT` distinctly from a
  scientific inference conflict.
- Return the common tagged `BackendProjectionResult`. Failure statuses contain
  findings and no selected projection/execution decision; never fabricate
  charge or UHF merely to populate the success object.

## Acceptance

- Changing backend invalidates/rebuilds only the projection.
- GFN-FF requests never receive a spin option.
- GFN1/2 requests always receive legal charge/UHF.
- Gaussian/ECP fixtures use effective, not all-nuclear, electron parity.
- `PROJECTED` has exactly one projection; `PROJECTION_CONFLICT` and
  `BACKEND_UNSUPPORTED` have none.
