# EXECUTION-001: exact-context backend history

Type: optional backend execution-selection policy. Planned default: **enabled
when matching evidence is supplied**.

## Theory and basis

A previously verified executed state for the same scientific-state candidate
and exact backend context is useful execution evidence. It is not backend-
independent scientific evidence and does not prove a physical ground state.

## Implementation

- Run only in backend projection/execution selection, after scientific
  resolution and hard validation.
- Require matching `ScientificStateIdentity`, atom map, backend/version,
  method, parameter set or basis, ECP/removed-core definition and relevant
  environment settings.
- Preserve requested and actually executed settings separately. A verified
  executed value may rank one hard-legal backend projection.
- Reject filename-only, coordinate-only or partial-context association.
- Never feed backend convergence back into charge/OS/spin scientific truth.

## Acceptance

- Exact-context evidence ranks the matching legal backend state and records
  artifact digests.
- Any scientific-state/backend/method/ECP mismatch causes abstention.
- Conflicting exact-context histories remain visible alternatives/conflict.
- Raw artifacts remain immutable, and disabling the policy changes no
  scientific result.
