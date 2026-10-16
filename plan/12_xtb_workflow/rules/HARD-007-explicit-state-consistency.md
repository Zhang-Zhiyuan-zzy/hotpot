# HARD-007: explicit and source-state consistency

Type: hard, non-toggleable.

## Theory and basis

Explicit or verified source settings have high authority, but authority cannot
override mathematics or a contradictory fact bound to the same identity.

## Implementation

- Apply simultaneous explicit $Q$, atom/component charges, OS overrides,
  molecular $2S$ and multiplicity as constraints on the joint scientific
  candidate space before path expansion or default selection.
- Treat a scalar request as one authoritative constraint. Treat an attributable
  request candidate tuple as the caller's allowed domain; alternatives inside
  that tuple are not mutually contradictory source facts and remain available
  for hard filtering and named default selection.
- Store explicit UHF as a separate backend request. Do not compare it with
  scientific $2S$/multiplicity in this scientific-state rule; HARD-013 owns
  that check after the projection convention and `BackendContext` are known.
- Preserve conflicting evidence and return reason-coded `CONFLICT`; never
  silently choose one or rewrite it.
- Reject an illegal member of a multi-value allowed domain without rejecting
  legal siblings. If an authoritative allowed domain has no legal member,
  return `INADMISSIBLE`; an empty optional-rule proposal is not an authoritative
  conflict.
- Treat requested and actually executed external states as distinct evidence.

## Acceptance

- Agreeing explicit fields are preserved with their source.
- Scientific charge-sum and $2S$/multiplicity conflicts are reported during
  resolution; explicit-UHF compatibility is reported by backend projection
  before launch.
- A lower-authority default never overwrites valid explicit input.
- No selected path combines explicit fields with independently valid but
  mutually incompatible inferred fields.
- A multi-value request candidate domain is not reported as `CONFLICT` merely
  because it contains alternatives.
- External requested/executed mismatches remain auditable as two facts.
