# OPTIONAL-014: exact-identity trusted scientific history

Type: optional empirical/evidence rule. Planned screening default: **enabled**.

| Metadata | Value |
|---|---|
| Rule version | `draft-1` |
| Phase | `SOURCE_EVIDENCE` |
| Requires rules | none |
| Consumes | molecular-facts identity and backend-independent scientific source records |
| Produces | attributable charge/OS/scientific electronic-state evidence |
| Required providers/data | immutable curated/reference records and identity schema |
| Review state | `DRAFT` |
| Owner approval | pending identity/provenance review |
| Primary basis | exact-identity curated scientific source records |
| Applicability artifact | scientific identity/source-authority matching schema; digest not yet frozen |
| Candidate domain scope | `DOMAIN_BOUNDED` |

## Theory and basis

A curated or independently verified scientific state for the exact same
molecular-facts identity is stronger evidence than a generic prior. This rule
contains no backend, method, basis or ECP context. Backend-specific execution
history belongs to EXECUTION-001.

## Implementation

- Require matching molecular-facts identity, atom map, computational boundary
  and source authority.
- Preserve source alternatives and assumptions; propose them as scientific
  evidence without claiming a ground state.
- Reject filename-only, coordinate-only or partial-identity association.
- Any backend/method/ECP-dependent value is delegated to EXECUTION-001 rather
  than stripped of its context.

## Acceptance

- Exact-identity records propose scientific candidates with source digests.
- Any chemical-boundary or atom-map mismatch disables reuse.
- Backend-dependent execution records cannot enter this rule.
- Conflicting trusted records remain conflict/alternatives.
- Raw source artifacts remain immutable.
