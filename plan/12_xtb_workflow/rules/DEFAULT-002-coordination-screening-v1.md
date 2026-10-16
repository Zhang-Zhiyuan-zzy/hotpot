# DEFAULT-002: `COORDINATION_SCREENING_V1`

Type: named application decision policy.
Runtime registry: `StatePolicyRegistry`.

## Theory and basis

Routine coordination-chemistry screening benefits from one reproducible legal
execution choice, even when the physical ground state is not known. The policy
is a declared compromise, not a universal chemical model.

## Implementation

- Apply hard rules before and after every proposal/composition layer.
- Enable exactly the reviewed providers and optional rules checked in the
  normative plan, subject to their versioned applicability tests.
- Evaluate a machine-readable profile artifact covering required molecular
  facts, supported elements/OS and ligand templates, coordination/H-inventory
  semantics, exclusion motifs, `ScientificOperationRequirements`, provider/rule
  versions, decision order, candidate budget and tie policy. This profile
  requires resolved net charge and an attributable metal OS; molecular $2S$
  may remain explicitly unresolved for later backend projection.
- Caller operation-requirement/toggle/configuration overrides create a distinct
  effective-profile identity, cannot weaken mandatory profile quantities and do
  not inherit the validated profile's coverage guarantee.
- Treat verified explicit user values as constraints on coupled paths, not as
  isolated fields to be reconciled after selection.
- Build a factorized candidate space and compose compatible charge, component,
  coordination-counting, OS and electronic-state paths without constructing
  their full Cartesian product.
- Compute static profile eligibility from molecular facts, operation
  requirements and the versioned eligibility artifact before candidate
  selection. Candidate-dependent `RuleApplicability` belongs to the later rule
  trace and decision certification. Separately certify a
  decision only when every decision-relevant factor is `EXHAUSTIVE` or finite
  `DOMAIN_BOUNDED` and has a provable optimistic `FrontierBound`. `OPEN` or
  unboundable factors do not change static applicability, but they prevent a
  selection by truncation or canonical order.
- Apply DEFAULT-003 to operation-complete coupled paths. Then apply enabled
  domain-rule rank evidence. A stable canonical tie breaker may choose one
  execution path only among decision-equivalent alternatives with identical
  operation-relevant $Q$, OS-equivalence class and molecular $2S$.
- Use a validated OS-dependent spin prior, including OPTIONAL-011 where
  applicable, to rank molecular $2S$. Do not invoke DEFAULT-004 here; an xTB
  projection may invoke that backend-only fallback after scientific resolution.
- Require an attributable metal-OS assignment for an in-domain coordination-
  complex decision. Missing OS returns `NO_AUTOMATIC_STATE`; a different
  explicitly declared nonmetal/operation profile may permit unresolved OS.
- Label the decision `EXECUTION_DEFAULT` or `EXECUTION_PRIOR`; preserve
  scientific `AMBIGUOUS` status when the selected path is not scientifically
  unique.
- Decide `NO_AUTOMATIC_STATE` for static out-of-domain records, missing-
  evidence spaces with no authoritative conflict, and
  `RESOURCE_LIMIT_NO_CERTIFIED_DECISION`; intrinsically illegal input and
  incompatible authoritative constraints remain `INADMISSIBLE` and `CONFLICT`
  respectively. A
  `TRUNCATED_BUT_DECISION_CERTIFIED` search may use its proven selection.
- Decide `NO_AUTOMATIC_STATE` when chemically non-equivalent paths remain tied
  after every enabled preference; identity order is not a chemical prior.

## Acceptance

- Profile activation is explicit in manifests and reports.
- The versioned `CandidateBudget` and decision-policy version are frozen and
  recorded before profile activation.
- The budget artifact and frontier-bound schema are serialized in every
  candidate space; until their numerical values pass Phase-5 stress validation,
  this profile remains planned rather than active.
- Toggling any optional rule changes only attributable outcomes.
- Every statically in-domain, decision-certifiable fixture produces exactly one
  hard-valid selected path; in-domain non-equivalent ties remain in-domain and
  produce `NO_AUTOMATIC_STATE` with a certification finding.
- Equal scientific candidates have a deterministic execution tie result while
  scientific ambiguity remains visible.
- Paths differing in operation-relevant $Q$, OS-equivalence class or molecular
  $2S$ are never selected merely by canonical identity.
- Candidate budgets prevent unbounded expansion and distinguish certified from
  uncertified truncated searches.
- Out-of-domain fixtures produce `NO_AUTOMATIC_STATE` without stopping a batch.
- Reports never call the profile choice a ground state.
