# DEFAULT-006: preserve candidates after selection

Type: common result-retention decision policy.
Runtime registry: `StatePolicyRegistry`.

## Theory and basis

Choosing one practical calculation does not erase other scientifically legal
states or prove that the chosen state is unique. Preserving the scientific
space does not require materializing its full Cartesian product.

## Implementation

- Results store immutable factor domains, compatibility constraints, actually
  expanded paths, rejected-path summaries, frontier bounds, search status and
  the selected identity.
- Distinguish `ScientificResolutionResult` from `ExecutionStateDecision`.
- Merge only candidates with equivalent scientific paths; backend scheduling
  may deduplicate identical $(Q,\mathrm{UHF})$ jobs while retaining every
  provenance path.
- Constraint propagation, deterministic best-first/lazy joins and an explicit
  `CandidateBudget` decide which full paths are expanded.
- The same budget caps materialized values per factor. Candidate producers emit
  compact/lazy domains and a remainder summary; exceeding the cap sets
  `materialization_status=RESOURCE_TRUNCATED` without changing the factor's
  scientific `domain_scope`.
- `save_all_candidates` means all paths actually expanded under the declared
  budget, not every theoretical combination.
- Serialization preserves the factorized space, expanded paths, selection
  source, evidence, rule trace, numerical budget, budget-artifact identity,
  frontier-bound schema, frontier bounds and exact `CandidateSearchStatus`.

## Acceptance

- Selecting a default does not mutate factor domains or remove provenance.
- Round trip restores factors, constraints, expanded paths, budget, frontier,
  search status, scientific resolution and execution decision exactly.
- Two scientific paths sharing one backend job remain attributable.
- Expanded alternatives and compact unexpanded-factor summaries appear in
  user-facing diagnostics without pretending that every cross-product path was
  enumerated.
- Million-scale dry resolution remains bounded in peak paths and memory per
  record.
