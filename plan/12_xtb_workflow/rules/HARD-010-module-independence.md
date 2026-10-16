# HARD-010: independent scientific modules and one orchestrator

Type: hard, non-toggleable architecture rule.

## Theory and basis

Charge, oxidation state and spin are different scientific quantities. Hidden
cross-calls make assumptions untraceable and make one ambiguity appear to be a
fact in another module.

## Implementation

- Charge, oxidation-state and electronic-state inference are independently
  callable pure operations over immutable inputs.
- They may import shared contracts and upstream result types, but not call a
  sibling inference implementation.
- `state_composition` is a pure compatibility/constraint layer over immutable
  module results. It does not infer, rank or select scientific values.
- `state_resolver.py` is the sole controller boundary and invokes composition,
  optional ranking and the named default decision in that order.
- Optional rules receive all prerequisites in `RuleContext`; they do not call
  the inference packages or backends themselves.

## Acceptance

- Unit tests instantiate each module from constructed inputs.
- Import/call audits find no hidden sibling-estimator calls.
- Composition tests prove that it cannot introduce an unconstrained value;
  every value absent from the input factors must be uniquely derived by a
  declared hard equation with complete parent identities.
- Disabling one optional rule cannot invoke another by fallback.
- CLI, pipeline and Python API use the same resolver/projection path.
