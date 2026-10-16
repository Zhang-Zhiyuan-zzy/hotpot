# DEFAULT-003: source authority and selection order

Type: common default decision-order policy.
Runtime registry: `StatePolicyRegistry`.

## Theory and basis

Attributable information about the exact system has greater authority than a
generic prior, provided it is hard-admissible.

## Implementation

Use this authority order when constraining and comparing operation-complete
coupled paths:

```text
verified explicit user value
  > verified scientific source value
  > exact-identity trusted scientific source
  > enabled domain rule
  > named general execution default
  > unresolved
```

- Hard validation precedes authority comparison.
- A valid explicit value first removes every incompatible path. It never gets
  paired later with independently selected incompatible OS/spin values.
- Build an authority vector over the requested quantities. Remove a path only
  when another path has equal-or-higher authority on every requested quantity
  and strictly higher authority on at least one; do not hide cross-quantity
  tradeoffs inside an arbitrary field order.
- Pass mutually non-dominated paths to enabled attributable domain-rule ranks
  and the profile's declared tie policy.
- Mutually exclusive authoritative facts bound to the same identity remain
  `CONFLICT`; no profile or canonical key may choose between them.
- Multiple legal scientific hypotheses without contradictory authoritative
  facts remain `AMBIGUOUS`. A named screening profile may use a stable canonical
  key to choose among decision-equivalent execution paths, but that key is not
  scientific evidence and does not resolve the ambiguity scientifically.
- Paths that still differ in operation-relevant $Q$, OS-equivalence class or
  molecular $2S$ after all approved preferences are not decision-equivalent;
  canonical identity cannot choose between them.
- A value uniquely completed by a hard conservation equation has authority
  category `DERIVED_CONSTRAINT`. It contributes no independent preference
  rank; retain and compare the authorities of its parent evidence separately.
- An explicit/source value for the same quantity acts as a composition
  constraint. A compatible derived value is corroborating lineage; an
  incompatible one rejects the whole path rather than entering an authority
  contest.
- `0` is a value, not evidence of authority.
- Record `selection_source`, source identity and assumptions.
- Backend/method/ECP-specific execution history is handled by EXECUTION-001
  after scientific resolution; it is not inserted into this scientific order.

## Acceptance

- A valid explicit setting wins over every default.
- An intrinsically hard-invalid explicit value returns `INADMISSIBLE` and is
  not replaced; individually legal but mutually incompatible authoritative
  constraints return `CONFLICT`.
- Independently valid but mutually incompatible Q/OS/spin results never form a
  selectable path.
- Equal-authority legal alternatives remain visible after an execution tie
  decision; mutually exclusive authoritative facts produce no decision.
- A derived charge cannot outrank its parents or override an explicit/source
  charge, and its complete parent/equation lineage survives serialization.
- Selection order is deterministic and serializable.
