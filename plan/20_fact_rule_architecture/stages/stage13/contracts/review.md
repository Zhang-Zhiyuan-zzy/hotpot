# Stage13A private aromatic/Kekulé protocol

Root defines this bounded contract from actual `B_exp`
`5a7b19396328eee964a8f3c65bc3345c013eaec6` after its main-scope acceptance
and successful push. Main-plan Section17 authorizes the private domain experiment.
No public signature, algorithm, default, return, permission or user setting changes.

## Exact connection

```python
# hotpot.cheminfo.operations._aromatic_protocol
_evaluate_aromatic_assignment(snapshot, *, kekule) -> _AromaticAssignment | None
```

This is a direct function binding to the existing sole
`rules.chemistry.bonding.kekule._calculate_aromatic_assignment`; it adds no
call wrapper, dispatch table, Protocol class, registry, plugin discovery or
scientific implementation. Implementation identity is the actual bound
function's existing module/qualified name plus the attributed Hotpot source
version. No per-result identity label is added. There are zero new class,
dataclass, Protocol, enum, TypedDict or ABI definitions.

| Connection | Contract and operation response |
|---|---|
| Input | Existing immutable `_BondingSnapshot`; ordered atom UIDs and bond UID/row pairs describe the selected topology. Atomic numbers, declared H or `None`, supplied aromatic flags, rings, coordinates and `ring_has_3d` retain their existing conventions. Coordinates are Cartesian angstrom; bond order is dimensionless. |
| Parameters | Required keyword-only `kekule`; false perceives flags, true also proposes the existing greedy per-ring bond-order assignment. No new defaults. |
| Defined result | Existing immutable `_AromaticAssignment`, source-bound with the complete ordered atom and selected bond UIDs, aromatic flags, bond orders/kinds, covered rows and existing method. Nonaromatic rows remain explicit; no generic success/confidence envelope. |
| No complete proposal | Existing `None`, including unknown required H on the nonspatial rule branch. This does not invent an empty assignment or authorize partial writes. |
| Scientific limits | Preserve per-ring order dependence, fused-ring limits, original H/coordinate conventions, supplied flags and the existing model; no uniqueness proof or new science gate. |
| Operation response | Existing preparation, source validation and coupled `_commit_aromatic_assignment` remain with `operations.chemistry`. Explicit assignment marks the original unresolved scope on `None`; lazy preparation retains its narrower inferred scope and source protections. |
| Failure | Existing errors propagate through the operation's factual transaction; original committed state and rollback behavior remain. There is no catch/default/retry or fallback implementation. |
| Evolution | A replacement implementation must satisfy this same input/mapping/proposal contract and identify genuine unsupported scope. New required output or public behavior needs an owning decision; the experiment does not authorize another domain's protocol. |

## Definition and integration boundary

Stage13A adds only this binding, isolated contract tests and records.
`operations.chemistry` and every production call site remain byte-identical to
`B_exp` until the validated definition commit `D_exp`. Tests exercise the actual
rule on independent named positive/negative/boundary fixtures, mapping/order,
read-only behavior, `None` and exact exception propagation. They do not claim
all historical fused cases are supported or execute the optional scientific tiers.

After committed `D_exp`, Stage13B may replace the two current calculate-call
connections through this one binding, preserving the existing preparation,
source checks and commits. Pure flag reconstruction and joint-ring queries
retain their existing purpose and owners. There will be one production route;
no user-selectable old/new path. Stage13B owns actual public workflow/rollback,
affected source/default/installed validation, paired same-work costs and `I_exp`.
