# DEFAULT-004: lowest legal xTB UHF execution state

Type: backend execution decision policy.
Runtime registry: backend `ExecutionPolicyRegistry`.

## Theory and basis

When $N_e$ is known and no stronger spin evidence selects an execution state,
the least unpaired value satisfying UHF parity is deterministic:

$$u_{\min}=N_e\bmod2.$$

This is an economical legal starting point, not a ground-state inference.

## Implementation

- Run only inside xTB `project_state()` after `BackendContext` is known; never
  run inside backend-independent scientific resolution.
- Apply only after a selected legal $Q$ determines all-electron xTB $N_e$.
- A HARD-013-valid explicit backend UHF wins when scientific spin is unresolved
  or projects compatibly. A conflicting explicit request terminates projection;
  this default never overrides selected molecular $2S$. Otherwise a selected
  molecular $2S$ candidate is projected under the named highest-$M_S$
  convention.
- Apply the parity fallback only when neither route supplies UHF.
- Select `u=0` for even $N_e$ and `u=1` for odd $N_e$.
- Mark `selection_source=EXECUTION_DEFAULT` and confidence as execution-only.
- Preserve all scientifically proposed spin candidates.
- Do not apply to ECP or other backends without an explicit projection rule.

## Acceptance

- Even/odd electron fixtures yield `0/1` and pass HARD-013.
- Scientific resolution is unchanged by applying this backend-only decision.
- A valid explicit, trusted or enabled-rule prior outranks this default.
- An explicit UHF conflicting with selected molecular $2S$ produces
  `PROJECTION_CONFLICT`, not a fallback value.
- The scientific candidate set is unchanged by execution selection.
- Output and docs contain no physical-ground-state claim.
