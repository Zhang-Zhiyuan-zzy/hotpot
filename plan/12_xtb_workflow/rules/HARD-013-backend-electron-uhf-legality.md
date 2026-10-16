# HARD-013: backend electron-count and xTB UHF legality

Type: hard, non-toggleable backend-projection rule.

## Theory and basis

For all-electron xTB, the projected electron count is

$$
N_e=\sum_i Z_i-Q,
$$

and xTB UHF $u=N_\alpha-N_\beta=2M_S$ must satisfy

$$
0\le u\le N_e,\qquad (N_e-u)\bmod 2=0.
$$

Then

$$
N_\alpha=(N_e+u)/2,\qquad N_\beta=(N_e-u)/2
$$

are non-negative integers.

## Implementation

- Run only during backend projection after `BackendContext` and any explicit
  backend request are known.
- Validate preconstructed projections again at the backend boundary without
  re-inferring the scientific state.
- Hard-validate an explicit backend UHF request before any execution policy.
  If a selected scientific $2S$/multiplicity is projected under a named pure-
  spin highest-$M_S$ convention, require explicit $u=2S=M-1$; an incompatible
  explicit request returns `PROJECTION_CONFLICT` and is not allowed to override
  the scientific path.
- Compute ECP effective-electron counts from explicit removed-core metadata;
  never reuse all-electron xTB parity blindly.
- A highest-$M_S$ projection may map selected molecular $2S$ to $u=2S$, but the
  projection convention and spin--orbit limitation remain explicit.

## Acceptance

- Even-electron `u=1`, odd-electron `u=0`, negative effective $N_e$ and
  `u>N_e` are rejected before backend probing/process launch.
- Legal projected states yield integral $N_\alpha,N_\beta$.
- Python, CLI, pipeline and stream projections produce the same result.
- A legal explicit UHF with unresolved scientific spin is preserved; an
  explicit UHF incompatible with selected projected $2S$ returns
  `PROJECTION_CONFLICT` before DEFAULT-004 can run.
- ECP parity tests are independent of all-electron xTB parity tests.
