# HARD-006: molecular electron-count and spin legality

Type: hard, non-toggleable scientific-state rule.

## Theory and basis

For an all-electron molecular scientific state,

$$
N_e=\sum_i Z_i-Q.
$$

A spin-only molecular value $2S$ is legal only when

$$
0\le 2S\le N_e,\qquad (N_e-2S)\bmod 2=0.
$$

Where a pure-spin multiplicity is claimed, $M=2S+1$.

## Implementation

- Provide one backend-independent electron-count/$2S$ validator used during
  scientific candidate composition.
- Bind $Q$, $2S$ and multiplicity to the same coupled scientific path.
- Preserve strong spin--orbit or otherwise non-pure-spin limitations in the
  electronic-state result rather than silently interpreting backend UHF as
  molecular $2S$.
- Do not inspect `BackendContext`, ECP metadata or xTB UHF here.

## Acceptance

- Even-electron `two_s=1`, odd-electron `two_s=0`, negative $N_e$ and
  `two_s>N_e` are rejected during scientific composition.
- Legal values have electron-count-compatible integer/half-integer $S$.
- A supplied pure-spin multiplicity agrees with $M=2S+1$.
- All scientific entry paths produce the same legality result.
