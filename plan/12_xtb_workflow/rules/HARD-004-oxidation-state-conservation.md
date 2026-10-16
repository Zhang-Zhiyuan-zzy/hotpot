# HARD-004: oxidation-state conservation

Type: hard, non-toggleable.

## Theory and basis

Oxidation state is integer ionic-approximation bookkeeping. For a complete
assignment under one declared convention,

$$x_i\in\mathbb Z,\qquad\sum_i x_i=Q.$$

This does not imply $x_i=q_i$ for formal charge.

## Implementation

- Local OS candidates may be proposed before a net charge is selected, but a
  complete selectable path binds exactly one $Q$, one complete OS assignment
  and one counting convention.
- Under an ionic coordination-counting convention, validate the complete
  metal-OS plus ligand/counterion bookkeeping sum against that path's $Q$;
  never equate OS with stored atom formal charge.
- Solve multimetal constraints globally; do not select each center
  independently.
- A partial explicit override is admissible only if at least one complete
  assignment can extend it.
- Return all non-equivalent legal assignments. Ambiguity is not conflict.

## Acceptance

- Complete assignments always sum to $Q$.
- Individually legal local OS and charge candidates that cannot form one
  complete conserved path are rejected before default selection.
- A non-integer or intrinsically illegal partial override is `INADMISSIBLE`.
  Individually legal partial OS and $Q$ constraints with no common extension
  are `CONFLICT`.
- Equivalent metal sites canonicalize localization alternatives without losing
  a mixed-valence/equivalence-class statement.
- Eu(II/III), Fe(II/III) and multimetal fixtures preserve ambiguity when facts
  do not decide it.
