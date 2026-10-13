# Formal-charge reference data

This source contains net integer formal charges for 187 extractants and their Eu/Am CBond
pairs. Proton-retaining and deprotonated variants are equally valid references when a pair
contains removable polar hydrogen atoms.

## Scope

- 187 source extractants: 183 neutral, three with charge `+2`, and one with charge `-4`.
- 182 Eu pairs and 182 Am pairs; extractant indices 139, 182, 185, 186, and 187 have no
  positive CBond assignment.
- 154 pairs per metal have one reference state.
- 28 pairs per metal have both proton-retaining and deprotonated states, for 56 variant
  structures per metal.

The charge convention assumes Eu(III) and Am(III):

```text
proton retained: Q = Q(extractant) + 3
deprotonated:    Q = Q(extractant) + 3 - N(removed H+)
```

Metal oxidation state is an explicit source convention, not a value inferred uniquely from
the molecular graph. The tabulated total charge is authoritative because ordinary SMILES
coordination bonds do not completely encode electron assignment.

Eu(III) and Am(III) produce the same charge classification for a given ligand and
protonation state. The classification is therefore stored once in `pair-charges.csv` and
`polar-H-variant-charges.csv`; the metal-specific SMILES remain separate structure files.

The local payload resides in `data/sources/formal-charge/` and is not tracked by Git. Its
source and pair structures are under `structures/`, alongside the two metal-specific
protonation-variant SMILES files.
