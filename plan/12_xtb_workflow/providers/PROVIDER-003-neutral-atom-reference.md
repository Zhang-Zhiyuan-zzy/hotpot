# PROVIDER-003: neutral isolated-atom reference data

Type: versioned scientific data provider; it is not a molecular default rule.

## Theory and basis

Evaluated neutral-atom configurations and ground terms provide useful priors
for an isolated neutral atom. They are not molecular spin defaults and cannot
be summed across atoms.

## Implementation

- Use a versioned NIST ASD/Handbook-derived registry with explicit blanks.
- The consuming atomic-reference operation may project a reviewed term to
  candidate spin-only $2S$ values while retaining spin-orbit/term metadata; a
  backend adapter performs any later UHF projection.
- Consumers must not use the table for an ion or molecule without an
  independent ionic-configuration rule.

## Acceptance

- Registry covers Z=1--118 with explicit unavailable entries.
- Representative C, Cr, Fe, Eu, Gd, U and Am records reproduce the reviewed
  source snapshot.
- Molecular and ionic inputs do not match.
- A selected atomic $2S$ value passes molecular-spin HARD-006 before any
  backend projection; its later xTB UHF projection independently passes
  HARD-013.

Reference: NIST Atomic Spectra Database and Handbook of Basic Atomic
Spectroscopic Data, with version/access date recorded in the data manifest.
