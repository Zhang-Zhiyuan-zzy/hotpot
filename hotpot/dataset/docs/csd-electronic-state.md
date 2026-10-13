# CSD electronic-state text evidence

This local source stores reported oxidation- and spin-related text extracted from accepted B/C
CSD entries and links it conservatively to metal-site candidates. It is evidence for review,
not a complete set of expert-adjudicated electronic-state truths.

## Interpretation

- `OX-T1` identifies an explicit integer oxidation state in a chemical-name field for an
  element with one retained site orbit. It is a unique localization candidate, not independent
  spectroscopic validation.
- `SP-T1` identifies one fixed `high_spin`, `low_spin`, or `intermediate_spin` phrase with one
  plausible metal-site candidate. It is not numerical `S`, multiplicity, or proof of a
  ground-state assignment.
- Conditional, crossover, population, isomer, irradiation, temperature, pressure, and
  metastability language remains unresolved evidence unless reviewed with its conditions.

The source contains no confirmed numerical `S` or multiplicity labels. Distinct CSD refcodes
are source records; polymorphs, redeterminations, and versioned refcodes are not automatically
deduplicated as chemical entities.

## Local assets

The portable SQLite source and its audit records reside in
`data/sources/csd-electronic-state/`. Its tracked relational definition is
`schemas/csd-electronic-state.sql`. Build and assessment tools live under
`hotpot/ds/tools/electronic_state/`.

The portable database contains CSD-derived text and therefore must be treated as restricted
unless the holder verifies redistribution rights. Metadata describing a license is not itself
legal authorization.

See [the quality assessment](csd-electronic-state-quality.md) for admission limits.
