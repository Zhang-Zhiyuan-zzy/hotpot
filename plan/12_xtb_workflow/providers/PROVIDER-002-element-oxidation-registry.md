# PROVIDER-002: element oxidation-state registry

Type: versioned scientific data provider; it is not a selection rule.

## Theory and basis

Element-level oxidation-state compilations can expand a finite search domain,
but an element symbol alone does not select an oxidation state. Rare and
superheavy-element entries may be predicted rather than experimentally
established.

## Implementation

- Load a versioned 1--118 source-tagged registry from machine-readable data.
- Each value records source, release/access date, evidence tier, applicability
  and whether it is observed or predicted.
- Registry output is `DOMAIN_BOUNDED`, never `EXHAUSTIVE`: absence from the
  reviewed snapshot does not prove an oxidation state impossible.
- The oxidation-state engine may use the provider to expand candidates only
  when structural/accounting evidence can compose them into a hard-valid
  assignment; the provider never ranks by numeric size.
- The table in the archived oxidation-state document is an acquisition seed,
  not runtime normative data. Production data require a reviewed snapshot and
  digest.

## Acceptance

- All 118 rows parse; missing values remain missing, not zero.
- Registry expansion alone never produces a unique selected OS.
- Predicted/rare states are visibly lower-tier evidence.
- Source updates produce a new data/rule version and reproducible diff.
