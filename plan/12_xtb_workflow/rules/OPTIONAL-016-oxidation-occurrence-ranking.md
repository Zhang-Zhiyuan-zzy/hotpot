# OPTIONAL-016: oxidation-state occurrence ranking

Type: optional empirical rule. Planned screening default: **disabled**.

| Metadata | Value |
|---|---|
| Rule version | `draft-1` |
| Phase | `OXIDATION_STATE_RANKING` |
| Requires rules | none |
| Consumes | hard-valid OS candidates and domain classification |
| Produces | OS occurrence rank adjustments |
| Required providers/data | versioned occurrence corpus/split/dedup manifest |
| Review state | `DRAFT` |
| Owner approval | pending corpus/domain review |
| Primary basis | versioned domain-matched occurrence corpus required |
| Applicability artifact | corpus split/dedup/domain manifest; digest not yet frozen |
| Candidate domain scope | not applicable; ranking-only rule |

## Theory and basis

Observed oxidation-state frequencies can rank candidates in a corpus similar
to the target domain. Frequencies from ionic solids are not generic priors for
molecular coordination or organometallic chemistry.

## Implementation

- Require a versioned occurrence dataset, deduplication policy and domain
  classifier.
- Rank only already hard-valid candidates within the validated matching
  domain.
- Report corpus coverage, source bias and smoothing. Never turn zero occurrence
  into a hard rejection.
- Keep disabled for molecular coordination until an appropriate held-out
  validation exists.

## Acceptance

- Train/reference and held-out compounds are source-separated.
- Coverage and ranking accuracy are reported by chemical domain.
- Cross-domain examples abstain.
- Disabling the rule removes only the occurrence rank contribution.
