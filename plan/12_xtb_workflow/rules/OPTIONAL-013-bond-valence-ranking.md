# OPTIONAL-013: bond-valence oxidation-state ranking

Type: optional empirical rule. Planned screening default: **disabled**.

| Metadata | Value |
|---|---|
| Rule version | `draft-1` |
| Phase | `OXIDATION_STATE_RANKING` |
| Requires rules | none |
| Consumes | hard-valid OS candidates and calibrated geometry |
| Produces | OS rank adjustments |
| Required providers/data | versioned bond-valence parameter/domain registry |
| Review state | `DRAFT` |
| Owner approval | pending domain review |
| Primary basis | Brown--Altermatt and Brese--O'Keeffe parameterizations |
| Applicability artifact | versioned pair-parameter/domain registry; digest not yet frozen |
| Candidate domain scope | not applicable; ranking-only rule |

## Theory and basis

Within calibrated ionic/inorganic domains, a bond-valence contribution is

$$s_{ij}=\exp\left(\frac{R_{0,ij}-R_{ij}}{B_{ij}}\right),\qquad
V_i=\sum_j s_{ij}.$$

Agreement between $V_i$ and an oxidation magnitude is soft geometric evidence,
not a universal molecular or organometallic rule.

## Implementation

- Load versioned element-pair parameters and applicability metadata.
- Rank already hard-valid OS candidates by a declared residual; never reject a
  candidate solely by BVS outside its calibrated domain.
- Abstain for missing parameters, f-element/organometallic/non-innocent domains
  not explicitly validated.

## Acceptance

- Reproduce source reference structures within documented tolerance.
- Held-out ionic-domain ranking reports coverage and error.
- Out-of-domain molecular complexes abstain.
- Disabling the rule leaves candidates unchanged.

References: Brown and Altermatt, DOI `10.1107/S0108768185002063`; Brese and
O'Keeffe, DOI `10.1107/S0108768190011041`.
