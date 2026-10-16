# OPTIONAL-009: ionic-configuration spin candidates

Type: optional empirical rule. Planned screening default: **enabled**.

| Metadata | Value |
|---|---|
| Rule version | `draft-1` |
| Phase | `ELECTRONIC_STATE_CANDIDATE` |
| Requires rules | none |
| Consumes | hard-valid partial charge/OS paths from the composition barrier |
| Produces | backend-independent `SpinContributionCandidate(scope=ATOM, two_s=...)` sets |
| Required providers/data | versioned `(element, OS)` ionic-configuration registry |
| Review state | `DRAFT` |
| Owner approval | pending individual review |
| Primary basis | NIST ASD/Handbook evaluated configurations and terms |
| Applicability artifact | `(element, OS)` ionic-configuration registry; digest not yet frozen |
| Candidate domain scope | `DOMAIN_BOUNDED` |

## Theory and basis

An element plus an attributable oxidation state can imply candidate ionic
electron configurations. Open $d^n/f^n$ occupancies generate possible
spin-only unpaired-electron sets, but usually do not select a unique molecular
spin.

For one shell, the maximum spin-only $2S$ value is

$$
(2S)_{\max}^{(d)}=\min(n,10-n),\qquad
(2S)_{\max}^{(f)}=\min(n,14-n),
$$

with candidates descending by two to the parity minimum.

## Implementation

- Consume an attributable OS assignment on a surviving partial path; do not
  infer or independently select OS.
- Load versioned `(element, OS)` configuration records with NIST or reviewed
  source provenance and explicit exceptions to simple electron removal.
- Propose `SpinContributionCandidate(scope=ATOM, ...)` values and couple them
  in the electronic-state module; every value names its atom subject and
  charge/OS parent candidate identities.
- Do not emit xTB UHF. A backend projection may later map a molecular $2S$
  candidate to UHF under a named highest-$M_S$ convention.
- Expand spin candidates lazily only for partial paths requested by the
  best-first composer/decision policy; do not form an OS-by-spin Cartesian
  product eagerly.
- Abstain for unregistered transition/f-block ions rather than fabricate an
  Aufbau configuration.

## Acceptance

- Fe(II) $d^6$ proposes `two_s={0,2,4}` without choosing high/low spin.
- Eu(III) $4f^6$ proposes `two_s={0,2,4,6}`.
- Missing/ambiguous OS does not trigger a hidden OS guess.
- Candidate parity remains legal after molecular coupling.
- Every local spin candidate retains the identity of its charge/OS parent path.
