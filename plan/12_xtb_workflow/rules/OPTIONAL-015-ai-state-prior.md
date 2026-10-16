# OPTIONAL-015: AI scientific-state prior

Type: optional AI rule. Planned screening default: **disabled**.

| Metadata | Value |
|---|---|
| Rule version | `draft-1` |
| Phase | declared by each registered model specification |
| Requires rules | declared, versioned and cycle-checked per model |
| Consumes | model-manifest-declared immutable facts/candidates |
| Produces | model-manifest-declared proposals or rank adjustments |
| Required providers/data | model runtime, artifact, model card and calibration manifest |
| Review state | `DRAFT` |
| Owner approval | pending model and scientific review |
| Primary basis | model card, held-out calibration and applicability-domain report required |
| Applicability artifact | no approved model artifact or digest yet |
| Candidate domain scope | not applicable unless a future model proposes candidates; then model manifest must declare it |

## Theory and basis

A calibrated model may rank charge, oxidation-state or spin candidates inside
its declared training/applicability domain. A prediction is optional evidence;
it cannot relax conservation or backend constraints.

## Implementation

- Implement the common `InferenceRule` interface with immutable features and
  outcomes.
- Record model repository/revision, artifact SHA-256, feature schema, units,
  training domain, calibration set and license.
- Consume and rank an explicit candidate space; never emit an unchecked raw
  integer directly to a backend.
- Out-of-domain, missing model or high uncertainty returns abstention/error
  according to the caller's explicit backend policy, never a silent heuristic.
- Keep training code/data outside the inference package.

## Acceptance

- Every proposal passes the same hard validators as empirical rules.
- Held-out calibration reports coverage, error and abstention separately.
- Model absence in an explicitly requested AI mode is visible.
- CPU/GPU inference parity and deterministic serialization are tested.
- Disabling/removing the model leaves non-AI behavior unchanged.
