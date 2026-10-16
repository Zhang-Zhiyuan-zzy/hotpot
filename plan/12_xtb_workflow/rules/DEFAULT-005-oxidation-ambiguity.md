# DEFAULT-005: oxidation-state ambiguity is not a universal blocker

Type: orchestration decision policy.
Runtime registry: `StatePolicyRegistry`.

## Theory and basis

GFN-FF and GFN1/2-xTB consume total charge and, for GFN1/2, UHF; they do not
require a unique formal oxidation-state assignment.

## Implementation

- Preserve `AMBIGUOUS` or `UNRESOLVED` oxidation results.
- Read operation-required quantities from the registered
  `ScientificOperationRequirements`; never infer them from whether a backend
  has already been supplied.
- Permit backend projection when every value consumed by that backend belongs
  to the same selected coupled path and that path is hard-valid.
- Block only a rule/projection that explicitly needs a unique OS.
- This permissive backend fact does not override a named application profile,
  such as `COORDINATION_SCREENING_V1`, that explicitly requires an OS-bearing
  path for its own decision contract.
- If an enabled profile preference needs OS to choose charge or spin, retain
  only complete compatible OS-bearing paths or decide `NO_AUTOMATIC_STATE`;
  do not bypass the preference with an independently chosen field.
- Never convert missing OS into zero or a common metal state.

## Acceptance

- Legal explicit $(Q,u)$ can launch xTB with unresolved OS.
- An OS-dependent optional spin rule abstains when OS is unresolved.
- The unresolved OS remains in provenance and reports.
- A backend that explicitly requires OS still blocks appropriately.
