# Backend execution-selection policies

These policies run after scientific-state resolution and before or during
backend projection. They may rank legal execution representations using exact
backend context, but they do not create scientific facts.

- [EXECUTION-001: exact-context backend history](EXECUTION-001-exact-context-history.md)
- [DEFAULT-004: lowest legal xTB UHF fallback](../rules/DEFAULT-004-lowest-legal-xtb-uhf.md)

For xTB projection, a HARD-013-valid explicit backend UHF has highest authority
only when scientific spin is unresolved or the request is compatible with its
declared projection. Exact-context evidence may rank only projections
compatible with the selected scientific path. A selected molecular $2S$ state
is projected under the declared convention; incompatible explicit UHF returns
`PROJECTION_CONFLICT`, and DEFAULT-004 is used only when none of the valid
routes supplies UHF.
