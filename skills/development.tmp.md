# Development rule staging ledger

> Non-normative working file. New rules discovered through conversation, code
> review, or failure analysis are staged here before they enter the normative
> Chinese and English development standards.

## Workflow

1. Add a proposed rule with a stable ID, scope, rationale, and evidence commits.
2. Decide whether it is repository-wide or belongs to a module contract.
3. Commit the staged batch so the integration has a stable source hash.
4. Rebuild `development.md` and `development.en.md` together: consolidate,
   reorder, and delete obsolete text instead of appending indefinitely.
5. Record the integration date, staging commit, and evidence commits in both
   normative documents.
6. Replace this file with an empty inbox and the latest compact receipt.

## Inbox

### Amend `DEV-ARCH-001`: mathematical facts versus scientific semantics

- **Scope:** repository-wide architecture and module ownership.
- **Rule:** mathematical modules such as geometry MUST report mathematical
  objects, measurements, relations, degeneracy, numerical tolerance, and
  uncertainty. They MUST NOT decide chemical or physical reasonableness,
  realism, quality, applicability, or repair policy.
- **Boundary:** mathematical implementation and scientific interpretation
  SHOULD be separated in code. A threshold that exists for numerical stability
  belongs to the mathematical layer; a threshold or score that expresses a
  chemical or physical standard belongs to the chemistry, force-field, or
  other scientific layer that owns that interpretation.
- **Rationale:** prevent scientific policy from being hidden in reusable
  mathematical infrastructure and prevent mathematical facts from being
  presented as scientific conclusions.
- **Evidence:** clarification of the existing `DEV-ARCH-001` contract during
  the 2026-09-23 force-field and geometry architecture review. The staging
  commit for this amendment is recorded when it is integrated.

## Most recent integration receipt

This receipt replaces the full staged proposals recorded by commit `339c53e`.
It will be replaced, not appended to, during the next integration cycle.

| Rule | Scope | Integrated | Source commits |
|---|---|---|---|
| `DEV-COMP-001` | Repository-wide evidence-based compatibility | 2026-09-23 | `339c53e`; `0e90c63`, `c604ef5`, `c5e53c9`, `f5a66a7`, `15cb23c` |
| `DEV-ARCH-001` | Facts, evaluation, control, recording, and presentation | 2026-09-23 | `339c53e`; `aa72c67`, `b8dc637`, `5e344ad`, `bf67119`, `ebb4a53`, `dc3fc2b` |
| `DEV-ERR-001` | Explicit failure semantics and retained evidence | 2026-09-23 | `339c53e`; `43b83d9`, `186d7b4`, `1ea39cc`, `81fd33b` |
| `DEV-OBS-001` | Optional, cost-aware scientific history and persistence | 2026-09-23 | `339c53e`; `81e02de`, `3631c7c`, `5795d8f`, `ebb4a53`, `dc3fc2b` |
| `DEV-MOD-001` | Explicit public module surface and source layout | 2026-09-23 | `339c53e`; `e789e7a`, `d08792a`, `f26ca0f`, `6b8755e`, `ba67c91` |
| `DEV-CLEAN-001` | Removal of superseded internal paths | 2026-09-23 | `339c53e`; `08639d7`, `87277f0`, `c1ace9f`, `f5a66a7`, `15cb23c` |
| `DEV-GOV-001` | Staging and holistic bilingual integration | 2026-09-23 | `339c53e`; `df78c7c`, `6e2ebac`, `a34375b`, `58a43df` |

## Scope decisions for this batch

- The force-field working-copy transaction was not generalized. It belongs in a
  force-field-specific contract before being treated as normative behavior.
- Full trajectory capture is not a default requirement. The integrated rule
  requires an explicit retention decision and resource assessment.
- Detailed test matrices and force-field-specific constants remain in their
  module documentation.
