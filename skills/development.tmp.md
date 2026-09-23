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

### Amend `DEV-GOV-001`: explicit authorization for normative integration

- **Scope:** governance of `development.md` and `development.en.md`.
- **Rule:** proposed rules remain in `development.tmp.md`. The normative Chinese
  and English documents may be modified only when the user explicitly says
  `将暂存规则整理并更新development`.
- **Boundary:** requests to record, add, refine, or stage a development rule do
  not by themselves authorize normative integration; they update only the
  staging ledger unless the exact integration instruction is present.
- **Rationale:** prevent a discussion-stage rule from being promoted into the
  active repository contract without an explicit user decision.
- **Evidence:** explicit user instruction on 2026-09-23. The staging commit is
  recorded when this amendment is integrated.

## Most recent integration receipt

This receipt records the amendment staged by commit `882d9dd` and retains the
active provenance from commit `339c53e`. It will be replaced, not appended to,
during the next integration cycle.

| Rule | Scope | Integrated | Source commits |
|---|---|---|---|
| `DEV-COMP-001` | Repository-wide evidence-based compatibility | 2026-09-23 | `339c53e`; `0e90c63`, `c604ef5`, `c5e53c9`, `f5a66a7`, `15cb23c` |
| `DEV-ARCH-001` | Mathematical facts, scientific semantics, control, recording, and presentation | 2026-09-23 | `339c53e`, `882d9dd`; `aa72c67`, `b8dc637`, `5e344ad`, `bf67119`, `ebb4a53`, `dc3fc2b` |
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
- Numerical tolerances remain with mathematical facts; chemical or physical
  acceptance thresholds remain with the scientific module that owns them.
- Detailed test matrices and force-field-specific constants remain in their
  module documentation.
