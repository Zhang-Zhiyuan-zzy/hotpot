# Stage record catalog

The main plan and normative companions remain in the project root (Plan21
remains in its own project). This catalog organizes Stage0–13. Existing execution records cover Stage0–3;
Stage4–13 currently have navigation indexes only. Folder creation supplies no
stage-entry or execution evidence and changes no objective, gate, contract,
scientific route or accepted decision.

## Markdown types

Every Markdown file below `stages/` uses one of these exact filenames. A stage
needs only the types for which it has real content. Repeat a type under a named
topic directory; do not overwrite another review or add numbered filename
variants.

| Filename | Meaning | Read when |
|---|---|---|
| `README.md` | Stage/topic navigation and a complete record index | Selecting the stage and locating all Markdown/JSON records |
| `entry_review.md` | Stage/sub-stage entry assessment of documents, current source, prerequisites and bounded scope | Before entering that stage or separately gated work |
| `review.md` | Design, implementation or contract review for one bounded topic | Before its implementation or to inspect its settled boundary |
| `decisions.md` | Exact accepted decisions and explicitly marked open proposals | Before changing a dependent public or private contract |
| `report.md` | Narrative of executed validation, observations and remaining gates | Assessing what actually ran and its coverage limits |

JSON names follow their record's purpose; entry/evidence records commonly use
`entry_review.json` and `evidence.json`. Every JSON file must be linked from its
stage README, including JSON in topic directories. Catalog-wide JSON files
are linked here. A Markdown type need not be duplicated as JSON or created
merely to fill a directory.

## Stages

| Stage | Index | Scope of existing records |
|---|---|---|
| Stage0 | [stage00](stage00/README.md) | Reviewed entry, route and project-state assessment |
| Stage1 | [stage01](stage01/README.md) | API/assets/interfaces, accepted decisions, baseline sampling and evidence |
| Stage2 | [stage02](stage02/README.md) | Dead/placeholder cleanup entry and execution evidence |
| Stage3 | [stage03](stage03/README.md) | Foundation, storage, identities, closed updates/builders, table cutover and native entry |
| Stage4 | [stage04](stage04/README.md) | Planned: Extract mathematical rules and structure adapters |
| Stage5 | [stage05](stage05/README.md) | Planned: Extract elemental, bonding, charge, and hydrogen rules |
| Stage6 | [stage06](stage06/README.md) | Planned: Split stereochemistry facts and rules |
| Stage7 | [stage07](stage07/README.md) | Planned: Remove second truth sources and clarify results |
| Stage8 | [stage08](stage08/README.md) | Planned: Split rule policy, workflow, and backend execution |
| Stage9 | [stage09](stage09/README.md) | Planned: Split the carriers and hard-cut `core.py` |
| Stage10 | [stage10](stage10/README.md) | Planned: Complete physical package and native build migration |
| Stage11 | [stage11](stage11/README.md) | Planned: Clean outer commands and data ownership |
| Stage12 | [stage12](stage12/README.md) | Planned: Explicit exports, documentation, and final hardening |
| Stage13 | [stage13](stage13/README.md) | Planned: kekulize/aromatic experimental rule-interface protocol |

Read the current main plan and normative companions as required by the plan,
then the selected stage README, its entry assessment, relevant review/decisions
and evidence. Use the root verification map and implementation log for current
execution requirements and recorded outcomes. This catalog is navigation, not
a second source of stage-completion or approval semantics.

## Historical paths and actual layout checks

- [Path migration index](path_migration.json): all28 former stage filenames,
  current homes and original/relocated document identities.
- [Executed layout evidence](layout_evidence.json): actual naming/index/link/
  JSON audit and byte-preservation results for this mechanical migration.
- [Read-only audit tool](../audit_stage_records.py): run
  `python plan/20_fact_rule_architecture/audit_stage_records.py` from the
  repository root; `--output <path>` retains its JSON result.
  `--check-migration-bytes` additionally checks the original JSON bytes,
  historical Markdown command/digest text and prior log prefix against the
  migration checkpoint. It is an archival check, not a requirement to freeze
  every current document forever.

Executed commands, argv/cwd, captured source/worktree paths, artifact names and
hash-bound evidence identities remain verbatim. Existing JSON snapshots are
moved byte for byte. Historical Markdown command fences, identity tables and
the existing append-only implementation-log entries are preserved. Consult the
migration index when a historical checkout filename has moved; use the stated
historical source commit or retained artifact for its original bytes. Active
navigation and maintained root-ledger references point to current locations.
No alias files or invented reruns are introduced.

## Root boundary

The root keeps the main plan, chemical-type interface protocol, Plan17/20
alignment and data-object scope audit, plus the separate Plan21 companion's
existing location. Cross-stage verification/performance/Python manifests,
architecture/interface/data-object ledgers, append-only implementation log,
baseline inventory and shared runners/auditors remain at the root. Stage-owned
reviews, entry snapshots, sampling records and execution evidence live under
their stage. New stage records follow this same boundary.
