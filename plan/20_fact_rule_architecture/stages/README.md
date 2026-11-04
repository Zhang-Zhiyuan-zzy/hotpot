# Stage record catalog

The main plan and normative companions remain in the project root; Plan21
remains separate. This catalog organizes Stage0–15. Stage0–12 main-scope delivery
is [root accepted](stage12/final-delivery/validation/main-scope-acceptance.json).
Stage13A/B and the owner-directed Stage14A–E/Stage15A–F branches are accepted
within their declared domains; their [final combined delivery](stage15/integration/report.md)
is also accepted. Stage-owned historical checkpoints retain their original
source, runtime, scientific and support boundaries; their former remaining-work
statements do not override the current acceptance records.

## Markdown types

Every Markdown file below `stages/` uses one of these exact filenames or an
explicit owner-relative path registered below. A stage
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
| `stage07/constraints/contracts/constraint_consolidation_design.md` | Existing named record at its exact owning path | Reading that bounded design or reference record |
| `stage07/preentry-hydrogen/consumer_matrix.md` | Existing named record at its exact owning path | Reading that bounded design or reference record |
| `stage07/preentry-selections/current_source_20261103.md` | Existing named record at its exact owning path | Reading that bounded design or reference record |
| `stage07/topology-selection/contracts/hidden_topology_design.md` | Existing named record at its exact owning path | Reading that bounded design or reference record |
| `stage08/scientific-state/validation/confidence_report.md` | Existing named record at its exact owning path | Reading that bounded design or reference record |
| `stage08/scientific-state/validation/diagnostics_report.md` | Existing named record at its exact owning path | Reading that bounded design or reference record |
| `stage08/scientific-state/validation/h_presence_report.md` | Existing named record at its exact owning path | Reading that bounded design or reference record |
| `stage08/scientific-state/validation/lewis_report.md` | Existing named record at its exact owning path | Reading that bounded design or reference record |
| `stage15/operation_table.md` | Existing named record at its exact owning path | Reading that bounded design or reference record |
| `stage15/space_group_catalogue.md` | Existing named record at its exact owning path | Reading that bounded design or reference record |

JSON names follow their record's purpose; entry/evidence records commonly use
`entry_review.json` and `evidence.json`. Every JSON file must be linked from its
stage README, including JSON in topic directories. Other stage assets follow
the same stage ownership and README indexing rule. Catalog-wide JSON files
are linked here. A Markdown type need not be duplicated as JSON or created
merely to fill a directory.

## Stages

| Stage | Index | Scope of existing records |
|---|---|---|
| Stage0 | [stage00](stage00/README.md) | Reviewed entry, route and project-state assessment |
| Stage1 | [stage01](stage01/README.md) | API/assets/interfaces, accepted decisions, baseline sampling and evidence |
| Stage2 | [stage02](stage02/README.md) | Dead/placeholder cleanup entry and execution evidence |
| Stage3 | [stage03](stage03/README.md) | Foundation, storage, identities, closed updates/builders, table cutover and native entry |
| Stage4 | [stage04](stage04/README.md) | Completed owning gates, validation and focused comparison; final global performance remains open |
| Stage5 | [stage05](stage05/README.md) | Completed owning technical gates,finite H-default correction,retained failures and open final performance/science |
| Stage6 | [stage06](stage06/README.md) | Completed owning gates; C62 delivered/remote-verified; C61 remains validated runtime source |
| Stage7 | [stage07](stage07/README.md) | Stage7A–E accepted within recorded CP311 scopes; final delivery/version/resource obligations remain |
| Stage8 | [stage08](stage08/README.md) | Stage8A–E accepted within recorded CP311 scopes; later/global obligations remain |
| Stage9 | [stage09](stage09/README.md) |9A–C semantic/physical/collection migrations accepted at recorded CP311source/installed boundaries |
| Stage10 | [stage10](stage10/README.md) | Native/build migration and final affected five-ABI/SDKdelivery accepted |
| Stage11 | [stage11](stage11/README.md) | Commands/reference/query ownership delivered; optional research-runtime limits retained |
| Stage12 | [stage12](stage12/README.md) | Main-scope final report/15criteria accepted with explicit limitations |
| Stage13 | [stage13](stage13/README.md) | Accepted: private aromatic/Kekulé definition and integration, source/default/five-ABI delivery and retained paired costs |
| Stage14 | [stage14](stage14/README.md) | Accepted: schema2 ordinary PyG, molecular/Crystal training and inspection, CPU/CUDA/support/cost delivery |
| Stage15 | [stage15](stage15/README.md) | Accepted: exact Crystal/SpaceGroup/SymmUnit, generation, geometry/topology and current CIF; [final combined delivery accepted](stage15/integration/report.md) |

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
- [Follow-up layout evidence](layout_followup_evidence.json): current Stage0–13
  naming, complete asset indexing and historical-byte checks; records any
  unresolved links from concurrent source migration.
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
